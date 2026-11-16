"""03 — 입력 사진과 생성 mesh 렌더링을 참고해 UV texture/geometry를 보정한다.

읽는 순서: RefinementNet -> constrain_geometry -> refinement_loss -> validate -> train.
FiCA C.3의 실제 Diffusers UNet2DConditionModel + frozen Sapiens ViT를 사용한다.
Photo/mask/keypoint와 texture/Laplacian/normal regularization의 gradient를 모두 보존한다.
Geometry는 원래 FLAME topology로 복원하고 tracking pose adapter를 통해 render한다.

CUDA_VISIBLE_DEVICES=0 python 03_refine_uv.py --stage cache
CUDA_VISIBLE_DEVICES=0 python 03_refine_uv.py --stage train --steps 1000 --init-from checkpoints/03/latest.pt
CUDA_VISIBLE_DEVICES=0 python 03_refine_uv.py --stage predict
"""

import importlib.util
import math
from pathlib import Path
import time

from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

spec = importlib.util.spec_from_file_location("fica_00", Path(__file__).with_name("00_prepare_nersemble.py"))
L00 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L00)


class RefinementNet(nn.Module):
    """입력은 UV 6ch, condition은 이미지 feature tokens. 출력은 보정된 UV 6ch.

    512 -> 256 -> 128은 convolution, 64에서는 self/cross-attention을 쓴다.
    Skip connection으로 고해상도 정보를 복원하여 최종 출력도 512다. 입력 UV를
    64로 줄이거나 residual을 단순 확대하는 구조가 아니다.
    정확한 FiCA 채널/attention 배치/출력 parameterization은 미공개인 구현 선택이다.
    """
    def __init__(self, config):
        super().__init__()
        from diffusers import UNet2DConditionModel
        settings = config["refinement"]
        channels = settings["channels"]
        stages = len(channels)
        size = config["uv_size"]
        maximum = settings.get("attention_max_size", 64)
        if stages < 1 or size % 2**(stages-1) or size // 2**(stages-1) > maximum:
            raise ValueError("U-Net의 가장 낮은 해상도가 attention_max_size 이하가 되어야 한다.")
        self.uv_size = size
        self.resolutions = [size // 2**index for index in range(stages)]
        self.attention_resolutions = [value for value in self.resolutions if value <= maximum]
        # Diffusers 기본 배치는 latent 해상도에 적합하다. UV512에서 그대로 사용하면
        # 첫 self-attention의 query 길이가 262,144가 된다. 다운샘플 후에만 배치한다.
        down = ["CrossAttnDownBlock2D" if value <= maximum else "DownBlock2D"
                for value in self.resolutions]
        up = ["CrossAttnUpBlock2D" if value <= maximum else "UpBlock2D"
              for value in reversed(self.resolutions)]
        self.unet = UNet2DConditionModel(
            in_channels=6, out_channels=6, block_out_channels=tuple(channels),
            down_block_types=tuple(down), up_block_types=tuple(up),
            layers_per_block=settings["layers_per_block"], cross_attention_dim=1536,
            attention_head_dim=settings["attention_head_dim"], norm_num_groups=32)
        # 같은 encoder의 reference/render features라도 역할을 식별할 수 있게 한다.
        self.feature_type = nn.Parameter(torch.zeros(2, 1536))
        nn.init.normal_(self.feature_type, std=.01)
        nn.init.zeros_(self.unet.conv_out.weight)
        nn.init.zeros_(self.unet.conv_out.bias)
        self.unet.enable_gradient_checkpointing()

    def forward(self, texture, xyz, reference_features, render_features, mean, std):
        if texture.ndim != 4 or texture.shape[1] != 3 or texture.shape != xyz.shape or texture.shape[-2:] != (self.uv_size, self.uv_size):
            raise ValueError("texture/XYZ는 config uv_size의 동일한 Bx3xHxW여야 한다.")
        uv = torch.cat((texture*2-1, (xyz-mean)/std), 1)
        reference = reference_features + self.feature_type[0][None, None]
        rendered = render_features + self.feature_type[1][None, None]
        # Diffusion sampling이 아니다. 시간 embedding은 상수 0을 사용한 feed-forward U-Net이다.
        residual = self.unet(uv, torch.zeros(uv.shape[0], device=uv.device), encoder_hidden_states=torch.cat((reference, rendered), 1)).sample
        # 학습 중 hard clamp로 색 gradient가 죽지 않도록 한다. PNG 저장 시에만 clip한다.
        # Residual=0이면 bit 수준으로 초기 UV를 유지해 작은 삼각형의 수치 잡음을 줄인다.
        return texture+residual[:,:3]/2, xyz+residual[:,3:]*std


def image_features(encoder, rgb):
    # ViT patch16의 배수로 resize한다. 종횡비는 유지한다.
    h, w = rgb.shape[-2:]
    size = (max(32, round(h/16)*16), max(32, round(w/16)*16))
    image = F.interpolate(rgb, size=size, mode="bilinear", align_corners=False)
    return encoder(image).flatten(2).transpose(1, 2)


@torch.no_grad()
def feature_pair(encoder, sample, initial, bridge, context):
    vertices = bridge(initial["xyz"])
    posed = L00.posed_from_canonical(vertices, sample)
    h, w = sample["rgb"].shape[-2:]
    rendered = L00.render_mesh(posed, bridge.faces, bridge.vt, bridge.ft, initial["texture"], sample["K"], sample["w2c"], h, w, context)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reference = image_features(encoder, sample["rgb"][None])
        features = image_features(encoder, rendered["rgb"][None])
    return reference.float(), features.float(), rendered


def constrain_geometry(xyz, initial_xyz, bridge):
    """고정된 미분 가능 연산: 보정 mesh의 중심 = 초기 mesh의 중심.

    UV texel의 단순 평균은 mesh vertex의 평균과 다르다. MeshBridge의 선형부를 B라
    하면 vertex 중심의 변화는 mean(B @ delta_UV)다. 한 번 계산한 계수 c로
    이 평균을 구하고, 모든 유효 texel에서 같은 translation을 뺀다.
    이 과정은 3개 translation 자유도만 제거하며, 개인 GT geometry를 사용하지 않는다.
    """
    if not hasattr(bridge, "refinement_center_weights"):
        with torch.no_grad():
            count = len(bridge.template)
            average = bridge.factor.new_full((count, 1), 1/count)
            inverse_average = torch.cholesky_solve(average, bridge.factor)[:, 0]
            weights = (inverse_average[bridge.corner_ids]*bridge.weights).sum(-1)
            gain = weights.sum()
            if not torch.isfinite(gain) or gain.abs() < 1e-6:
                raise ValueError("MeshBridge의 translation response를 계산할 수 없다.")
            bridge.register_buffer("refinement_center_weights", weights/gain)
    delta = xyz-initial_xyz[None]
    shift = (delta[:, :, bridge.valid]*bridge.refinement_center_weights).sum(-1)
    return initial_xyz[None] + (delta-shift[:, :, None, None])*bridge.valid[None, None]


def refine(config, model, initial, features, mean, std, bridge):
    """학습/validation/predict에서 동일한 Network + 고정 geometry 제약을 사용한다."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        texture, xyz = model(initial["texture"][None], initial["xyz"][None],
                             features["reference"], features["rendered"], mean, std)
    texture, xyz = texture.float(), xyz.float()
    if config["refinement"].get("objective", "legacy") == "surface_refinement_v2":
        xyz = constrain_geometry(xyz, initial["xyz"], bridge)
    return texture, xyz


def refinement_loss(config, texture, xyz, initial, identity, sample, bridge, context):
    settings = config["refinement"]
    initial_vertices = bridge(initial["xyz"]).detach()
    vertices = bridge(xyz[0])
    posed = L00.posed_from_canonical(vertices, sample)
    h, w = sample["rgb"].shape[-2:]
    rendered = L00.render_mesh(posed, bridge.faces, bridge.vt, bridge.ft, texture[0], sample["K"], sample["w2c"], h, w, context)
    # Fitting target은 mesh가 관측되는 영역이다. FLAME로 hair 전체를 설명하도록 강요하지 않는다.
    target_alpha = sample["labels"]["valid"].float()
    expected = sample["rgb"] * target_alpha + 1 - target_alpha
    landmarks=(posed[bridge.faces[sample["landmark_faces"]]]*sample["landmark_bary"][...,None]).sum(1)
    keypoints, depth = L00.elite(3).project(landmarks, sample["K"], sample["w2c"])
    valid_kp = sample["keypoint_valid"] & (depth > .01)
    if valid_kp.any():
        keypoint = (keypoints - sample["keypoints"]).abs()[valid_kp].mean()  # Eq.2의 pixel 단위 L1
    else:
        keypoint = keypoints.sum()*0
    valid = identity["valid"][None, None]
    delta_texture = texture-initial["texture"][None]
    photo_full = F.l1_loss(rendered["rgb"], expected)
    repaired = settings.get("objective", "legacy") == "surface_refinement_v2"
    if repaired:
        # 배경의 쉬운 흰 pixel로 얼굴 오차가 희석되지 않게 관측 얼굴 영역에서 평균한다.
        photo = ((rendered["rgb"]-sample["rgb"]).abs().mean(0, keepdim=True)*target_alpha).sum()/target_alpha.sum().clamp_min(1)
        # 기존 pixel별 vector norm은 작은 delta에서도 기울기가 일정했다. 제곱 오차는
        # delta -> 0일 때 규제 기울기도 작아져 색상 보정을 가로막는 현상을 줄인다.
        texture_reg = (delta_texture.square()*valid).sum()/(3*valid.sum().clamp_min(1))
    else:
        photo = photo_full
        texture_reg = (delta_texture.norm(dim=1,keepdim=True)*valid).sum()/valid.sum().clamp_min(1)
    laplacian = (L00.cotangent_laplacian_values(vertices,bridge.faces)-L00.cotangent_laplacian_values(initial_vertices,bridge.faces)).norm()/math.sqrt(len(vertices))
    normals = L00.vertex_normals(vertices, bridge.faces)
    initial_normals = L00.vertex_normals(initial_vertices, bridge.faces)
    normal_reg = (normals-initial_normals).norm(dim=-1).mean()
    scale = settings.get("geometry_anchor_scale", .01)  # 현재 FLAME world 단위에서 1 cm.
    geometry_reg = ((vertices-initial_vertices)/scale).square().sum(-1).mean()
    terms = {"photo": photo, "photo_full": photo_full, "mask": F.l1_loss(rendered["alpha"], target_alpha),
             "keypoint": keypoint, "texture_reg": texture_reg, "laplacian_reg": laplacian,
             "normal_reg": normal_reg, "geometry_reg": geometry_reg}
    total = settings["photo"]*terms["photo"] + settings["mask"]*terms["mask"] + settings["keypoint"]*keypoint
    if repaired:
        total = total + settings["regularization"]*(settings["texture_reg_weight"]*texture_reg
            + settings["laplacian_reg_weight"]*laplacian + settings["normal_reg_weight"]*normal_reg
            + settings["geometry_reg_weight"]*geometry_reg)
    else:
        total = total + settings["regularization"]*(texture_reg+laplacian+normal_reg)
    return total, terms, rendered


def read_initial(data, row, device):
    return L00.load_tensor(data.path / "generated" / f"{Path(row['file']).stem}.pt", device)


def read_features(data, row, device):
    """02의 생성 결과가 바뀌면 train/validation/predict 모두 cache를 거부한다."""
    stem = Path(row["file"]).stem
    features = L00.load_tensor(data.path / "refine_features" / f"{stem}.pt", device)
    if features["initial_sha256"] != L00.sha256(data.path / "generated" / f"{stem}.pt"):
        raise ValueError("생성 mesh 이후 feature cache가 stale이다. 03 --stage cache를 다시 실행하세요.")
    return features


def stats(data, device):
    return (torch.tensor(data.manifest["xyz_mean"], device=device)[None, :, None, None],
            torch.tensor(data.manifest["xyz_std"], device=device)[None, :, None, None])


def run_paths(config, run_name=None):
    out, checkpoints = L00.work_root(config)/"outputs/03", L00.work_root(config)/"checkpoints/03"
    if run_name is not None:
        if not run_name or run_name in (".", "..") or Path(run_name).name != run_name:
            raise ValueError("run-name은 한 개의 폴더 이름이어야 한다.")
        out, checkpoints = out/run_name, checkpoints/run_name
    return out, checkpoints


def load_refiner(path, data, device, config=None):
    """Warm start/추론은 weight만 읽는다. optimizer resume는 엄격한 L00.restore를 쓴다.

    학습 loss나 코드 설명이 바뀐 뒤에도 기존 비교 모델을 읽을 수 있다. Dataset과
    Network parameter keys/shapes는 검사하고, source config/hash/step을 기록한다.
    """
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if state["format"] != "fica_checkpoint_v1" or state["data_sha256"] != L00.sha256(data.path/"manifest.json"):
        raise ValueError("refiner checkpoint format/dataset 불일치")
    settings = config or state["config"]
    if settings["uv_size"] != state["config"]["uv_size"]:
        raise ValueError("refiner checkpoint UV size 불일치")
    with torch.device("meta"):
        model = RefinementNet(settings)
    model.load_state_dict(state["model"], strict=True, assign=True)
    model.to(device)
    info = {"path": str(path.resolve()), "step": state["step"], "source_config": state["config"],
            "code_hashes": state["code_hashes"], "data_sha256": state["data_sha256"], "extra": state.get("extra", {})}
    return model, info


def geometry_metrics(xyz, initial, identity, bridge):
    """절대 위치 이동과 실제 형태 변화, coarse pseudo target 오차를 함께 확인한다."""
    vertices, before = bridge(xyz), bridge(initial["xyz"])
    delta = vertices-before
    shift = delta.mean(0)
    energy = delta.square().sum(-1).mean()
    return {"vertex_rmse": float((vertices-identity["canonical_vertices"]).square().sum(-1).mean().sqrt()),
            "centroid_shift": float(shift.norm()), "vertex_change_rms": float(energy.sqrt()),
            "shape_change_rms": float((delta-shift).square().sum(-1).mean().sqrt()),
            "translation_energy_fraction": float(shift.square().sum()/energy.clamp_min(1e-12))}


def cache_features(data, device):
    import nvdiffrast.torch as dr
    encoder = L00.lesson(1).pretrained_encoder(device)
    context = dr.RasterizeCudaContext(device=device)
    bridges = {}
    for row in tqdm(data.manifest["samples"], desc="03 Sapiens feature cache", unit="frame", dynamic_ncols=True):
        key = row["identity"]
        if key not in bridges:
            bridges[key] = data.bridge(key,device)
        sample, initial = data.sample(row, device), read_initial(data, row, device)
        reference, features, rendered = feature_pair(encoder, sample, initial, bridges[key], context)
        stem = Path(row["file"]).stem
        L00.save_tensor(data.path / "refine_features" / f"{stem}.pt", {"reference": reference.cpu(), "rendered": features.cpu(),
                        "source": row["file"], "initial_sha256": L00.sha256(data.path / "generated" / f"{stem}.pt")})
        L00.save_panels(L00.work_root(data.manifest["config"]) / "outputs/03" / f"{stem}_before.jpg", [sample["rgb"], rendered["rgb"]], ["Reference", "Generated mesh render"])
    del encoder, bridges
    torch.cuda.empty_cache()


def validation_rows(data, limit):
    """고정된 held-out frame. 0이면 전체 validation split을 평가한다."""
    rows = data.samples("validation")
    if not rows:
        raise ValueError("validation split이 없다. 00에서 held-out frame을 준비하세요.")
    return rows[:limit] if limit else rows


def save_validation_panel(path, images, titles, step, row, metrics, baseline, frame_count, kind="validation"):
    """한 step = JPEG 한 장. 세로 사진을 찌그러뜨리거나 얼굴을 crop하지 않는다."""
    side, header, label, columns = 300, 108, 28, 3
    rows = math.ceil(len(images)/columns)
    canvas = Image.new("RGB", (columns*side, header+rows*(side+label)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"03 | optimizer step {step} | {row['split']} {Path(row['file']).stem}", fill="white")
    draw.text((10, 27), f"{kind}: {frame_count} frame(s) | photo=foreground for surface_refinement_v2", fill="white")
    draw.text((10, 46), f"Mean photo L1: {baseline['photo']:.5f} -> {metrics['photo']:.5f} | "
              f"mask L1: {baseline['mask']:.5f} -> {metrics['mask']:.5f}", fill="white")
    draw.text((10, 65), f"Keypoint pixels: {baseline['keypoint']:.3f} -> {metrics['keypoint']:.3f} | "
              f"vertex RMSE: {baseline['vertex_rmse']:.5f} -> {metrics['vertex_rmse']:.5f}", fill="white")
    draw.text((10, 84), f"Centroid shift={metrics.get('centroid_shift', 0):.3g} | "
              f"shape change RMS={metrics.get('shape_change_rms', 0):.4g} | texture change={metrics.get('texture_change', 0):.4g}", fill="white")
    for index, (value, title) in enumerate(zip(images, titles)):
        value = value.detach().float().cpu().expand(3, -1, -1)
        array = (value.clamp(0, 1).permute(1, 2, 0).numpy()*255).round().astype("uint8")
        tile = Image.fromarray(array)
        scale = min(side/tile.width, side/tile.height)
        tile = tile.resize((max(1, round(tile.width*scale)), max(1, round(tile.height*scale))), Image.Resampling.BILINEAR)
        x, y = index%columns*side, header+index//columns*(side+label)
        draw.text((x+5, y+6), title, fill="white")
        canvas.paste(tile, (x+(side-tile.width)//2, y+label+(side-tile.height)//2))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)


@torch.no_grad()
def validate(config, data, model, device, context, rows, out, step):
    """같은 frame/초기 UV를 비교한다. GT UV를 입력하지 않고 학습 RNG/모드를 보존한다."""
    if not rows:
        raise ValueError("validation frame이 없다. Train frame으로 대체하지 않는다.")
    mean, std = stats(data, device)
    totals, baseline, preview = {}, {}, None
    was_training = model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=[device]):
            for row in rows:
                sample, initial = data.sample(row, device), read_initial(data, row, device)
                features = read_features(data, row, device)
                identity, bridge = data.identity(row["identity"], device), data.bridge(row["identity"], device)
                texture, xyz = refine(config, model, initial, features, mean, std, bridge)
                loss, terms, rendered = refinement_loss(config, texture, xyz, initial, identity, sample, bridge, context)
                before_loss, before_terms, before = refinement_loss(config, initial["texture"][None], initial["xyz"][None], initial, identity, sample, bridge, context)
                if not all(torch.isfinite(value).all() for value in (texture, xyz, loss, before_loss)):
                    raise FloatingPointError(f"03 validation NaN/Inf: {row['file']}")
                values = {"loss": float(loss), **{key: float(value) for key, value in terms.items()}}
                before_values = {"loss": float(before_loss), **{key: float(value) for key, value in before_terms.items()}}
                # FLAME pseudo target과의 거리는 진단용이며 03의 학습 loss에 추가하지 않는다.
                for position, metrics in ((xyz[0], values), (initial["xyz"], before_values)):
                    metrics.update(geometry_metrics(position, initial, identity, bridge))
                values["texture_change"] = float((texture[0]-initial["texture"])[:, identity["valid"]].abs().mean())
                before_values["texture_change"] = 0.
                for key, value in values.items():
                    totals[key] = totals.get(key, 0.) + value
                    baseline[key] = baseline.get(key, 0.) + before_values[key]
                if preview is None:
                    valid = identity["valid"][None].float()
                    alpha = sample["labels"]["valid"].float()
                    expected = sample["rgb"]*alpha + 1-alpha
                    def heatmap(error, maximum):
                        ratio = (error/maximum).clamp(0, 1)
                        return torch.cat((ratio, 1-(2*ratio-1).abs(), 1-ratio), 0)
                    def xyz_color(position):
                        return ((position-mean[0])/(std[0]*6)+.5)*valid
                    displacement = (xyz[0]-initial["xyz"]).square().sum(0, keepdim=True).sqrt()
                    limit = float(std.mean())*.1
                    images = [sample["rgb"], before["rgb"], rendered["rgb"], expected,
                              heatmap((before["rgb"]-expected).abs().mean(0, keepdim=True), .25),
                              heatmap((rendered["rgb"]-expected).abs().mean(0, keepdim=True), .25),
                              initial["texture"]*valid, texture[0]*valid,
                              heatmap((texture[0]-initial["texture"]).abs().mean(0, keepdim=True), .1)*valid,
                              xyz_color(initial["xyz"]), xyz_color(xyz[0]), heatmap(displacement, limit)*valid]
                    titles = ["Reference", "Initial mesh render", "Refined mesh render", "Photo target (FLAME visibility)",
                              "Initial photo error: red >= 0.25", "Refined photo error: same scale",
                              "Initial UV texture", "Refined UV texture", "Texture change: red >= 0.1",
                              "Initial XYZ (fixed display scale)", "Refined XYZ (same scale)", f"XYZ change: red >= {limit:.4g}"]
                    preview = images, titles, row
        metrics = {key: value/len(rows) for key, value in totals.items()}
        baseline = {key: value/len(rows) for key, value in baseline.items()}
        directory = out/"validation"
        save_validation_panel(directory/f"{step}.jpg", *preview[:2], step, preview[2], metrics, baseline, len(rows))
        L00.append_log(directory/"metrics.jsonl", {"step": step, **metrics, "initial": baseline,
            "frames": [row["file"] for row in rows], "preview_frame": preview[2]["file"],
            "scope": data.manifest["scope"], "vertex_target": "coarse FLAME pseudo target; diagnostic only",
            "attention_resolutions": model.attention_resolutions, "objective": config["refinement"].get("objective", "legacy"),
            "photo_metric": "foreground mean" if config["refinement"].get("objective") == "surface_refinement_v2" else "full image mean",
            "photo_full_metric": "full image white-composite L1, comparable with the legacy photo metric"})
        return metrics
    finally:
        model.train(was_training)


def train(config, data, device, steps, resume=False, val_every=None, val_frames=None, run_name=None, init_from=None):
    import nvdiffrast.torch as dr
    settings = config["refinement"]
    val_every = settings.get("val_every", 100) if val_every is None else val_every
    val_frames = settings.get("val_frames", 1) if val_frames is None else val_frames
    checkpoint_every = settings.get("checkpoint_every", 100)
    if steps < 1 or val_every < 1 or val_frames < 0 or checkpoint_every < 1:
        raise ValueError("steps/val_every/checkpoint_every >= 1, val_frames >= 0이어야 한다.")
    rows = validation_rows(data, val_frames)
    out, checkpoints = run_paths(config, run_name)
    path = checkpoints/"latest.pt"
    if init_from is not None and (resume or Path(init_from).resolve() == path.resolve()):
        raise ValueError("init-from은 별도 run을 weight만 초기화하는 용도다. --resume와 함께 사용하지 않는다.")
    if path.exists() and not resume and config.get("profile") != "smoke":
        raise FileExistsError(f"{path}가 있다. --resume 또는 새 --run-name을 사용하세요.")
    if not data.samples("train"):
        raise ValueError("train frame이 없다.")
    # 비싼 모델을 할당하기 전에 train/validation cache의 입력 일치를 확인한다.
    for row in data.samples("train") + rows:
        read_features(data, row, "cpu")
    context = dr.RasterizeCudaContext(device=device)
    parent = None
    if init_from is None:
        model = RefinementNet(config).to(device)
    else:
        model, parent = load_refiner(init_from, data, device, config)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"])
    bridges = {}
    mean, std = stats(data, device)
    start=L00.training_start(path,model,optimizer,config,data,resume)
    if start >= steps:
        raise ValueError(f"이미 {start} step이다. --steps는 추가 횟수가 아닌 총 목표이므로 더 큰 값을 지정하세요.")
    best_score, best_step = math.inf, None
    if resume:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        extra = state["extra"]
        if extra.get("validation_frames") != [row["file"] for row in rows]:
            raise ValueError("resume 중 validation frame을 바꾸면 best score를 비교할 수 없다.")
        best_score, best_step = extra["best_score"], extra["best_step"]
        parent = extra.get("parent")
        del state

    def checkpoint_info():
        return {"best_score": best_score, "best_step": best_step, "selection": "mean validation total loss",
                "validation_frames": [row["file"] for row in rows], "parent": parent, "run_name": run_name,
                "attention_resolutions": model.attention_resolutions}

    def update_best(metrics, step):
        nonlocal best_score, best_step
        if math.isfinite(metrics["loss"]) and metrics["loss"] < best_score:
            best_score, best_step = metrics["loss"], step
            # Best는 추론용 weight만, latest는 optimizer/RNG까지 저장한다.
            L00.checkpoint(checkpoints/"best.pt", model, None, config, data, step, extra=checkpoint_info())
            L00.save_json(out/"best.json", {"step": step, "score": best_score,
                "selection": "mean validation total loss", "metrics": metrics,
                "frames": [row["file"] for row in rows]})
    accumulation=config["training"]["grad_accumulation"]
    with tqdm(total=steps, initial=start, desc="03 UV refinement", unit="step", dynamic_ncols=True, mininterval=1) as bar:
        recent = validate(config, data, model, device, context, rows, out, start)
        update_best(recent, start)
        bar.set_postfix(val_photo=f"{recent['photo']:.5f}", refresh=False)
        for step in range(start,steps):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            mean_loss=0.;mean_terms={}
            for micro in range(accumulation):
                row=data.training_row()
                identity = data.identity(row["identity"], device)
                if row["identity"] not in bridges:
                    bridges[row["identity"]] = data.bridge(row["identity"],device)
                bridge = bridges[row["identity"]]
                sample, initial = data.sample(row, device), read_initial(data, row, device)
                features = read_features(data, row, device)
                texture, xyz = refine(config, model, initial, features, mean, std, bridge)
                # CUDA rasterization / Cholesky solve는 float32로 실행한다.
                loss, terms, _ = refinement_loss(config, texture.float(), xyz.float(), initial, identity, sample, bridge, context)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"03 training loss NaN/Inf at step {step+1}")
                (loss/accumulation).backward()
                values = torch.stack([loss.detach(), *[value.detach() for value in terms.values()]]).cpu().tolist()
                mean_loss += values[0]/accumulation
                for name, value in zip(terms, values[1:]):
                    mean_terms[name] = mean_terms.get(name, 0.) + value/accumulation
            # 수백 개 tensor마다 isfinite -> CPU sync하지 않는다. 전체 grad norm으로
            # NaN/Inf를 검사하면서 clip하고, 상세 통계는 JSONL에만 남긴다.
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_grad"], error_if_nonfinite=True)
            report = {"norm": float(norm), "tensors": sum(p.grad is not None for p in model.parameters())}
            optimizer.step()
            torch.cuda.synchronize(device)
            seconds = time.perf_counter()-started
            L00.append_log(out/"training.jsonl", {"step": step+1, "loss": mean_loss, "terms": mean_terms,
                "gradient": report, "step_seconds": seconds})
            if (step+1) % val_every == 0 or step == steps-1:
                recent = validate(config, data, model, device, context, rows, out, step+1)
                update_best(recent, step+1)
            bar.set_postfix(loss=f"{mean_loss:.5f}", val_photo=f"{recent['photo']:.5f}", refresh=False)
            bar.update(1)
            if (step+1) % checkpoint_every == 0 or step == steps-1:
                L00.checkpoint(path, model, optimizer, config, data, step+1, extra=checkpoint_info())
    return model.eval()


@torch.no_grad()
def predict(config, data, model, device, run_name=None, checkpoint_info=None):
    import nvdiffrast.torch as dr
    mean, std = stats(data, device)
    context = dr.RasterizeCudaContext(device=device)
    out, _ = run_paths(config, run_name)
    destination = data.path/"refined"
    # 첫 quality run의 export 전에 이전 1K 결과를 그대로 보존한다. 이후 04는
    # 기존과 같은 refined/*.pt 경로로 새 결과를 읽으므로 별도 설정이 필요 없다.
    existing = sorted(destination.glob("*.pt")) if destination.exists() else []
    if run_name and config.get("profile") != "smoke" and existing:
        old = L00.load_tensor(existing[0])
        if "refinement_run" not in old:
            backup = data.path/"refined_legacy_1000"
            if backup.exists():
                raise FileExistsError(f"이전 결과 보존 경로가 이미 있다: {backup}")
            destination.rename(backup)
    info = checkpoint_info or {}
    records = []
    for row in tqdm(data.manifest["samples"], desc="03 Export refined UV", unit="frame", dynamic_ncols=True):
        initial = read_initial(data, row, device)
        stem = Path(row["file"]).stem
        features = read_features(data, row, device)
        identity, bridge = data.identity(row["identity"], device), data.bridge(row["identity"], device)
        texture, xyz = refine(config, model, initial, features, mean, std, bridge)
        if not all(torch.isfinite(value).all() for value in (texture, xyz)):
            raise FloatingPointError(f"03 predict NaN/Inf: {row['file']}")
        L00.save_tensor(destination/f"{stem}.pt", {"texture": texture[0].cpu(), "xyz": xyz[0].cpu(), "source": row["file"],
            "refinement_run": run_name, "checkpoint_step": info.get("step"), "checkpoint": info.get("path"),
            "objective": config["refinement"].get("objective", "legacy"), "geometry_constraint": "mesh_centroid_fixed"
            if config["refinement"].get("objective") == "surface_refinement_v2" else "none"})
        sample = data.sample(row, device)
        loss, terms, rendered = refinement_loss(config, texture, xyz, initial, identity, sample, bridge, context)
        before_loss, before_terms, before = refinement_loss(config, initial["texture"][None], initial["xyz"][None], initial, identity, sample, bridge, context)
        metrics = {"loss": float(loss), **{key: float(value) for key,value in terms.items()},
                   **geometry_metrics(xyz[0], initial, identity, bridge),
                   "texture_change": float((texture[0]-initial["texture"])[:, identity["valid"]].abs().mean())}
        baseline = {"loss": float(before_loss), **{key: float(value) for key,value in before_terms.items()},
                    **geometry_metrics(initial["xyz"], initial, identity, bridge), "texture_change": 0.}
        records.append({"file": row["file"], "split": row["split"], "initial": baseline, "refined": metrics})
        valid = identity["valid"][None].float()
        displacement = (xyz[0]-initial["xyz"]).norm(dim=0, keepdim=True)
        images = [sample["rgb"], before["rgb"], rendered["rgb"], initial["texture"]*valid, texture[0]*valid,
                  (displacement/(float(std.mean())*.1)).clamp(0,1).expand(3,-1,-1)*valid]
        save_validation_panel(out/"predictions"/f"{stem}.jpg", images,
            ["Reference", "Initial mesh", "Refined mesh", "Initial UV texture", "Refined UV texture", "XYZ change: white >= 0.1*mean std"],
            info.get("step", "unknown"), row, metrics, baseline, 1, kind="predict")
    summary = {}
    for split in ("train", "validation"):
        subset = [record for record in records if record["split"] == split]
        if subset:
            summary[split] = {kind: {key: sum(record[kind][key] for record in subset)/len(subset)
                                   for key in subset[0][kind]} for kind in ("initial", "refined")}
    L00.save_json(out/"predictions/metrics.json", {"checkpoint": info.get("path"), "step": info.get("step"),
        "run_name": run_name, "objective": config["refinement"].get("objective", "legacy"), "summary": summary, "frames": records})
    return summary


def main():
    parser = L00.arguments(__doc__)
    parser.add_argument("--stage", choices=("cache", "train", "predict", "validate"), default="cache")
    parser.add_argument("--run-name", default="quality_repair", help="학습/평가 결과를 분리한다. 기본 quality_repair")
    parser.add_argument("--init-from", type=Path, help="별도 run의 초기 weight. optimizer/RNG는 새로 시작한다.")
    parser.add_argument("--weights", choices=("best", "latest"), default="best", help="predict/validate에 사용할 checkpoint")
    parser.add_argument("--val-every", type=int, default=None, help="고정 validation 주기. 기본 100 optimizer steps")
    parser.add_argument("--val-frames", type=int, default=None, help="고정 held-out frame 수. config 기본 0=전체")
    args = parser.parse_args()
    config = L00.read_config(args.config, args.smoke)
    device = L00.setup_device(args.device, config["seed"])
    data = L00.PreparedData(args.data)
    if args.stage == "cache":
        cache_features(data, device)
    elif args.stage == "train":
        steps = config["refinement"].get("steps", 1000) if args.steps is None else args.steps
        train(config, data, device, steps, resume=args.resume, val_every=args.val_every, val_frames=args.val_frames,
              run_name=args.run_name, init_from=args.init_from)
    else:
        out, checkpoints = run_paths(config, args.run_name)
        model, info = load_refiner(checkpoints/f"{args.weights}.pt", data, device)
        model.eval()
        # 추론은 weight를 학습할 당시의 objective/geometry 제약을 그대로 사용한다.
        source_config = info["source_config"]
        if args.stage == "predict":
            predict(source_config, data, model, device, args.run_name, info)
        else:
            import nvdiffrast.torch as dr
            context = dr.RasterizeCudaContext(device=device)
            limit = source_config["refinement"].get("val_frames", 0) if args.val_frames is None else args.val_frames
            validate(source_config, data, model, device, context, validation_rows(data, limit), out, info["step"])


if __name__ == "__main__":
    main()
