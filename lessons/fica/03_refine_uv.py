"""03 — 입력 사진과 생성 mesh 렌더링을 참고해 UV texture/geometry를 보정한다.

읽는 순서: RefinementNet -> feature_pair -> refinement_loss -> train.
FiCA C.3의 실제 Diffusers UNet2DConditionModel + frozen Sapiens ViT를 사용한다.
Photo/mask/keypoint와 texture/Laplacian/normal regularization의 gradient를 모두 보존한다.
Geometry는 원래 FLAME topology로 복원하고 tracking pose adapter를 통해 render한다.

CUDA_VISIBLE_DEVICES=0 python 03_refine_uv.py --stage cache
CUDA_VISIBLE_DEVICES=0 python 03_refine_uv.py --stage train
CUDA_VISIBLE_DEVICES=0 python 03_refine_uv.py --stage predict
"""

import importlib.util
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location("fica_00", Path(__file__).with_name("00_prepare_nersemble.py"))
L00 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L00)


class RefinementNet(nn.Module):
    """입력은 UV 6ch, condition은 이미지 feature tokens. 출력은 보정된 UV 6ch.

    안정적인 초기값을 위해 residual parameterization을 쓴다. 정확한 FiCA U-Net
    channel schedule/출력 parameterization은 미공개이므로 config에 구현 선택으로 기록한다.
    """
    def __init__(self, config):
        super().__init__()
        from diffusers import UNet2DConditionModel
        settings = config["refinement"]
        channels = settings["channels"]
        stages = len(channels)
        self.unet = UNet2DConditionModel(
            in_channels=6, out_channels=6, block_out_channels=tuple(channels),
            down_block_types=tuple(["CrossAttnDownBlock2D"]*(stages-1) + ["DownBlock2D"]),
            up_block_types=tuple(["UpBlock2D"] + ["CrossAttnUpBlock2D"]*(stages-1)),
            layers_per_block=settings["layers_per_block"], cross_attention_dim=1536,
            attention_head_dim=settings["attention_head_dim"], norm_num_groups=32)
        # 같은 encoder의 reference/render features라도 역할을 식별할 수 있게 한다.
        self.feature_type = nn.Parameter(torch.zeros(2, 1536))
        nn.init.normal_(self.feature_type, std=.01)
        nn.init.zeros_(self.unet.conv_out.weight)
        nn.init.zeros_(self.unet.conv_out.bias)
        self.unet.enable_gradient_checkpointing()

    def forward(self, texture, xyz, reference_features, render_features, mean, std):
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
    texture_reg = ((texture-initial["texture"][None]).norm(dim=1,keepdim=True)*valid).sum()/valid.sum().clamp_min(1)
    laplacian = (L00.cotangent_laplacian_values(vertices,bridge.faces)-L00.cotangent_laplacian_values(initial_vertices,bridge.faces)).norm()/math.sqrt(len(vertices))
    normals = L00.vertex_normals(vertices, bridge.faces)
    initial_normals = L00.vertex_normals(initial_vertices, bridge.faces)
    normal_reg = (normals-initial_normals).norm(dim=-1).mean()
    terms = {"photo": F.l1_loss(rendered["rgb"], expected), "mask": F.l1_loss(rendered["alpha"], target_alpha),
             "keypoint": keypoint, "texture_reg": texture_reg, "laplacian_reg": laplacian, "normal_reg": normal_reg}
    total = settings["photo"]*terms["photo"] + settings["mask"]*terms["mask"] + settings["keypoint"]*keypoint
    total = total + settings["regularization"]*(texture_reg+laplacian+normal_reg)
    return total, terms, rendered


def read_initial(data, row, device):
    return L00.load_tensor(data.path / "generated" / f"{Path(row['file']).stem}.pt", device)


def stats(data, device):
    return (torch.tensor(data.manifest["xyz_mean"], device=device)[None, :, None, None],
            torch.tensor(data.manifest["xyz_std"], device=device)[None, :, None, None])


def cache_features(data, device):
    import nvdiffrast.torch as dr
    encoder = L00.lesson(1).pretrained_encoder(device)
    context = dr.RasterizeCudaContext(device=device)
    bridges = {}
    for row in data.manifest["samples"]:
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


def train(config, data, device, steps, resume=False):
    import nvdiffrast.torch as dr
    context = dr.RasterizeCudaContext(device=device)
    model = RefinementNet(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["refinement"]["lr"])
    bridges = {}
    mean, std = stats(data, device)
    rows = data.samples("train")
    start=L00.training_start(L00.work_root(config)/"checkpoints/03/latest.pt",model,optimizer,config,data,resume)
    accumulation=config["training"]["grad_accumulation"]
    for step in range(start,steps):
        optimizer.zero_grad(set_to_none=True)
        mean_loss=0.;mean_terms={}
        for micro in range(accumulation):
            row=data.training_row()
            identity = data.identity(row["identity"], device)
            if row["identity"] not in bridges:
                bridges[row["identity"]] = data.bridge(row["identity"],device)
            bridge = bridges[row["identity"]]
            sample, initial = data.sample(row, device), read_initial(data, row, device)
            features = L00.load_tensor(data.path / "refine_features" / f"{Path(row['file']).stem}.pt", device)
            if features["initial_sha256"] != L00.sha256(data.path / "generated" / f"{Path(row['file']).stem}.pt"):
                raise ValueError("생성 mesh 이후 feature cache가 stale이다. 03 --stage cache를 다시 실행하세요.")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                texture, xyz = model(initial["texture"][None], initial["xyz"][None], features["reference"], features["rendered"], mean, std)
            # CUDA rasterization / Cholesky solve는 float32로 실행한다.
            loss, terms, rendered = refinement_loss(config, texture.float(), xyz.float(), initial, identity, sample, bridge, context)
            (loss/accumulation).backward()
            mean_loss+=float(loss.detach())/accumulation
            for name,value in terms.items():
                mean_terms[name]=mean_terms.get(name,0.)+float(value.detach())/accumulation
        report = L00.gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_grad"])
        optimizer.step()
        L00.append_log(L00.work_root(config) / "outputs/03/training.jsonl", {"step": step+1, "loss": mean_loss, "terms": mean_terms, "gradient": report})
        if (step+1) % config["training"]["preview_every"] == 0 or step == 0:
            L00.save_panels(L00.work_root(config) / "outputs/03/preview" / f"{step+1}.jpg", [sample["rgb"], rendered["rgb"], initial["texture"], texture[0]],
                            ["Reference", "Refined mesh", "Initial texture", "Refined texture"])
        print(f"03 step {step+1}/{steps}: {mean_loss:.6f}", flush=True)
        if (step+1) % config["training"]["checkpoint_every"] == 0 or step == steps-1:
            L00.checkpoint(L00.work_root(config) / "checkpoints/03/latest.pt", model, optimizer, config, data, step+1)
    return model.eval()


@torch.no_grad()
def predict(config, data, model, device):
    mean, std = stats(data, device)
    for row in data.manifest["samples"]:
        initial = read_initial(data, row, device)
        stem = Path(row["file"]).stem
        features = L00.load_tensor(data.path / "refine_features" / f"{stem}.pt", device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            texture, xyz = model(initial["texture"][None], initial["xyz"][None], features["reference"], features["rendered"], mean, std)
        L00.save_tensor(data.path / "refined" / f"{stem}.pt", {"texture": texture[0].float().cpu(), "xyz": xyz[0].float().cpu(), "source": row["file"]})


def main():
    parser = L00.arguments(__doc__)
    parser.add_argument("--stage", choices=("cache", "train", "predict"), default="cache")
    args = parser.parse_args()
    config = L00.read_config(args.config, args.smoke)
    device = L00.setup_device(args.device, config["seed"])
    data = L00.PreparedData(args.data)
    if args.stage == "cache":
        cache_features(data, device)
    elif args.stage == "train":
        train(config, data, device, args.steps or config["training"]["steps"],resume=args.resume)
    else:
        model = RefinementNet(config).to(device).eval()
        L00.restore(L00.work_root(config) / "checkpoints/03/latest.pt", model, config, data)
        predict(config, data, model, device)


if __name__ == "__main__":
    main()
