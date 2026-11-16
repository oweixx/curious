"""02 — 실제 SDXL VAE/CLIP + DiT/ControlMLP로 UV texture와 geometry를 생성한다.

읽는 순서: FrozenConditioners -> UVFlowModel -> flow_loss -> sample_domains.
CLIP과 VAE는 frozen. DiT와 ControlMLP는 scratch 학습. FiCA 전용 pretrained weight 없음.
두 domain은 같은 Network를 두 번 호출한다. 6채널 이미지를 한 번 출력하는 모델이 아니다.
Pippo의 공식 DiTBlock과 ControlMLP를 revision 고정해 직접 사용한다.
학습은 tqdm bar로 표시하며 고정 validation을 0/100/200/...jpg로 저장한다.

CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage cache --partial-source predicted
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage train
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage sample
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage stabilize
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage finetune-geometry
"""

import importlib.util
import json
import math
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as recompute
from tqdm.auto import tqdm

spec = importlib.util.spec_from_file_location("fica_00", Path(__file__).with_name("00_prepare_nersemble.py"))
L00 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L00)


class FrozenConditioners(nn.Module):
    """VAE는 RGB처럼 보이는 XYZ의 수치 복원을 담당한다. 단위/normalization을 보존한다."""
    def __init__(self, data, device, load_clip=True, xyz_scale=1.):
        super().__init__()
        if xyz_scale <= 0:
            raise ValueError("XYZ codec scale은 양수여야 한다.")
        self.xyz_scale = float(xyz_scale)
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(str(L00.asset_path("vae")), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
        if load_clip:
            from transformers import CLIPVisionModelWithProjection
            self.clip = CLIPVisionModelWithProjection.from_pretrained(str(L00.asset_path("clip")), torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor(data.manifest["xyz_mean"], device=device)[None, :, None, None])
        self.register_buffer("std", torch.tensor(data.manifest["xyz_std"], device=device)[None, :, None, None])
        self.scale = self.vae.config.scaling_factor

    def xyz_to_vae(self, xyz):
        # 색처럼 임의로 [0,1] clip하지 않는다. Normalized coordinates를 그대로 인코딩한다.
        return (xyz - self.mean) / (self.std*self.xyz_scale)

    def decode_xyz(self, latent):
        return self.decode(latent)*self.std*self.xyz_scale + self.mean

    def encode(self, value):
        return self.vae.encode(value).latent_dist.mode() * self.scale

    def decode(self, latent):
        # no_grad decorator 없음: 03/확장실험에서 input latent로 gradient가 흐를 수 있다.
        return self.vae.decode(latent / self.scale).sample

    def image_embedding(self, rgb):
        side = min(rgb.shape[-2:])
        y = (rgb.shape[-2] - side) // 2
        x = (rgb.shape[-1] - side) // 2
        crop = F.interpolate(rgb[:, :, y:y+side, x:x+side], size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        mean = crop.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
        std = crop.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]
        return self.clip(pixel_values=(crop - mean) / std).image_embeds

    @torch.no_grad()
    def encode_sample(self, sample, identity, partial, full_geometry=False):
        texture = identity["texture"][None]
        xyz = identity["xyz"][None]
        texture_latent = self.encode(texture * 2 - 1)
        xyz_latent = self.encode(self.xyz_to_vae(xyz))
        parts = [partial[None, :3]*2-1, partial[None, 3:4].expand(-1, 3, -1, -1)*2-1,
                 partial[None, 4:7], self.xyz_to_vae(partial[None, 7:10])]
        # Missing texel을 background geometry로 학습하지 않도록 조건의 mask를 함께 전달한다.
        parts[-1] = parts[-1] * partial[None, 3:4]
        partial_latent = torch.cat([self.encode(value) for value in parts], 1)
        texture_weight = F.interpolate(identity["observed"][None].float(), size=texture_latent.shape[-2:], mode="area")
        geometry_weight = F.interpolate(identity["valid"][None, None].float(), size=xyz_latent.shape[-2:], mode="area")
        if full_geometry:
            # Atlas 밖은 미관측 표면이 아니라 알려진 padding이다. 이 latent를 noise로
            # 방치하면 VAE의 넓은 receptive field를 통해 유효 XYZ까지 오염된다.
            geometry_weight = torch.ones_like(geometry_weight)
        return {"texture": texture_latent[0].cpu(), "geometry": xyz_latent[0].cpu(),
                "texture_weight": texture_weight[0].cpu(), "geometry_weight": geometry_weight[0].cpu(),
                "control": partial_latent[0].cpu(), "clip": self.image_embedding(sample["rgb"][None])[0].cpu(),
                "geometry_codec": {"xyz_scale": self.xyz_scale, "full_geometry": full_geometry}}


def pippo_modules():
    L00.verify_sources()
    source = L00.CACHE / "reference/pippo"
    required = source / "latent_diffusion/models/control_mlp.py"
    if not required.is_file():
        raise FileNotFoundError("00 --stage assets로 pinned Pippo source도 준비하세요.")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from latent_diffusion.models.dit import DiTBlock
    from latent_diffusion.models.control_mlp import ControlMLP
    from latent_diffusion.blocks.timestep import TimestepEmbedder
    return DiTBlock, ControlMLP, TimestepEmbedder


def sine_positions(height, width, dim, device, dtype):
    if dim % 4:
        raise ValueError("2D sin/cos positional width는 4의 배수여야 한다.")
    frequency = 1 / (10000 ** (torch.arange(dim // 4, device=device).float() / (dim // 4)))
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
    x, y = x.flatten()[:, None] * frequency, y.flatten()[:, None] * frequency
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), -1)[None].to(dtype)


class UVFlowModel(nn.Module):
    """Shared texture/geometry model. ControlMLP는 UV와 같은 위치의 token에 작용한다.

    SDXL latent4x64x64 -> pack16x32x32 -> patch64x16x16 -> DiT tokens.
    pixel_unshuffle는 가역적이다. 논문의 latent 표기를 해석한 구현 선택임을 config에 기록.
    """
    def __init__(self, config):
        super().__init__()
        settings = config["diffusion"]
        DiTBlock, ControlMLP, TimestepEmbedder = pippo_modules()
        self.pack, self.patch = settings["latent_pack"], settings["patch_size"]
        self.width = settings["width"]
        self.latent_channels = 4 * self.pack ** 2
        self.x_embed = nn.Linear(self.latent_channels * self.patch ** 2, self.width)
        self.time_embed = TimestepEmbedder(self.width)
        self.clip_embed = nn.Linear(settings["clip_dim"], self.width)
        self.domain_embed = nn.Embedding(2, self.width)
        self.control = ControlMLP(16 * self.pack ** 2, self.width, settings["depth"], control_encoder_latent=True)
        self.blocks = nn.ModuleList([DiTBlock(self.width, settings["heads"], cond_attn_enabled=False) for _ in range(settings["depth"])])
        self.final_norm = nn.LayerNorm(self.width, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(self.width, 2*self.width))
        self.output = nn.Linear(self.width, self.latent_channels * self.patch ** 2)
        self.checkpoint_blocks = settings["checkpoint_blocks"]
        # adaLN-zero를 유지한다. 첫 step의 gradient가 output에 먼저 도달하는 것은 정상이다.
        for block in self.blocks:
            nn.init.zeros_(block.cond_adaln[-1].weight)
            nn.init.zeros_(block.cond_adaln[-1].bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, latent, time, clip, control, domain):
        x = F.pixel_unshuffle(latent, self.pack)
        batch, channels, height, width = x.shape
        patch = self.patch
        if height % patch or width % patch:
            raise ValueError("latent 크기가 packing/patch 크기의 배수가 아니다.")
        tokens = F.unfold(x, kernel_size=patch, stride=patch).transpose(1, 2)
        tokens = self.x_embed(tokens)
        tokens = tokens + sine_positions(height//patch, width//patch, self.width, x.device, tokens.dtype)
        conditioning = self.time_embed(time * 1000) + self.clip_embed(clip) + self.domain_embed(domain)
        controls = self.control(F.pixel_unshuffle(control, self.pack), time * 1000)
        if controls[0].shape[1] != tokens.shape[1]:
            raise ValueError("ControlMLP와 target patch의 공간 주소가 일치하지 않는다.")
        for block, dense in zip(self.blocks, controls):
            if self.training and self.checkpoint_blocks:
                tokens = recompute(lambda a, b, c, module=block: module(a, b, control=c), tokens, conditioning, dense, use_reentrant=False)
            else:
                tokens = block(tokens, conditioning, control=dense)
        shift, scale = self.final_mod(conditioning).chunk(2, -1)
        tokens = self.final_norm(tokens) * (1 + scale[:, None]) + shift[:, None]
        output = self.output(tokens).transpose(1, 2)
        output = F.fold(output, (height, width), kernel_size=patch, stride=patch)
        return F.pixel_shuffle(output, self.pack)


def flow_loss(model, cache, noise=None, time=None, geometry=None, return_terms=False):
    """Linear conditional flow: x_t=(1-t)epsilon+t*z, v*=z-epsilon.

    Texture/geometry 각각 forward한다. Sampling은 t=0 noise에서 t=1 data로 적분한다.
    """
    target = torch.stack((cache["texture"], cache["geometry"]))
    noise = torch.randn_like(target) if noise is None else noise
    time = torch.rand(2, device=target.device) if time is None else time
    noisy = (1-time[:, None, None, None])*noise + time[:, None, None, None]*target
    result = model(noisy, time, cache["clip"][None].expand(2, -1), cache["control"][None].expand(2, -1, -1, -1), torch.arange(2, device=target.device))
    weight = torch.stack((cache["texture_weight"], cache["geometry_weight"]))
    error = (result.float() - (target-noise).float()).square()
    # Complete texture GT가 없는 영역을 검정 정답으로 감독하지 않는다.
    flow = (error * weight).sum() / (weight.sum().clamp_min(1e-6) * target.shape[1])
    terms = {"flow": flow}
    total = flow
    if geometry is not None:
        conditioners, identity, bridge, settings = geometry
        # x_t + (1-t)*v는 정답 velocity일 때 정확히 x_1이다. 별도 학습 network가
        # 아니라 같은 DiT 출력에 endpoint supervision을 더한다. VAE weight는 고정.
        endpoint = (noisy[1:2] + (1-time[1])*result[1:2]).float()
        with torch.autocast("cuda", enabled=False):
            xyz = recompute(conditioners.decode_xyz, endpoint, use_reentrant=False)[0]
            valid = identity["valid"][None]
            xyz_loss = (((xyz-identity["xyz"])/conditioners.std[0]).abs()*valid).sum()/(valid.sum()*3)
            vertices = bridge(xyz)
            reference = identity["canonical_vertices"]
            vertex_loss = ((vertices-reference)/conditioners.std.mean()).abs().mean()
            edges = bridge.edges
            delta = (vertices-reference)[edges[:, 0]]-(vertices-reference)[edges[:, 1]]
            edge_loss = (delta/conditioners.std.mean()).abs().mean()
            decoded = xyz_loss + settings["vertex_weight"]*vertex_loss + settings["edge_weight"]*edge_loss
        total = total + settings["geometry_weight"]*decoded
        terms.update(xyz=xyz_loss, vertex=vertex_loss, edge=edge_loss)
    return (total, terms) if return_terms else total


def run_paths(config, run_name=None):
    if run_name is not None and (not run_name or Path(run_name).name != run_name or run_name in (".", "..")):
        raise ValueError("run-name은 폴더 이름 하나로 지정하세요.")
    root = L00.work_root(config)
    suffix = Path(run_name) if run_name else Path()
    return root/"outputs/02"/suffix, root/"checkpoints/02"/suffix/"latest.pt"


def latent_path(data, row, run_name=None):
    directory = data.path/"latents"
    if run_name:
        run_paths(data.manifest["config"], run_name)  # 경로 이름 검증
        directory = directory/run_name
    return directory/f"{Path(row['file']).stem}.pt"


def codec_settings(config, run_name=None):
    return {"xyz_scale": float(config["geometry_repair"]["xyz_scale"]) if run_name else 1.,
            "full_geometry": bool(run_name)}


def load_flow_model(path, data, device):
    """구 checkpoint의 weight를 엄격히 읽는다. 평가/warm start는 코드 hash 변경 허용.

    Optimizer resume는 별도의 L00.restore가 config/data/code/RNG까지 검사한다.
    여기서는 dataset과 UV 해상도를 확인하고 checkpoint의 정확한 구조를 구성한다.
    """
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if state["format"] != "fica_checkpoint_v1" or state["data_sha256"] != L00.sha256(data.path/"manifest.json"):
        raise ValueError("02 checkpoint format/dataset 불일치")
    if state["config"]["uv_size"] != data.manifest["config"]["uv_size"]:
        raise ValueError("02 checkpoint UV 해상도 불일치")
    with torch.device("meta"):
        model = UVFlowModel(state["config"])
    model.load_state_dict(state["model"], strict=True, assign=True)
    extra = state.get("extra", {})
    metadata = {"checkpoint": str(path.resolve()), "source_step": state["step"],
                "source_code_hashes": state["code_hashes"], "source_config": state["config"],
                "geometry_codec": extra.get("geometry_codec", {"xyz_scale": 1., "full_geometry": False}),
                "cache_run": extra.get("run"), "extra": extra}
    return model.to(device), metadata


@torch.no_grad()
def sample_domains(model, cache, steps, seed=42):
    """같은 model에 domain=0/1을 넣어 두 domain을 병렬 평가한다. Euler ODE sampling."""
    if steps < 1:
        raise ValueError("sampling steps>=1 필요")
    device = cache["texture"].device
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn((2, *cache["texture"].shape), generator=generator, device=device)
    history = [latent.cpu()]
    for index in range(steps):
        time = torch.full((2,), index / steps, device=device)
        velocity = model(latent, time, cache["clip"][None].expand(2, -1), cache["control"][None].expand(2, -1, -1, -1), torch.arange(2, device=device))
        latent = latent + velocity / steps
        if index == steps-1 or index % max(1, steps//5) == 0:
            history.append(latent.cpu())
    return latent, history


def build_cache(data, device, source="predicted", run_name=None, config=None):
    config = config or L00.read_config()
    codec = codec_settings(config, run_name)
    conditioners = FrozenConditioners(data, device, xyz_scale=codec["xyz_scale"])
    for row in tqdm(data.manifest["samples"], desc="02 latent cache", dynamic_ncols=True, mininterval=1):
        sample = data.sample(row, device)
        identity = data.identity(row["identity"], device)
        stem = Path(row["file"]).stem
        if source == "predicted":
            partial_path = data.path / "partial_pred" / f"{stem}.pt"
            partial = L00.load_tensor(partial_path, device)["partial"]
        elif source == "gt_correspondence":
            partial = sample["partial_gt"]
        else:
            raise ValueError(source)
        cache = conditioners.encode_sample(sample, identity, partial, full_geometry=codec["full_geometry"])
        cache.update(partial_source=source, identity=row["identity"], source=row["file"])
        cache["data_sha256"] = L00.sha256(data.path/"manifest.json")
        if source == "predicted":
            cache["partial_sha256"] = L00.sha256(partial_path)
        L00.save_tensor(latent_path(data, row, run_name), cache)
    del conditioners
    torch.cuda.empty_cache()


def read_cache(data, row, device, run_name=None, codec=None):
    stem = Path(row["file"]).stem
    cache = L00.load_tensor(latent_path(data, row, run_name), device)
    actual_codec = cache.get("geometry_codec", {"xyz_scale": 1., "full_geometry": False})
    if codec is not None and actual_codec != codec:
        raise ValueError("XYZ codec가 다른 latent cache다. 해당 --run-name으로 02 --stage cache를 다시 실행하세요.")
    if cache.get("data_sha256", L00.sha256(data.path/"manifest.json")) != L00.sha256(data.path/"manifest.json"):
        raise ValueError("latent cache의 dataset 불일치")
    if cache["partial_source"] == "predicted":
        partial_path = data.path / "partial_pred" / f"{stem}.pt"
        if cache.get("partial_sha256") != L00.sha256(partial_path):
            raise ValueError("partial UV가 바뀌었다. 02_generate_uv.py --stage cache를 다시 실행하세요.")
    return cache


def decoded_result(conditioners, latent, identity):
    texture = (conditioners.decode(latent[:1]) + 1) / 2
    xyz = conditioners.decode_xyz(latent[1:2])
    # Invalid UV는 topology 밖이다. geometry는 template/bary lookup과 함께 해석한다.
    return texture[0], xyz[0]


def mesh_metrics(vertices, identity, bridge):
    reference = identity["canonical_vertices"]
    error = (vertices-reference).norm(dim=-1)
    edges = bridge.edges
    target_length = (reference[edges[:, 0]]-reference[edges[:, 1]]).norm(dim=-1).clamp_min(1e-8)
    length = (vertices[edges[:, 0]]-vertices[edges[:, 1]]).norm(dim=-1)
    return {"vertex_rmse": float(error.square().mean().sqrt()), "vertex_max": float(error.max()),
            "stretched_edges_10x": int((length > 10*target_length).sum())}


@torch.no_grad()
def codec_check(config, data, device, run_name="geometry_repair"):
    """DiT 학습 전에 raw XYZ -> VAE -> mesh 오차를 확인한다. 개인 정답은 평가에만 사용."""
    codec = codec_settings(config, run_name)
    conditioners = FrozenConditioners(data, device, load_clip=False)
    out, _ = run_paths(config, run_name)
    import nvdiffrast.torch as dr
    context = dr.RasterizeCudaContext(device=device)
    reports, images, titles = {}, [], []
    for key in data.manifest["identities"]:
        identity = data.identity(key, device)
        bridge = data.bridge(key, device)
        row = next(row for row in data.manifest["samples"] if row["identity"] == key)
        sample = data.sample(row, device)
        cases = {}
        for name, scale in (("raw", None), ("legacy_vae", 1.), ("repaired_vae", codec["xyz_scale"])):
            xyz = identity["xyz"]
            if scale is not None:
                conditioners.xyz_scale = scale
                xyz = conditioners.decode_xyz(conditioners.encode(conditioners.xyz_to_vae(xyz[None])))[0]
            vertices = bridge(xyz)
            valid = identity["valid"]
            values = mesh_metrics(vertices, identity, bridge)
            values["xyz_rmse"] = float((xyz[:, valid]-identity["xyz"][:, valid]).square().sum(0).mean().sqrt())
            cases[name] = values
            if key == data.manifest["identities"][0]:
                posed = L00.posed_from_canonical(vertices, sample)
                image = L00.render_mesh(posed, bridge.faces, bridge.vt, bridge.ft, identity["texture"], sample["K"], sample["w2c"], *sample["rgb"].shape[-2:], context)["rgb"]
                images.append(image); titles.append(f"{name}: scale={scale or '-'}")
        diameter = float((identity["canonical_vertices"].amax(0)-identity["canonical_vertices"].amin(0)).norm())
        cases["relative_vertex_rmse"] = cases["repaired_vae"]["vertex_rmse"]/diameter
        reports[key] = cases
    L00.save_json(out/"codec_check.json", {"geometry_codec": codec, "mesh_laplacian": bridge.laplacian, "identities": reports})
    L00.save_panels(out/"codec_check.jpg", images, titles)
    if config.get("profile") != "smoke" and any(value["relative_vertex_rmse"] > .05 for value in reports.values()):
        raise ValueError(f"XYZ codec 왕복 오차가 head 크기의 5%를 넘는다. {out/'codec_check.json'}을 먼저 확인하세요.")
    return reports


def validation_rows(data, limit):
    """매번 같은 held-out frame을 사용한다. Train frame으로 대체하지 않는다."""
    rows = data.samples("validation")
    if not rows:
        raise ValueError("validation split이 없다. 00에서 held-out frame을 준비하세요.")
    return rows[:limit] if limit else rows


def save_validation_panel(path, images, titles, step, row, metrics, seed, sample_steps, frame_count):
    """한 step을 JPEG 한 장에 저장한다. 사진의 종횡비와 error 색 범위를 유지한다."""
    side, header, label_height, columns = 300, 96, 28, 3
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns*side, header+rows*(side+label_height)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"02 | optimizer step {step} | held-out {Path(row['file']).stem}", fill="white")
    draw.text((10, 27), f"Fixed seed={seed} | Euler steps={sample_steps} | flow loss at t=0.5 | {frame_count} validation frame(s)", fill="white")
    displayed = {key:value for key,value in metrics.items() if key not in ("vertex_max","stretched_edges_10x")}
    draw.text((10, 46), "Mean validation: " + " | ".join(f"{key}={value:.5g}" for key, value in displayed.items()), fill="white")
    if "stretched_edges_10x" in metrics:
        draw.text((10, 66), f"Mesh: max vertex error={metrics['vertex_max']:.5g} | edges stretched >10x={metrics['stretched_edges_10x']:.0f}", fill="white")
    for index, (value, title) in enumerate(zip(images, titles)):
        value = value.detach().float().cpu().expand(3, -1, -1)
        array = (value.clamp(0, 1).permute(1, 2, 0).numpy()*255).round().astype("uint8")
        tile = Image.fromarray(array)
        scale = min(side/tile.width, side/tile.height)
        tile = tile.resize((max(1, round(tile.width*scale)), max(1, round(tile.height*scale))), Image.Resampling.BILINEAR)
        x, y = (index % columns)*side, header+(index//columns)*(side+label_height)
        draw.text((x+5, y+6), title, fill="white")
        canvas.paste(tile, (x+(side-tile.width)//2, y+label_height+(side-tile.height)//2))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)


@torch.no_grad()
def validate(config, data, model, device, conditioners, context, rows, out, step, sample_steps, run_name=None):
    """고정 noise/time으로 loss와 생성 결과를 평가한다. 학습 RNG/모드는 복원한다.

    Loss는 t=0.5의 flow velocity 오차다. Texture/XYZ 지표는 noise에서 시작해
    sampling한 결과를 VAE로 decode한 뒤 계산하므로 서로 다른 항목이다.
    """
    if not rows:
        raise ValueError("validation split이 없다. Train frame으로 대체하지 않는다.")
    was_training = model.training
    model.eval()
    totals, preview = {}, None
    try:
        with torch.random.fork_rng(devices=[device]):
            for index, row in enumerate(rows):
                cache = read_cache(data, row, device, run_name, codec_settings(config, run_name))
                identity = data.identity(row["identity"], device)
                seed = config["seed"] + index
                generator = torch.Generator(device=device).manual_seed(seed)
                noise = torch.randn((2, *cache["texture"].shape), device=device, generator=generator)
                time = torch.full((2,), .5, device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = flow_loss(model, cache, noise=noise, time=time)
                    latent, _ = sample_domains(model, cache, sample_steps, seed=seed)
                texture, xyz = decoded_result(conditioners, latent.float(), identity)
                if not all(torch.isfinite(value).all() for value in (loss, latent, texture, xyz)):
                    raise FloatingPointError(f"02 validation NaN/Inf: {row['file']}")
                observed, valid = identity["observed"].float(), identity["valid"][None].float()
                if not observed.any() or not valid.any():
                    raise ValueError(f"02 validation에 감독 가능한 UV가 없다: {row['file']}")
                texture_error = (texture-identity["texture"]).abs().mean(0, keepdim=True)
                xyz_error = (xyz-identity["xyz"]).square().sum(0, keepdim=True).sqrt()
                values = {"flow_loss": float(loss),
                          "texture_l1": float((texture_error*observed).sum()/observed.sum()),
                          "xyz_rmse": float(((xyz_error.square()*valid).sum()/valid.sum()).sqrt())}
                bridge = data.bridge(row["identity"], device)
                values.update(mesh_metrics(bridge(xyz), identity, bridge))
                for key, value in values.items():
                    totals[key] = totals.get(key, 0.) + value
                if preview is None:
                    sample = data.sample(row, device)
                    stem = Path(row["file"]).stem
                    partial = (L00.load_tensor(data.path/"partial_pred"/f"{stem}.pt", device)["partial"]
                               if cache["partial_source"] == "predicted" else sample["partial_gt"])
                    bridge = data.bridge(row["identity"], device)
                    def render(position, color):
                        vertices = L00.posed_from_canonical(bridge(position), sample)
                        return L00.render_mesh(vertices, bridge.faces, bridge.vt, bridge.ft, color,
                                               sample["K"], sample["w2c"], *sample["rgb"].shape[-2:], context)["rgb"]
                    def heatmap(error, mask, maximum):
                        ratio = (error/maximum).clamp(0, 1)
                        return torch.cat((ratio, 1-(2*ratio-1).abs(), 1-ratio), 0)*mask
                    def xyz_color(position):
                        return ((position-conditioners.mean[0])/(conditioners.std[0]*6)+.5)*valid
                    xyz_limit = float(conditioners.std.mean())*.25
                    images = [sample["rgb"], partial[:3], partial[3:4],
                              identity["texture"]*observed, texture*valid, heatmap(texture_error, observed, .25),
                              xyz_color(identity["xyz"]), xyz_color(xyz), heatmap(xyz_error, valid, xyz_limit),
                              render(identity["xyz"], identity["texture"]), render(xyz, texture), observed]
                    titles = ["Reference", "Input partial UV RGB", "Input UV visibility",
                              "Target texture (observed only)", "Generated texture", "Texture error: red >= 0.25",
                              "Target XYZ (fixed display scale)", "Generated XYZ (same scale)", f"XYZ error: red >= {xyz_limit:.4g}",
                              "Target mesh render", "Generated mesh render", "Texture supervision mask"]
                    preview = (images, titles, row, seed)
        metrics = {key: value/len(rows) for key, value in totals.items()}
        directory = out/"validation"
        save_validation_panel(directory/f"{step}.jpg", preview[0], preview[1], step, preview[2], metrics,
                              preview[3], sample_steps, len(rows))
        L00.append_log(directory/"metrics.jsonl", {"step": step, **metrics, "frames": [row["file"] for row in rows],
            "preview_frame": preview[2]["file"], "seed": config["seed"], "sample_steps": sample_steps,
            "flow_time": .5, "partial_source": cache["partial_source"], "scope": data.manifest["scope"],
            "geometry_codec": codec_settings(config, run_name), "mesh_laplacian": bridge.laplacian,
            "texture_metric": "unclamped RGB L1 over train-observed target texels",
            "xyz_metric": "Euclidean RMSE in world units over valid UV texels"})
        return metrics
    finally:
        model.train(was_training)


def train(config, data, device, steps, resume=False, val_every=100, val_frames=1, val_sample_steps=None,
          run_name=None, init_from=None, repair=False):
    if steps < 1 or val_every < 1 or val_frames < 0 or (val_sample_steps is not None and val_sample_steps < 1):
        raise ValueError("steps/val_every/val_sample_steps >= 1, val_frames >= 0 이어야 한다.")
    rows = validation_rows(data, val_frames)
    sample_steps = config["diffusion"]["sample_steps"] if val_sample_steps is None else val_sample_steps
    out, path = run_paths(config, run_name)
    codec = codec_settings(config, run_name)
    if repair and not run_name:
        raise ValueError("geometry 추가 학습은 별도 run-name이 필요하다.")
    if path.exists() and not resume and config.get("profile") != "smoke":
        raise FileExistsError(f"{path}가 있다. --resume 또는 새 --run-name을 사용하세요.")
    for row in data.manifest["samples"]:
        read_cache(data, row, "cpu", run_name, codec)
    parent = None
    if init_from is not None:
        if resume or Path(init_from).resolve() == path.resolve():
            raise ValueError("init-from은 새 run의 weight 초기화 전용이다.")
        model, parent = load_flow_model(init_from, data, device)
        for key in ("width", "depth", "heads", "patch_size", "latent_pack", "clip_dim"):
            if parent["source_config"]["diffusion"][key] != config["diffusion"][key]:
                raise ValueError(f"warm start Network 구조 불일치: {key}")
        model.checkpoint_blocks = config["diffusion"]["checkpoint_blocks"]
    else:
        model = UVFlowModel(config).to(device)
    model.train()
    settings = config["geometry_repair"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"] if repair else config["diffusion"]["lr"], weight_decay=config["diffusion"]["weight_decay"])
    start=L00.training_start(path,model,optimizer,config,data,resume)
    if resume:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if state["extra"].get("geometry_codec", {"xyz_scale":1.,"full_geometry":False}) != codec:
            raise ValueError("resume checkpoint/cache의 XYZ codec 불일치")
        parent = state["extra"].get("parent")
        del state
    if start > steps:
        raise ValueError("목표 steps가 이미 학습한 step보다 작다.")
    # CLIP/partial은 이미 cache에 있으므로 validation에는 VAE만 한 번 로드한다.
    # 이 초기화와 validation이 학습의 noise/frame 추출 순서를 바꾸지 않도록 한다.
    with torch.random.fork_rng(devices=[device]):
        import nvdiffrast.torch as dr
        context = dr.RasterizeCudaContext(device=device)
        conditioners = FrozenConditioners(data, device, load_clip=False, xyz_scale=codec["xyz_scale"])
    accumulation=config["training"]["grad_accumulation"]
    with tqdm(total=steps, initial=start, desc="02 UV generation", unit="step", dynamic_ncols=True, mininterval=1) as bar:
        recent = validate(config, data, model, device, conditioners, context, rows, out, start, sample_steps, run_name)
        bar.set_postfix(val=f"{recent['flow_loss']:.5f}", refresh=False)
        for step in range(start,steps):
            optimizer.zero_grad(set_to_none=True)
            mean_loss=0.; mean_terms={}
            for micro in range(accumulation):
                row = data.training_row()
                cache=read_cache(data,row,device,run_name,codec)
                geometry = None
                if repair and step >= settings["warmup_steps"] and (step-settings["warmup_steps"]) % settings["geometry_every"] == 0:
                    geometry = (conditioners, data.identity(row["identity"],device), data.bridge(row["identity"],device), settings)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, terms = flow_loss(model, cache, geometry=geometry, return_terms=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"02 training loss NaN/Inf at step {step+1}")
                (loss/accumulation).backward()
                mean_loss+=float(loss.detach())/accumulation
                for key, value in terms.items():
                    mean_terms[key] = mean_terms.get(key,0.)+float(value.detach())/accumulation
            report = L00.gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_grad"])
            optimizer.step()
            L00.append_log(out/"training.jsonl", {"step": step+1, "loss": mean_loss, "terms": mean_terms,
                "gradient": report, "partial_source": cache["partial_source"], "geometry_codec": codec})
            if (step+1) % val_every == 0 or step == steps-1:
                recent = validate(config, data, model, device, conditioners, context, rows, out, step+1, sample_steps, run_name)
            bar.set_postfix(loss=f"{mean_loss:.5f}", val=f"{recent['flow_loss']:.5f}", refresh=False)
            bar.update(1)
            checkpoint_every = settings["checkpoint_every"] if repair else config["training"]["checkpoint_every"]
            if (step+1) % checkpoint_every == 0 or step == steps-1:
                L00.checkpoint(path, model, optimizer, config, data, step+1,
                    extra={"run":run_name,"geometry_codec":codec,"parent":parent,"geometry_supervision":repair})
    del conditioners, context
    return model.eval()


@torch.no_grad()
def generate(config, data, model, device, steps=None, run_name=None, cache_run=None,
             codec=None, output_name="generated", source=None):
    import nvdiffrast.torch as dr
    context=dr.RasterizeCudaContext(device=device)
    codec = codec or {"xyz_scale":1.,"full_geometry":False}
    conditioners = FrozenConditioners(data, device, load_clip=False, xyz_scale=codec["xyz_scale"])
    out, _ = run_paths(config, run_name)
    reports = []
    for row in tqdm(data.manifest["samples"], desc="02 UV sampling", dynamic_ncols=True, mininterval=1):
        cache = read_cache(data, row, device, cache_run, codec)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent, history = sample_domains(model, cache, steps or config["diffusion"]["sample_steps"])
        texture, xyz = decoded_result(conditioners, latent.float(), data.identity(row["identity"], device))
        stem = Path(row["file"]).stem
        if not torch.isfinite(texture).all() or not torch.isfinite(xyz).all():
            raise FloatingPointError(f"02 generation NaN/Inf: {stem}")
        identity = data.identity(row["identity"], device)
        bridge=data.bridge(row["identity"],device)
        sample=data.sample(row,device)
        canonical_vertices = bridge(xyz)
        values = mesh_metrics(canonical_vertices, identity, bridge)
        values.update(frame=row["file"],split=row["split"])
        reports.append(values)
        L00.save_tensor(data.path / output_name / f"{stem}.pt", {"texture": texture.cpu(), "xyz": xyz.cpu(), "history": history,
             "xyz_surface":bridge.vertices_to_uv(canonical_vertices).cpu(), "source": row["file"],
             "partial_source": cache["partial_source"], "model": "UVFlowModel", "geometry_codec":codec,
             "mesh_bridge":{"ridge":bridge.ridge,"laplacian":bridge.laplacian}, "predictor":source or {},
             "gt_used_for_prediction":False})
        vertices=L00.posed_from_canonical(canonical_vertices,sample)
        rendered=L00.render_mesh(vertices,bridge.faces,bridge.vt,bridge.ft,texture,sample["K"],sample["w2c"],*sample["rgb"].shape[-2:],context)
        L00.save_panels(out/"predictions"/f"{stem}.jpg", [sample["rgb"],rendered["rgb"], texture, (xyz-conditioners.mean[0])/(conditioners.std[0]*6)+.5],
                        ["Reference", "Generated mesh render", "Generated texture", "Generated XYZ (display)"])
    L00.save_json(out/"generation_metrics.json", {"geometry_codec":codec,"mesh_laplacian":bridge.laplacian,
                 "metrics_use_gt_for_evaluation_only":True,"frames":reports})
    del conditioners
    torch.cuda.empty_cache()


def main():
    parser = L00.arguments(__doc__)
    parser.add_argument("--stage", choices=("cache", "train", "sample", "stabilize", "codec-check", "finetune-geometry"), default="cache")
    parser.add_argument("--partial-source", choices=("predicted", "gt_correspondence"), default="predicted")
    parser.add_argument("--val-every", type=int, default=100, help="Validation 간격 (optimizer step, 기본 100)")
    parser.add_argument("--val-frames", type=int, default=1, help="고정 held-out frame 수; 0은 전체 (기본 1)")
    parser.add_argument("--val-sample-steps", type=int, help="Validation Euler step 수 (기본 config의 sample_steps=50)")
    parser.add_argument("--run-name", help="추가 학습/cache의 폴더 이름. finetune-geometry 기본 geometry_repair")
    parser.add_argument("--init-from", type=Path, help="추가 학습의 weight 초기화. 기본 기존 checkpoints/02/latest.pt")
    parser.add_argument("--checkpoint", type=Path, help="sample/stabilize에서 읽을 checkpoint")
    args = parser.parse_args()
    if args.val_every < 1 or args.val_frames < 0 or (args.val_sample_steps is not None and args.val_sample_steps < 1):
        parser.error("--val-every/--val-sample-steps >= 1, --val-frames >= 0 이어야 한다.")
    if args.init_from and (args.stage not in ("train","finetune-geometry") or args.resume):
        parser.error("--init-from은 새 train/finetune-geometry 전용이다. --resume과 함께 쓰지 않는다.")
    if args.checkpoint and args.stage not in ("sample","stabilize"):
        parser.error("--checkpoint는 sample/stabilize 전용이다.")
    if args.resume and args.stage not in ("train","finetune-geometry"):
        parser.error("--resume은 train/finetune-geometry 전용이다.")
    config = L00.read_config(args.config, args.smoke)
    device = L00.setup_device(args.device, config["seed"])
    data = L00.PreparedData(args.data)
    run_name = args.run_name or ("geometry_repair" if args.stage in ("finetune-geometry","codec-check") else None)
    run_paths(config, run_name)
    if args.stage == "cache":
        build_cache(data, device, args.partial_source, run_name, config)
    elif args.stage == "codec-check":
        codec_check(config,data,device,run_name)
    elif args.stage in ("train","finetune-geometry"):
        repair = args.stage == "finetune-geometry"
        if repair:
            codec_check(config,data,device,run_name)
            torch.cuda.empty_cache()
            if any(not latent_path(data,row,run_name).exists() for row in data.manifest["samples"]):
                build_cache(data,device,args.partial_source,run_name,config)
        init_from = args.init_from or (run_paths(config)[1] if repair and not args.resume else None)
        model = train(config, data, device, args.steps or (config["geometry_repair"]["steps"] if repair else config["training"]["steps"]),
              resume=args.resume, val_every=args.val_every, val_frames=args.val_frames, val_sample_steps=args.val_sample_steps,
              run_name=run_name,init_from=init_from,repair=repair)
        if repair:
            # 완료된 새 모델의 raw world XYZ를 03/04가 그대로 읽을 수 있게 내보낸다.
            generate(config,data,model,device,run_name=run_name,cache_run=run_name,codec=codec_settings(config,run_name),
                     source={"checkpoint":str(run_paths(config,run_name)[1]),"run":run_name})
    else:
        path = args.checkpoint or run_paths(config,None if args.stage=="stabilize" else run_name)[1]
        model, source = load_flow_model(path,data,device)
        model.eval()
        stable = args.stage == "stabilize"
        generate(config,data,model,device,args.steps,run_name="stabilized" if stable else run_name,
                 cache_run=source["cache_run"],codec=source["geometry_codec"],
                 output_name="generated_stabilized" if stable else "generated",source=source)


if __name__ == "__main__":
    main()
