"""02 — 실제 SDXL VAE/CLIP + DiT/ControlMLP로 UV texture와 geometry를 생성한다.

읽는 순서: FrozenConditioners -> UVFlowModel -> flow_loss -> sample_domains.
CLIP과 VAE는 frozen. DiT와 ControlMLP는 scratch 학습. FiCA 전용 pretrained weight 없음.
두 domain은 같은 Network를 두 번 호출한다. 6채널 이미지를 한 번 출력하는 모델이 아니다.
Pippo의 공식 DiTBlock과 ControlMLP를 revision 고정해 직접 사용한다.
학습은 tqdm bar로 표시하며 고정 validation을 0/100/200/...jpg로 저장한다.

CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage cache --partial-source predicted
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage train
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage sample
"""

import importlib.util
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
    def __init__(self, data, device, load_clip=True):
        super().__init__()
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
        return (xyz - self.mean) / self.std

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
    def encode_sample(self, sample, identity, partial):
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
        return {"texture": texture_latent[0].cpu(), "geometry": xyz_latent[0].cpu(),
                "texture_weight": texture_weight[0].cpu(), "geometry_weight": geometry_weight[0].cpu(),
                "control": partial_latent[0].cpu(), "clip": self.image_embedding(sample["rgb"][None])[0].cpu()}


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


def flow_loss(model, cache, noise=None, time=None):
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
    return (error * weight).sum() / (weight.sum().clamp_min(1e-6) * target.shape[1])


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


def build_cache(data, device, source="predicted"):
    conditioners = FrozenConditioners(data, device)
    for row in data.manifest["samples"]:
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
        cache = conditioners.encode_sample(sample, identity, partial)
        cache.update(partial_source=source, identity=row["identity"], source=row["file"])
        if source == "predicted":
            cache["partial_sha256"] = L00.sha256(partial_path)
        L00.save_tensor(data.path / "latents" / f"{stem}.pt", cache)
    del conditioners
    torch.cuda.empty_cache()


def read_cache(data, row, device):
    stem = Path(row["file"]).stem
    cache = L00.load_tensor(data.path / "latents" / f"{stem}.pt", device)
    if cache["partial_source"] == "predicted":
        partial_path = data.path / "partial_pred" / f"{stem}.pt"
        if cache.get("partial_sha256") != L00.sha256(partial_path):
            raise ValueError("partial UV가 바뀌었다. 02_generate_uv.py --stage cache를 다시 실행하세요.")
    return cache


def decoded_result(conditioners, latent, identity):
    texture = (conditioners.decode(latent[:1]) + 1) / 2
    xyz = conditioners.decode(latent[1:2]) * conditioners.std + conditioners.mean
    # Invalid UV는 topology 밖이다. geometry는 template/bary lookup과 함께 해석한다.
    return texture[0], xyz[0]


def validation_rows(data, limit):
    """매번 같은 held-out frame을 사용한다. Train frame으로 대체하지 않는다."""
    rows = data.samples("validation")
    if not rows:
        raise ValueError("validation split이 없다. 00에서 held-out frame을 준비하세요.")
    return rows[:limit] if limit else rows


def save_validation_panel(path, images, titles, step, row, metrics, seed, sample_steps, frame_count):
    """한 step을 JPEG 한 장에 저장한다. 사진의 종횡비와 error 색 범위를 유지한다."""
    side, header, label_height, columns = 300, 78, 28, 3
    rows = math.ceil(len(images) / columns)
    canvas = Image.new("RGB", (columns*side, header+rows*(side+label_height)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"02 | optimizer step {step} | held-out {Path(row['file']).stem}", fill="white")
    draw.text((10, 27), f"Fixed seed={seed} | Euler steps={sample_steps} | flow loss at t=0.5 | {frame_count} validation frame(s)", fill="white")
    draw.text((10, 46), "Mean validation: " + " | ".join(f"{key}={value:.5g}" for key, value in metrics.items()), fill="white")
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
def validate(config, data, model, device, conditioners, context, rows, out, step, sample_steps):
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
                cache = read_cache(data, row, device)
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
            "texture_metric": "unclamped RGB L1 over train-observed target texels",
            "xyz_metric": "Euclidean RMSE in world units over valid UV texels"})
        return metrics
    finally:
        model.train(was_training)


def train(config, data, device, steps, resume=False, val_every=100, val_frames=1, val_sample_steps=None):
    if steps < 1 or val_every < 1 or val_frames < 0 or (val_sample_steps is not None and val_sample_steps < 1):
        raise ValueError("steps/val_every/val_sample_steps >= 1, val_frames >= 0 이어야 한다.")
    rows = validation_rows(data, val_frames)
    sample_steps = config["diffusion"]["sample_steps"] if val_sample_steps is None else val_sample_steps
    out = L00.work_root(config)/"outputs/02"
    model = UVFlowModel(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["diffusion"]["lr"], weight_decay=config["diffusion"]["weight_decay"])
    start=L00.training_start(L00.work_root(config)/"checkpoints/02/latest.pt",model,optimizer,config,data,resume)
    # CLIP/partial은 이미 cache에 있으므로 validation에는 VAE만 한 번 로드한다.
    # 이 초기화와 validation이 학습의 noise/frame 추출 순서를 바꾸지 않도록 한다.
    with torch.random.fork_rng(devices=[device]):
        import nvdiffrast.torch as dr
        context = dr.RasterizeCudaContext(device=device)
        conditioners = FrozenConditioners(data, device, load_clip=False)
    accumulation=config["training"]["grad_accumulation"]
    with tqdm(total=steps, initial=start, desc="02 UV generation", unit="step", dynamic_ncols=True, mininterval=1) as bar:
        recent = validate(config, data, model, device, conditioners, context, rows, out, start, sample_steps)
        bar.set_postfix(val=f"{recent['flow_loss']:.5f}", refresh=False)
        for step in range(start,steps):
            optimizer.zero_grad(set_to_none=True)
            mean_loss=0.
            for micro in range(accumulation):
                cache=read_cache(data,data.training_row(),device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = flow_loss(model, cache)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"02 training loss NaN/Inf at step {step+1}")
                (loss/accumulation).backward()
                mean_loss+=float(loss.detach())/accumulation
            report = L00.gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_grad"])
            optimizer.step()
            L00.append_log(out/"training.jsonl", {"step": step+1, "loss": mean_loss, "gradient": report, "partial_source": cache["partial_source"]})
            if (step+1) % val_every == 0 or step == steps-1:
                recent = validate(config, data, model, device, conditioners, context, rows, out, step+1, sample_steps)
            bar.set_postfix(loss=f"{mean_loss:.5f}", val=f"{recent['flow_loss']:.5f}", refresh=False)
            bar.update(1)
            if (step+1) % config["training"]["checkpoint_every"] == 0 or step == steps-1:
                L00.checkpoint(L00.work_root(config) / "checkpoints/02/latest.pt", model, optimizer, config, data, step+1)
    del conditioners, context
    return model.eval()


@torch.no_grad()
def generate(config, data, model, device, steps=None):
    import nvdiffrast.torch as dr
    context=dr.RasterizeCudaContext(device=device)
    conditioners = FrozenConditioners(data, device)
    for row in data.manifest["samples"]:
        cache = read_cache(data, row, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent, history = sample_domains(model, cache, steps or config["diffusion"]["sample_steps"])
        texture, xyz = decoded_result(conditioners, latent.float(), data.identity(row["identity"], device))
        stem = Path(row["file"]).stem
        L00.save_tensor(data.path / "generated" / f"{stem}.pt", {"texture": texture.cpu(), "xyz": xyz.cpu(), "history": history,
             "source": row["file"], "partial_source": cache["partial_source"], "model": "UVFlowModel"})
        identity = data.identity(row["identity"], device)
        bridge=data.bridge(row["identity"],device)
        sample=data.sample(row,device)
        vertices=L00.posed_from_canonical(bridge(xyz),sample)
        rendered=L00.render_mesh(vertices,bridge.faces,bridge.vt,bridge.ft,texture,sample["K"],sample["w2c"],*sample["rgb"].shape[-2:],context)
        L00.save_panels(L00.work_root(config) / "outputs/02" / f"{stem}.jpg", [sample["rgb"],rendered["rgb"], texture, (xyz-conditioners.mean[0])/(conditioners.std[0]*6)+.5],
                        ["Reference", "Generated mesh render", "Generated texture", "Generated XYZ (display)"])
    del conditioners
    torch.cuda.empty_cache()


def main():
    parser = L00.arguments(__doc__)
    parser.add_argument("--stage", choices=("cache", "train", "sample"), default="cache")
    parser.add_argument("--partial-source", choices=("predicted", "gt_correspondence"), default="predicted")
    parser.add_argument("--val-every", type=int, default=100, help="Validation 간격 (optimizer step, 기본 100)")
    parser.add_argument("--val-frames", type=int, default=1, help="고정 held-out frame 수; 0은 전체 (기본 1)")
    parser.add_argument("--val-sample-steps", type=int, help="Validation Euler step 수 (기본 config의 sample_steps=50)")
    args = parser.parse_args()
    if args.val_every < 1 or args.val_frames < 0 or (args.val_sample_steps is not None and args.val_sample_steps < 1):
        parser.error("--val-every/--val-sample-steps >= 1, --val-frames >= 0 이어야 한다.")
    config = L00.read_config(args.config, args.smoke)
    device = L00.setup_device(args.device, config["seed"])
    data = L00.PreparedData(args.data)
    if args.stage == "cache":
        build_cache(data, device, args.partial_source)
    elif args.stage == "train":
        train(config, data, device, args.steps or config["training"]["steps"], resume=args.resume,
              val_every=args.val_every, val_frames=args.val_frames, val_sample_steps=args.val_sample_steps)
    else:
        model = UVFlowModel(config).to(device).eval()
        L00.restore(L00.work_root(config) / "checkpoints/02/latest.pt", model, config, data)
        generate(config, data, model, device, args.steps)


if __name__ == "__main__":
    main()
