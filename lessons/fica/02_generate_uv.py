"""02 — 실제 SDXL VAE/CLIP + DiT/ControlMLP로 UV texture와 geometry를 생성한다.

읽는 순서: FrozenConditioners -> UVFlowModel -> flow_loss -> sample_domains.
CLIP과 VAE는 frozen. DiT와 ControlMLP는 scratch 학습. FiCA 전용 pretrained weight 없음.
두 domain은 같은 Network를 두 번 호출한다. 6채널 이미지를 한 번 출력하는 모델이 아니다.
Pippo의 공식 DiTBlock과 ControlMLP를 revision 고정해 직접 사용한다.

CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage cache --partial-source predicted
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage train
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage sample
"""

import importlib.util
import math
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as recompute

spec = importlib.util.spec_from_file_location("fica_00", Path(__file__).with_name("00_prepare_nersemble.py"))
L00 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L00)


class FrozenConditioners(nn.Module):
    """VAE는 RGB처럼 보이는 XYZ의 수치 복원을 담당한다. 단위/normalization을 보존한다."""
    def __init__(self, data, device):
        super().__init__()
        from diffusers import AutoencoderKL
        from transformers import CLIPVisionModelWithProjection
        self.vae = AutoencoderKL.from_pretrained(str(L00.asset_path("vae")), subfolder="vae", variant="fp16", torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
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


def train(config, data, device, steps, resume=False):
    model = UVFlowModel(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["diffusion"]["lr"], weight_decay=config["diffusion"]["weight_decay"])
    rows = data.samples("train")
    start=L00.training_start(L00.work_root(config)/"checkpoints/02/latest.pt",model,optimizer,config,data,resume)
    accumulation=config["training"]["grad_accumulation"]
    for step in range(start,steps):
        optimizer.zero_grad(set_to_none=True)
        mean_loss=0.
        for micro in range(accumulation):
            cache=read_cache(data,data.training_row(),device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = flow_loss(model, cache)
            (loss/accumulation).backward()
            mean_loss+=float(loss.detach())/accumulation
        report = L00.gradients(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["clip_grad"])
        optimizer.step()
        L00.append_log(L00.work_root(config) / "outputs/02/training.jsonl", {"step": step+1, "loss": mean_loss, "gradient": report, "partial_source": cache["partial_source"]})
        print(f"02 step {step+1}/{steps}: {mean_loss:.6f}", flush=True)
        if (step+1) % config["training"]["checkpoint_every"] == 0 or step == steps-1:
            L00.checkpoint(L00.work_root(config) / "checkpoints/02/latest.pt", model, optimizer, config, data, step+1)
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
    args = parser.parse_args()
    config = L00.read_config(args.config, args.smoke)
    device = L00.setup_device(args.device, config["seed"])
    data = L00.PreparedData(args.data)
    if args.stage == "cache":
        build_cache(data, device, args.partial_source)
    elif args.stage == "train":
        train(config, data, device, args.steps or config["training"]["steps"],resume=args.resume)
    else:
        model = UVFlowModel(config).to(device).eval()
        L00.restore(L00.work_root(config) / "checkpoints/02/latest.pt", model, config, data)
        generate(config, data, model, device, args.steps)


if __name__ == "__main__":
    main()
