"""04 — 기존 DiT block을 velocity predictor로 사용해 실제 MNIST image flow를 학습한다.

repo root:
    CUDA_VISIBLE_DEVICES=7 python lessons/flow_matching/04_train_image_flow.py
lesson 폴더:
    CUDA_VISIBLE_DEVICES=7 python 04_train_image_flow.py --steps 6000
    CUDA_VISIBLE_DEVICES=7 python 04_train_image_flow.py --resume outputs/04/latest.pt --steps 12000
짧은 실행 확인:
    CUDA_VISIBLE_DEVICES=7 python 04_train_image_flow.py --profile smoke --steps 8 \
        --batch-size 16 --val-every 4 --preview-steps 4 --out .cache/image_smoke

출력: outputs/04/{latest.pt,best.pt,0.png,100.png,...,loss.png,history.json,training_batch.png}.
읽기: FlowDiT -> image_batch -> training loop 6줄 -> update_ema -> validate.
실제 MNIST train split을 train/validation으로 나눈다. 공식 test split은 읽지 않는다.
28px 이미지는 검은 padding으로 32px로 만든다. VAE 없이 pixel space에서 학습한다.
DiT block/adaLN-Zero는 ../dit/models.py를 직접 재사용한다. 이 모델은 원 DiT checkpoint가 아니다.
"""
from contextlib import nullcontext
from copy import deepcopy
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import trange

from common import (ROOT, cpu_state, digest, finish, image_grid, json_file, lesson, load, module,
                    parser, pyplot, restore_rng, save, setup, state_rng, step)

L00 = lesson(0)
DIT_SOURCE = ROOT.parent / "dit/models.py"
PROFILES = {"mini": dict(width=192, depth=4, heads=6), "smoke": dict(width=64, depth=2, heads=4)}


class FlowDiT(nn.Module):
    """Transformer는 backbone이고 FM은 학습 목적이다. 서로 대체하는 두 모델 이름이 아니다.

    learn_sigma=False -> 입력과 같은 1채널 출력. 의미는 noise/variance가 아닌 dx/dt.
    t∈[0,1]을 embedding 직전에 1000*t로 표현한다. 1000은 solver step 수가 아니다.
    시간 표기만 바꾸는 것이므로 velocity target에 1000을 곱하지 않는다.
    """
    def __init__(self, width=192, depth=4, heads=6):
        super().__init__()
        original = module(DIT_SOURCE, "fm_dit_backbone")
        self.backbone = original.DiT(input_size=32, patch_size=4, in_channels=1,
            hidden_size=width, depth=depth, num_heads=heads, num_classes=10,
            class_dropout_prob=.1, learn_sigma=False)

    def forward(self, x, t, labels):
        t = t.expand(len(x)) if t.ndim == 0 else t
        return self.backbone(x, t*1000, labels)


def mnist(root, limit=None, download=False):
    """데이터는 uint8 CPU로 보관. 매 batch만 float/GPU로 옮긴다."""
    from torchvision.datasets import MNIST
    try:
        dataset = MNIST(str(root), train=True, download=download)
    except RuntimeError as error:
        raise RuntimeError(f"MNIST cache를 확인하세요: {root}. 첫 실행은 --download를 추가하세요.") from error
    generator = torch.Generator().manual_seed(2026)
    order = torch.randperm(len(dataset.data), generator=generator)
    train_ids, val_ids = order[:-5000], order[-5000:]
    if limit is not None:
        if not 10 <= limit <= len(train_ids):
            raise ValueError("train-limit은 10~55000")
        train_ids = train_ids[:limit]
    return dataset.data, dataset.targets, train_ids, val_ids


def image_batch(images, labels, indices, device):
    # background=0(uint8) -> -1(float). resize/stretch로 숫자 모양을 바꾸지 않는다.
    x = images[indices, None].to(device, dtype=torch.float32)/127.5-1
    return F.pad(x, (2, 2, 2, 2), value=-1), labels[indices].to(device)


@torch.no_grad()
def update_ema(ema, model, decay):
    for target, source in zip(ema.parameters(), model.parameters()):
        target.lerp_(source, 1-decay) if source.requires_grad else target.copy_(source)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


@torch.no_grad()
def validate(model, ema, data, device, out, index, preview_steps):
    images, labels, _, ids = data
    old_mode = model.training
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    # validation 빈도/생성 step이 학습의 다음 random batch를 바꾸지 않게 한다.
    with torch.random.fork_rng(devices=devices):
        model.eval()
        ema.eval()
        g = torch.Generator().manual_seed(901)
        x1, y = image_batch(images, labels, ids[:128], device)
        x0 = torch.randn(x1.shape, generator=g).to(device)
        t = torch.linspace(.02, .98, len(x1), device=device)
        xt, velocity = L00.conditional_path(x0, x1, t)
        result, previews = {}, []
        for name, network in (("model", model), ("ema", ema)):
            result[name] = float(F.mse_loss(network(xt, t, y).float(), velocity))
            generated, _, _ = lesson(5).sample(network, torch.arange(10, device=device),
                steps=preview_steps, solver="heun", guidance=1., seed=777, keep=False)
            previews.append(generated)
        if not all(math.isfinite(v) for v in result.values()):
            raise FloatingPointError("nonfinite validation")
        image_grid(torch.cat(previews), out / f"{index}.png", labels=[f"{s}: {i}" for s in ("model", "EMA") for i in range(10)],
                   title=f"Step {index} | fixed noise/classes | Heun NFE={2*preview_steps} | top model / bottom EMA")
        model.train(old_mode)
    return result


def plot_loss(history, out):
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    train = [h for h in history if h["kind"] == "train"]
    val = [h for h in history if h["kind"] == "validation"]
    axes[0].plot([h["step"] for h in train], [h["loss"] for h in train])
    axes[0].set(title="Random t/noise CFM training", xlabel="optimizer step", ylabel="velocity MSE")
    for name in ("model", "ema"):
        axes[1].plot([h["step"] for h in val], [h[name] for h in val], label=name)
    axes[1].set(title="Fixed heldout images/t/noise", xlabel="optimizer step", ylabel="velocity MSE")
    axes[1].legend()
    finish(fig, out / "loss.png")


def train(a):
    device = setup(a.seed, a.device)
    if min(a.steps, a.batch_size, a.val_every, a.preview_steps) < 1 or a.lr <= 0 or not 0 <= a.ema_decay < 1:
        raise ValueError("학습 수치 설정이 잘못됐다")
    if a.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16은 지원하는 CUDA GPU에서만 사용. --precision fp32 사용 가능")
    out = a.out or (a.resume.parent if a.resume else ROOT / "outputs/04")
    if not a.resume and (out / "latest.pt").exists():
        raise FileExistsError("새 --out 또는 --resume outputs/04/latest.pt 사용")
    data = mnist(a.data_root, a.train_limit, a.download)
    config = PROFILES[a.profile]
    settings = dict(batch_size=a.batch_size, lr=a.lr, ema_decay=a.ema_decay, seed=a.seed,
                    train_limit=a.train_limit, precision=a.precision)
    raw = a.data_root / "MNIST/raw"
    fingerprint = dict(images=digest(raw / "train-images-idx3-ubyte"), labels=digest(raw / "train-labels-idx1-ubyte"),
                       dit=digest(DIT_SOURCE), path=digest(Path(L00.__file__)),
                       model=digest(Path(__file__)))
    model = FlowDiT(**config).to(device)
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.)
    begin, history, best = 0, [], float("inf")
    if a.resume:
        ck = load(a.resume)
        if ck["format"] != "fm_image_v1" or ck["model_config"] != config or ck["settings"] != settings or ck["fingerprint"] != fingerprint:
            raise ValueError("resume의 model/data/source/optimizer 설정 불일치. 기존 설정을 유지하세요.")
        if (out / "latest.pt").is_file() and load(out / "latest.pt")["step"] > ck["step"]:
            raise ValueError("더 최근 checkpoint가 있는 폴더를 과거 checkpoint로 덮을 수 없다. 새 --out 사용")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer"])
        begin, history, best = ck["step"], ck["history"], ck["best"]
        restore_rng(ck["rng"], device)
    if begin >= a.steps:
        raise ValueError("--steps는 checkpoint보다 큰 누적 목표 step이어야 한다")
    out.mkdir(parents=True, exist_ok=True)
    if not begin:
        values = validate(model, ema, data, device, out, 0, a.preview_steps)
        history.append(dict(kind="validation", step=0, **values))
    model.train()
    images, labels, train_ids, _ = data
    progress = trange(begin+1, a.steps+1, desc="DiT velocity / CFM")
    for i in progress:
        ids = train_ids[torch.randint(len(train_ids), (a.batch_size,))]
        x1, y = image_batch(images, labels, ids, device)
        # 학습은 random t 하나의 회귀다. ODE solver는 validation과 05에서만 사용한다.
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1), device=device)
        xt, target_velocity = L00.conditional_path(x0, x1, t)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if a.precision == "bf16" else nullcontext()
        with context:
            prediction = model(xt, t, y)  # train=True이면 원 DiT의 10% label dropout도 동작
            loss = F.mse_loss(prediction.float(), target_velocity)
        grad_norm = step(loss, model, optimizer)
        update_ema(ema, model, a.ema_decay)
        history.append(dict(kind="train", step=i, loss=float(loss.detach()), grad_norm=grad_norm))
        if i == 1:
            n = min(10, len(x1))
            image_grid(torch.cat([x1[:n], x0[:n], xt[:n], target_velocity[:n]/3]), out / "training_batch.png", columns=n,
                       title="Rows: data x1 / noise x0 / x_t / target velocity divided by 3 (display only)")
        if i % 10 == 0:
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
        if i % a.val_every == 0 or i == a.steps:
            values = validate(model, ema, data, device, out, i, a.preview_steps)
            history.append(dict(kind="validation", step=i, **values))
            improved = values["ema"] < best
            best = min(best, values["ema"])
            ck = dict(format="fm_image_v1", path="linear", time_direction="noise0_to_data1", prediction="velocity",
                step=i, model=cpu_state(model), ema=cpu_state(ema), optimizer=optimizer.state_dict(),
                model_config=config, settings=settings, fingerprint=fingerprint, rng=state_rng(device),
                best=best, history=history, data_root=str(a.data_root.resolve()))
            save(out / "latest.pt", ck)
            if improved:
                save(out / "best.pt", ck)
            json_file(out / "history.json", history)
            plot_loss(history, out)
    print(f"{out / 'latest.pt'} | train={len(train_ids)}, validation={len(data[3])}, official test unused")


def main():
    shared = ROOT.parent / "dit/data"
    p = parser(__doc__)
    p.add_argument("--data-root", type=Path, default=shared if (shared / "MNIST/raw/train-images-idx3-ubyte").is_file() else ROOT / "data")
    p.add_argument("--download", action="store_true", help="MNIST가 없을 때 torchvision으로 다운로드")
    p.add_argument("--out", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--profile", choices=list(PROFILES), default="mini")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema-decay", type=float, default=.995)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--preview-steps", type=int, default=24)
    p.add_argument("--train-limit", type=int)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=7)
    train(p.parse_args())


if __name__ == "__main__":
    main()
