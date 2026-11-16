r"""06 — 공식 DiT의 실제 학습: random t/noise, epsilon+VB loss, AdamW, EMA, CFG dropout.

    CUDA_VISIBLE_DEVICES=4 python 06_train.py
    # 중단된 학습 재개; --steps는 누적 목표 step:
    CUDA_VISIBLE_DEVICES=4 python 06_train.py --resume checkpoints/mini/latest.pt --steps 40000
    # 같은 원본 DiT-S 구조로 확장:
    CUDA_VISIBLE_DEVICES=4 python 06_train.py --profile S --out checkpoints/small
    # 짧은 실행 검사(최종 생성 품질을 뜻하지 않는다):
    CUDA_VISIBLE_DEVICES=4 python 06_train.py --steps 4 --val-every 2 --preview-steps 10 --out .cache/train_smoke

기본: data/mnist64, 20,000 optimizer steps, original DiT block 4개/width192/head6, batch128, lr1e-4,
weight_decay0, mini의 EMA0.999, 1000 diffusion timesteps, learned variance.
원본 EMA0.9999는 --ema-decay 0.9999로 지정. S/B/L/XL은 기본0.9999.
짧은 mini 실습의 EMA 시간척도를 줄인 선택이며 모델/loss 변경이 아니다.
원본 DiT-S/B/L/XL profile도 지원. 작은 mini는 lesson용 크기이고 논문 모델명이 아니다.
VAE posterior는 매 visit 다시 sample한다. MNIST는 flip하면 의미가 바뀌므로 horizontal flip 없음.

결과: --out/{latest.pt,best.pt,metrics.jsonl,curves.png,samples/0.png,100.png,...}.
고정 validation t/noise/seed, EMA의 MSE+VB 기준 best. 생성 샘플은 같은 초기 noise와 class.
샘플 preview는50-step DDIM이고 원논문의250-step DDPM 평가/FID와 다르다.
validation에는 dropout이 꺼진 conditional model을 사용한다. training은 CFG label-dropout 활성.
sampling을 매번 학습하는 것이 아니다: train step에서는 random t에서 단1회 forward.

읽는 순서: posterior -> training_losses -> backward/step -> update_ema -> validate.
원본 참고: original train.py의 계산을 단일 GPU + 저장/재개/시각화로 확장했다.
"""

from pathlib import Path
from copy import deepcopy
import json
import math
import torch
from tqdm.auto import tqdm
from diffusion import create_diffusion
from common import (
    ROOT,
    PROFILES,
    setup,
    parser,
    data,
    make_model,
    posterior,
    vae,
    decode,
    image_grid,
    lesson,
    update_ema,
    save,
    load,
    digest,
    rng_state,
    restore_rng,
    pyplot,
    savefig,
)


@torch.no_grad()
def validate(model, ema, val, v, step, out, preview_steps, seed):
    """학습과 분리된 두 관찰: 고정 noising 문제의 loss + noise부터 생성한 이미지.

    loss를 측정할 때는 정답 이미지가 있고, sampling에는 정답 이미지가 없다.
    RNG를 복원하므로 validation 빈도를 바꿔도 다음 training batch/noise는 같다.
    """
    state = rng_state()
    old = model.training
    model.eval()
    ema.eval()
    try:
        torch.manual_seed(seed)
        device = next(model.parameters()).device
        n = min(128, len(val["labels"]))
        x, y = posterior(val, slice(0, n), device)
        t = torch.arange(n, device=device) * 999 // max(n - 1, 1)
        noise = torch.randn_like(x)
        d = create_diffusion("")
        values = {}
        for name, network in (("model", model), ("ema", ema)):
            terms = d.training_losses(network, x, t, dict(y=y), noise=noise)
            for key, value in terms.items():
                values[f"{name}_{key}"] = float(value.mean())
        if not all(math.isfinite(x) for x in values.values()):
            raise FloatingPointError("validation nonfinite")
        previews = []
        for network in (model, ema):
            z, _ = lesson(7).sample(network, list(range(10)), steps=preview_steps, seed=seed)
            previews.append(decode(v, z))
        image_grid(
            torch.cat(previews),
            out / "samples" / f"{step}.png",
            labels=[f"{name} c{i}" for name in ("model", "EMA") for i in range(10)],
            title=f"Current model / EMA | step {step} | fixed seed {seed} | DDIM {preview_steps}",
        )
        return values
    finally:
        model.train(old)
        restore_rng(state)


def plot_history(out):
    rows = [
        json.loads(line)
        for line in (out / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    plt = pyplot()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    tr = [r for r in rows if r["kind"] == "train"]
    va = [r for r in rows if r["kind"] == "validation"]
    for key in ("mse", "vb"):
        ax[0].plot([r["step"] for r in tr], [r[key] for r in tr], label=key)
    for key in ("model_loss", "ema_loss"):
        ax[1].plot([r["step"] for r in va], [r[key] for r in va], label=key)
    for a in ax:
        a.set_xlabel("Optimizer step")
        a.legend()
        a.grid(alpha=0.2)
    ax[0].set_title("Training (random t/noise)")
    ax[1].set_title("Held-out fixed t/noise")
    savefig(fig, out / "curves.png")


def main():
    p = parser(__doc__)
    p.add_argument("--data")
    p.add_argument("--out", type=Path, default=ROOT / "checkpoints/mini")
    p.add_argument("--profile", choices=list(PROFILES), default="mini")
    p.add_argument("--patch", type=int, default=2)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ema-decay", type=float)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--preview-steps", type=int, default=50)
    p.add_argument("--resume", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    setup(a.seed)
    if a.ema_decay is None:
        a.ema_decay = 0.999 if a.profile == "mini" else 0.9999
    if (
        min(a.steps, a.batch_size, a.val_every) < 1
        or not 0 <= a.ema_decay < 1
        or not 2 <= a.preview_steps <= 1000
    ):
        p.error("invalid training settings")
    folder, tr, val = data(a.data)
    size = tr["mean"].shape[-1]
    if a.patch < 1 or size % a.patch:
        p.error("patch must divide latent size")
    cfg = dict(input_size=size, profile=a.profile, patch=a.patch, classes=10)
    fingerprint = dict(
        train=digest(folder / "train.pt"),
        validation=digest(folder / "validation.pt"),
        vae=digest(ROOT / "weights/vae_revision.json"),
        model=digest(ROOT / "models.py"),
        diffusion={f.name: digest(f) for f in (ROOT / "diffusion").glob("*.py")},
    )
    settings = dict(lr=a.lr, ema_decay=a.ema_decay, batch_size=a.batch_size, seed=a.seed)
    if (a.out / "latest.pt").exists() and not a.resume:
        raise FileExistsError("--resume 또는 새 --out 사용")
    model = make_model(**cfg).to(a.device)
    # EMA는 두 번째 학습 network가 아니다. optimizer에는 model만 들어간다.
    # 같은 weight에서 출발하고, optimizer step 뒤 과거/현재 weight를 섞어 갱신한다.
    ema = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0)
    d = create_diffusion("")
    v = vae(a.device)
    start = 0
    best = float("inf")
    if a.resume:
        ck = load(a.resume)
        if (
            ck["model_config"] != cfg
            or ck["fingerprint"] != fingerprint
            or ck["settings"] != settings
        ):
            raise ValueError("resume model/data/optimizer 설정 불일치")
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer"])
        start = ck["step"]
        best = ck["best"]
        restore_rng(ck["rng"])
    if start >= a.steps:
        raise ValueError("steps는 checkpoint step보다 커야 합니다")
    a.out.mkdir(parents=True, exist_ok=True)
    model.train()

    def record(row):
        with (a.out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")

    if not a.resume:
        record(
            dict(
                kind="validation",
                step=0,
                **validate(model, ema, val, v, 0, a.out, a.preview_steps, a.seed + 100),
            )
        )
    bar = tqdm(range(start + 1, a.steps + 1), desc="DiT training")
    for step in bar:
        # 1. x는 RGB가 아니라 VAE posterior에서 새로 뽑은 clean latent z_0.
        #    cache는 mean/logvar이고, 여기서는 매번 다른 posterior sample을 쓴다.
        # sampling 
        ids = torch.randint(len(tr["labels"]), (a.batch_size,))
        x, y = posterior(tr, ids, a.device)

        # 2. sample마다 t 하나를 뽑는다. 1000번의 noising loop가 아니다.
        #    training_losses 내부 순서:
        #    epsilon 생성 -> q_sample(x,t,epsilon) -> model(z_t,t,y)
        #    -> epsilon MSE + variance VB. 식/gradient는 01과05에서 확인한다.
        #    model.train()이므로 LabelEmbedder가 label의10%를 null로 바꾼다.
        # landom t를 뽑는다
        t = torch.randint(d.num_timesteps, (len(x),), device=a.device)
        terms = d.training_losses(model, x, t, dict(y=y))
        loss = terms["loss"].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at {step}")

        # 3. 이번 random t에서의 오차만으로 모든 학습 가능한 DiT weight를 update.
        #    VAE decode, 반복 reverse sampling은 이 backward에 들어가지 않는다.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # threshold=inf라 gradient 크기를 제한하지 않고 NaN/Inf만 검사한다.
        torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        optimizer.step()
        update_ema(ema, model, a.ema_decay)
        values = {k: float(v.mean()) for k, v in terms.items()}
        record(dict(kind="train", step=step, **values))
        bar.set_postfix(loss=f"{float(loss):.4f}", mse=f"{values['mse']:.4f}")
        if step % a.val_every == 0 or step == a.steps:
            # 4. eval/no_grad에서 실제 reverse sampling. 학습과 inference의 경계다.
            metrics = validate(model, ema, val, v, step, a.out, a.preview_steps, a.seed + 100)
            record(dict(kind="validation", step=step, **metrics))
            improved = metrics["ema_loss"] < best
            best = min(best, metrics["ema_loss"])
            ck = dict(
                model=model.state_dict(),
                ema=ema.state_dict(),
                optimizer=optimizer.state_dict(),
                step=step,
                best=best,
                model_config=cfg,
                settings=settings,
                fingerprint=fingerprint,
                rng=rng_state(),
            )
            save(a.out / "latest.pt", ck)
            if improved:
                save(a.out / "best.pt", ck)
            plot_history(a.out)
    print("학습 완료:", a.out, "| best는 EMA validation loss 기준")


if __name__ == "__main__":
    main()
