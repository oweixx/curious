r"""07 — noise에서 이미지를 생성한다. 하나의 diffusion step마다 전체 DiT를 호출한다.

    CUDA_VISIBLE_DEVICES=4 python 07_reverse_sampling.py --checkpoint checkpoints/mini/best.pt
    CUDA_VISIBLE_DEVICES=4 python 07_reverse_sampling.py --checkpoint checkpoints/mini/best.pt --sampler ddpm --steps 250 --out outputs/07_ddpm
    # 현재 weight와 EMA 비교:
    CUDA_VISIBLE_DEVICES=4 python 07_reverse_sampling.py --checkpoint checkpoints/mini/best.pt --weights model --out outputs/07_model

결과: samples.png, trajectory.png, trajectory.gif, reverse_step.png, run.json.
reverse_step_check.json은 epsilon/variance -> reverse Gaussian 수식의 일치를 검사한다.
trajectory는 각 sampling step의 predicted z0를 VAE decode한 이미지다.
z_t 자체의 decode와 구분한다. 마지막 것은 최종 sample이다.
DDPM: 학습된 reverse variance를 이용해 매 step noise 추가 (마지막 step 제외).
DDIM eta=0: 같은 initial noise/조건에서 결정론적 경로. learned variance는 쓰지 않는다.
steps=50도 model에는 원본1000 clock의 mapped timestep이 들어간다. 0..49로 바꾸지 않는다.
CFG paper: 원본 forward_with_cfg의 첫3개 epsilon 채널만 guide. all: 4개 모두 guide.
모델 전체 dropout은 eval에서 꺼져 있어야 한다.
"""

from pathlib import Path
import numpy as np
import torch
from diffusion import create_diffusion
from common import (
    ROOT,
    parser,
    setup,
    model_from_checkpoint,
    vae,
    decode,
    image_grid,
    write_json,
    pyplot,
    savefig,
)


def guided_forward(model, x, t, y, cfg_scale, mode="paper"):
    """한 z_t에 conditional/null 두 예측을 구하고 epsilon을 결합한다 (08에서 비교)."""
    if mode == "paper":
        return model.forward_with_cfg(x, t, y, cfg_scale)
    # Explicit all-channel variant; keep learned variance outputs unchanged.
    half = x[: len(x) // 2]
    out = model(torch.cat([half, half]), t, y)
    eps, rest = out[:, : model.in_channels], out[:, model.in_channels :]
    conditional, unconditional = eps.chunk(2)
    guided = unconditional + cfg_scale * (conditional - unconditional)
    return torch.cat([torch.cat([guided, guided]), rest], dim=1)


@torch.no_grad()
def sample(
    model,
    labels,
    steps=50,
    cfg=4.0,
    seed=42,
    sampler="ddim",
    cfg_mode="paper",
    progress=False,
    trace=False,
):
    """학습된 network를 반복 호출하는 고정 sampler. optimizer/backward가 없다."""
    if not 2 <= steps <= 1000:
        raise ValueError("sampling steps는 2..1000")
    if cfg_mode not in ("paper", "all") or sampler not in ("ddpm", "ddim"):
        raise ValueError("sampling mode")
    device = next(model.parameters()).device
    size = model.x_embedder.img_size[0]
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    n = len(labels)
    classes = model.y_embedder.num_classes
    if n == 0 or labels.min() < 0 or labels.max() >= classes:
        raise ValueError("class label 범위")
    # 이것은 새로운 network를 만드는 호출이 아니다. reverse 시간 격자를 만든다.
    # SpacedDiffusion은 줄어든 index를 original 1000-step clock으로 매핑한다.
    d = create_diffusion(str(steps))
    old = model.training
    model.eval()
    # Do not perturb training's RNG when called for fixed validation previews.
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    frames = []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            noise = torch.randn(n, 4, size, size, device=device)
            # 앞 n개는 class, 뒤 n개는 null. 같은 noisy latent를 두 조건으로 조회한다.
            noise = torch.cat([noise, noise])
            y = torch.cat([labels, torch.full_like(labels, classes)])
            call = lambda x, t, **kw: guided_forward(model, x, t, **kw, mode=cfg_mode)
            iterator = (
                d.p_sample_loop_progressive if sampler == "ddpm" else d.ddim_sample_loop_progressive
            )
            selected = set(torch.linspace(0, steps - 1, 8).round().int().tolist())
            for i, out in enumerate(
                iterator(
                    call,
                    noise.shape,
                    noise=noise,
                    clip_denoised=False,
                    model_kwargs=dict(y=y, cfg_scale=cfg),
                    device=device,
                    progress=progress,
                )
            ):
                # iterator 안에서 ① model로 epsilon/v 예측 ② z0/mean/variance 계산
                # ③ DDPM 또는 DDIM 수식으로 다음 latent 생성이 이루어진다.
                # 아래 explain_reverse_step은 ②를 풀어 써 원본 결과와 비교한다.
                if not torch.isfinite(out["sample"]).all():
                    raise FloatingPointError(f"nonfinite sample at step {i}")
                if trace and i in selected:
                    frames.append(
                        (
                            int(d.timestep_map[steps - i - 1]),
                            out["pred_xstart"][:n].detach().clone(),
                        )
                    )
            return out["sample"][:n], frames
    finally:
        model.train(old)


@torch.no_grad()
def explain_reverse_step(model, cfg, cfg_mode, out_dir):
    """한 시점에서 network 출력과 sampler 계산을 분리해서 관찰한다.

    공간은 latent이고, 이 예제의 x는 수식 검사용 입력이다. t=500의 실제 데이터
    분포에서 뽑았다고 해석하지 않는다. 단일 step 비교에는 원본1000 schedule을 쓴다.
    """
    device = next(model.parameters()).device
    size = model.x_embedder.img_size[0]
    x = torch.randn(1, 4, size, size, device=device).repeat(2, 1, 1, 1)
    t = torch.full((2,), 500, device=device, dtype=torch.long)
    y = torch.tensor([0, model.y_embedder.num_classes], device=device)
    d = create_diffusion("")
    prediction = guided_forward(model, x, t, y, cfg, cfg_mode)
    epsilon, v = prediction.chunk(2, dim=1)

    def coefficient(array):
        # 원본 _extract_into_tensor와 같은 float64 schedule -> float32 계수.
        return torch.as_tensor(array, device=device)[t].float()[:, None, None, None]

    # epsilon-prediction으로 clean latent를 추정한다. 01의 역산과 같은 식이다.
    z0 = coefficient(d.sqrt_recip_alphas_cumprod) * x
    z0 -= coefficient(d.sqrt_recipm1_alphas_cumprod) * epsilon

    # q(z_{t-1}|z_t,z0)의 z0 자리에 network의 추정 z0를 대입한다.
    # mu = coef1(t)*z0_hat + coef2(t)*z_t. z0를 추정한 뒤에도 sampler 계산이 남아 있다.
    mean = coefficient(d.posterior_mean_coef1) * z0 + coefficient(d.posterior_mean_coef2) * x
    lower = coefficient(d.posterior_log_variance_clipped)
    upper = coefficient(np.log(d.betas))
    fraction = (v + 1) / 2
    log_variance = fraction * upper + (1 - fraction) * lower
    std = (0.5 * log_variance).exp()
    # DDPM 다음 latent는 mu + std * fresh_noise (마지막 t=0에서는 noise 항을0으로).
    # DDIM eta=0는 위 std를 사용하지 않고 다른 고정 업데이트 식을 사용한다.

    reference = d.p_mean_variance(
        lambda z, time, **kw: guided_forward(model, z, time, **kw, mode=cfg_mode),
        x,
        t,
        clip_denoised=False,
        model_kwargs=dict(y=y, cfg_scale=cfg),
    )
    errors = {}
    for name, manual in (("pred_xstart", z0), ("mean", mean), ("log_variance", log_variance)):
        errors[name] = float((manual - reference[name]).abs().max())
        assert torch.allclose(manual, reference[name], atol=1e-6), (name, errors[name])
    plt = pyplot()
    fig, axes = plt.subplots(1, 6, figsize=(17, 3))
    for ax, value, name in zip(
        axes,
        (x, epsilon, v, z0, mean, std),
        (
            "Input z_t",
            "Predicted epsilon",
            "Predicted variance v",
            "Estimated z0",
            "Reverse mean",
            "Reverse std",
        ),
    ):
        im = ax.imshow(value[0, 0].cpu(), cmap="coolwarm")
        fig.colorbar(im, ax=ax, shrink=0.7, format="%.6f" if name == "Reverse std" else "%.2g")
        ax.set_title(name)
    fig.suptitle("One reverse step, t=500 | latent channel0 | independent color scales; not RGB")
    savefig(fig, out_dir / "reverse_step.png")
    write_json(out_dir / "reverse_step_check.json", dict(t=500, max_absolute_errors=errors))


def main():
    p = parser(__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/mini/best.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--weights", choices=["ema", "model"], default="ema")
    p.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--cfg-mode", choices=["paper", "all"], default="paper")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--labels", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--out", type=Path, default=ROOT / "outputs/07")
    a = p.parse_args()
    setup()
    model, ck = model_from_checkpoint(a.checkpoint, a.device, a.weights == "ema")
    v = vae(a.device)
    z, trace = sample(model, a.labels, a.steps, a.cfg, a.seed, a.sampler, a.cfg_mode, True, True)
    label = [f"class {c}" for c in a.labels]
    title = f"step {ck['step']} | {a.weights} | {a.sampler} | CFG {a.cfg} ({a.cfg_mode})"
    image_grid(decode(v, z), a.out / "samples.png", columns=len(label), labels=label, title=title)
    images = []
    caption = []
    animation = []
    for t, pred in trace:
        rgb = decode(v, pred)
        images.append(rgb)
        caption.extend([f"t={t} c={c}" for c in a.labels])
        animation.append(
            image_grid(
                rgb,
                a.out / ".frame.png",
                columns=len(label),
                labels=label,
                title=f"Predicted clean image | t={t}",
            )
        )
    image_grid(
        torch.cat(images),
        a.out / "trajectory.png",
        columns=len(label),
        labels=caption,
        title=title,
        cell=72,
    )
    animation[0].save(
        a.out / "trajectory.gif", save_all=True, append_images=animation[1:], duration=450, loop=0
    )
    (a.out / ".frame.png").unlink()
    explain_reverse_step(model, a.cfg, a.cfg_mode, a.out)
    write_json(
        a.out / "run.json",
        dict(
            checkpoint=str(a.checkpoint),
            training_step=ck["step"],
            weights=a.weights,
            seed=a.seed,
            sampler=a.sampler,
            steps=a.steps,
            cfg=a.cfg,
            cfg_mode=a.cfg_mode,
            labels=a.labels,
        ),
    )


if __name__ == "__main__":
    main()
