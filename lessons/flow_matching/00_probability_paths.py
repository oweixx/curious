"""00 — noise와 data 사이의 '경로'를 먼저 정한다. 아직 network는 없다.

repo root:
    python lessons/flow_matching/00_probability_paths.py
lesson 폴더:
    python 00_probability_paths.py

출력: outputs/00/{paths.png,conditional_paths.png,schedules.png,path.gif,checks.json}.
읽기: coefficients -> conditional_path -> main.
관례: x0=noise, x1=data, t=0→1이 생성 방향. DiT lesson의 noising 시간과 방향이 다르다.
공식 flow_matching의 AffineProbPath/CondOTScheduler/CosineScheduler 수식과 대응한다.
"""
import math
from pathlib import Path

import numpy as np
import torch

from common import ROOT, finish, json_file, parser, pyplot, scatter, setup


def coefficients(t, kind="linear", sigma_min=0.):
    """x_t = alpha(t) x1 + sigma(t) x0; derivative도 직접 제공한다.

    linear: alpha=t, sigma=1-t. 각 (x0,x1) pair의 경로는 직선.
    residual: 원 FM 논문의 작은 endpoint noise. x(1)=x1+sigma_min*x0.
    cosine: alpha²+sigma²=1인 대안. FM이 linear path 하나를 뜻하지 않음을 보여준다.
    """
    if not 0 <= sigma_min < 1:
        raise ValueError("sigma_min은 [0,1)")
    if kind == "linear":
        return t, 1-t, torch.ones_like(t), -torch.ones_like(t)
    if kind == "residual":
        return t, 1-(1-sigma_min)*t, torch.ones_like(t), -(1-sigma_min)*torch.ones_like(t)
    if kind == "cosine":
        angle = t*(math.pi/2)
        return angle.sin(), angle.cos(), (math.pi/2)*angle.cos(), -(math.pi/2)*angle.sin()
    raise ValueError(f"unknown path: {kind}")


def conditional_path(x0, x1, t, kind="linear", sigma_min=0.):
    """batch의 각 pair마다 t 하나를 뽑아 x_t와 그 점의 정답 속도를 바로 만든다.

    train step에서 ODE를 풀지 않는다. x1-x0는 입력 tensor와 같은 shape의 velocity다.
    t를 [B,1,...]로 바꾸지 않으면 image의 width 축에 잘못 broadcast될 수 있다.
    """
    if x0.shape != x1.shape or t.shape != (len(x0),):
        raise ValueError("x0/x1 shape가 같고 t는 [batch]여야 한다")
    if not torch.isfinite(t).all() or (t < 0).any() or (t > 1).any():
        raise ValueError("t must be finite and in [0,1]")
    values = [z.reshape(len(t), *([1]*(x0.ndim-1))) for z in coefficients(t, kind, sigma_min)]
    a, s, da, ds = values
    return a*x1+s*x0, da*x1+ds*x0


def mixture(n, device="cpu", generator=None):
    """정답 분포가 알려진 2D 8-Gaussian mixture. 데이터 파일이나 외부 모델 불필요."""
    k = torch.randint(8, (n,), generator=generator, device=device)
    angle = k*math.pi/4
    centers = 2.5*torch.stack([angle.cos(), angle.sin()], -1)
    return centers + .12*torch.randn(n, 2, generator=generator, device=device), k


def main():
    p = parser(__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/00")
    a = p.parse_args()
    setup()
    x0, (x1, labels) = torch.randn(1500, 2), mixture(1500)
    plt = pyplot()
    fig, axes = plt.subplots(3, 5, figsize=(14, 8))
    for row, kind in enumerate(("linear", "residual", "cosine")):
        for col, t in enumerate((0., .25, .5, .75, 1.)):
            xt, _ = conditional_path(x0, x1, torch.full((len(x0),), t), kind, .15)
            scatter(axes[row, col], xt, f"{kind}, t={t:g}", labels)
    finish(fig, a.out / "paths.png")
    times = torch.linspace(0, 1, 51)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, kind in zip(axes, ("linear", "residual", "cosine")):
        trace = torch.stack([conditional_path(x0[:12], x1[:12], t.expand(12), kind, .15)[0] for t in times])
        for i in range(12):
            ax.plot(trace[:, i, 0], trace[:, i, 1], alpha=.7)
        ax.scatter(x0[:12, 0], x0[:12, 1], marker="x", color="black", label="noise")
        ax.scatter(x1[:12, 0], x1[:12, 1], marker="o", facecolors="none", edgecolors="black", label="data")
        ax.set(title=f"Known endpoint pairs: {kind}", aspect="equal")
        ax.legend()
    finish(fig, a.out / "conditional_paths.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for kind in ("linear", "cosine"):
        alpha, sigma, da, ds = coefficients(times, kind)
        axes[0].plot(times, alpha, label=f"{kind}: alpha (data)")
        axes[0].plot(times, sigma, "--", label=f"{kind}: sigma (noise)")
        axes[1].plot(times, da, label=f"{kind}: d_alpha")
        axes[1].plot(times, ds, "--", label=f"{kind}: d_sigma")
    for ax in axes:
        ax.set_xlabel("generation time t")
        ax.legend(fontsize=8)
    finish(fig, a.out / "schedules.png")
    # Pillow GIF: training path interpolation임을 명시한다. 생성 network 결과가 아니다.
    from PIL import Image
    frames = []
    for t in torch.linspace(0, 1, 21):
        fig, ax = plt.subplots(figsize=(4, 4))
        xt, _ = conditional_path(x0, x1, t.expand(len(x0)))
        scatter(ax, xt, f"Training path (known pairs), t={float(t):.2f}", labels)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    frames[0].save(a.out / "path.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    start, _ = conditional_path(x0, x1, torch.zeros(len(x0)))
    end, _ = conditional_path(x0, x1, torch.ones(len(x0)))
    assert torch.equal(start, x0) and torch.equal(end, x1)
    json_file(a.out / "checks.json", dict(start_equals_noise=True, end_equals_data=True,
        residual_endpoint="x1 + sigma_min * x0; not exactly clean data"))
    print(a.out)


if __name__ == "__main__":
    main()
