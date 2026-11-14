"""02 — 학습한 velocity를 ODE로 적분하면 생성이 된다. 생성 시 x1을 주지 않는다.

repo root:
    python lessons/flow_matching/02_ode_sampling.py
lesson 폴더:
    python 02_ode_sampling.py --checkpoint outputs/01/field.pt

출력: outputs/02/{trajectories.png,sampling.gif,nfe.png,metrics.json}.
읽기: integrate -> sliced_w2 -> main.
Euler/Heun/RK4를 동일한 NFE(network field evaluation 수)로 비교한다.
모델 오차와 solver 오차는 다르다. 같은 모델의 촘촘한 RK4 결과는 수치 기준이지 데이터 정답이 아니다.
"""
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from common import ROOT, finish, json_file, lesson, load, parser, pyplot, scatter, setup


@torch.no_grad()
def integrate(field, x0, steps=32, method="heun", start=0., end=1., keep=True):
    """dx/dt = field(x,t). solver에는 target endpoint나 정답 velocity를 넘기지 않는다.

    Heun: 현재 기울기로 predictor -> endpoint 기울기 -> 평균 기울기로 corrector.
    RK4: 구간 안에서 기울기를 네 번 평가한다. 'steps=32'와 'NFE=32'를 구분한다.
    end<start도 허용해 numerical round-trip을 검사할 수 있다.
    """
    if steps < 1 or method not in ("euler", "heun", "rk4"):
        raise ValueError("positive steps and euler/heun/rk4 required")
    x = x0.clone()
    times = torch.linspace(start, end, steps+1, device=x.device, dtype=x.dtype)
    trace = [x.clone()] if keep else None
    nfe = 0
    for t0, t1 in zip(times[:-1], times[1:]):
        dt = t1-t0
        k1 = field(x, t0)
        nfe += 1
        if method == "euler":
            x = x + dt*k1
        elif method == "heun":
            k2 = field(x+dt*k1, t1)
            nfe += 1
            x = x + .5*dt*(k1+k2)
        else:
            k2 = field(x+.5*dt*k1, t0+.5*dt)
            k3 = field(x+.5*dt*k2, t0+.5*dt)
            k4 = field(x+dt*k3, t1)
            nfe += 3
            x = x+dt*(k1+2*k2+2*k3+k4)/6
        if not torch.isfinite(x).all():
            raise FloatingPointError("ODE diverged. 작은 step/학습 상태를 확인하세요.")
        if keep:
            trace.append(x.clone())
    return x, torch.stack(trace) if keep else None, nfe


def field_from_checkpoint(path, device):
    ck = load(path)
    if ck["format"] != "fm_2d_v1" or ck["path"] != "linear":
        raise ValueError("01의 linear flow checkpoint가 필요하다")
    model = lesson(1).VelocityMLP(ck["width"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck


@torch.no_grad()
def sliced_w2(x, y, projections=128):
    """무작위 1D projection별 정렬된 empirical W2의 RMS. point-to-point GT 오차가 아니다."""
    if x.shape != y.shape:
        raise ValueError("같은 수의 samples 필요")
    g = torch.Generator(device=x.device).manual_seed(912)
    direction = torch.randn(x.shape[-1], projections, device=x.device, dtype=x.dtype, generator=g)
    direction /= direction.norm(dim=0, keepdim=True)
    a, b = (x@direction).sort(dim=0).values, (y@direction).sort(dim=0).values
    return float((a-b).square().mean().sqrt())


def path_bending(trace):
    """trajectory와 자기 endpoints의 직선 보간 간 RMS. coupling의 OT 여부 판정은 아니다."""
    t = torch.linspace(0, 1, len(trace), device=trace.device)[:, None, None]
    line = (1-t)*trace[:1]+t*trace[-1:]
    return float((trace-line).square().mean().sqrt())


def main():
    p = parser(__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/01/field.pt")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/02")
    p.add_argument("--samples", type=int, default=1024)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    device = setup(device=a.device)
    if a.samples < 16:
        raise ValueError("samples>=16 필요")
    model, ck = field_from_checkpoint(a.checkpoint, device)
    x0 = torch.randn(a.samples, 2, device=device)
    target, _ = lesson(0).mixture(a.samples, device)
    reference, _, _ = integrate(model, x0, 256, "rk4", keep=False)
    _, trace, _ = integrate(model, x0, 64, "heun")
    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    scatter(axes[0], x0, "Noise only at t=0")
    scatter(axes[1], trace[-1], "Generated t=1")
    scatter(axes[2], target, "Independent target samples")
    paths = trace[:, :24].cpu().numpy()
    for i in range(paths.shape[1]):
        axes[1].plot(paths[:, i, 0], paths[:, i, 1], lw=.7, alpha=.5)
    finish(fig, a.out / "trajectories.png")
    images = []
    for i in range(0, len(trace), 2):
        fig, ax = plt.subplots(figsize=(4, 4))
        scatter(ax, trace[i], f"Learned ODE, t={i/64:.2f}")
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(fig)
    images[0].save(a.out / "sampling.gif", save_all=True, append_images=images[1:], duration=100, loop=0)
    rows = []
    for method, cost in (("euler", 1), ("heun", 2), ("rk4", 4)):
        for budget in (4, 8, 16, 32, 64, 128):
            output, _, nfe = integrate(model, x0, budget//cost, method, keep=False)
            rows.append(dict(method=method, nfe=nfe, steps=budget//cost,
                endpoint_rmse=float((output-reference).square().mean().sqrt()), sliced_w2=sliced_w2(output, target)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in ("euler", "heun", "rk4"):
        selected = [r for r in rows if r["method"] == method]
        for ax, key in zip(axes, ("endpoint_rmse", "sliced_w2")):
            ax.plot([r["nfe"] for r in selected], [r[key] for r in selected], marker="o", label=method)
            ax.set(xscale="log", xlabel="field NFE", ylabel=key)
            ax.legend()
    axes[0].set_title("Numerical error vs same model RK4 reference")
    axes[1].set_title("Distribution discrepancy vs new data samples")
    finish(fig, a.out / "nfe.png")
    json_file(a.out / "metrics.json", dict(rows=rows, learned_path_bending=path_bending(trace),
        training_steps=ck["step"], reference="same model RK4, 256 steps; not ground truth"))
    print(a.out)


if __name__ == "__main__":
    main()
