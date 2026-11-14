"""01 — endpoint를 모르는 network가 conditional velocity로 학습할 수 있는 이유.

repo root:
    python lessons/flow_matching/01_conditional_flow_matching.py
GPU 사용 (CPU로도 실행 가능):
    CUDA_VISIBLE_DEVICES=7 python 01_conditional_flow_matching.py --device cuda

출력: outputs/01/{field.pt,0.png,100.png,...,loss.png,conditional_vs_marginal.png}.
읽기: oracle_velocity -> VelocityMLP -> train_field.
Network 입력은 x_t와 t뿐이다. loss 정답에만 x0/x1이 들어간다.
CFM의 conditional은 class conditioning과 다른 개념이다. 이 단계는 unconditional 생성이다.
"""
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import trange

from common import ROOT, cpu_state, finish, fresh, json_file, lesson, parser, pyplot, save, setup, step

L00 = lesson(0)


def oracle_velocity(x, t):
    """independent linear coupling의 정확한 marginal field E[x1-x0 | x_t=x].

    x1|k ~ N(mu_k, tau² I), x0 ~ N(0,I)
    x_t|k ~ N(t*mu_k, s² I), s²=(1-t)²+t²*tau²
    E[u|x_t,k] = mu_k + (t*tau²-(1-t))/s² * (x_t-t*mu_k)
    마지막에 Bayes posterior p(k|x_t)로 평균한다. train label로 사용하지 않는다.
    """
    angle = torch.arange(8, device=x.device, dtype=x.dtype)*math.pi/4
    mu = 2.5*torch.stack([angle.cos(), angle.sin()], -1)
    t = t[:, None, None]
    variance = (1-t).square()+t.square()*.12**2
    centered = x[:, None]-t*mu[None]
    log_prob = -centered.square().sum(-1)/(2*variance[:, :, 0])
    posterior = log_prob.softmax(-1)
    conditional = mu[None] + (t*.12**2-(1-t))/variance*centered
    return (posterior[..., None]*conditional).sum(1)


class VelocityMLP(nn.Module):
    """[x,y,t,sin(2πkt),cos(2πkt)] -> 현재 위치에서 이동할 2D velocity."""
    def __init__(self, width=128):
        super().__init__()
        self.width = width
        self.net = nn.Sequential(nn.Linear(11, width), nn.SiLU(), nn.Linear(width, width),
                                 nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))

    def forward(self, x, t):
        t = t.expand(len(x)) if t.ndim == 0 else t
        angles = t[:, None]*x.new_tensor([1., 2., 4., 8.])[None]*(2*math.pi)
        return self.net(torch.cat([x, t[:, None], angles.sin(), angles.cos()], -1))


def pair_batch(n, device, coupling="independent", pool=None, generator=None):
    if pool is not None:
        ids = torch.randint(len(pool[0]), (n,), device=device, generator=generator)
        return pool[0][ids], pool[1][ids]
    x0 = torch.randn(n, 2, device=device, generator=generator)
    x1, _ = L00.mixture(n, device, generator)
    if coupling == "minibatch_ot":
        # 대응만 바꾼다. 두 batch의 empirical marginals는 동일하다.
        from scipy.optimize import linear_sum_assignment
        cost = torch.cdist(x0, x1).square().detach().cpu().numpy()
        rows, cols = linear_sum_assignment(cost)
        order = torch.empty(n, device=device, dtype=torch.long)
        order[torch.as_tensor(rows, device=device)] = torch.as_tensor(cols, device=device)
        x1 = x1[order]
    elif coupling != "independent":
        raise ValueError(coupling)
    return x0, x1


def train_field(steps, device, *, coupling="independent", pool=None, validation_pool=None,
                width=128, batch_size=256, seed=7, every=100, callback=None):
    if min(steps, batch_size, width, every) < 1:
        raise ValueError("steps/batch/width/every must be positive")
    if pool is not None and validation_pool is None:
        raise ValueError("reflow validation에는 별도의 teacher pair가 필요하다")
    torch.manual_seed(seed)
    model = VelocityMLP(width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    g = torch.Generator(device=device).manual_seed(777)
    va0, va1 = pair_batch(512, device, coupling, validation_pool, g)
    vt = torch.rand(512, device=device, generator=g)
    vx, vu = L00.conditional_path(va0, va1, vt)
    oracle = oracle_velocity(vx, vt) if coupling == "independent" and pool is None else None
    history = []

    @torch.no_grad()
    def validate(i, train_loss):
        model.eval()
        prediction = model(vx, vt)
        row = dict(step=i, train_loss=train_loss, cfm_validation=float(F.mse_loss(prediction, vu)),
                   marginal_mse=float(F.mse_loss(prediction, oracle)) if oracle is not None else None)
        history.append(row)
        if callback:
            callback(model, i, history)
        model.train()

    validate(0, None)
    progress = trange(1, steps+1, desc=f"CFM {coupling if pool is None else 'reflow'}")
    for i in progress:
        # 이 6줄이 학습의 핵심: ODE integration이나 teacher rollout이 없다.
        x0, x1 = pair_batch(batch_size, device, coupling, pool)
        t = torch.rand(batch_size, device=device)
        xt, target_velocity = L00.conditional_path(x0, x1, t)
        prediction = model(xt, t)
        loss = F.mse_loss(prediction, target_velocity)
        step(loss, model, optimizer)
        if i % every == 0 or i == steps:
            validate(i, float(loss.detach()))
            progress.set_postfix(cfm=f"{history[-1]['cfm_validation']:.3f}")
    return model.eval(), history


@torch.no_grad()
def plot_field(model, out, index):
    device = next(model.parameters()).device
    line = torch.linspace(-3.1, 3.1, 19, device=device)
    yy, xx = torch.meshgrid(line, line, indexing="ij")
    grid = torch.stack([xx.flatten(), yy.flatten()], -1)
    plt = pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for col, time in enumerate((.15, .5, .85)):
        t = grid.new_full((len(grid),), time)
        for row, velocity in enumerate((oracle_velocity(grid, t), model(grid, t))):
            g, v = grid.cpu().numpy(), velocity.cpu().numpy()
            axes[row, col].quiver(g[:, 0], g[:, 1], v[:, 0], v[:, 1], scale=45)
            axes[row, col].set(title=f"{'Analytic marginal' if row == 0 else 'Learned'} t={time}",
                               xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), aspect="equal")
    finish(fig, out / f"{index}.png")


def ambiguity_plot(out):
    # 같은 작은 공간 cell에도 서로 다른 pair velocity가 들어온다.
    g = torch.Generator().manual_seed(44)
    x0, x1 = pair_batch(50000, "cpu", generator=g)
    xt, u = L00.conditional_path(x0, x1, torch.full((len(x0),), .5))
    center = torch.tensor([1., 0.])
    near = (xt-center).norm(dim=-1) < .22
    conditional = u[near][:120]
    if len(conditional) < 2:
        raise RuntimeError("시각화할 sample이 부족하다")
    expected = oracle_velocity(center[None], torch.tensor([.5]))[0]
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    positions = xt[near][:len(conditional)]
    axes[0].quiver(positions[:, 0], positions[:, 1], conditional[:, 0], conditional[:, 1], alpha=.25, scale=50)
    axes[0].set(title="Different training pair velocities near x", xlim=(.4, 1.6), ylim=(-.6, .6), aspect="equal")
    axes[1].scatter(conditional[:, 0], conditional[:, 1], s=9, alpha=.4, label="conditional targets")
    axes[1].scatter(*expected, color="red", marker="x", s=90, label="analytic E[u | x,t]")
    axes[1].set(title="What MSE regression learns", xlabel="velocity x", ylabel="velocity y")
    axes[1].legend()
    finish(fig, out / "conditional_vs_marginal.png")


def main():
    p = parser(__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/01")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--every", type=int, default=100)
    p.add_argument("--device", default="cpu")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    device = setup(device=a.device)
    fresh(a.out / "field.pt", a.overwrite)
    ambiguity_plot(a.out)
    model, history = train_field(a.steps, device, every=a.every,
                                callback=lambda m, i, h: plot_field(m, a.out, i))
    save(a.out / "field.pt", dict(format="fm_2d_v1", model=cpu_state(model), width=model.width,
         step=a.steps, history=history, path="linear", coupling="independent"))
    json_file(a.out / "history.json", history)
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ("cfm_validation", "marginal_mse"):
        ax.plot([h["step"] for h in history], [h[key] for h in history], label=key)
    ax.set(xlabel="optimizer step", ylabel="MSE", title="Conditional targets have irreducible variance")
    ax.legend()
    finish(fig, a.out / "loss.png")
    print(a.out / "field.pt")


if __name__ == "__main__":
    main()
