"""03 — 경로의 모양만큼 '어떤 noise와 어떤 data를 짝짓는가'도 중요하다.

repo root:
    CUDA_VISIBLE_DEVICES=7 python lessons/flow_matching/03_coupling_and_reflow.py --device cuda
lesson 폴더 (01 먼저 실행):
    python 03_coupling_and_reflow.py --teacher outputs/01/field.pt

출력: outputs/03/{couplings.png,samples.png,nfe.png,metrics.json,ot.pt,reflow.pt}.
읽기: plot_pairs -> teacher pair 생성 -> pair_batch(minibatch_ot) -> train_field -> 평가.
원 FM의 conditional OT path와, batch 전체의 OT matching은 다른 선택이다.
Reflow는 teacher의 생성 endpoint로 새 coupling을 만들고 다시 학습하는 추가 단계다.
이 파일은 Rectified Flow/OT-CFM의 작은 2D 실험이며 원 논문의 대규모 재현이 아니다.
"""
from pathlib import Path
import time

import torch

from common import ROOT, cpu_state, finish, fresh, json_file, lesson, parser, pyplot, save, scatter, setup

L01, L02 = lesson(1), lesson(2)


def clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def plot_pairs(out, teacher, device):
    from scipy.optimize import linear_sum_assignment
    g = torch.Generator(device=device).manual_seed(61)
    x0, x1 = L01.pair_batch(64, device, generator=g)
    _, order = linear_sum_assignment(torch.cdist(x0, x1).square().cpu().numpy())
    ot = x1[torch.as_tensor(order, device=device)]
    generated, _, _ = L02.integrate(teacher, x0, 64, "heun", keep=False)
    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    costs = {}
    for ax, name, endpoint in zip(axes, ("Independent pairs", "Minibatch OT pairs", "Teacher/reflow pairs"), (x1, ot, generated)):
        start, end = x0.cpu().numpy(), endpoint.cpu().numpy()
        for a, b in zip(start, end):
            ax.plot([a[0], b[0]], [a[1], b[1]], lw=.6, alpha=.35)
        ax.scatter(start[:, 0], start[:, 1], s=9, color="gray")
        ax.scatter(end[:, 0], end[:, 1], s=12, color="darkorange")
        cost = float((x0-endpoint).square().sum(-1).mean())
        costs[name] = cost
        ax.set(title=f"{name}\nmean squared displacement={cost:.3f}", aspect="equal")
    finish(fig, out / "couplings.png")
    return costs


def main():
    p = parser(__doc__)
    p.add_argument("--teacher", type=Path, default=ROOT / "outputs/01/field.pt")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/03")
    p.add_argument("--steps", type=int, help="각 새 모델의 step. 생략하면 teacher training step과 동일")
    p.add_argument("--pairs", type=int, default=4096)
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    device = setup(device=a.device)
    if a.pairs < 32 or a.samples < 16 or (a.steps is not None and a.steps < 1):
        raise ValueError("pairs>=32, samples>=16, steps>=1 필요")
    fresh(a.out / "metrics.json", a.overwrite)
    teacher, ck = L02.field_from_checkpoint(a.teacher, device)
    if ck["coupling"] != "independent":
        raise ValueError("비교의 teacher는 01 independent coupling checkpoint를 사용하세요")
    steps = ck["step"] if a.steps is None else a.steps
    pair_costs = plot_pairs(a.out, teacher, device)
    start = clock(device)
    ot, ot_history = L01.train_field(steps, device, coupling="minibatch_ot", width=ck["width"])
    ot_seconds = clock(device)-start
    # Teacher 생성은 추가 비용이다. 학생의 training loop 안에서 매번 ODE를 풀지 않는다.
    start = clock(device)
    g = torch.Generator(device=device).manual_seed(31)
    source = torch.randn(a.pairs+512, 2, device=device, generator=g)
    end, _, teacher_nfe = L02.integrate(teacher, source, 64, "heun", keep=False)
    pair_seconds = clock(device)-start
    pool, val_pool = (source[:a.pairs], end[:a.pairs]), (source[a.pairs:], end[a.pairs:])
    start = clock(device)
    reflow, rf_history = L01.train_field(steps, device, pool=pool, validation_pool=val_pool, width=ck["width"])
    reflow_seconds = clock(device)-start
    for name, model, history in (("ot", ot, ot_history), ("reflow", reflow, rf_history)):
        save(a.out / f"{name}.pt", dict(format="fm_2d_v1", model=cpu_state(model), width=model.width,
             step=steps, path="linear", coupling=name, history=history, teacher_steps=ck["step"]))
    g = torch.Generator(device=device).manual_seed(989)
    noise = torch.randn(a.samples, 2, device=device, generator=g)
    data, _ = lesson(0).mixture(a.samples, device, g)
    plt = pyplot()
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    rows, bends = [], {}
    for row, (name, model) in enumerate((("independent", teacher), ("minibatch_ot", ot), ("reflow", reflow))):
        _, trace, _ = L02.integrate(model, noise, 64, "heun")
        bends[name] = L02.path_bending(trace)
        for nfe in (1, 2, 4, 8, 16, 32, 64):
            value, _, evaluations = L02.integrate(model, noise, nfe, "euler", keep=False)
            rows.append(dict(method=name, nfe=evaluations, sliced_w2=L02.sliced_w2(value, data)))
            if nfe in (1, 8, 64):
                scatter(axes[row, (1, 8, 64).index(nfe)], value, f"{name}, Euler NFE={nfe}")
    finish(fig, a.out / "samples.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in bends:
        selected = [r for r in rows if r["method"] == name]
        ax.plot([r["nfe"] for r in selected], [r["sliced_w2"] for r in selected], marker="o", label=name)
    ax.set(xscale="log", xlabel="Euler NFE", ylabel="empirical sliced W2", title="Coupling changes low-NFE behavior")
    ax.legend()
    finish(fig, a.out / "nfe.png")
    json_file(a.out / "metrics.json", dict(rows=rows, path_bending=bends, pair_costs=pair_costs,
        teacher_training_steps=ck["step"], ot_training_steps=steps, reflow_training_steps=steps,
        ot_training_seconds=ot_seconds, teacher_pair_seconds=pair_seconds, reflow_training_seconds=reflow_seconds,
        teacher_pair_nfe=teacher_nfe, teacher_pair_sample_count=len(source),
        interpretation="Reflow has teacher-training + pair-generation + student-training cost. "
                       "Minibatch assignment is not global distribution OT. Lower bending does not prove data quality."))
    print(a.out)


if __name__ == "__main__":
    main()
