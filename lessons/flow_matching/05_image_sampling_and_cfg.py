"""05 — noise만으로 이미지 생성: ODE, velocity CFG, NFE, 한 step 예측의 한계.

repo root (04 먼저 학습):
    CUDA_VISIBLE_DEVICES=7 python lessons/flow_matching/05_image_sampling_and_cfg.py
lesson 폴더:
    CUDA_VISIBLE_DEVICES=7 python 05_image_sampling_and_cfg.py --checkpoint outputs/04/best.pt \
        --steps 32 --solver heun --guidance 2

출력: outputs/05/{samples.png,sampling.gif,cfg.png,solvers.png,extrapolation.png,samples.pt,metrics.json}.
읽기: guided_velocity -> sample -> main. 여기에는 MNIST 정답 이미지 입력이 없다.
04에서 network weight를 학습했다. 이 단계에서 변하는 것은 ODE의 image state x다.
"""
from pathlib import Path

import torch

from common import ROOT, digest, image_grid, json_file, lesson, load, parser, save, setup


def guided_velocity(model, x, t, labels, guidance):
    """v_cfg = v_null + w*(v_cond-v_null). w=0 unconditional, w=1 ordinary conditional.

    null label=10은 04에서 label dropout으로 학습했다. 임의의 '빈 class'를 새로 만든 게 아니다.
    두 branch는 하나의 batch로 forward하지만 두 배의 sample 계산량이 필요하다.
    원 DiT의 noise/variance용 forward_with_cfg를 그대로 호출하지 않고 velocity 전체에 적용한다.
    """
    t = t.expand(len(x)) if t.ndim == 0 else t
    null = torch.full_like(labels, 10)
    if guidance == 1:
        return model(x, t, labels)
    if guidance == 0:
        return model(x, t, null)
    uncond, cond = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([null, labels])).chunk(2)
    return uncond + guidance*(cond-uncond)


@torch.no_grad()
def sample(model, labels, steps=32, solver="heun", guidance=2., seed=777, keep=True, noise=None):
    device = next(model.parameters()).device
    labels = labels.to(device, dtype=torch.long)
    if ((labels < 0) | (labels > 9)).any():
        raise ValueError("생성 class는 0~9")
    if noise is None:
        g = torch.Generator().manual_seed(seed)
        noise = torch.randn(len(labels), 1, 32, 32, generator=g).to(device)
    if noise.shape != (len(labels), 1, 32, 32):
        raise ValueError("noise는 [batch,1,32,32]")
    previous = model.training
    model.eval()
    try:
        field = lambda x, t: guided_velocity(model, x, t, labels, guidance)
        result, trace, nfe = lesson(2).integrate(field, noise.to(device), steps, solver, keep=keep)
    finally:
        model.train(previous)
    branches = 1 if guidance in (0, 1) else 2
    return result, trace, dict(field_nfe=nfe, network_calls=nfe, branches_per_call=branches,
                              single_branch_batch_equivalent_evals=nfe*branches)


def model_from_checkpoint(path, device, use_ema=True):
    ck = load(path)
    if ck["format"] != "fm_image_v1" or ck["prediction"] != "velocity" or ck["path"] != "linear":
        raise ValueError("04의 velocity checkpoint가 필요하다. DiT noise checkpoint는 사용할 수 없다.")
    L04 = lesson(4)
    if ck["fingerprint"]["dit"] != digest(L04.DIT_SOURCE):
        raise ValueError("학습 이후 DiT source가 변경됐다. checkpoint와 같은 코드를 사용하세요.")
    model = L04.FlowDiT(**ck["model_config"]).to(device)
    model.load_state_dict(ck["ema" if use_ema else "model"])
    return model.eval(), ck


@torch.no_grad()
def main():
    p = parser(__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/04/best.pt")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/05")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--solver", choices=["euler", "heun", "rk4"], default="heun")
    p.add_argument("--guidance", type=float, default=2.)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--raw-model", action="store_true", help="EMA 대신 현재 weight로 생성")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    device = setup(device=a.device)
    model, ck = model_from_checkpoint(a.checkpoint, device, not a.raw_model)
    labels = torch.arange(10, device=device).repeat(2)
    result, trace, cost = sample(model, labels, a.steps, a.solver, a.guidance, a.seed)
    image_grid(result, a.out / "samples.png", labels=labels.cpu().tolist(),
               title=f"Training step {ck['step']} | {a.solver}, field NFE={cost['field_nfe']}, CFG={a.guidance}")
    frames = [image_grid(trace[i], labels=labels.cpu().tolist(), title=f"Learned flow t={i/a.steps:.3f}")
              for i in sorted(set(torch.linspace(0, a.steps, min(25, a.steps+1)).round().int().tolist()))]
    frames[0].save(a.out / "sampling.gif", save_all=True, append_images=frames[1:], duration=120, loop=0)
    # 같은 noise를 모든 class/guidance에 사용. w=0이면 각 열의 이미지는 같아야 한다.
    same_noise = trace[0, :1].expand(10, -1, -1, -1).contiguous()
    y = labels[:10]
    images, names, costs = [], [], []
    for w in (0., 1., 2., 4.):
        value, _, info = sample(model, y, a.steps, a.solver, w, keep=False, noise=same_noise)
        images.append(value)
        names.extend([f"w={w:g}, c={i}" for i in range(10)])
        costs.append(dict(guidance=w, **info))
    image_grid(torch.cat(images), a.out / "cfg.png", labels=names, title="Same initial noise for every cell")
    images, names, solver_costs = [], [], []
    for solver, steps in (("euler", 1), ("euler", 8), ("euler", 32), ("heun", 16), ("rk4", 8)):
        value, _, info = sample(model, y, steps, solver, a.guidance, keep=False, noise=trace[0, :10])
        images.append(value)
        names.extend([f"{solver}: {info['field_nfe']}" for _ in range(10)])
        solver_costs.append(dict(solver=solver, steps=steps, **info))
    image_grid(torch.cat(images), a.out / "solvers.png", labels=names,
               title="Cell labels = solver: NFE | last three rows: equal NFE=32")
    images, names = [], []
    for index in sorted(set([0, a.steps//4, a.steps//2, 3*a.steps//4, a.steps])):
        t = result.new_tensor(index/a.steps)
        x = trace[index, :10]
        v = guided_velocity(model, x, t, y, a.guidance)
        endpoint_estimate = x+(1-t)*v
        images.append(endpoint_estimate)
        names.extend([f"t={float(t):.2f}" for _ in range(10)])
    image_grid(torch.cat(images), a.out / "extrapolation.png", labels=names,
               title="x+(1-t)*v at different t; not an exact future endpoint")
    save(a.out / "samples.pt", dict(samples=result.cpu(), trace=trace.cpu(), labels=labels.cpu(),
                                   training_step=ck["step"], cost=cost))
    json_file(a.out / "metrics.json", dict(training_step=ck["step"], main=cost, guidance=costs, solvers=solver_costs,
        value_min=float(result.min()), value_max=float(result.max()), clipping="display only; ODE states not clipped",
        note="NFE counts a vector-field call. CFG>1 doubles batch branches, not the Python call count. No FID claim."))
    print(a.out)


if __name__ == "__main__":
    main()
