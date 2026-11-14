"""실제 수식·optimizer·이미지 생성·checkpoint 재개를 검사한다.

repo root (기존 elite Conda):
    CUDA_VISIBLE_DEVICES=7 python lessons/flow_matching/smoke.py
이미 생성한 결과의 계약 검사만:
    CUDA_VISIBLE_DEVICES=7 python lessons/flow_matching/smoke.py --reuse
다른 폴더에 새 검증:
    CUDA_VISIBLE_DEVICES=7 python lessons/flow_matching/smoke.py --out .cache/check2

6개 lesson의 CLI를 작은 설정으로 실제 실행한다. 데이터는 MNIST이며 합성 이미지로 대체하지 않는다.
2D 학습 200 step / 이미지 smoke 모델 8 step은 실행 검사이지 최종 생성 품질 검증이 아니다.
"""
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F

from common import ROOT, cpu_state, json_file, lesson, load, parser, setup


def contracts(out, device):
    L00, L01, L02, L04, L05 = [lesson(i) for i in (0, 1, 2, 4, 5)]
    # 미분이 맞지 않으면 다른 velocity를 학습한다. 수치 미분으로 image broadcast도 검사.
    x0, x1 = torch.randn(5, 1, 4, 4, dtype=torch.float64), torch.randn(5, 1, 4, 4, dtype=torch.float64)
    t = torch.linspace(.1, .9, 5, dtype=torch.float64)
    derivative_errors = {}
    for kind in ("linear", "cosine", "residual"):
        xt, velocity = L00.conditional_path(x0, x1, t, kind, .1)
        plus = L00.conditional_path(x0, x1, t+1e-6, kind, .1)[0]
        minus = L00.conditional_path(x0, x1, t-1e-6, kind, .1)[0]
        error = float(((plus-minus)/2e-6-velocity).abs().max())
        assert error < 1e-8
        derivative_errors[kind] = error
    points = torch.randn(20, 2)
    assert torch.allclose(L01.oracle_velocity(points, torch.zeros(20)), -points, atol=2e-6)
    assert torch.allclose(L01.oracle_velocity(points, torch.ones(20)), points, atol=2e-6)
    # dx/dt=x -> x(1)=e*x(0). 같은 NFE=32에서 구현의 수렴 성질과 정확한 call 수 검사.
    solver_errors = {}
    for method, steps in (("euler", 32), ("heun", 16), ("rk4", 8)):
        calls = [0]
        def field(x, t):
            calls[0] += 1
            return x
        result, _, nfe = L02.integrate(field, torch.ones(1, 1, dtype=torch.float64), steps, method)
        assert calls[0] == nfe == 32
        solver_errors[method] = abs(float(result)-float(torch.exp(torch.tensor(1., dtype=torch.float64))))
    assert solver_errors["rk4"] < solver_errors["heun"] < solver_errors["euler"]
    # OT는 동일 batch의 순열이어야 하고 최소화한 비용이 independent pairing보다 크면 안 된다.
    g = torch.Generator(device=device).manual_seed(82)
    a, b = L01.pair_batch(64, device, generator=g)
    g.manual_seed(82)
    ao, bo = L01.pair_batch(64, device, "minibatch_ot", generator=g)
    assert torch.equal(a, ao)
    assert torch.allclose(b.sort(dim=0).values, bo.sort(dim=0).values)
    assert (a-bo).square().sum() <= (a-b).square().sum()+1e-5
    field_ck = load(out / "01/field.pt")
    history = field_ck["history"]
    assert history[-1]["marginal_mse"] < history[0]["marginal_mse"], "2D field가 학습되지 않았다"
    model, image_ck = L05.model_from_checkpoint(out / "04/latest.pt", device)
    y = torch.arange(4, device=device)
    noise = torch.randn(4, 1, 32, 32, device=device)
    time = noise.new_full((4,), .4)
    with torch.no_grad():
        conditional = model(noise, time, y)
        unconditional = model(noise, time, torch.full_like(y, 10))
        assert torch.equal(L05.guided_velocity(model, noise, time, y, 1), conditional)
        assert torch.equal(L05.guided_velocity(model, noise, time, y, 0), unconditional)
        assert torch.allclose(L05.guided_velocity(model, noise, time, y, 2), 2*conditional-unconditional, atol=2e-6, rtol=1e-5)
    # EMA가 아닌 학습 weight에서 transformer 내부까지 gradient가 이어지는지 확인.
    model.load_state_dict(image_ck["model"])
    prediction = model(noise, time, y)
    assert prediction.shape == noise.shape
    loss = F.mse_loss(prediction, torch.randn_like(prediction))
    loss.backward()
    grads = [p.grad for p in model.backbone.blocks.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0
    before = cpu_state(model)
    one, _, _ = L05.sample(model, y, steps=2, seed=12)
    two, _, _ = L05.sample(model, y, steps=2, seed=12)
    assert torch.equal(one, two)
    assert all(torch.equal(value, cpu_state(model)[key]) for key, value in before.items())
    # 중단 없는 8 steps와 4+4 steps 비교. optimizer/EMA/RNG까지 복원해야 한다.
    resumed = load(out / "resume/latest.pt")
    differences = {name: max(float((image_ck[name][k]-resumed[name][k]).abs().max()) for k in image_ck[name])
                   for name in ("model", "ema")}
    assert max(differences.values()) < 2e-6, differences
    report = dict(path_derivative_max_error=derivative_errors, equal_nfe_solver_errors=solver_errors,
        minibatch_ot_cost_decreased=True, marginal_mse_initial=history[0]["marginal_mse"],
        marginal_mse_final=history[-1]["marginal_mse"], dit_block_gradients=True,
        cfg_identities=True, deterministic_sampling=True, inference_weights_unchanged=True,
        resume_max_abs_difference=differences)
    json_file(out / "smoke_report.json", report)
    return report


def main():
    p = parser(__doc__)
    p.add_argument("--out", type=Path, default=ROOT / ".cache/smoke")
    p.add_argument("--reuse", action="store_true")
    a = p.parse_args()
    a.out = a.out.resolve()
    def run(i, extra):
        script = next(ROOT.glob(f"{i:02d}_*.py"))
        subprocess.run([sys.executable, str(script), *map(str, extra)], check=True)
    if not a.reuse:
        run(0, ["--out", a.out / "00"])
        run(1, ["--device", "cuda", "--steps", 200, "--every", 100, "--out", a.out / "01"])
        run(2, ["--device", "cuda", "--samples", 64, "--checkpoint", a.out / "01/field.pt", "--out", a.out / "02"])
        run(3, ["--device", "cuda", "--steps", 100, "--pairs", 128, "--samples", 64,
                "--teacher", a.out / "01/field.pt", "--out", a.out / "03"])
        config = ["--download", "--profile", "smoke", "--batch-size", 16, "--val-every", 4,
                  "--preview-steps", 4, "--train-limit", 256, "--precision", "fp32"]
        run(4, [*config, "--steps", 8, "--out", a.out / "04"])
        run(5, ["--checkpoint", a.out / "04/latest.pt", "--steps", 4, "--out", a.out / "05"])
        run(4, [*config, "--steps", 4, "--out", a.out / "resume"])
        run(4, [*config, "--steps", 8, "--resume", a.out / "resume/latest.pt"])
    report = contracts(a.out, setup(device="cuda"))
    print(report)
    print(a.out / "smoke_report.json")


if __name__ == "__main__":
    main()
