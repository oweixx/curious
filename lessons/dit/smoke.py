r"""검증용. 원본 source hash, 수식/gradient, CFG, sampler, 실제 CLI resume를 검사한다.

    python smoke.py
    # 00 준비 후 GPU4에서 실제06의 uninterrupted/resume 동등성까지:
    CUDA_VISIBLE_DEVICES=4 python smoke.py --device cuda --training

출력: .cache/smoke/report.json, checks.png. 학습 품질 평가/FID를 뜻하지 않는다.
짧은 학습 checkpoint는 임시 폴더에서 검사 후 제거한다. 사용자 checkpoint는 건드리지 않는다.
"""

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import torch
from diffusion import create_diffusion
from common import (
    ROOT,
    parser,
    setup,
    make_model,
    digest,
    lesson,
    load,
    pyplot,
    savefig,
    write_json,
)


def main():
    p = parser(__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--training", action="store_true")
    a = p.parse_args()
    setup()
    report = {}
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    for name, sha in manifest["files"].items():
        assert digest(ROOT / name) == sha, name
    report["verbatim_source_hashes"] = True
    m = make_model().to(a.device)
    m.eval()
    x = torch.randn(2, 4, 8, 8, device=a.device)
    t = torch.tensor([0, 999], device=a.device)
    y = torch.tensor([1, 2], device=a.device)
    c = m.t_embedder(t) + m.y_embedder(y, False)
    tokens = m.x_embedder(x) + m.pos_embed
    assert torch.equal(tokens, m.blocks[0](tokens, c))
    assert m(x, t, y).count_nonzero() == 0
    report["zero_identity_and_output"] = True
    d = create_diffusion("")
    eps = torch.randn_like(x)
    mixed = d.q_sample(x, t, eps)
    # 원본은 schedule을 float64로 계산한 뒤 계수를 float32로 변환한다.
    # t=0 근처의 1-alpha를 float32에서 먼저 빼면 cancellation 오차가 커진다.
    alpha = torch.tensor(d.alphas_cumprod, device=a.device, dtype=torch.float64)[
        t, None, None, None
    ]
    manual = alpha.sqrt().float() * x + (1 - alpha).sqrt().float() * eps
    assert torch.allclose(mixed, manual, atol=1e-6)
    leaf = torch.randn(2, 8, 8, 8, device=a.device, requires_grad=True)
    terms = d.training_losses(lambda *args, **kw: leaf, x, t, noise=eps)
    gradient = torch.autograd.grad(terms["vb"].mean(), leaf)[0]
    assert gradient[:, :4].count_nonzero() == 0 and gradient[:, 4:].abs().sum() > 0
    report["q_sample_and_variance_gradient"] = True
    # Avoid a vacuous CFG test on the zero-output initial network.
    with torch.no_grad():
        torch.nn.init.normal_(m.final_layer.linear.weight, std=0.01)
        torch.nn.init.normal_(m.final_layer.adaLN_modulation[-1].weight, std=0.01)
    yy = torch.tensor([1, 10], device=a.device)
    tt = torch.full_like(yy, 400)
    joined = torch.cat([x[:1], x[:1]])
    raw = m(joined, tt, yy)
    s = lesson(7)
    all0 = s.guided_forward(m, x, tt, yy, 0.0, "all")
    all1 = s.guided_forward(m, x, tt, yy, 1.0, "all")
    assert not torch.equal(raw[0, :4], raw[1, :4])
    assert torch.allclose(all0[0, :4], raw[1, :4], atol=1e-6)
    assert torch.allclose(all1[0, :4], raw[0, :4], atol=1e-6)
    paper = s.guided_forward(m, x, tt, yy, 4.0, "paper")
    assert torch.equal(paper[:, 3:], raw[:, 3:])
    report["cfg_formulas_and_original_three_channels"] = True
    seen = []
    hook = m.register_forward_pre_hook(lambda module, args: seen.append(int(args[1][0])))
    before = torch.get_rng_state().clone()
    # all 경로는 model(...)을 호출하므로 hook으로 실제 입력 t를 관찰할 수 있다.
    # 원본 forward_with_cfg는 self.forward(...)를 직접 호출해 module hook을 우회한다.
    z1, _ = s.sample(m, [1], steps=5, seed=3, cfg_mode="all")
    after = torch.get_rng_state()
    z2, _ = s.sample(m, [1], steps=5, seed=3, cfg_mode="all")
    hook.remove()
    assert torch.equal(z1, z2) and torch.equal(before, after)
    assert seen[:5] == list(reversed(create_diffusion("5").timestep_map)), seen
    report["ddim_repeatability_rng_and_timestep_mapping"] = True
    if a.training:
        dest = ROOT / ".cache/smoke"
        dest.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest) as td:
            td = Path(td)
            base = [
                sys.executable,
                str(ROOT / "06_train.py"),
                "--device",
                a.device,
                "--batch-size",
                "4",
                "--val-every",
                "2",
                "--preview-steps",
                "2",
            ]
            commands = [
                base + ["--steps", "3", "--out", str(td / "full")],
                base + ["--steps", "2", "--out", str(td / "split")],
                base
                + [
                    "--steps",
                    "3",
                    "--out",
                    str(td / "split"),
                    "--resume",
                    str(td / "split/latest.pt"),
                ],
            ]
            for i, cmd in enumerate(commands):
                with (dest / f"train_{i}.log").open("w") as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            full = load(td / "full/latest.pt")
            resumed = load(td / "split/latest.pt")
            errors = {
                name: max(float((full[name][k] - resumed[name][k]).abs().max()) for k in full[name])
                for name in ("model", "ema")
            }
            assert max(errors.values()) == 0, errors
            report["actual_training_resume_exact"] = errors
    out = ROOT / ".cache/smoke"
    write_json(out / "report.json", report)
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(10, 4))
    names = list(report)
    ax.barh(names, [1] * len(names), color="seagreen")
    ax.set_xlim(0, 1.1)
    ax.set_title("Executable checks passed (not generation quality)")
    savefig(fig, out / "checks.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
