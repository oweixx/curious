"""실제 NeRSemble RGB + FLAME + 2DGS로 작은 전체 경로를 검사한다.

repo root (기존 elite Conda, GPU 7):
    CUDA_VISIBLE_DEVICES=7 python lessons/online_avatar/smoke.py
새 작업 폴더에서 재실행:
    CUDA_VISIBLE_DEVICES=7 python lessons/online_avatar/smoke.py --work .cache/my_smoke
이미 생성한 smoke 결과 검사만:
    CUDA_VISIBLE_DEVICES=7 python lessons/online_avatar/smoke.py --reuse

5시각/128px/64UV/12 warm-start steps/각 관측 3 update steps.
이것은 최적화·누수 방지·출력 계약 검사다. 수렴/novelty/실시간 성능 검증이 아니다.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from common import ROOT, lesson, load_pt, save_json, setup


def contract_checks(work, runs, device):
    L02 = lesson(2)
    L03 = lesson(3)
    stream, tracking, initial, surface = L02.load_components(work, device)
    # 평가 camera RGB에 실수로 접근하면 값이 반환되는 대신 즉시 실패해야 한다.
    try:
        stream.rgb(0, stream.heldout)
    except ValueError:
        pass
    else:
        raise AssertionError("heldout camera read guard 없음")
    models = {mode: L02.Avatar(initial, mode).to(device)
              for mode in ("frozen", "representation", "adapter")}
    base = models["frozen"].raw().detach()
    assert all(torch.equal(m.raw().detach(), base) for m in models.values()), "baseline initial mismatch"
    checks = {}
    for mode in ("representation", "adapter"):
        m = models[mode]
        loss = L02.photo_loss(surface, m.raw(), 0, L02.Lighting().to(device))
        loss.backward()
        grads = [p.grad for p in m.trainable() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0
        assert all(p.grad is None for p in m.field.parameters()), "frozen field에 gradient 발생"
        if mode == "representation":
            assert m.delta.grad[:, :3].abs().sum() > 0, "geometry까지 gradient가 안 왔다"
            assert m.delta.grad[:, 3:].abs().sum() > 0, "appearance까지 gradient가 안 왔다"
        checks[mode] = dict(photo_loss=float(loss.detach()), grad_l1=sum(float(g.abs().sum()) for g in grads))
    # 각 방법의 최초 snapshot도 실제로 같아야 한다.
    snapshots = []
    for mode in ("frozen", "representation", "adapter", "adapter_replay"):
        run = json.loads((runs / mode / "run.json").read_text())
        snapshots.append(load_pt(runs / mode / "states" / f"{stream.prefix-1}.pt")["raw"])
        final = load_pt(runs / mode / "states" / f"{stream.count-1}.pt")
        assert torch.isfinite(final["raw"]).all()
        for row in run["rows"]:
            assert all(i < row["t"] for i in row["memory_before"])
            assert all(i <= row["t"] for i in row["memory_after"])
        if mode == "frozen":
            assert torch.equal(snapshots[-1], final["raw"]), "frozen persistent state가 바뀌었다"
        else:
            assert not torch.equal(snapshots[-1], final["raw"]), f"{mode} 실제 갱신이 없다"
    assert all(torch.equal(snapshots[0], s) for s in snapshots[1:])
    # 불확실한 관측을 거절하는 분기도 실제 update 함수로 확인한다.
    model = L02.Avatar(initial, "adapter").to(device)
    before = model.raw().detach().clone()
    options = SimpleNamespace(seconds=2., light_steps=0, no_gate=False, no_replay=False,
                              min_confidence=1., max_steps=1, replay_weight=.5)
    row = L03.update_one_time(options, "adapter_replay", model, surface, stream.prefix, {},
                             torch.optim.Adam(model.trainable(), lr=.01), [0])
    assert not row["accepted"] and row["steps"] == 0 and torch.equal(before, model.raw().detach())
    assert L03.update_memory([0], 3, 1) == [0], "capacity=1에서도 first anchor를 보존해야 한다"
    checks.update(heldout_read_guard=True, same_initial_state=True, frozen_state_unchanged=True,
                  real_geometry_and_appearance_gradients=True, past_only_replay=True, confidence_rejection=True,
                  final_reprojection_median_px=float(np.median([r["median_error_px"] for r in tracking["frames"]])))
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=ROOT / ".cache/smoke")
    p.add_argument("--reuse", action="store_true")
    a = p.parse_args()
    a.work = a.work.resolve()
    runs, evaluation = a.work / "runs", a.work / "evaluation"
    if not a.reuse:
        commands = [
            (0, ["--frames", "5", "--stride", "4", "--size", "128"]),
            (1, ["--initial-steps", "150", "--track-steps", "20"]),
            (2, ["--uv-size", "64", "--steps", "12", "--save-every", "6"]),
            (3, ["--out", str(runs), "--seconds", "2", "--max-steps", "3", "--light-steps", "2"]),
            (4, ["--runs", str(runs), "--out", str(evaluation)])]
        for number, args in commands:
            script = next(ROOT.glob(f"{number:02d}_*.py"))
            subprocess.run([sys.executable, str(script), "--work", str(a.work), *args], check=True)
    checks = contract_checks(a.work, runs, setup())
    report = json.loads((evaluation / "metrics.json").read_text())
    assert len(report["methods"]) == 4
    for result in report["methods"]:
        assert all(np.isfinite(r["current_heldout_psnr"]) for r in result["records"])
    save_json(a.work / "smoke_report.json", checks)
    print(json.dumps(checks, indent=2))
    print(f"시각적 점검: {evaluation / 'comparison.jpg'}")


if __name__ == "__main__":
    main()
