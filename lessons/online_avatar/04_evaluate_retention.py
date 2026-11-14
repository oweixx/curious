"""04 — 새 관측이 정말 이득인가? 새로운 camera, 고정된 과거 관측, 실행 비용을 분리.

repo root에서:
    CUDA_VISIBLE_DEVICES=7 python lessons/online_avatar/04_evaluate_retention.py
lesson 폴더에서:
    CUDA_VISIBLE_DEVICES=7 python 04_evaluate_retention.py --runs outputs/stream

출력: outputs/evaluation/{comparison.jpg,curves.png,metrics.json}, method별 progress.gif.
읽는 순서: masked_psnr -> evaluate_method의 triangular matrix -> evaluate의 공통 비교.
이 단계에서만 세 번째 camera의 RGB를 연다. 조명을 평가 RGB로 다시 맞추지 않는다.
ROI는 입력 camera로 추적한 FLAME에서 고정한다. full-head benchmark 수치가 아니다.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from common import (ROOT, arguments, figure, lesson, load_pt, panels, save_json, setup, sha, timer)

L02 = lesson(2)


def masked_psnr(prediction, target, mask):
    mse = ((prediction-target).square()*mask).sum()/(3*mask.sum().clamp_min(1))
    return float(-10*torch.log10(mse.clamp_min(1e-10)))


def light_from_state(state, device):
    light = L02.Lighting().to(device)
    light.load_state_dict(state)
    return light.eval().requires_grad_(False)


@torch.no_grad()
def evaluate_method(a, folder, stream, surface, initial):
    run = json.loads((folder / "run.json").read_text())
    stream.check_artifact(run)
    if not run.get("complete") or run["initial_sha256"] != sha(a.work / "initial.pt"):
        raise ValueError(f"{folder}: 미완료 run 또는 initial mismatch")
    out = a.out / folder.name
    indices = list(range(stream.prefix-1, stream.count))
    matrix = np.full((len(indices), stream.count), np.nan)
    records, gifs, final_rgb = [], [], None
    initial_model = L02.Avatar(initial).to(stream.device).eval()
    initial_raw = initial_model.raw()
    initial_offset, initial_color = surface.decode(initial_raw)
    for row, t in enumerate(tqdm(indices, desc=f"heldout {folder.name}")):
        state = load_pt(folder / "states" / f"{t}.pt", stream.device)
        stream.check_artifact(state)
        if state["seen_through"] != t or any(int(k) > t for k in state["lights"]):
            raise ValueError("snapshot에 미래 상태가 들어 있다")
        raw = state["raw"]
        # 관측한 시각까지만 평가한다. 각 열 j를 고정하면 forgetting을 측정할 수 있다.
        for j in range(t+1):
            light = light_from_state(state["lights"][j], stream.device)
            rendered = surface.render(raw, j, stream.heldout, light)
            target = stream.rgb(j, stream.heldout, evaluation=True)
            roi = surface.roi(j, stream.heldout)
            matrix[row, j] = masked_psnr(rendered["rgb"], target, roi)
            if j == t:
                output, current_light = rendered, light
        old = matrix[row, :t]
        if row:
            history = matrix[:row, :t]
            observed = np.isfinite(history).any(0)
            best = np.max(np.where(np.isfinite(history[:, observed]), history[:, observed], -np.inf), axis=0)
            forgetting = float(np.maximum(0, best-old[observed]).mean())
        else:
            forgetting = 0.
        offset, color = surface.decode(raw)
        records.append(dict(t=t, current_heldout_psnr=float(matrix[row, t]),
            anchor_heldout_psnr=float(matrix[row, 0]), mean_seen_psnr=float(np.nanmean(matrix[row])),
            forgetting_db=forgetting,
            canonical_offset_change_mm=float((offset-initial_offset).norm(dim=-1).mean()*1000),
            canonical_color_change=float((color-initial_color).abs().mean())))
        base = surface.render(initial_raw, t, stream.heldout, current_light)
        # normal map는 alpha로 나눈 뒤 normalize한다. 색은 시각화에만 사용.
        normal = torch.nn.functional.normalize(output["normal"], dim=0)
        normal = (normal*.5+.5)*output["alpha"].clamp(0, 1)
        image_path = out / f"{t}.jpg"
        panels(image_path, [stream.rgb(t, stream.heldout, evaluation=True), base["rgb"], output["rgb"],
               surface.roi(t, stream.heldout), normal, surface.uv(color)],
               [f"heldout time {t}", "initial raw / same lighting", folder.name,
                "fixed evaluation ROI", "render normal", "canonical appearance"], size=192)
        with Image.open(image_path) as im:
            gifs.append(im.copy())
        final_rgb = output["rgb"].detach().cpu()
    gifs[0].save(out / "progress.gif", save_all=True, append_images=gifs[1:], duration=300, loop=0)
    # render timing: pose/FLAME/cache가 이미 준비된 상태의 decoder+2DGS만 측정한다.
    for _ in range(3):
        surface.render(raw, indices[-1], stream.heldout, current_light)
    start = timer()
    for _ in range(20):
        surface.render(raw, indices[-1], stream.heldout, current_light)
    fps = 20/(timer()-start)
    updates = run["rows"]
    report = dict(method=folder.name, records=records,
        retention_matrix=[[float(v) if np.isfinite(v) else None for v in r] for r in matrix],
        row_times=indices, column_times=list(range(stream.count)),
        trainable_parameters=run["trainable_parameters"],
        total_update_seconds=sum(r["update_seconds"] for r in updates),
        mean_update_seconds=float(np.mean([r["update_seconds"] for r in updates])),
        mean_tracking_seconds=float(np.mean([r["tracking_seconds"] for r in updates])),
        total_steps=sum(r["steps"] for r in updates),
        max_budget_overrun_seconds=max(r["overrun_seconds"] for r in updates),
        cached_pose_decode_render_fps=fps, peak_gpu_mib=run["peak_gpu_mib"],
        timing_scope="update includes current RGB load, transient light fit, canonical/replay optimization; "
                     "tracking measured separately; landmark detection, snapshot I/O, 04 evaluation excluded",
        metric_scope="sRGB masked PSNR on input-tracked eroded FLAME surface; not hair/full-image/geometry accuracy",
        update_records=updates, config=run["config"])
    save_json(out / "metrics.json", report)
    return report, final_rgb


def evaluate(a):
    device = setup(a.device)
    stream, tracking, initial, surface = L02.load_components(a.work, device)
    order = ["frozen", "representation", "adapter", "adapter_replay"]
    folders = [a.runs / name for name in order if (a.runs / name / "run.json").is_file()]
    if not folders:
        raise FileNotFoundError("완료한 03 run이 없다. --runs 경로를 확인하세요.")
    reports, images = [], [stream.rgb(stream.count-1, stream.heldout, evaluation=True).cpu()]
    for folder in folders:
        report, image = evaluate_method(a, folder, stream, surface, initial)
        reports.append(report)
        images.append(image)
    panels(a.out / "comparison.jpg", images, ["heldout RGB"]+[r["method"] for r in reports])
    def series(key):
        return [(r["method"], [v["t"] for v in r["records"]], [v[key] for v in r["records"]]) for r in reports]
    figure(a.out / "curves.png", [
        dict(title="Heldout camera (different time at each point)", ylabel="masked PSNR (dB)", series=series("current_heldout_psnr")),
        dict(title="Same first frame: retention", ylabel="masked PSNR (dB)", series=series("anchor_heldout_psnr")),
        dict(title="Past-frame forgetting (lower is better)", ylabel="mean best-to-current drop (dB)", series=series("forgetting_db"))])
    figure(a.out / "canonical_drift.png", [
        dict(title="Canonical geometry change (not GT error)", ylabel="mean mm", series=series("canonical_offset_change_mm")),
        dict(title="Canonical color change (not GT error)", ylabel="mean |RGB delta|", series=series("canonical_color_change"))])
    figure(a.out / "efficiency.png", [dict(title="Fixed anchor quality vs update cost (prefix excluded)",
        xlabel="cumulative update seconds", ylabel="anchor masked PSNR (dB)", series=[
            (r["method"], [0.]+np.cumsum([v["update_seconds"] for v in r["update_records"]]).tolist(),
             [v["anchor_heldout_psnr"] for v in r["records"]]) for r in reports])])
    save_json(a.out / "metrics.json", dict(stream_sha256=stream.fingerprint, methods=reports,
        interpretation="No heldout gradients/checkpoint selection. A short smoke test is not evidence of research quality."))
    print(f"comparison / curves / progress.gif / metrics -> {a.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    arguments(p)
    p.add_argument("--runs", type=Path, default=ROOT / "outputs/stream")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/evaluation")
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
