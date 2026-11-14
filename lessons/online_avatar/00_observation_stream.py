"""00 — 무엇을 언제 관측했는가? 입력 2개 / 평가 1개를 데이터 입구에서 나눈다.

repo root에서 (elite Conda):
    python lessons/online_avatar/00_observation_stream.py
lesson 폴더에서:
    python 00_observation_stream.py --frames 24 --stride 3 --prefix 2

출력: data/042_FREE/{stream.json,observations.pt}, inspection.jpg.
기존 VDA의 동기화 RGB와 2D landmark만 재사용한다. all-view face_world.npz,
mono_flame.npz, ELITE 학습 checkpoint/UV texture는 읽지 않는다.
읽을 부분: triangulate_pair -> crop_camera -> prepare. 여기는 학습 단계가 아니다.
"""
import argparse
import json

import numpy as np
import torch

from common import DEFAULT_SOURCE, Stream, arguments, panels, require_new, save_json, save_pt, sha


def triangulate_pair(xy, projections, threshold=20.):
    """두 시점의 같은 inner-51 landmark로 DLT. 윤곽 17점은 view-dependent여서 제외.

    (u P_3 - P_1) X = 0, (v P_3 - P_2) X = 0을 두 camera에서 모은다.
    잘못된 대응/음수 depth는 NaN으로 남기고 학습 loss에서 명시적으로 제외한다.
    """
    xyz = np.full((xy.shape[1], 3), np.nan, np.float32)
    for j in range(len(xyz)):
        if not np.isfinite(xy[:, j]).all():
            continue
        A = np.stack([xy[:, j, 0, None]*projections[:, 2]-projections[:, 0],
                      xy[:, j, 1, None]*projections[:, 2]-projections[:, 1]], 1).reshape(4, 4)
        _, _, vt = np.linalg.svd(A)
        if abs(vt[-1, 3]) < 1e-9:
            continue
        p = vt[-1, :3]/vt[-1, 3]
        q = projections @ np.r_[p, 1.]
        if (q[:, 2] > 0).all() and np.max(np.linalg.norm(q[:, :2]/q[:, 2:]-xy[:, j], axis=-1)) < threshold:
            xyz[j] = p
    return xyz


def crop_camera(K, w2c, points0, size):
    """첫 입력 시각의 3D landmark를 투영해 고정 square crop. 평가 RGB/landmark 불필요.

    crop 후 K': cx-=left, cy-=top; fx,fy,cx,cy *= resize_scale.
    프레임마다 crop을 바꾸면 camera 자체가 움직이므로 여기서는 고정한다.
    """
    cam = points0 @ w2c[:3, :3].T + w2c[:3, 3]
    p = cam @ K.T
    xy = p[:, :2]/p[:, 2:]
    lo, hi = xy.min(0), xy.max(0)
    center = (lo+hi)/2
    side = int(np.ceil(max(hi-lo)*2.2))
    center[1] -= .14 * side  # inner landmarks보다 위에 있는 이마/두피도 포함
    left, top = np.round(center-side/2).astype(int)
    Kc = K.copy()
    Kc[:2, 2] -= [left, top]
    Kc[:2] *= size/side
    return Kc, [int(left), int(top), int(left+side), int(top+side)]


def prepare(a):
    source = a.source.resolve()
    run = json.loads((source / "run.json").read_text())
    cameras = a.inputs + [a.heldout]
    if len(a.inputs) != 2 or len(set(cameras)) != 3:
        raise ValueError("서로 다른 입력 2 camera, 평가 1 camera가 필요하다.")
    if a.frames < 3 or not 1 <= a.prefix < a.frames or a.stride < 1 or a.start < 0 or a.size < 64:
        raise ValueError("frames>=3, 1<=prefix<frames, stride>=1, start>=0, size>=64 필요")
    indices = list(range(a.start, a.start + a.frames*a.stride, a.stride))
    if indices[-1] >= run["frames"]:
        raise ValueError(f"source는 {run['frames']} frames. --frames/--stride를 줄이세요.")
    require_new(a.work / "stream.json", a.overwrite)
    calibration = {}
    for c in cameras:
        with np.load(source / "cameras" / f"{c}.npz") as z:
            calibration[c] = {k: z[k] for k in ("K", "world_2_cam")}
        for i in indices:
            if not (source / "frames" / c / f"{i:05d}.jpg").is_file():
                raise FileNotFoundError(f"{c} frame {i}가 없다. VDA 전처리 frames 단계 필요.")
    # 평가 camera의 landmark 파일은 열지 않는다.
    xy = np.stack([np.load(source / "landmarks" / f"{c}.npz")["xy"][indices, 17:] for c in a.inputs])
    P = np.stack([calibration[c]["K"] @ calibration[c]["world_2_cam"][:3] for c in a.inputs])
    world = np.stack([triangulate_pair(xy[:, t], P) for t in range(a.frames)])
    good = np.isfinite(world).all(-1)
    if (good.sum(-1) < 20).any():
        raise ValueError("유효한 two-view inner landmarks가 20개 미만인 frame이 있다.")
    first = world[0, good[0]]
    cam_info = {}
    for c in cameras:
        cal = calibration[c]
        K, crop = crop_camera(cal["K"], cal["world_2_cam"], first, a.size)
        cam_info[c] = dict(K=K.tolist(), w2c=cal["world_2_cam"].tolist(), crop=crop)
    for v, c in enumerate(a.inputs):
        left, top, right, bottom = cam_info[c]["crop"]
        xy[v] = (xy[v]-[left, top])*a.size/(right-left)
    records = [dict(index=i, source_index=run["source_indices"][i], timestamp=run["timestamps"][i],
                    input_sha256={c: sha(source / "frames" / c / f"{i:05d}.jpg") for c in a.inputs}) for i in indices]
    manifest = dict(format="online_avatar_v1", source=str(source), input_cameras=a.inputs,
                    heldout_camera=a.heldout, prefix=a.prefix, image_size=a.size, frames=records,
                    cameras=cam_info, initialization="FLAME neutral; only first input frame may fit identity shape",
                    provenance={"run": sha(source / "run.json"),
                                "input_landmarks": {c: sha(source / "landmarks" / f"{c}.npz") for c in a.inputs},
                                "calibration": {c: sha(source / "cameras" / f"{c}.npz") for c in cameras}},
                    protocol="causal tracking and prefix initialization; heldout RGB is evaluation-only")
    save_pt(a.work / "observations.pt", {"xy": torch.from_numpy(xy), "xyz": torch.from_numpy(world)})
    manifest["observations_sha256"] = sha(a.work / "observations.pt")
    save_json(a.work / "stream.json", manifest)
    stream = Stream(a.work)
    panels(a.work / "inspection.jpg", [stream.rgb(0, c) for c in a.inputs],
           [f"input {c}: first frame" for c in a.inputs])
    print(f"{a.frames} timestamps | inputs={a.inputs}, heldout={a.heldout} | {a.work / 'inspection.jpg'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    arguments(p)
    p.add_argument("--source", type=type(DEFAULT_SOURCE), default=DEFAULT_SOURCE)
    p.add_argument("--inputs", nargs=2, default=["222200037", "222200042"])
    p.add_argument("--heldout", default="222200036")
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--prefix", type=int, default=2)
    p.add_argument("--size", type=int, default=256)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
