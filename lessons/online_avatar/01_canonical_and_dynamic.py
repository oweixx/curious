"""01 — canonical identity와 expression/pose를 서로 다른 변수로 만든다.

repo root에서:
    CUDA_VISIBLE_DEVICES=7 python lessons/online_avatar/01_canonical_and_dynamic.py
lesson 폴더에서:
    CUDA_VISIBLE_DEVICES=7 python 01_canonical_and_dynamic.py --initial-steps 600 --track-steps 120

출력: data/042_FREE/tracking.pt, tracking/0.jpg ... , tracking/dynamics.png.
읽는 순서: flame_model -> fit_stream의 첫 시각 shape fitting -> 다음 시각 transient fitting.
공개 pixel3DMM의 실제 FLAME2020/LBS만 사용한다. dense tracker나 이미 학습한 사람의
shape를 가져오지 않는다. 이 작은 tracker는 landmark 기반 교육용 baseline이다.
얼굴 표정/머리 자세는 현재 입력 두 camera에서만 추정하며, 미래 frame을 smoothing하지 않는다.
"""
import argparse
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from tqdm import trange

from common import (UPSTREAM, Stream, arguments, figure, finite_step, panels, project,
                    require_new, save_pt, setup, timer)


def flame_model():
    """기존에 설치된 원본 FLAME 코드를 사용한다. 가중치를 학습하는 network가 아니다.

    shape/expr 선형 blendshapes + jaw/neck의 LBS를 평가하는 미분 가능한 함수다.
    모델 버퍼는 고정, 이 함수에 넣는 shape/expr/pose tensor만 optimizer에 들어간다.
    """
    if not (UPSTREAM / "vda").is_dir():
        raise FileNotFoundError("기존 lessons/vda setup과 FLAME2020 assets가 필요하다. GUIDE 참고.")
    sys.path.insert(0, str(UPSTREAM)) if str(UPSTREAM) not in sys.path else None
    for path in (UPSTREAM / "third_party/pixel3dmm/src", UPSTREAM.parent / "runtime"):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from vda.preprocess.geometry import flame20
    return flame20().eval().requires_grad_(False)


def forward_flame(model, shape, expr, jaw):
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
    return model(shape_params=shape[None], cameras=torch.eye(3, device=shape.device)[None],
                 expression_params=expr[None],
                 jaw_pose_params=matrix_to_rotation_6d(axis_angle_to_matrix(jaw[None])))[:2]


def similarity(x, y):
    """Umeyama: y = scale * R @ x + t. 두 입력에서 삼각측량한 점만 사용."""
    mx, my = x.mean(0), y.mean(0)
    X, Y = x-mx, y-my
    u, s, vt = np.linalg.svd(Y.T @ X / len(x))
    d = np.diag([1., 1., np.linalg.det(u @ vt)])
    r = u @ d @ vt
    scale = np.sum(s*np.diag(d))/np.mean(np.sum(X*X, -1))
    return r, my-scale*r@mx, scale


def landmark_preview(path, stream, t, predicted):
    images = []
    for k, c in enumerate(stream.inputs):
        array = (stream.rgb(t, c).cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8)
        im = Image.fromarray(array)
        draw = ImageDraw.Draw(im)
        for xy, color in ((stream.landmarks["xy"][k, t], "#40ff80"),
                          (predicted[k].detach().cpu(), "#ff7040")):
            for x, y in xy:
                if np.isfinite([x, y]).all():
                    draw.ellipse((float(x)-1, float(y)-1, float(x)+1, float(y)+1), fill=color)
        images.append(im)
    panels(path, images, [f"{c}: detected green / FLAME orange" for c in stream.inputs])


def fit_stream(a):
    device = setup(a.device)
    stream = Stream(a.work, device)
    require_new(a.work / "tracking.pt", a.overwrite)
    if min(a.initial_steps, a.track_steps) < 1 or a.confidence_px <= 0:
        raise ValueError("fitting step/confidence-px는 양수여야 한다.")
    model = flame_model()
    from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
    shape = torch.nn.Parameter(torch.zeros(300, device=device))
    expr = torch.nn.Parameter(torch.zeros(100, device=device))
    jaw = torch.nn.Parameter(torch.zeros(3, device=device))
    with torch.no_grad():
        _, lm = forward_flame(model, shape, expr, jaw)
    xyz0 = stream.landmarks["xyz"][0].numpy()
    valid = np.isfinite(xyz0).all(-1)
    R, t, scale = similarity(lm[0, 17:].cpu().numpy()[valid], xyz0[valid])
    if not .4 < scale < 2.5:
        raise ValueError(f"좌표계/대응 오류: FLAME world scale={scale:.3f}")
    tt = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)
    rotation = torch.nn.Parameter(matrix_to_rotation_6d(tt(R)))
    translation = torch.nn.Parameter(tt(t))
    log_scale = torch.nn.Parameter(tt(np.log(scale)))
    transient = [expr, jaw, rotation, translation]
    history = []
    progress = trange(stream.count, desc="causal FLAME (2 input cameras)")
    for i in progress:
        start = timer()
        # shape/scale은 첫 관측에서만 갱신. 이후에는 같은 사람의 canonical 기준으로 고정.
        shape.requires_grad_(i == 0)
        log_scale.requires_grad_(i == 0)
        params = transient + ([shape, log_scale] if i == 0 else [])
        optimizer = torch.optim.Adam([
            {"params": [expr, rotation] + ([shape] if i == 0 else []), "lr": .01},
            {"params": [jaw, translation] + ([log_scale] if i == 0 else []), "lr": .001}])
        obs = stream.landmarks["xy"][:, i].to(device)
        mask = torch.isfinite(obs).all(-1)
        obs = torch.nan_to_num(obs)
        previous = [p.detach().clone() for p in transient]

        def evaluate():
            vertices, lm = forward_flame(model, shape, expr, jaw)
            r = rotation_6d_to_matrix(rotation)
            world = log_scale.exp()*(lm[0, 17:] @ r.T) + translation
            p = torch.stack([project(world, stream.K[c], stream.w2c[c])[0] for c in stream.inputs])
            return vertices[0], r, p

        for _ in range(a.initial_steps if i == 0 else a.track_steps):
            _, _, p = evaluate()
            reprojection = F.smooth_l1_loss(p[mask]/5, obs[mask]/5, beta=.5)
            prior = .002*expr.square().mean() + .004*shape.square().mean() + .02*jaw.square().mean()
            temporal = sum((p-q).square().mean() for p, q in zip(transient, previous)) if i else 0.
            finite_step(reprojection+prior+.005*temporal, params, optimizer)
            with torch.no_grad():
                jaw[0].clamp_(0, .8)
                jaw[1:].clamp_(-.15, .15)
                log_scale.clamp_(np.log(.4), np.log(2.5))
        with torch.no_grad():
            vertices, r, p = evaluate()
            px = float((p[mask]-obs[mask]).norm(dim=-1).median())
            row = dict(local_vertices=vertices.cpu(), R=r.cpu(), translation=translation.detach().cpu().clone(),
                       expr=expr.detach().cpu().clone(), jaw=jaw.detach().cpu().clone(),
                       median_error_px=px, tracking_seconds=timer()-start,
                       confidence=float(np.exp(-(px/a.confidence_px)**2)))
            history.append(row)
            landmark_preview(a.work / "tracking" / f"{i}.jpg", stream, i, p)
        progress.set_postfix(error_px=f"{px:.2f}")
    with torch.no_grad():
        neutral, neutral_lm = forward_flame(model, shape, torch.zeros_like(expr), torch.zeros_like(jaw))
    # expression/pose를 0으로 만든 canonical mesh. 첫 frame의 posed mesh와 구분한다.
    save_pt(a.work / "tracking.pt", dict(stream_sha256=stream.fingerprint, shape=shape.detach().cpu(),
            scale=log_scale.detach().exp().cpu(), neutral=neutral[0].cpu(), neutral_landmarks=neutral_lm[0].cpu(),
            faces=model.faces.cpu(), frames=history, config=vars(a) | {"work": str(a.work)},
            provenance="shape: input timestamp 0 only; pose/expression: current and previous inputs only"))
    figure(a.work / "tracking/dynamics.png", [
        dict(title="Transient parameters; identity shape frozen", ylabel="parameter norm",
             series=[("expression", list(range(stream.count)), [float(x["expr"].norm()) for x in history]),
                     ("jaw (radians)", list(range(stream.count)), [float(x["jaw"].norm()) for x in history])]),
        dict(title="Input-view fitting quality", ylabel="pixels", series=[
             ("median reprojection", list(range(stream.count)), [x["median_error_px"] for x in history])])])
    print(f"tracking -> {a.work / 'tracking.pt'}; overlay와 confidence를 먼저 확인하세요.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    arguments(p)
    p.add_argument("--initial-steps", type=int, default=600)
    p.add_argument("--track-steps", type=int, default=120)
    p.add_argument("--confidence-px", type=float, default=8.)
    fit_stream(p.parse_args())


if __name__ == "__main__":
    main()
