"""04 — 공식 pretrained PointDiT-B + 실제 DINOv3로 RGB -> 상대 XYZ geometry.

repo root (DINO access 승인 및 assets 준비 필요):
    python lessons/pointdit/assets.py source
    python lessons/pointdit/assets.py dinov3
    python lessons/pointdit/assets.py pointdit --resolution 512
    CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/04_pretrained_and_geometry.py --device cuda
내 이미지:
    CUDA_VISIBLE_DEVICES=1 python 04_pretrained_and_geometry.py --image /absolute/path/photo.jpg --device cuda

--image를 생략하면 00의 연습 RGB만 입력한다. Depth/K/GT normalization은 inference에 전달하지 않는다.
출력: input.png,comparison.png,trajectory.gif,geometry.png,points.ply,prediction.npz,metadata.json.
Paper checkpoint는 02의 mini checkpoint와 다르다. 이 단계는 공식 checkpoint 구조를 엄격하게 검사한다.
읽기: load_official -> resize_preserving_frame -> sample(03) -> restore_resolution -> sky filter.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from assets import DINO_REPO, DINO_WEIGHTS, POINTDIT_FILES
from common import (ROOT, figure_image, finish, lesson, panels, parser, ply, point_rgb,
                    pointcloud, pyplot, setup, sha256, write_json)


def load_official(checkpoint, resolution, dino_weights, dino_repo, device, use_ema=True):
    # 공개 checkpoint는 argparse.Namespace를 포함할 수 있다. allowlist를 해당 class에만 한정한다.
    with torch.serialization.safe_globals([argparse.Namespace]):
        ck=torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(ck, dict) or 'model' not in ck:
        raise ValueError('Official PointDiT checkpoint must contain model')
    model=lesson(1).build_model('B', resolution, 'dino', dino_weights, dino_repo)
    chosen=dict(ck['model'])
    ema_used=use_ema and 'model_ema1' in ck
    if ema_used:
        if set(ck['model_ema1'])-set(chosen): raise ValueError('Unexpected EMA keys in official checkpoint')
        chosen.update(ck['model_ema1'])  # matches upstream engine.py's parameter swap
    state={}
    for key, value in chosen.items():
        key=key.removeprefix('module.')
        if not key.startswith('net.'):
            raise ValueError(f'Unexpected official key: {key}')
        key=key.removeprefix('net.')
        if key.startswith('y_embedder.'):
            raise ValueError('Use the official -nodinov3- checkpoint and explicit licensed DINO weights')
        state[key]=value
    lesson(1).load_trainable_state(model, state)
    model.official_ema_used=ema_used
    return model.to(device).eval()


def resize_preserving_frame(rgb, resolution):
    """동일 aspect-ratio resize -> bottom/right pad to /16. RGB 전체를 보존하고 stretch/crop하지 않는다.

    Official wild-image demo는 가까운 /16 rectangle로 resize한다. 여기서는 비율을 정확히 유지한 후
    최대 15px을 pad하는 표시용 차이다. 모델 입력과 결과의 pad를 함께 관리한다.
    """
    _, h, w=rgb.shape
    factor=resolution/(h*w)**.5
    hh, ww=max(16, round(h*factor)), max(16, round(w*factor))
    scaled=F.interpolate(rgb[None], size=(hh, ww), mode='bilinear', align_corners=False, antialias=True)
    ph, pw=(-hh)%16, (-ww)%16
    padded=F.pad(scaled, (0, pw, 0, ph), mode='replicate')
    return padded, dict(original=[h, w], resized=[hh, ww], padded=list(padded.shape[-2:]), pad=[ph, pw])


def restore_resolution(points, geometry):
    hh, ww=geometry['resized']
    return F.interpolate(points[..., :hh, :ww], size=geometry['original'], mode='bilinear', align_corners=False)


@torch.no_grad()
def main():
    p=parser(__doc__)
    p.add_argument('--image', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--resolution', type=int, choices=[256,512], default=512)
    p.add_argument('--dino-weights', type=Path, default=DINO_WEIGHTS)
    p.add_argument('--dino-repo', type=Path, default=DINO_REPO)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--initial-std', type=float, default=0.)
    p.add_argument('--keep-sky', action='store_true')
    p.add_argument('--raw-model', action='store_true', help='Official EMA1 대신 raw model weights 사용')
    p.add_argument('--out', type=Path, default=ROOT/'outputs/04')
    p.add_argument('--device', default='cpu')
    a=p.parse_args(); device=setup(device=a.device)
    checkpoint=a.checkpoint or ROOT/'weights'/POINTDIT_FILES[a.resolution]
    if not Path(a.dino_weights).is_file():
        raise FileNotFoundError('DINOv3 weights missing. Run python assets.py dinov3 after approval, or set --dino-weights')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Run python assets.py pointdit --resolution {a.resolution}')
    if a.image:
        rgb=torch.from_numpy(np.array(Image.open(a.image).convert('RGB'))).permute(2,0,1).float()/255
    else:
        # Only RGB: deliberately do not call read_sample, which would also load geometry GT.
        with np.load(ROOT/'data/practice/train/0000.npz', allow_pickle=False) as d:
            rgb=torch.from_numpy(d['rgb'].copy()).permute(2,0,1).float()/255
    condition, geometry=resize_preserving_frame(rgb, a.resolution)
    model=load_official(checkpoint, a.resolution, a.dino_weights, a.dino_repo, device, not a.raw_model)
    prediction, trace, cost=lesson(3).sample(model, condition.to(device), a.steps, a.initial_std)
    raw=restore_resolution(prediction, geometry)[0]
    # threshold는 normalized prediction에서 적용한다. unknown GT μ,s로 복원하지 않는다.
    keep=torch.isfinite(raw).all(0)
    if a.resolution==512 and not a.keep_sky: keep &= raw.norm(dim=0)<=2.9
    a.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb.permute(1,2,0).numpy()*255).round().astype(np.uint8)).save(a.out/'input.png')
    finish(panels(rgb, None, raw, keep, 'Official PointDiT-B | normalized coordinates; no metric scale'), a.out/'geometry.png')
    pointcloud(raw, rgb, keep, a.out/'pointcloud.png')
    ply(a.out/'points.ply', raw, rgb, keep)
    images=[]
    for i, state in enumerate(trace):
        view=restore_resolution(state, geometry)[0]
        images.append(figure_image(panels(rgb, None, view, torch.isfinite(view).all(0), f'Official Euler state t={i/a.steps:.3f}')))
    images[0].save(a.out/'trajectory.gif', save_all=True, append_images=images[1:], duration=300, loop=0)
    plt=pyplot(); fig, axes=plt.subplots(1,3,figsize=(11,4))
    for ax, steps in zip(axes,[1,3,8]):
        value, _, _=lesson(3).sample(model,condition.to(device),steps,a.initial_std,keep=False)
        ax.imshow(point_rgb(restore_resolution(value,geometry)[0])); ax.set_title(f'Euler NFE={steps}'); ax.axis('off')
    finish(fig,a.out/'comparison.png')
    np.savez_compressed(a.out/'prediction.npz',points=raw.cpu().numpy(),valid=keep.cpu().numpy(),rgb=rgb.numpy())
    write_json(a.out/'metadata.json',dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha256(checkpoint),
        dino_sha256=sha256(a.dino_weights),resize=geometry, sampling=cost,
        coordinates='centroid-shifted, mean-distance-normalized XYZ; unknown metric scale and translation',
        sky_removed=a.resolution==512 and not a.keep_sky, official_ema1_used=model.official_ema_used,
        model='official B/16, all non-DINO state keys strictly matched; not lesson mini'))
    print(a.out)


if __name__=='__main__':
    main()
