"""00 — RGB pixel에 대응하는 XYZ, normalization, sky 처리를 직접 계산한다.

repo root:
    python lessons/pointdit/00_pointmaps.py
lesson 폴더:
    python 00_pointmaps.py --size 64 --train-scenes 128 --val-scenes 16
실제 RGB-D sample 확인:
    python 00_pointmaps.py --sample data/my_rgbd/train/000.npz --out outputs/00_real

출력: outputs/00/{geometry.png,pointcloud.png,normalization.png,points.ply,checks.json}.
기본 실행은 ray/sphere/plane intersection으로 작은 RGB-D 연습 dataset도 만든다.
이 데이터는 SceneNet/논문 성능 재현이 아니다. 실제 dataset도 같은 NPZ 계약으로 읽을 수 있다.
읽기: backproject -> normalize_points -> read_sample -> render_scene.
Point map의 2D 인덱스는 image pixel이다. FLAME의 canonical UV texel이 아니다.
"""
from pathlib import Path
import numpy as np
import torch

from common import ROOT, finish, panels, parser, ply, pointcloud, pyplot, setup, write_json


def backproject(depth, K):
    """Z-depth + pixel-unit K -> [3,H,W] camera XYZ. pixel centers는 정수 (u,v)."""
    h, w = depth.shape
    if K.shape != (3, 3) or not torch.isfinite(K).all() or abs(float(torch.linalg.det(K))) < 1e-10:
        raise ValueError('K must be a finite invertible 3x3 matrix')
    yy, xx = torch.meshgrid(torch.arange(h, device=depth.device, dtype=depth.dtype),
                            torch.arange(w, device=depth.device, dtype=depth.dtype), indexing='ij')
    pixel = torch.stack([xx, yy, torch.ones_like(xx)]).reshape(3, -1)
    ray = torch.linalg.solve(K, pixel).reshape(3, h, w)
    ray = ray / ray[2:3]
    return ray * depth[None]


def normalize_points(points, valid, sky=None, radius=3.):
    """μ=valid non-sky centroid, s=mean ||P-μ||. 회전/비등방 scale을 제거하는 식은 아니다.

    μ,s는 GT 전처리 메타데이터다. unknown RGB inference에서 가져다 쓰지 않는다.
    Invalid는 0 placeholder + 별도 mask. Sky는 원 camera ray 방향의 radius=3 구면.
    """
    sky = torch.zeros_like(valid) if sky is None else sky
    valid = valid.bool() & torch.isfinite(points).all(0)
    scene = valid & ~sky
    if scene.sum() < 2:
        raise ValueError('At least two valid non-sky points are required')
    center = points[:, scene].mean(1)[:, None, None]
    scale = (points[:, scene]-center[:, 0, 0, None]).norm(dim=0).mean()
    if not torch.isfinite(scale) or scale <= 1e-6:
        raise ValueError('Degenerate point-map scale')
    clean = torch.where(scene[None], (torch.nan_to_num(points)-center)/scale, 0.)
    if sky.any():
        directions = points[:, sky]
        if not torch.isfinite(directions).all() or (directions.norm(dim=0) < 1e-6).any():
            raise ValueError('Sky needs a finite nonzero ray direction')
        clean[:, sky] = radius * directions / directions.norm(dim=0)
    return clean, scene | sky, center, scale


def read_sample(path):
    """NPZ: rgb[H,W,3] uint8 or float[0,1], depth[H,W] Z-depth, K[3,3], optional valid/sky.

    Sky depth는 방향을 계산할 수 있는 finite positive 값으로 저장한다.
    Range sensor의 ray-distance는 Z-depth로 변환한 뒤 넣어야 한다.
    """
    with np.load(path, allow_pickle=False) as d:
        rgb = torch.from_numpy(d['rgb'].copy())
        rgb = rgb.float()/255 if rgb.dtype == torch.uint8 else rgb.float()
        depth = torch.from_numpy(d['depth'].copy()).float()
        K = torch.from_numpy(d['K'].copy()).float()
        valid = torch.from_numpy(d['valid'].copy()).bool() if 'valid' in d else torch.isfinite(depth) & (depth>0)
        sky = torch.from_numpy(d['sky'].copy()).bool() if 'sky' in d else torch.zeros_like(valid)
    if rgb.shape != (*depth.shape, 3) or not torch.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>1:
        raise ValueError(f'Invalid RGB shape/range: {path}')
    if valid.shape != depth.shape or sky.shape != depth.shape:
        raise ValueError('Mask sizes must match RGB/depth')
    valid = valid & torch.isfinite(depth) & (depth>0)
    raw = backproject(torch.where(valid | sky, depth, 0.), K)
    clean, valid, center, scale = normalize_points(raw, valid, sky)
    return dict(rgb=rgb.permute(2, 0, 1), points=clean, valid=valid, sky=sky,
                raw=raw, K=K, center=center, scale=scale)


def batch(paths, device):
    items = [read_sample(p) for p in paths]
    shapes = {tuple(i['rgb'].shape) for i in items}
    if len(shapes) != 1:
        raise ValueError('A batch needs equal image sizes; crop RGB/depth together and update K')
    return {key: torch.stack([d[key] for d in items]).to(device) for key in items[0]}


def render_scene(seed, size):
    """단일 pinhole camera의 rays와 실제 sphere/plane 교차. RGB에서 depth를 임의로 만들지 않는다."""
    rng = np.random.default_rng(seed)
    h = w = size
    focal = rng.uniform(.9, 1.3)*size
    K = np.array([[focal, 0, (w-1)/2], [0, focal, (h-1)/2], [0, 0, 1]], dtype=np.float32)
    yy, xx = np.mgrid[:h, :w]
    rays = np.stack([(xx-K[0, 2])/focal, (yy-K[1, 2])/focal, np.ones((h, w))], -1)
    depth = np.full((h, w), 6.)
    normal = np.broadcast_to([0., 0., -1.], (h, w, 3)).copy()
    checker = ((np.floor(rays[..., 0]*6*3)+np.floor(rays[..., 1]*6*3)) % 2)[..., None]
    albedo = np.broadcast_to(.25+.07*checker, (h, w, 3)).copy()
    for _ in range(rng.integers(2, 5)):
        center = np.array([rng.uniform(-1.3, 1.3), rng.uniform(-.9, .9), rng.uniform(2.6, 4.5)])
        radius = rng.uniform(.3, .75)
        a = np.sum(rays*rays, -1); b = -2*np.sum(rays*center, -1)
        c = center@center-radius**2
        disc = b*b-4*a*c
        z = (-b-np.sqrt(disc.clip(0)))/(2*a)
        hit = (disc>=0) & (z>0) & (z<depth)
        depth[hit] = z[hit]
        normal[hit] = (z[hit, None]*rays[hit]-center)/radius
        albedo[hit] = rng.uniform(.2, .95, 3)
    light = np.array([-.4, -.6, -1.]); light /= np.linalg.norm(light)
    shade = .25+.75*np.maximum(0, np.sum(normal*light, -1))
    rgb = np.clip(albedo*shade[..., None], 0, 1)
    sky = np.zeros((h, w), bool)
    if seed % 4 == 0:
        sky = (yy<size//6) & (depth==6)
        depth[sky] = 1000
        rgb[sky] = [.45, .65, .9]
    valid = np.ones((h, w), bool)
    # 작은 depth sensor hole. RGB는 남기고 GT validity만 제거한다.
    if seed % 3 == 0:
        valid[h//2:h//2+2, w//2:w//2+3] = False
    return dict(rgb=(rgb*255).round().astype(np.uint8), depth=depth.astype(np.float32), K=K, valid=valid, sky=sky)


def prepare_dataset(root, size, train_n, val_n):
    if size<32 or size%16 or min(train_n, val_n)<1:
        raise ValueError('size>=32 and divisible by 16; nonempty train/validation')
    root = Path(root)
    config = dict(kind='analytic ray-traced spheres + plane; not SceneNet', size=size,
                  train_n=train_n, val_n=val_n, train_seed=0, validation_seed=100000)
    meta = root/'dataset.json'
    if meta.exists():
        import json
        if json.loads(meta.read_text()) != config:
            raise FileExistsError('Dataset settings differ. Choose a new --data path.')
    for split, n, start in [('train', train_n, 0), ('validation', val_n, 100000)]:
        folder = root/split; folder.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            path = folder/f'{i:04d}.npz'
            if not path.exists():
                np.savez_compressed(path, **render_scene(start+i, size))
    write_json(meta, config)


def main():
    p = parser(__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'data/practice')
    p.add_argument('--sample', type=Path)
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--train-scenes', type=int, default=128)
    p.add_argument('--val-scenes', type=int, default=16)
    p.add_argument('--out', type=Path, default=ROOT/'outputs/00')
    a = p.parse_args(); setup()
    if a.sample is None:
        prepare_dataset(a.data, a.size, a.train_scenes, a.val_scenes)
    sample = read_sample(a.sample or a.data/'train/0000.npz')
    xyz, valid, sky = sample['points'], sample['valid'], sample['sky']
    finish(panels(sample['rgb'], xyz, xyz, valid, 'GT construction: image grid -> camera XYZ -> normalized XYZ'), a.out/'geometry.png')
    pointcloud(xyz, sample['rgb'], valid, a.out/'pointcloud.png')
    ply(a.out/'points.ply', xyz, sample['rgb'], valid)
    # 2.7*P+b의 정규화도 같다. 이 검사는 sky proxy를 제외한 실제 scene geometry에 적용한다.
    transform = sample['raw']*2.7 + torch.tensor([.8, -1.2, 3.])[:, None, None]
    normalized, _, _, _ = normalize_points(transform, valid & ~sky)
    err = (normalized[:, valid & ~sky]-xyz[:, valid & ~sky]).abs().max()
    plt = pyplot(); fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(sample['raw'][2][valid & ~sky].numpy(), bins=40); axes[0].set_title('Raw Z before centering')
    axes[1].hist(xyz[:, valid & ~sky].norm(dim=0).numpy(), bins=40); axes[1].set_title('Normalized point norms (mean=1)')
    finish(fig, a.out/'normalization.png')
    write_json(a.out/'checks.json', dict(normalized_mean_distance=float(xyz[:, valid & ~sky].norm(dim=0).mean()),
        positive_scale_translation_invariance_error=float(err), sky_count=int(sky.sum()),
        warning='Output uses relative scale and centroid; not metric depth or canonical UV coordinates'))
    print(a.out)


if __name__ == '__main__':
    main()
