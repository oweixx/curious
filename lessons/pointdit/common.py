"""공통 입출력/그림. 연구 수식은 00~03의 번호 파일에서 읽는다."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))


def parser(doc):
    return argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)


def setup(seed=7, device='cpu'):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = torch.device(device)
    if d.type == 'cuda':
        d = torch.device('cuda', d.index or 0)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use --device cpu')
        torch.cuda.set_device(d)
    return d


def lesson(i):
    name = f'pointdit_lesson_{i}'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, next(ROOT.glob(f'{i:02d}_*.py')))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp); tmp.replace(path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def finish(fig, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=140); pyplot().close(fig)


def array(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def point_rgb(points):
    """XYZ 색은 표시를 위해서만 [-3,3] -> [0,1]. 학습/적분 값은 clamp하지 않는다."""
    return np.clip((array(points).transpose(1, 2, 0)+3)/6, 0, 1)


def panels(rgb, target, prediction, mask, title='', error=True):
    rgb, mask = array(rgb).transpose(1, 2, 0), array(mask).astype(bool)
    pred = array(prediction); gt = None if target is None else array(target)
    plt = pyplot()
    n = 6 if gt is not None else 3
    fig, axs = plt.subplots(1, n, figsize=(3*n, 3.5))
    axs[0].imshow(rgb.clip(0, 1)); axs[0].set_title('RGB condition')
    if gt is None:
        axs[1].imshow(point_rgb(pred)); axs[1].set_title('Predicted normalized XYZ')
        p = axs[2].imshow(np.where(mask, pred[2], np.nan), cmap='turbo')
        axs[2].set_title('Centered Z (not metric depth)')
        fig.colorbar(p, ax=axs[2], shrink=.65)
    else:
        axs[1].imshow(point_rgb(gt)); axs[1].set_title('GT normalized XYZ')
        axs[2].imshow(point_rgb(pred)); axs[2].set_title('Predicted normalized XYZ')
        low, high = np.quantile(gt[2][mask], [.01, .99])
        for ax, z, name in [(axs[3], gt[2], 'GT centered Z'), (axs[4], pred[2], 'Predicted centered Z')]:
            ax.imshow(np.where(mask, z, np.nan), cmap='turbo', vmin=low, vmax=high); ax.set_title(name)
        err = np.linalg.norm(pred-gt, axis=0)
        p = axs[5].imshow(np.where(mask, err, np.nan), cmap='magma', vmin=0, vmax=1)
        axs[5].set_title('XYZ error (display max=1)'); fig.colorbar(p, ax=axs[5], shrink=.65)
    for ax in axs:
        ax.axis('off')
    fig.suptitle(title)
    return fig


def figure_image(fig):
    fig.tight_layout(); fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    pyplot().close(fig)
    return image


def pointcloud(points, rgb, mask, path, title='Normalized camera-oriented geometry'):
    pts, colors = array(points).transpose(1, 2, 0), array(rgb).transpose(1, 2, 0)
    valid = array(mask).astype(bool) & np.isfinite(pts).all(-1)
    pts, colors = pts[valid], colors[valid]
    if not len(pts):
        raise ValueError('No finite points to visualize')
    # A flat raster stride can create diagonal stripe artifacts in the scatter plot.
    ids = np.random.default_rng(17).choice(len(pts), min(12000, len(pts)), replace=False)
    pts, colors = pts[ids], colors[ids]
    plt = pyplot(); fig = plt.figure(figsize=(10, 4))
    for i, az in enumerate((-65, -20)):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=colors.clip(0, 1), s=2)
        ax.set(xlabel='X', ylabel='Z', zlabel='-Y'); ax.view_init(18, az)
        span = np.maximum(np.ptp(pts[:, [0, 2, 1]], axis=0), 1e-4)
        ax.set_box_aspect(span)
    fig.suptitle(title); finish(fig, path)


def ply(path, points, rgb, mask):
    xyz = array(points).transpose(1, 2, 0)
    color = (array(rgb).transpose(1, 2, 0).clip(0, 1)*255).round().astype(np.uint8)
    keep = array(mask).astype(bool) & np.isfinite(xyz).all(-1)
    rows = np.concatenate([xyz[keep], color[keep]], axis=1)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(rows)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        np.savetxt(f, rows, fmt='%.6f %.6f %.6f %d %d %d')


def rng_state(device):
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)


def restore_rng(state, device):
    torch.set_rng_state(state['cpu'])
    if device.type=='cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)
