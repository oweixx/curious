"""공통 입출력·그림·재현성 도구. 직접 실행하지 않고 00~05에서 import한다.

경로와 plot만 공유하며 path/velocity/loss/solver는 번호 파일에서 직접 읽는다.
기존 elite Conda 환경을 사용한다. venv나 자동 package 설치는 없다.
"""
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
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))


def parser(doc):
    return argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)


def setup(seed=7, device="cpu"):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = torch.device(device)
    if d.type == "cuda":
        d = torch.device("cuda", 0 if d.index is None else d.index)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA가 없다. 00~03은 --device cpu 가능. GPU 실행은 CUDA_VISIBLE_DEVICES=7 사용.")
        torch.cuda.set_device(d)
    return d


def module(path, name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def lesson(number):
    return module(next(ROOT.glob(f"{number:02d}_*.py")), f"fm_lesson_{number}")


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def load(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)


def json_file(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh(path, overwrite=False):
    if Path(path).exists() and not overwrite:
        raise FileExistsError(f"결과가 이미 있다: {path}. 새 --out 또는 --overwrite 사용.")


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def step(loss, model, optimizer):
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads or any(not torch.isfinite(g).all() for g in grads):
        raise FloatingPointError("gradient 누락 또는 NaN/Inf")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
    optimizer.step()
    return float(norm)


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def finish(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    pyplot().close(fig)


def scatter(ax, x, title, colors=None):
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    ax.scatter(x[:, 0], x[:, 1], s=3, alpha=.5, c=colors, cmap="hsv" if colors is not None else None)
    ax.set(xlim=(-4, 4), ylim=(-4, 4), title=title, aspect="equal")
    ax.grid(alpha=.15)


def image_grid(images, path=None, columns=10, labels=None, title="", cell=72):
    """[-1,1] tensor를 표시할 때만 clip. ODE state를 clip하는 함수가 아니다."""
    x = images.detach().float().cpu()
    if x.ndim == 3:
        x = x[:, None]
    x = x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x
    rows = (len(x)+columns-1)//columns
    canvas = Image.new("RGB", (columns*cell, rows*(cell+18)+28), "#161b24")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), title, fill="white")
    for i, value in enumerate(x):
        array = ((value.clamp(-1, 1).permute(1, 2, 0).numpy()+1)*127.5).round().astype(np.uint8)
        image = Image.fromarray(array).resize((cell, cell), Image.Resampling.NEAREST)
        left, top = (i % columns)*cell, 28+(i//columns)*(cell+18)
        canvas.paste(image, (left, top+18))
        if labels is not None:
            draw.text((left+3, top+2), str(labels[i]), fill="white")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
    return canvas


def state_rng(device):
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"].cpu())
    if state["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)
