"""공통 I/O와 시각화. 직접 실행하지 않고 00~10에서 import한다.

DiT 계산은 models.py, diffusion 수식은 diffusion/의 원본 코드에 있다.
ROOT 기준 기본 경로를 사용한다. 사용자 conda/다른 lesson의 파일은 수정하지 않는다.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parent
SCALE = 0.18215
VAE_ID = "stabilityai/sd-vae-ft-ema"
VAE_REVISION = "f04b2c4b98319346dad8c65879f680b1997b204a"  # 확인한 공식 VAE-EMA snapshot.
PROFILES = {
    "mini": dict(depth=4, hidden_size=192, num_heads=6),
    "S": dict(depth=12, hidden_size=384, num_heads=6),
    "B": dict(depth=12, hidden_size=768, num_heads=12),
    "L": dict(depth=24, hidden_size=1024, num_heads=16),
    "XL": dict(depth=28, hidden_size=1152, num_heads=16),
}


def parser(doc):
    return argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )


def lesson(number):
    path = next(ROOT.glob(f"{number:02d}_*.py"))
    spec = importlib.util.spec_from_file_location(f"dit_lesson_{number}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup(seed=42):
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def data(folder=None):
    folder = Path(folder or ROOT / "data/mnist64")
    if not (folder / "train.pt").is_file():
        raise FileNotFoundError("먼저 00_pixels_and_latents.py를 실행하세요: " + str(folder))
    return folder, load(folder / "train.pt"), load(folder / "validation.pt")


def make_model(input_size=8, profile="mini", patch=2, classes=10):
    from models import DiT

    return DiT(
        input_size=input_size,
        patch_size=patch,
        in_channels=4,
        num_classes=classes,
        learn_sigma=True,
        class_dropout_prob=0.1,
        **PROFILES[profile],
    )


def model_from_checkpoint(path, device="cpu", ema=True):
    ck = load(path)
    m = make_model(**ck["model_config"])
    m.load_state_dict(ck["ema" if ema else "model"], strict=True)
    return m.to(device).eval(), ck


def vae(device="cpu"):
    from diffusers import AutoencoderKL

    # Pin once, reuse the same VAE for cache/training/inference, without hidden updates.
    pointer = ROOT / "weights/vae_revision.json"
    revision = json.loads(pointer.read_text())["revision"] if pointer.exists() else VAE_REVISION
    model = AutoencoderKL.from_pretrained(
        VAE_ID, revision=revision, cache_dir=str(ROOT / "weights")
    )
    resolved = model.config.get("_commit_hash", None)
    if not pointer.exists():
        if not resolved:
            from huggingface_hub import model_info

            resolved = model_info(VAE_ID, revision=revision).sha
        write_json(pointer, dict(model=VAE_ID, revision=resolved))
    return model.to(device).eval().requires_grad_(False)


def posterior(blob, ids, device, mean_only=False):
    # Cache mean/logvar, not one fixed draw: original train.py samples q(z|image) every visit.
    mean = blob["mean"][ids].to(device)
    logvar = blob["logvar"][ids].to(device)
    z = mean if mean_only else mean + (0.5 * logvar).exp() * torch.randn_like(mean)
    return z * SCALE, blob["labels"][ids].to(device)


@torch.no_grad()
def decode(v, z):
    return torch.cat([v.decode(x / SCALE).sample.cpu() for x in z.split(16)]).clamp(-1, 1)


def image_grid(images, path, columns=10, labels=None, cell=96, title=""):
    images = images.detach().float().cpu()
    if images.ndim == 3:
        images = images[:, None]
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    n = len(images)
    rows = (n + columns - 1) // columns
    header = 28 if title else 0
    line = 20 if labels is not None else 0
    canvas = Image.new("RGB", (columns * cell, rows * (cell + line) + header), "#181c22")
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((5, 7), title, fill="white")
    for i, x in enumerate(images):
        im = Image.fromarray(
            ((x.clamp(-1, 1).permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
        )
        im = im.resize((cell, cell))
        left, top = i % columns * cell, header + i // columns * (cell + line)
        canvas.paste(im, (left, top + line))
        if labels is not None:
            draw.text((left + 3, top + 3), str(labels[i]), fill="white")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return canvas


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def savefig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    pyplot().close(fig)


def rng_state():
    return dict(
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        numpy=np.random.get_state(),
        python=random.getstate(),
    )


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def update_ema(ema, model, decay):
    # Adapted from official train.py; parameters with no gradient (fixed positions) are copied.
    for (name, target), (_, source) in zip(ema.named_parameters(), model.named_parameters()):
        if source.requires_grad:
            target.mul_(decay).add_(source, alpha=1 - decay)
        else:
            target.copy_(source)
