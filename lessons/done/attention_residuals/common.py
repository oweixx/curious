"""그림/실행 설정만 공유한다. 논문의 계산은 models.py와 각 번호 파일에 있다."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#e67e22", "#00b8b8", "#e685b5"]
VARIANTS = ("baseline", "full", "block", "static", "no_key_norm")


def parser(doc):
    return argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)


def setup(seed=42):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)


def savefig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def heatmap(ax, matrix, title, xlabel="source", ylabel="destination", **kwargs):
    image = ax.imshow(np.asarray(matrix), aspect="auto", interpolation="nearest", **kwargs)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return image


def rms(x):
    return x.detach().square().mean().sqrt().item()


def recall_batch(batch_size, pairs, generator):
    """새 key/color 대응을 매번 만든다. 마지막 query token에는 정답 색을 넣지 않는다.

    예: (C,red), (A,blue), (B,green), query=A -> blue.
    key ID: 0..pairs-1, color ID: 0..7, 8은 query의 unknown color.
    하나의 (key,color) pair를 하나의 token으로 표현하는 학습용 과제다.
    자연어 corpus/next-token language modeling의 재현은 아니다.
    """
    keys = torch.rand(batch_size, pairs, generator=generator).argsort(dim=1)
    colors = torch.randint(8, (batch_size, pairs), generator=generator)
    selected = torch.randint(pairs, (batch_size,), generator=generator)
    rows = torch.arange(batch_size)
    query = keys[rows, selected]
    target = colors[rows, selected]
    keys = torch.cat((keys, query[:, None]), dim=1)
    colors = torch.cat((colors, torch.full((batch_size, 1), 8)), dim=1)
    return keys, colors, target


def draw_predictions(ax, keys, colors, target, predicted):
    """색 표와 정답/예측을 직접 읽을 수 있는 inference 그림."""
    keys, colors, target, predicted = [x.detach().cpu() for x in (keys, colors, target, predicted)]
    n, length = keys.shape
    ax.set(xlim=(-0.6, length + 1.6), ylim=(n - 0.4, -1.25))
    for row in range(n):
        for col in range(length - 1):
            ax.add_patch(plt.Rectangle((col - .42, row - .38), .84, .76, color=COLORS[colors[row, col]]))
            ax.text(col, row, chr(65 + keys[row, col]), ha="center", va="center", color="black", weight="bold")
        ax.text(length - 1, row, chr(65 + keys[row, -1]) + "?", ha="center", va="center")
        for col, value in ((length, target[row]), (length + 1, predicted[row])):
            ax.add_patch(plt.Rectangle((col - .38, row - .32), .76, .64, color=COLORS[value]))
        if target[row] != predicted[row]:
            ax.text(length + 1, row, "x", ha="center", va="center", color="black", weight="bold")
    for col, label in ((0, "memory: key / color"), (length - 1, "query"), (length, "truth"), (length + 1, "pred")):
        ax.text(col, -0.85, label, fontsize=8, ha="center")
    ax.axis("off")
