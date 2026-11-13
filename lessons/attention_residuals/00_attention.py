r"""00 — attention은 '어디에서 읽을지'와 '무엇을 읽을지'를 분리하는 계산이다.

실행 (conda elite, CPU):
    cd /media/vilab/mh/curious/lessons/attention_residuals
    python 00_attention.py

결과: outputs/00/lookup.png, scaling.png, multihead.png, checks.json
읽는 순서: 손으로 만든 Q/K/V -> softmax -> AV -> learned projection/multi-head.
첫 그림은 고정된 숫자의 lookup, 마지막 그림은 무작위 초기화 모델이다. 학습 결과가 아니다.
04에서는 이 Q/K/V projection도 실제 loss로 학습한다.
질문: key와 value가 왜 따로 필요할까? softmax의 dim을 바꾸면 무엇이 달라질까?
"""

import math
from pathlib import Path
import torch
from torch.nn import functional as F
from common import ROOT, parser, setup, plt, heatmap, savefig, write_json
from models import TokenAttention


def main():
    p = parser(__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/00")
    a = p.parse_args()
    setup()

    # key는 A/B/C의 주소, value는 주소에 저장된 RGB다. 둘의 의미가 다르다.
    keys = torch.eye(3) * 3
    values = torch.eye(3)
    queries = keys[[2, 0, 1]]
    scores = queries @ keys.T / math.sqrt(keys.shape[-1])
    weights = scores.softmax(dim=-1)
    output = weights @ values
    assert torch.allclose(weights.sum(-1), torch.ones(3))
    assert weights.argmax(-1).tolist() == [2, 0, 1]
    fig, ax = plt.subplots(1, 4, figsize=(13, 3))
    for axis, matrix, title in zip(ax[:3], (scores, weights, output), ("QK^T / sqrt(d)", "softmax over keys", "A @ V (RGB numbers)")):
        heatmap(axis, matrix, title, "key / RGB channel", "query: C, A, B")
    ax[3].imshow(output[:, None, :]); ax[3].set_title("Retrieved colors"); ax[3].axis("off")
    savefig(fig, a.out / "lookup.png")

    # d가 커질 때 독립 random Q/K의 dot-product 분산이 증가한다.
    # sqrt(d)는 이 조건하에서 logit scale을 안정화한다. 학습된 모든 attention의 보장은 아니다.
    dims = [4, 16, 64, 256, 1024]
    entropies = {"unscaled": [], "scaled": []}
    for d in dims:
        q, k = torch.randn(128, 1, d), torch.randn(128, 32, d)
        logits = q @ k.transpose(-1, -2)
        for label, divisor in (("unscaled", 1), ("scaled", math.sqrt(d))):
            w = (logits / divisor).softmax(-1)
            entropies[label].append(float(-(w * w.clamp_min(1e-30).log()).sum(-1).mean()))
    fig, ax = plt.subplots(figsize=(7, 3))
    for label, entropy in entropies.items():
        ax.plot(dims, entropy, "o-", label=label)
    ax.set(xscale="log", xlabel="head dimension", ylabel="attention entropy (nats)", title="Random Q/K: scale changes selection sharpness")
    ax.legend(); savefig(fig, a.out / "scaling.png")

    # 실제 projection은 [B,T,D] -> [B,H,T,D/H]. head별 서로 다른 주소/내용을 배운다.
    module = TokenAttention(32, 4).double()
    x = torch.randn(2, 6, 32, dtype=torch.float64)
    actual, attention = module(x)
    q, k, v = module.qkv(x).reshape(2, 6, 3, 4, 8).permute(2, 0, 3, 1, 4).unbind(0)
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    reference = module.proj(reference.transpose(1, 2).reshape(2, 6, 32))
    error = (actual - reference).abs().max().item()
    assert torch.allclose(actual, reference, atol=1e-10)
    changed = x.clone(); changed[:, 4:] += 100
    assert torch.allclose(module(changed)[0][:, :4], actual[:, :4], atol=1e-10)
    fig, ax = plt.subplots(1, 4, figsize=(13, 3))
    for h, axis in enumerate(ax):
        heatmap(axis, attention[0, h].detach(), f"Random head {h}", "key token", "query token", vmin=0, vmax=1)
    savefig(fig, a.out / "multihead.png")
    write_json(a.out / "checks.json", dict(sdpa_max_error=error, causal_future_isolation=True, training=False))
    print(f"Q/K/V + causal mask 확인 -> {a.out}")


if __name__ == "__main__":
    main()
