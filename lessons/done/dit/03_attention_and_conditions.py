r"""03 — Q/K/V와 timestep/class condition이 서로 들어가는 위치를 본다.

    python 03_attention_and_conditions.py
    # 06 이후 같은 진단을 학습한 모델로 다시 보기:
    python 03_attention_and_conditions.py --checkpoint checkpoints/mini/best.pt --out outputs/03_trained

결과: attention.png, conditions.png, attention_check.json.
원본 DiT는 c=t_embed+y_embed로 adaLN을 조절한다. 텍스트 cross-attention을 쓰는
Wan과 다르다. null label은 class 수(여기서는10), dropout은 train에서만 활성화.
attention 그림은 첫 block, head별 token0의 조회다. 미학습 attention에 의미를 부여하지 않는다.
"""

from pathlib import Path
import torch
from models import modulate
from diffusion import create_diffusion
from common import (
    ROOT,
    parser,
    setup,
    data,
    posterior,
    make_model,
    model_from_checkpoint,
    pyplot,
    savefig,
    write_json,
)


def main():
    p = parser(__doc__)
    p.add_argument("--data")
    p.add_argument("--checkpoint")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/03")
    a = p.parse_args()
    setup()
    _, _, val = data(a.data)
    x, y = posterior(val, slice(0, 1), "cpu", mean_only=True)
    m, ck = (
        model_from_checkpoint(a.checkpoint)
        if a.checkpoint
        else (make_model(x.shape[-1]).eval(), None)
    )
    t = torch.tensor([300])
    x = create_diffusion("").q_sample(x, t)
    tokens = m.x_embedder(x) + m.pos_embed
    c = m.t_embedder(t) + m.y_embedder(y, False)
    block = m.blocks[0]
    shift, scale, gate, *_ = block.adaLN_modulation(c).chunk(6, dim=1)
    u = modulate(block.norm1(tokens), shift, scale)
    attn = block.attn
    b, n, d = u.shape
    heads = attn.num_heads
    q, k, v = attn.qkv(u).reshape(b, n, 3, heads, d // heads).permute(2, 0, 3, 1, 4).unbind(0)
    q, k = attn.q_norm(q), attn.k_norm(k)
    weights = ((q * attn.scale) @ k.transpose(-2, -1)).softmax(-1)
    manual = attn.proj((weights @ v).transpose(1, 2).reshape(b, n, d))
    actual = attn(u)
    err = float((manual - actual).abs().max())
    assert torch.allclose(manual, actual, atol=1e-6)
    plt = pyplot()
    fig, ax = plt.subplots(1, min(heads, 6), figsize=(14, 3))
    for i, axes in enumerate(ax):
        axes.imshow(
            weights[0, i, 0].reshape(int(n**0.5), -1).detach(), vmin=0, vmax=float(weights.max())
        )
        axes.set_title(f"head {i}; query token0")
    fig.suptitle(f"First-block attention; t=300; {'trained' if ck else 'random initialization'}")
    savefig(fig, a.out / "attention.png")
    classes = m.y_embedder.num_classes
    times = torch.tensor([0, 50, 200, 500, 999])
    labels = torch.tensor([0, 1, 2, 3, classes])
    te = m.t_embedder(times)
    ye = m.y_embedder(labels, False)
    dropped = m.y_embedder(labels, True, force_drop_ids=torch.ones_like(labels))
    assert torch.equal(dropped, m.y_embedder.embedding_table.weight[classes].expand_as(dropped))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3))
    for axes, value, title in zip(
        ax,
        (te, ye, dropped),
        ("t embeddings", "class 0,1,2,3,null", "forced CFG dropout: all null"),
    ):
        axes.imshow(value.detach(), aspect="auto")
        axes.set_title(title)
        axes.set_xlabel("feature")
    savefig(fig, a.out / "conditions.png")
    write_json(
        a.out / "attention_check.json",
        dict(manual_attention_max_error=err, null_embedding_check=True, checkpoint=a.checkpoint),
    )


if __name__ == "__main__":
    main()
