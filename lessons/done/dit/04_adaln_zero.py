r"""04 — adaLN-Zero: 조건부 shift/scale/gate와 zero initialization을 직접 관찰한다.

    python 04_adaln_zero.py
    # 06 이후 학습된 조건별 modulation 값도 확인:
    python 04_adaln_zero.py --checkpoint checkpoints/mini/best.pt --out outputs/04_trained

결과: outputs/04/{identity.png,gradient_flow.png,modulation.png,checks.json}.
block: x += gate_attn * Attention((1+scale)*LN(x)+shift); MLP branch도 같은 구성.
0 초기화는 모든 weight를0으로 만든다는 뜻이 아니다. adaLN의 최종 Linear,
최종 출력 head 등을0으로 초기화한다. 최초 block은 identity, network output은0.
첫 backward에서 output head부터 열리고, 이후 gate와 attention으로 gradient가 흐른다.
아래 4회의 optimizer step은 gradient 진단이며 학습 성능/생성 품질 실험이 아니다.
"""

from pathlib import Path
import torch
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


@torch.no_grad()
def inspect_modulation(model, out_dir, title):
    """조건만 바꾸면 shift/scale/gate가 어떻게 달라지는지 첫 block에서 관찰한다."""
    model.eval()
    times = torch.tensor([300, 300, 900, 300])
    labels = torch.tensor([0, 1, 0, model.y_embedder.num_classes])
    conditions = model.t_embedder(times) + model.y_embedder(labels, False)
    block = model.blocks[0]
    values = block.adaLN_modulation(conditions).chunk(6, dim=1)
    names = ("shift_attn", "scale_attn", "gate_attn", "shift_mlp", "scale_mlp", "gate_mlp")
    plt = pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, value, name in zip(axes.flat, values, names):
        limit = max(float(value.abs().max()), 1e-9)
        im = ax.imshow(value, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_yticks(range(4), ["t300 c0", "t300 c1", "t900 c0", "t300 null"])
        ax.set_xlabel("Feature channel")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(title + "\nEach row is broadcast over all spatial tokens")
    savefig(fig, out_dir / "modulation.png")

    # 원본 DiTBlock.forward를 분해해 계산한다. 이제 gate가0이 아니므로
    # identity 확인과 달리 두 residual branch가 실제로 연결되었는지 검사한다.
    from models import modulate

    latent = torch.randn(4, 4, *model.x_embedder.img_size)
    tokens = model.x_embedder(latent) + model.pos_embed
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = values
    attention_input = modulate(block.norm1(tokens), shift_a, scale_a)
    after_attention = tokens + gate_a[:, None] * block.attn(attention_input)
    mlp_input = modulate(block.norm2(after_attention), shift_m, scale_m)
    manual = after_attention + gate_m[:, None] * block.mlp(mlp_input)
    actual = block(tokens, conditions)
    error = float((manual - actual).abs().max())
    assert torch.allclose(manual, actual, atol=1e-6)
    return error


def main():
    p = parser(__doc__)
    p.add_argument("--data")
    p.add_argument("--checkpoint")
    p.add_argument("--out", type=Path, default=ROOT / "outputs/04")
    a = p.parse_args()
    setup()
    _, train, _ = data(a.data)
    z, y = posterior(train, slice(0, 8), "cpu")
    m = make_model(z.shape[-1])
    m.train()
    t = torch.arange(8) * 120
    tokens = m.x_embedder(z) + m.pos_embed
    c = m.t_embedder(t) + m.y_embedder(y, False)
    after = m.blocks[0](tokens, c)
    assert torch.equal(tokens, after)
    assert torch.count_nonzero(m(z, t, y)) == 0
    plt = pyplot()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    for axes, value, title in zip(
        ax,
        (tokens[0], after[0], after[0] - tokens[0]),
        ("block input", "block output at init", "difference = 0"),
    ):
        axes.imshow(value.detach(), aspect="auto")
        axes.set_title(title)
    savefig(fig, a.out / "identity.png")
    groups = {
        "output projection": m.final_layer.linear,
        "output modulation": m.final_layer.adaLN_modulation,
        "block0 modulation": m.blocks[0].adaLN_modulation,
        "block0 attention": m.blocks[0].attn,
        "block0 MLP": m.blocks[0].mlp,
    }
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0)
    d = create_diffusion("")
    history = []
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = d.training_losses(m, z, t, dict(y=y))["loss"].mean()
        loss.backward()
        history.append(
            [
                sum(float(p.grad.square().sum()) for p in module.parameters() if p.grad is not None)
                ** 0.5
                for module in groups.values()
            ]
        )
        optimizer.step()
    assert history[0][0] > 0 and all(v == 0 for v in history[0][1:])
    assert history[-1][3] > 0
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, name in enumerate(groups):
        ax.plot(range(1, 5), [max(row[i], 1e-16) for row in history], "o-", label=name)
    ax.set(yscale="log", xlabel="Backward pass", ylabel="Gradient norm (zeros plotted at 1e-16)")
    ax.legend()
    ax.set_title("Zero initialization opens gradient paths in stages")
    savefig(fig, a.out / "gradient_flow.png")
    if a.checkpoint:
        diagnostic_model, ck = model_from_checkpoint(a.checkpoint)
        title = f"First-block modulation | EMA checkpoint step {ck['step']}"
    else:
        diagnostic_model = m.eval()
        title = "First-block modulation | after 4 diagnostic updates; not a trained generator"
    manual_error = inspect_modulation(diagnostic_model, a.out, title)
    write_json(
        a.out / "checks.json",
        dict(
            block_identity=True,
            initial_model_output_zero=True,
            groups=list(groups),
            gradient_norms=history,
            manual_block_max_error=manual_error,
            modulation_checkpoint=a.checkpoint,
        ),
    )


if __name__ == "__main__":
    main()
