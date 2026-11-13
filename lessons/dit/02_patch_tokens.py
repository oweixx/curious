r"""02 — latent의 patch를 token으로 바꾸고 output token을 공간으로 복원한다.

    python 02_patch_tokens.py

결과: outputs/02/patches.png, embeddings.png, roundtrip.json.
핵심: Conv2d(kernel=p,stride=p) == 각 patch를 flatten한 뒤 같은 Linear.
raw patch의 재배열은 가역이다. 학습한 embedding 자체가 역변환된다는 뜻은 아니다.
DiT final_layer는 새 p*p*8 값을 예측하고 unpatchify가 이를 8-channel grid로 배치한다.
MNIST64의 latent8x8, patch2 => 16 tokens. ImageNet256의 latent32x32 =>256 tokens.
"""

import torch
from common import ROOT, parser, setup, data, posterior, make_model, pyplot, savefig, write_json


def patchify(x, p):
    b, c, h, w = x.shape
    return x.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1).reshape(b, -1, p * p * c)


def main():
    p = parser(__doc__)
    p.add_argument("--data")
    a = p.parse_args()
    setup()
    _, _, val = data(a.data)
    x, _ = posterior(val, slice(0, 1), "cpu", mean_only=True)
    size = x.shape[-1]
    model = make_model(size)
    patch = 2
    # conv flattens patches as C,p,p while output unpatchify uses p,p,C.
    conv_patches = torch.nn.functional.unfold(x, kernel_size=patch, stride=patch).transpose(1, 2)
    linear = conv_patches @ model.x_embedder.proj.weight.flatten(1).T + model.x_embedder.proj.bias
    tokens = model.x_embedder(x)
    assert torch.allclose(linear, tokens, atol=1e-6)
    raw = patchify(x, patch).reshape(1, -1, patch, patch, 4)
    # Make an 8-channel output layout to exercise the UNMODIFIED original unpatchify.
    packed = torch.cat([raw, raw], -1).flatten(2)
    back = model.unpatchify(packed)
    assert torch.equal(back[:, :4], x)
    plt = pyplot()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    for a, pval in zip(ax, (1, 2, 4)):
        a.imshow(x[0, 0], cmap="coolwarm")
        a.set_xticks(torch.arange(-0.5, size, pval))
        a.set_yticks(torch.arange(-0.5, size, pval))
        a.grid(color="black")
        a.set_title(f"patch={pval}; tokens={(size//pval)**2}")
    savefig(fig, ROOT / "outputs/02/patches.png")
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for axes, values, title in zip(
        ax,
        (tokens, model.pos_embed, tokens + model.pos_embed),
        ("learned patch embedding", "fixed 2D sine/cosine position", "sum: block input"),
    ):
        im = axes.imshow(values[0].detach(), aspect="auto")
        axes.set_title(title)
        axes.set_xlabel("feature")
        axes.set_ylabel("token")
        fig.colorbar(im, ax=axes)
    savefig(fig, ROOT / "outputs/02/embeddings.png")
    write_json(
        ROOT / "outputs/02/roundtrip.json",
        dict(
            patch_projection_equivalence=True, unpatchify_exact=True, token_shape=list(tokens.shape)
        ),
    )


if __name__ == "__main__":
    main()
