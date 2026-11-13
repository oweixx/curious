r"""05 — 원본 학습 objective: epsilon MSE + learned-variance VB.

    python 05_objective.py

결과: outputs/05/{loss_terms.png,gradient_routing.png,checks.json}.
4-channel latent를 입력해 8-channel을 출력: 앞4개=epsilon, 뒤4개=variance parameter.
뒤4개는 sigma 자체가 아니라 log-variance 범위를 보간하는 값이다.
VB에서 epsilon output은 detach되지만 공유 trunk에는 variance branch로 gradient가 간다.
t=0에서 decoder NLL, 그 외 posterior KL을 사용한다. 논문과 실제 기본 loss를 읽는다:
diffusion/gaussian_diffusion.py::training_losses, _vb_terms_bpd, p_mean_variance.
기본 create_diffusion은 MSE+VB이며 임의로 VB에0.001을 곱하지 않는다.
"""

import torch
from diffusion import create_diffusion
from common import ROOT, parser, setup, data, posterior, pyplot, savefig, write_json


def main():
    p = parser(__doc__)
    p.add_argument("--data")
    a = p.parse_args()
    setup()
    _, train, _ = data(a.data)
    z, _ = posterior(train, slice(0, 7), "cpu", mean_only=True)
    t = torch.tensor([0, 1, 50, 200, 500, 750, 999])
    eps = torch.randn_like(z)
    # Leaf outputs isolate the gradients received by each head from the original objective.
    output = torch.zeros(len(z), 8, *z.shape[-2:], requires_grad=True)
    d = create_diffusion("")
    terms = d.training_losses(lambda *args, **kwargs: output, z, t, noise=eps)
    expected = eps.square().flatten(1).mean(1)
    assert torch.allclose(expected, terms["mse"])
    assert torch.allclose(terms["loss"], terms["mse"] + terms["vb"])
    mse_grad = torch.autograd.grad(terms["mse"].mean(), output, retain_graph=True)[0]
    vb_grad = torch.autograd.grad(terms["vb"].mean(), output)[0]
    assert vb_grad[:, :4].count_nonzero() == 0 and vb_grad[:, 4:].abs().sum() > 0
    assert mse_grad[:, 4:].count_nonzero() == 0
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, color in (("mse", "tab:blue"), ("vb", "tab:orange"), ("loss", "black")):
        ax.plot(range(len(t)), terms[name].detach(), "o-", label=name, color=color)
    ax.set_xticks(range(len(t)), [str(int(x)) for x in t])
    ax.set_xlabel("t (nonuniform spacing)")
    ax.set_ylabel("Loss with zero network output")
    ax.legend()
    savefig(fig, ROOT / "outputs/05/loss_terms.png")
    fig, ax = plt.subplots(figsize=(8, 3))
    gradients = torch.stack([mse_grad.abs().mean((0, 2, 3)), vb_grad.abs().mean((0, 2, 3))])
    im = ax.imshow(gradients.detach(), aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(8), ["eps0", "eps1", "eps2", "eps3", "var0", "var1", "var2", "var3"])
    ax.set_yticks([0, 1], ["MSE gradient", "VB gradient"])
    savefig(fig, ROOT / "outputs/05/gradient_routing.png")
    write_json(
        ROOT / "outputs/05/checks.json",
        dict(
            mse_matches_noise_target=True,
            loss_is_mse_plus_vb=True,
            vb_epsilon_detached=True,
            mse_variance_gradient_zero=True,
            timesteps=t.tolist(),
        ),
    )


if __name__ == "__main__":
    main()
