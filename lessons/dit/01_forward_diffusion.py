r"""01 — q(z_t|z_0): 학습 입력은 한 번의 계산으로 원하는 t에서 만든다.

    python 01_forward_diffusion.py
    # 다른 00 cache: python 01_forward_diffusion.py --data data/mnist64_full

결과: outputs/01/noising.png, schedule.png, algebra_check.json.
윗줄은 같은 z0에 같은 epsilon을 섞되 t만 바꾼 latent의 첫 채널이다.
아랫줄은 같은 주변분포의 독립 Gaussian noise를 사용한다. 둘 다 시간에 따른
한 Markov trajectory가 아니다. q_sample은 바로 z_t를 생성한다.
sqrt(alpha_bar)*z0 + sqrt(1-alpha_bar)*epsilon; beta와 alpha_bar를 혼동하지 않는다.
정답 epsilon으로 z0 복원이 되는 것은 대수 검증이며 학습한 denoiser 성능이 아니다.
"""

import torch
from diffusion import create_diffusion
from common import ROOT, parser, setup, data, posterior, pyplot, savefig, write_json


def main():
    p = parser(__doc__)
    p.add_argument("--data")
    a = p.parse_args()
    setup()
    _, _, val = data(a.data)
    z, _ = posterior(val, slice(0, 1), "cpu", mean_only=True)
    d = create_diffusion("")
    times = torch.tensor([0, 50, 150, 300, 500, 750, 999])
    noise = torch.randn_like(z).expand(len(times), -1, -1, -1)
    clean = z.expand_as(noise)
    mixed = d.q_sample(clean, times, noise)
    # schedule은 원본처럼 float64로 계산하고, sqrt 이후 모델의 float32로 바꾼다.
    # alpha가 1에 가까울 때 float32에서 먼저 1-alpha를 빼면 오차가 커질 수 있다.
    alpha = torch.from_numpy(d.alphas_cumprod)[times][:, None, None, None]
    signal_weight = alpha.sqrt().float()
    noise_weight = (1 - alpha).sqrt().float()
    manual = signal_weight * clean + noise_weight * noise
    recovered = (mixed - noise_weight * noise) / signal_weight
    assert torch.allclose(mixed, manual, atol=1e-6)
    error = float((recovered - clean).abs().max())
    assert error < 1e-4
    plt = pyplot()
    fig, ax = plt.subplots(2, len(times), figsize=(14, 4))
    independent = d.q_sample(clean, times, torch.randn_like(clean))
    for j, t in enumerate(times):
        for row, values in enumerate((mixed, independent)):
            ax[row, j].imshow(values[j, 0], cmap="coolwarm", vmin=-3, vmax=3)
            ax[row, j].axis("off")
            ax[row, j].set_title(f"t={int(t)}")
    fig.suptitle(
        "Latent channel 0 | top: shared epsilon; bottom: independent epsilon (not a Markov trajectory)"
    )
    savefig(fig, ROOT / "outputs/01/noising.png")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].plot(d.betas)
    axes[0].set_title("beta_t: per-step variance")
    axes[1].plot(d.alphas_cumprod)
    axes[1].set_title("alpha_bar: accumulated signal power")
    axes[2].semilogy(d.alphas_cumprod / (1 - d.alphas_cumprod))
    axes[2].set_title("SNR = alpha_bar / (1-alpha_bar)")
    for ax in axes:
        ax.set_xlabel("Diffusion timestep (not network depth)")
    savefig(fig, ROOT / "outputs/01/schedule.png")
    write_json(
        ROOT / "outputs/01/algebra_check.json",
        dict(original_q_sample_matches_formula=True, max_roundtrip_error=error),
    )


if __name__ == "__main__":
    main()
