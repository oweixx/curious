r"""09 — DiT의 핵심 주장: parameter 수보다 token 수와 계산량을 함께 봐야 한다.

    python 09_scaling.py

결과: outputs/09/{scaling.png,depth_vs_steps.png,table.json}.
ImageNet256 latent32x32 기준 원본 S/B/L/XL x patch2/4/8을 비교한다.
meta device로 실제 원본 모델의 parameter 수만 센다. GPU 메모리에 XL12개를 올리지 않는다.
계산량은 주요 Linear/attention의 analytic MAC 추정이며 timing이나 FID가 아니다.
여기서1 multiply-add=1 unit. 2 FLOPs로 세는 profiler와 숫자를 바로 비교하지 않는다.
patch 절반 -> tokens4배, attention score의 quadratic term16배, 전체 network는 그 사이.
이 계산만으로 생성 품질 개선을 검증했다고 말하지 않는다. 논문의 FID 결과와 구분한다.
"""

import torch
from common import ROOT, setup, parser, make_model, PROFILES, pyplot, savefig, write_json


def compute(size, patch, width, depth):
    tokens = (size // patch) ** 2
    # QKV + output:4*T*D²; MLP:8*T*D²; QK/AV:2*T²*D; modulation:6*D².
    block = 12 * tokens * width**2 + 2 * tokens**2 * width + 6 * width**2
    patch_projection = tokens * patch**2 * 4 * width
    output = tokens * width * patch**2 * 8
    condition = 256 * width + width**2  # timestep MLP; embeddings cost negligible
    final_modulation = 2 * width**2
    return dict(
        tokens=tokens,
        approx_gmac=(depth * block + patch_projection + output + condition + final_modulation)
        / 1e9,
        attention_score_elements=tokens**2,
    )


def main():
    parser(__doc__).parse_args()
    setup()
    rows = []
    for profile in ("S", "B", "L", "XL"):
        for patch in (2, 4, 8):
            with torch.device("meta"):
                m = make_model(32, profile, patch, 1000)
            params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            row = dict(
                profile=profile,
                patch=patch,
                parameters=params,
                **compute(32, patch, PROFILES[profile]["hidden_size"], PROFILES[profile]["depth"]),
            )
            rows.append(row)
    plt = pyplot()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for profile in ("S", "B", "L", "XL"):
        selected = [r for r in rows if r["profile"] == profile]
        ax[0].plot(
            [r["tokens"] for r in selected],
            [r["parameters"] / 1e6 for r in selected],
            "o-",
            label=profile,
        )
        ax[1].plot(
            [r["tokens"] for r in selected],
            [r["approx_gmac"] for r in selected],
            "o-",
            label=profile,
        )
    for a in ax:
        a.set_xlabel("Latent patch tokens")
        a.legend()
        a.grid(alpha=0.2)
    ax[0].set_ylabel("Trainable parameters (million)")
    ax[1].set_ylabel("Approx. GMAC / forward (FMA=1)")
    savefig(fig, ROOT / "outputs/09/scaling.png")
    fig, ax = plt.subplots(figsize=(10, 4))
    for depth in (4, 12, 28):
        steps = torch.tensor([10, 50, 100, 250])
        ax.plot(steps, steps * depth, "o-", label=f"{depth} blocks/network")
    ax.set(
        xlabel="Sampling steps (one full model evaluation each, before CFG batching)",
        ylabel="Block evaluations per trajectory",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    savefig(fig, ROOT / "outputs/09/depth_vs_steps.png")
    write_json(
        ROOT / "outputs/09/table.json",
        dict(convention="analytic MAC estimate, not measured FID/latency", rows=rows),
    )


if __name__ == "__main__":
    main()
