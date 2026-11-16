r"""08 — 같은 noise에서 class와 CFG scale만 바꿔 조건의 효과를 분리한다.

    CUDA_VISIBLE_DEVICES=4 python 08_classifier_free_guidance.py --checkpoint checkpoints/mini/best.pt

결과: outputs/08/{cfg_sweep.png,channel_comparison.png,checks.json}.
행은 CFG0/1/2/4/8, 열은 class0..9. 모든 cell은 동일한 initial noise에서 시작.
scale0 = unconditional epsilon(원본 paper모드는 첫3채널만), 1 = conditional,
큰 scale은 conditional - unconditional 방향을 외삽한다. 높을수록 항상 좋은 것은 아니다.
원본3채널 관례와 일반적인4채널 guidance를 별도로 보여준다. variance는 guide하지 않는다.
classifier를 따로 학습하지 않는다. 06의 label dropout이 null branch 학습을 제공한다.
"""

from pathlib import Path
import torch
from tqdm.auto import tqdm
from common import (
    ROOT,
    parser,
    setup,
    model_from_checkpoint,
    vae,
    decode,
    image_grid,
    lesson,
    write_json,
)


def main():
    p = parser(__doc__)
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/mini/best.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/08")
    a = p.parse_args()
    setup()
    m, ck = model_from_checkpoint(a.checkpoint, a.device)
    v = vae(a.device)
    sampler = lesson(7)
    images = []
    labels = []
    # Separate class calls deliberately reuse the same initial noise (not merely same batch seed).
    with tqdm(total=50, desc="Same-noise class/CFG sweep") as bar:
        for scale in (0.0, 1.0, 2.0, 4.0, 8.0):
            zs = []
            for label in range(10):
                z, _ = sampler.sample(
                    m, [label], steps=a.steps, cfg=scale, seed=a.seed, cfg_mode="all"
                )
                zs.append(z)
                bar.update(1)
            images.append(decode(v, torch.cat(zs)))
            labels.extend([f"cfg={scale:g} c={i}" for i in range(10)])
    image_grid(
        torch.cat(images),
        a.out / "cfg_sweep.png",
        labels=labels,
        title=f"All 4 epsilon channels guided | checkpoint step {ck['step']} | identical starting noise",
    )
    comparison = []
    for mode in ("paper", "all"):
        z, _ = sampler.sample(
            m, list(range(10)), steps=a.steps, cfg=4.0, seed=a.seed, cfg_mode=mode
        )
        comparison.append(decode(v, z))
    image_grid(
        torch.cat(comparison),
        a.out / "channel_comparison.png",
        labels=[f"{mode} c={i}" for mode in ("paper", "all") for i in range(10)],
        title="Top: original 3-channel CFG / bottom: all 4 epsilon channels; same batch noise",
    )
    with torch.no_grad():
        n = 2
        size = m.x_embedder.img_size[0]
        x = torch.randn(n, 4, size, size, device=a.device)
        t = torch.tensor([400] * n, device=a.device)
        y = torch.tensor([1, m.y_embedder.num_classes], device=a.device)
        combined = torch.cat([x[:1], x[:1]])
        raw = m(combined, t, y)
        out0 = sampler.guided_forward(m, x, t, y, 0.0, "all")
        out1 = sampler.guided_forward(m, x, t, y, 1.0, "all")
        assert torch.allclose(out0[0, :4], raw[1, :4], atol=1e-6)
        assert torch.allclose(out1[0, :4], raw[0, :4], atol=1e-6)
        assert torch.equal(out1[:, 4:], raw[:, 4:])
    write_json(
        a.out / "checks.json",
        dict(
            cfg0_is_unconditional_epsilon=True,
            cfg1_is_conditional_epsilon=True,
            variance_unchanged=True,
            training_step=ck["step"],
        ),
    )


if __name__ == "__main__":
    main()
