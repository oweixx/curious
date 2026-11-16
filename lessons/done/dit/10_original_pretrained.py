r"""10 — 원본 DiT-XL/2와 ImageNet checkpoint로 지금까지의 계산을 실제 생성에 연결한다.

    CUDA_VISIBLE_DEVICES=4 python 10_original_pretrained.py
    # 빠른 생성 검사: sampling step만 줄이고 원본 XL/2는 유지
    CUDA_VISIBLE_DEVICES=4 python 10_original_pretrained.py --steps 50 --out outputs/10_fast

최초 실행은 저자 공개 checkpoint 약2.7GB를 weights/에 다운로드한다.
원본256x256, 4x32x32 latent, 256 tokens, 28 blocks, width1152, 16heads,
1000 ImageNet classes, CFG4/첫3 epsilon channels, learned reverse variance.
원본 sample.py의 공개 예제 label 중207/360/387/88을 사용한다.
00~09의 작은 MNIST 실험과 결과를 섞지 않는다. --steps250 DDPM이 원본 sampling 기본값.
VAE는 00의 EMA를 사용한다. 원본 sample.py 기본은 MSE이고 원논문은 두 VAE를 비교한다.

결과: outputs/10/{samples.png,token_attention.png,run.json}.
첫 attention은 t=999 noisy latent에서 관찰하므로 semantic segmentation으로 해석하지 않는다.
code clone/import가 아니라 이 lesson 안의 원본 models.py와 diffusion/를 직접 실행한다.
"""

from pathlib import Path
import json
import torch
from models import DiT_XL_2, modulate
from common import (
    ROOT,
    parser,
    setup,
    vae,
    decode,
    image_grid,
    lesson,
    digest,
    write_json,
    pyplot,
    savefig,
)


def main():
    p = parser(__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--labels", type=int, nargs="+", default=[207, 360, 387, 88])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/10")
    a = p.parse_args()
    setup(a.seed)
    path = ROOT / "weights/DiT-XL-2-256x256.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt"
    if not path.exists():
        tmp = path.with_suffix(".partial")
        torch.hub.download_url_to_file(url, str(tmp), progress=True)
        tmp.replace(path)
    m = DiT_XL_2(input_size=32, num_classes=1000)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    m.to(a.device).eval()
    v = vae(a.device)
    with torch.no_grad():
        z, _ = lesson(7).sample(m, a.labels, a.steps, a.cfg, a.seed, "ddpm", "paper", True)
        image_grid(
            decode(v, z),
            a.out / "samples.png",
            columns=len(a.labels),
            cell=256,
            labels=[f"ImageNet class {y}" for y in a.labels],
            title=f"Official DiT-XL/2 | DDPM {a.steps} | CFG {a.cfg} | VAE-EMA",
        )
        torch.manual_seed(a.seed)
        x = torch.randn(1, 4, 32, 32, device=a.device)
        t = torch.tensor([999], device=a.device)
        y = torch.tensor(a.labels[:1], device=a.device)
        tokens = m.x_embedder(x) + m.pos_embed
        c = m.t_embedder(t) + m.y_embedder(y, False)
        block = m.blocks[0]
        shift, scale, *_ = block.adaLN_modulation(c).chunk(6, dim=1)
        u = modulate(block.norm1(tokens), shift, scale)
        attn = block.attn
        n = u.shape[1]
        d = u.shape[2]
        h = attn.num_heads
        q, k, vv = attn.qkv(u).reshape(1, n, 3, h, d // h).permute(2, 0, 3, 1, 4).unbind(0)
        scores = ((attn.q_norm(q) * attn.scale) @ attn.k_norm(k).transpose(-2, -1)).softmax(-1)
        plt = pyplot()
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for i, ax in enumerate(axes):
            ax.imshow(scores[0, i, 0].reshape(16, 16).cpu())
            ax.set_title(f"head{i}, query0")
        fig.suptitle(
            "Official XL/2 first-block attention at noisy t=999 (not an explanation of generated objects)"
        )
        savefig(fig, a.out / "token_attention.png")
    write_json(
        a.out / "run.json",
        dict(
            model="official DiT-XL/2",
            checkpoint_url=url,
            checkpoint_sha256=digest(path),
            labels=a.labels,
            steps=a.steps,
            cfg=a.cfg,
            seed=a.seed,
            vae="sd-vae-ft-ema",
            vae_revision=json.loads((ROOT / "weights/vae_revision.json").read_text())["revision"],
            cfg_channels=3,
        ),
    )


if __name__ == "__main__":
    main()
