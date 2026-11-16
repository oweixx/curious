r"""00 — RGB와 VAE latent를 구분한다. VAE는 고정, 이후 DiT만 학습한다.

    conda activate elite
    cd /media/vilab/mh/curious/lessons/done/dit
    CUDA_VISIBLE_DEVICES=4 python 00_pixels_and_latents.py

처음 실행: MNIST와 공식 SD VAE-EMA를 다운로드하고 posterior mean/logvar를 저장.
기본 train 10,000 / validation 1,000: MNIST train/test에서 각각 추출한다.
    CUDA_VISIBLE_DEVICES=4 python 00_pixels_and_latents.py --train-count 60000 --val-count 10000 --out data/mnist64_full
후속 01~06에도 --data data/mnist64_full을 지정한다.

결과: outputs/00/{reconstruction.png,latent_channels.png}, data/mnist64/*.pt.
흑백 숫자를 RGB 64x64로 바꾼다. 논문의 ImageNet256 실험과 구분한다.
64 RGB -> 4x8x8 latent; scale=0.18215. 채널의 색은 RGB 의미가 없다.
posterior.sample()과 mode(mean)의 차이, encode/decode의 정보 손실을 본다.
cache에는 scaled sample 한 장을 고정하지 않고 분포 자체를 저장한다.
"""

from pathlib import Path
import json
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from common import (
    ROOT,
    SCALE,
    setup,
    parser,
    vae,
    save,
    load,
    write_json,
    decode,
    image_grid,
    pyplot,
    savefig,
)


@torch.no_grad()
def main():
    p = parser(__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--train-count", type=int, default=10000)
    p.add_argument("--val-count", type=int, default=1000)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", type=Path, default=ROOT / "data/mnist64")
    a = p.parse_args()
    setup()
    if a.size < 32 or a.size % 16:
        p.error("size >= 32 and divisible by 16 (VAE stride8 + lesson patch2)")
    if a.batch_size < 1:
        p.error("batch-size must be positive")
    if not 10 <= a.train_count <= 60000 or not 10 <= a.val_count <= 10000:
        p.error("MNIST split 크기를 확인하세요")
    v = vae(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        dataset="MNIST",
        size=a.size,
        train_count=a.train_count,
        val_count=a.val_count,
        seed=42,
        latent_scale=SCALE,
        vae=json.loads((ROOT / "weights/vae_revision.json").read_text()),
    )
    meta = a.out / "meta.json"
    if meta.exists() and json.loads(meta.read_text()) != metadata:
        raise ValueError("기존 cache 설정과 다릅니다. 새 --out 사용")
    write_json(meta, metadata)
    transform = Compose(
        [
            Resize((a.size, a.size)),
            Lambda(lambda x: x.convert("RGB")),
            ToTensor(),
            Normalize(0.5, 0.5),
        ]
    )
    for split, count in (("train", a.train_count), ("validation", a.val_count)):
        dest = a.out / f"{split}.pt"
        if dest.exists():
            print("재사용:", dest)
            continue
        ds = MNIST(str(ROOT / "data"), train=split == "train", download=True, transform=transform)
        ids = torch.randperm(len(ds), generator=torch.Generator().manual_seed(42))[:count]
        loader = DataLoader(
            Subset(ds, ids.tolist()), batch_size=a.batch_size, shuffle=False, num_workers=0
        )
        means = []
        logvars = []
        labels = []
        examples = []
        for x, y in tqdm(loader, desc=f"VAE {split}"):
            posterior = v.encode(x.to(a.device)).latent_dist
            means.append(posterior.mean.cpu())
            logvars.append(posterior.logvar.cpu())
            labels.append(y)
            remaining = 20 - sum(len(batch) for batch in examples)
            if remaining > 0:
                examples.append(x[:remaining])
        save(
            dest,
            dict(
                mean=torch.cat(means),
                logvar=torch.cat(logvars),
                labels=torch.cat(labels),
                example_rgb=torch.cat(examples),
                source_indices=ids,
            ),
        )
    blob = load(a.out / "validation.pt")
    m = blob["mean"][:20].to(a.device)
    sampled = m + (blob["logvar"][:20].to(a.device) * 0.5).exp() * torch.randn_like(m)
    reconstruction = decode(v, m * SCALE)
    sample_rgb = decode(v, sampled * SCALE)
    image_grid(
        torch.cat([blob["example_rgb"][:10], reconstruction[:10], sample_rgb[:10]]),
        ROOT / "outputs/00/reconstruction.png",
        title="Rows: original / VAE mean reconstruction / VAE sample reconstruction",
    )
    plt = pyplot()
    fig, ax = plt.subplots(2, 4, figsize=(11, 5))
    for i in range(4):
        im = ax[0, i].imshow((m[0, i] * SCALE).cpu(), cmap="coolwarm")
        fig.colorbar(im, ax=ax[0, i])
        ax[0, i].set_title(f"scaled latent channel {i}")
        ax[1, i].hist((blob["mean"][:, i] * SCALE).flatten().numpy(), bins=60)
        ax[1, i].set_title("posterior mean distribution")
    savefig(fig, ROOT / "outputs/00/latent_channels.png")
    print("00 완료:", a.out, "| 시각화:", ROOT / "outputs/00")


if __name__ == "__main__":
    main()
