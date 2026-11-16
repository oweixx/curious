# Sources and reproduction boundaries

William Peebles and Saining Xie, **Scalable Diffusion Models with Transformers**,
ICCV 2023. Paper: https://arxiv.org/abs/2212.09748

Official source: https://github.com/facebookresearch/DiT
Pinned revision: **ed81ce2229091fd4ecc9a223645f95cf379d582b**.

- models.py and diffusion/*.py are verbatim copies from that revision.
- original/train.py, original/sample.py and original/download.py are retained as
  reading references. The numbered lessons are the runnable entry points here.
- source_manifest.json records SHA-256 hashes; smoke.py checks the copied sources.
- The original copyright headers and full LICENSE.txt (CC BY-NC 4.0) are retained.
  The numbered lessons adapt and illustrate this implementation, with attribution
  to Meta AI and the authors, under the same license.
- The original diffusion utilities credit OpenAI's diffusion implementations.
  Those attribution headers remain intact.

The educational training loop in 06_train.py adapts the official optimizer,
posterior sampling, diffusion loss and EMA. Added behavior includes single-device
execution, exact RNG resume, fixed validation previews, traces and diagnostics.
The mini profile (depth4/width192/heads6), MNIST64 data, short runs and no digit
flipping are educational choices. The mini preset uses EMA decay 0.999 for a
shorter averaging timescale; the original 0.9999 is available explicitly and
remains the default for the S/B/L/XL presets. This is not a reproduction of ImageNet FID
or a claim of paper-scale training. Default previews use DDIM; original sampling
is DDPM. CFG on all channels is an explicitly labeled alternative to the original
three-channel convention.

External assets are downloaded into ignored directories, never committed:
MNIST via torchvision; the frozen stabilityai/sd-vae-ft-ema VAE via Hugging Face
(default revision f04b2c4b98319346dad8c65879f680b1997b204a, recorded locally);
DiT-XL/2 via the authors' checkpoint URL.
Respective data/model terms continue to apply.
