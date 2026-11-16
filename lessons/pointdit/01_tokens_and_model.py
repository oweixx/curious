"""01 — 원본 PointDiT: patch tokens, DINOv3 4개 layer, channel fusion, clean XYZ head.

repo root (DINO assets 먼저 준비):
    CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/01_tokens_and_model.py --device cuda
lesson 폴더:
    CUDA_VISIBLE_DEVICES=1 python 01_tokens_and_model.py --profile B --device cuda
가중치 없이 별도 RGB linear-conditioning ablation을 볼 때:
    python 01_tokens_and_model.py --encoder rgb --profile smoke --out outputs/01_rgb

출력: tokens.png, feature_pca.png, tensor_shapes.json. 랜덤 PointDiT의 출력은 아직 geometry 예측이 아니다.
읽기: build_model -> upstream/model.py의 forward/PointDiTBlock/FinalLayer -> condition_tokens.
원본 B=12 blocks/768 width/12 heads/p16/bottleneck128. mini/smoke는 명시적인 축소 실험이다.
"""
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch import nn

from assets import DINO_REPO, DINO_WEIGHTS, meta_encoder
from common import ROOT, finish, lesson, load, parser, pyplot, setup, sha256, write_json
from upstream.model import PointDiT

PROFILES = dict(B=dict(hidden_size=768, depth=12, num_heads=12, bottleneck_dim=128),
                mini=dict(hidden_size=192, depth=4, num_heads=6, bottleneck_dim=128),
                smoke=dict(hidden_size=64, depth=2, num_heads=4, bottleneck_dim=32))


class RGBPatchEncoder(nn.Module):
    """논문 Table 3의 linear image conditioning을 따로 실험. DINOv3로 부르지 않는다."""
    def __init__(self, width):
        super().__init__()
        self.num_features = width
        self.proj = nn.Conv2d(3, width, 16, 16)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class LessonPointDiT(PointDiT):
    def train(self, mode=True):
        super().train(mode)
        if not self.finetune_feature_embedding:
            self.y_embedder.eval()
        return self


def build_model(profile='mini', size=64, encoder='dino', dino_weights=DINO_WEIGHTS, dino_repo=DINO_REPO):
    if size<32 or size%16:
        raise ValueError('Size must be >=32 and divisible by the original patch size 16')
    if encoder=='dino':
        if not Path(dino_weights).is_file():
            raise FileNotFoundError(f'DINOv3 weights missing: {dino_weights}\n'
                'Run python assets.py dinov3 after access approval, or supply --dino-weights /path/model.pth.\n'
                'An explicit --encoder rgb is available for a separate ablation; no random DINO fallback.')
        image_encoder = meta_encoder(dino_repo)
        image_encoder.load_state_dict(load(dino_weights), strict=True)
        image_encoder.eval().requires_grad_(False)
        feature_type, intermediate, tune = 'dinov3_vitb16', True, False
    elif encoder=='rgb':
        image_encoder = RGBPatchEncoder(PROFILES[profile]['hidden_size'])
        feature_type, intermediate, tune = 'rgb_linear', False, True
    else:
        raise ValueError(encoder)
    model = LessonPointDiT(input_size=size, patch_size=16, in_channels=3, **PROFILES[profile],
        feature_embedding_type=feature_type, dinov3_use_intermediate_layers=intermediate,
        dinov3_num_intermediate_layers=4, finetune_feature_embedding=tune, image_encoder=image_encoder)
    model.lesson_config = dict(profile=profile, size=size, encoder=encoder,
        dino_weights=str(Path(dino_weights).resolve()) if encoder=='dino' else None,
        dino_repo=str(Path(dino_repo).resolve()) if encoder=='dino' else None,
        dino_sha256=sha256(dino_weights) if encoder=='dino' else None)
    return model


def condition_tokens(model, rgb):
    """rgb in [0,1] -> frozen DINO hierarchy + position; sampler에서 한번만 계산한다.

    RGB ablation은 encoder도 trainable이므로 no_grad를 적용하면 안 된다.
    """
    with nullcontext() if model.finetune_feature_embedding else torch.no_grad():
        return model.extract_y_embedding(rgb*2-1, rgb.shape[-2]//16, rgb.shape[-1]//16)


def state_without_frozen_encoder(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if model.finetune_feature_embedding or not k.startswith('y_embedder.')}


def load_trainable_state(model, state):
    incompatible = model.load_state_dict(state, strict=False)
    allowed = {k for k in model.state_dict() if k.startswith('y_embedder.')} if not model.finetune_feature_embedding else set()
    if set(incompatible.missing_keys)!=allowed or incompatible.unexpected_keys:
        raise ValueError(f'Checkpoint mismatch: {incompatible}')


def from_lesson_checkpoint(path, device, ema=True):
    ck = load(path)
    if ck.get('format')!='pointdit_lesson_v1':
        raise ValueError('02 lesson checkpoint required')
    config = dict(ck['model_config']); expected = config.pop('dino_sha256')
    if config['encoder']=='rgb':
        config.pop('dino_repo'); config.pop('dino_weights')
    model = build_model(**config)
    if model.lesson_config['dino_sha256']!=expected:
        raise ValueError('DINO checkpoint changed since training')
    load_trainable_state(model, ck['ema' if ema else 'model'])
    return model.to(device).eval(), ck


def pca_image(tokens, h, w):
    x = tokens.detach().float().cpu()[0]
    x = x-x.mean(0)
    _, _, v = torch.linalg.svd(x, full_matrices=False)
    color = (x@v[:3].T).reshape(h, w, 3)
    lo, hi = color.amin((0, 1)), color.amax((0, 1))
    return ((color-lo)/(hi-lo).clamp_min(1e-8)).numpy()


@torch.no_grad()
def inspect(model, rgb, out):
    model.eval()
    h, w = rgb.shape[-2:]; n = (h//16)*(w//16)
    image_tokens = condition_tokens(model, rgb)
    noise = torch.randn_like(rgb)
    point_tokens = model.x_embedder(noise)+model.interpolate_pos_encoding(h//16, w//16)
    fused = model.concat_proj(torch.cat([image_tokens, point_tokens], dim=-1))
    output = model(noise, torch.zeros(len(rgb), device=rgb.device), rgb*2-1, cached_y_emb=image_tokens)
    shapes = dict(rgb=list(rgb.shape), noisy_points=list(noise.shape), point_tokens=list(point_tokens.shape),
        image_tokens=list(image_tokens.shape), concatenated_channels=image_tokens.shape[-1]+point_tokens.shape[-1],
        fused=list(fused.shape), clean_prediction=list(output.shape), sequence_length=n,
        intermediate_layers=getattr(model, 'dinov3_intermediate_layer_indices', []),
        encoder=model.lesson_config['encoder'], trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad))
    write_json(out/'tensor_shapes.json', shapes)
    plt = pyplot(); fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb[0].permute(1, 2, 0).cpu()); axes[0].set_title('RGB / aligned 16x16 patches')
    for pos in range(0, h+1, 16): axes[0].axhline(pos-.5, color='yellow', lw=.5)
    for pos in range(0, w+1, 16): axes[0].axvline(pos-.5, color='yellow', lw=.5)
    axes[1].imshow(image_tokens[0, :, :min(96, image_tokens.shape[-1])].cpu(), aspect='auto', cmap='coolwarm')
    axes[1].set(title='Image tokens: positions x channels', xlabel='first channels', ylabel='patch index')
    axes[2].imshow(fused[0, :, :64].cpu(), aspect='auto', cmap='coolwarm')
    axes[2].set(title='Channel-concat -> linear projection', xlabel='first 64 channels', ylabel='same patch index')
    finish(fig, out/'tokens.png')
    if model.lesson_config['encoder']=='dino':
        image = (rgb-rgb.new_tensor([.485, .456, .406])[None, :, None, None])/rgb.new_tensor([.229, .224, .225])[None, :, None, None]
        features = model.y_embedder.get_intermediate_layers(image, n=model.dinov3_intermediate_layer_indices, norm=True)
    else:
        features = (model.y_embedder(rgb*2-1),)
    fig, axs = plt.subplots(1, len(features), figsize=(3.5*len(features), 3.5), squeeze=False)
    for i, feat in enumerate(features):
        axs[0, i].imshow(pca_image(feat, h//16, w//16), interpolation='nearest')
        axs[0, i].set_title(f'DINO block {model.dinov3_intermediate_layer_indices[i]+1}' if len(features)>1 else 'Untrained RGB linear ablation')
        axs[0, i].axis('off')
    fig.suptitle('Feature PCA: colors are not XYZ predictions'); finish(fig, out/'feature_pca.png')
    return shapes


def main():
    p=parser(__doc__)
    p.add_argument('--sample', type=Path, default=ROOT/'data/practice/train/0000.npz')
    p.add_argument('--profile', choices=list(PROFILES), default='mini')
    p.add_argument('--encoder', choices=['dino', 'rgb'], default='dino')
    p.add_argument('--dino-weights', type=Path, default=DINO_WEIGHTS)
    p.add_argument('--dino-repo', type=Path, default=DINO_REPO)
    p.add_argument('--out', type=Path, default=ROOT/'outputs/01')
    p.add_argument('--device', default='cpu')
    a=p.parse_args(); device=setup(device=a.device)
    sample=lesson(0).read_sample(a.sample); size=sample['rgb'].shape[-1]
    if sample['rgb'].shape[-2]!=size: raise ValueError('01 initialization sample should be square; 04 supports rectangular inference')
    model=build_model(a.profile, size, a.encoder, a.dino_weights, a.dino_repo).to(device)
    print(inspect(model, sample['rgb'][None].to(device), a.out))


if __name__=='__main__':
    main()
