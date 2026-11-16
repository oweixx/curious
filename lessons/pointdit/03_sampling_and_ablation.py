"""03 — clean prediction -> velocity -> Euler. 1/few steps, zero/random initialization을 비교한다.

repo root (02 먼저 학습):
    CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/03_sampling_and_ablation.py --device cuda
lesson 폴더 (RGB ablation checkpoint):
    CUDA_VISIBLE_DEVICES=1 python 03_sampling_and_ablation.py --checkpoint outputs/02_rgb/best.pt \
        --device cuda --out outputs/03_rgb

출력: comparison.png, trajectory.gif, pointcloud.png,target_pointcloud.png,points.ply, metrics.json,conditioning.png.
읽기: sample -> evaluate -> main. 이 sampler에는 GT point map/centroid/scale/intrinsics를 주지 않는다.
같은 RGB, 같은 initial state로 step 수만 바꾼다. Inference 시 conditioning feature는 한 번 계산한다.
"""
from pathlib import Path
import numpy as np
import torch

from common import (ROOT, figure_image, finish, lesson, panels, parser, ply,
                    point_rgb, pointcloud, pyplot, save, setup, write_json)


@torch.no_grad()
def sample(model, rgb, steps=3, initial_std=0., seed=71, prediction_type='clean', keep=True, sample_eps=0.):
    if steps<1 or initial_std<0 or sample_eps<0:
        raise ValueError('steps>=1; nonnegative noise std and epsilon')
    old=model.training; model.eval()
    try:
        generator=torch.Generator().manual_seed(seed)
        z=torch.randn(rgb.shape, generator=generator).to(rgb.device)*initial_std
        times=torch.linspace(0, 1, steps+1, device=rgb.device)
        cond=lesson(1).condition_tokens(model, rgb)  # cached once, includes image position encoding
        states=[z.clone()] if keep else None
        for t, next_t in zip(times[:-1], times[1:]):
            prediction=model(z, t.expand(len(rgb)), rgb*2-1, cached_y_emb=cond)
            # t=1에서는 평가하지 않는다. 마지막 Euler update의 시작점은 항상 t<1.
            if prediction_type=='clean':
                velocity=(prediction-z)/(1-t).clamp_min(sample_eps)
            elif prediction_type=='velocity':
                velocity=prediction
            else: raise ValueError(prediction_type)
            z=z+(next_t-t)*velocity
            if not torch.isfinite(z).all(): raise FloatingPointError('Nonfinite Euler state')
            if keep: states.append(z.clone())
    finally:
        model.train(old)
    return z, torch.stack(states) if keep else None, dict(denoiser_nfe=steps, image_encoder_calls=1, initial_std=initial_std)


@torch.no_grad()
def evaluate(model, data, out, prediction='clean'):
    rgb=data['rgb']; target=data['points']; valid=data['valid'] & ~data['sky']
    plt=pyplot(); fig, axes=plt.subplots(2, 4, figsize=(12, 6))
    metrics=[]
    for row, std in enumerate([0., 1.]):
        for col, steps in enumerate([1, 2, 4, 8]):
            result, _, cost=sample(model, rgb, steps, std, prediction_type=prediction, keep=False)
            mse=(result-target).square().mean(1)[valid].mean()
            axes[row, col].imshow(point_rgb(result[0]))
            axes[row, col].set_title(f'std={std:g}, NFE={steps}, MSE={float(mse):.3f}'); axes[row, col].axis('off')
            metrics.append(dict(**cost, normalized_xyz_mse=float(mse)))
    fig.suptitle('Fixed RGB and seed; same XYZ display range [-3,3]'); finish(fig, out/'comparison.png')
    result, trace, cost=sample(model, rgb, 8, 0., prediction_type=prediction)
    images=[figure_image(panels(rgb[0], target[0], state[0], data['valid'][0], f'Euler state t={i/8:.3f}')) for i, state in enumerate(trace)]
    images[0].save(out/'trajectory.gif', save_all=True, append_images=images[1:], duration=220, loop=0)
    pointcloud(result[0], rgb[0], data['valid'][0], out/'pointcloud.png')
    pointcloud(target[0], rgb[0], data['valid'][0], out/'target_pointcloud.png', title='GT of the same validation image')
    ply(out/'points.ply', result[0], rgb[0], data['valid'][0])
    # Image prior이 실제로 쓰이는지 blank condition도 같은 state에서 비교한다.
    blank, _, _=sample(model, torch.full_like(rgb, .5), 8, 0., prediction_type=prediction, keep=False)
    fig, axes=plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(rgb[0].permute(1, 2, 0).cpu()); axes[0].set_title('RGB condition')
    axes[1].imshow(point_rgb(result[0])); axes[1].set_title('With original RGB')
    axes[2].imshow(point_rgb(blank[0])); axes[2].set_title('With blank RGB (OOD probe)')
    for ax in axes: ax.axis('off')
    finish(fig, out/'conditioning.png')
    write_json(out/'metrics.json', dict(rows=metrics, blank_condition_effect_mse=float((blank-result).square().mean()),
        note='Same normalized target frame; no GT affine alignment. Practice data is not a paper benchmark.'))
    save(out/'samples.pt', dict(points=result.cpu(), trace=trace.cpu(), rgb=rgb.cpu(), cost=cost))
    return metrics


def main():
    p=parser(__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'outputs/02/best.pt')
    p.add_argument('--data', type=Path)
    p.add_argument('--out', type=Path, default=ROOT/'outputs/03')
    p.add_argument('--device', default='cpu')
    a=p.parse_args(); device=setup(device=a.device)
    model, ck=lesson(1).from_lesson_checkpoint(a.checkpoint, device)
    root=a.data or Path(ck['data']); files=sorted((root/'validation').glob('*.npz'))[:4]
    if not files: raise FileNotFoundError('Validation samples missing')
    evaluate(model, lesson(0).batch(files, device), a.out, ck['train_config']['prediction'])
    print(a.out)


if __name__=='__main__':
    main()
