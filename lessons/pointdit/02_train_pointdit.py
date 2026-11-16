"""02 — clean XYZ를 출력하는 PointDiT를 velocity loss + relative point loss로 실제 학습.

repo root (00 데이터, DINOv3 자산 준비 후):
    CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/02_train_pointdit.py --device cuda
lesson 폴더, 공식 B backbone을 처음부터 학습:
    CUDA_VISIBLE_DEVICES=1 python 02_train_pointdit.py --profile B --device cuda --batch-size 2
가중치 없는 별도 linear-RGB ablation:
    CUDA_VISIBLE_DEVICES=1 python 02_train_pointdit.py --encoder rgb --device cuda --out outputs/02_rgb
재개 (steps는 누적 목표; 기존 profile/encoder/batch/precision 설정을 유지):
    CUDA_VISIBLE_DEVICES=1 python 02_train_pointdit.py --device cuda --resume outputs/02/latest.pt --steps 4000
가중치 없이 timestep/loss 수식만 시각화:
    python 02_train_pointdit.py --inspect-objective

출력: latest.pt,best.pt,history.json,loss.png,0.png,100.png,... (고정 validation 장면/조건).
읽기: sample_time -> noisy_pair -> pointdit_loss -> train loop -> validate.
원본 코드에 맞춘 loss가 기본이다. --recipe paper로 논문의 식과 세부 차이를 비교할 수 있다.
--prediction velocity는 별도 parameterization ablation이며 원 PointDiT의 기본 설정이 아니다.
"""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import math
from pathlib import Path

import torch
from tqdm import trange

from assets import DINO_REPO, DINO_WEIGHTS
from common import (ROOT, finish, lesson, load, panels, parser, pyplot, restore_rng, rng_state,
                    save, setup, sha256, write_json)
from upstream.loss import rel_point_loss


def sample_time(n, device, mean=-.8, std=.8, p_zero=.1):
    t = torch.sigmoid(torch.randn(n, device=device)*std+mean)
    return torch.where(torch.rand(n, device=device)<p_zero, 0., t)


def noisy_pair(clean, valid, t, noise, fill_invalid=True):
    t4 = t[:, None, None, None]
    filled = torch.where(valid[:, None], clean, noise) if fill_invalid else clean
    return t4*filled + (1-t4)*noise


def region_mean(values, mask):
    """빈 mask에서도 graph를 유지하되 해당 region loss는 0. 유효 sample 자체가 없으면 상위에서 거절."""
    mask = mask.expand_as(values)
    return values[mask].mean() if mask.any() else values.sum()*0


def pointdit_loss(prediction, clean, noisy, t, valid, sky, *, recipe='released',
                  prediction_type='clean', noise=None, t_eps=.05, sky_weight=.01, relative_weight=.1):
    """x-prediction != x-loss. 출력은 clean, 비교는 v로 바꾼 뒤 수행한다.

    Released denoiser.py는 GT와 prediction 모두 같은 clamp denominator를 사용한다.
    Paper Eq.(5)는 GT=x-noise를 쓴다. t>0.95에서 두 정의는 달라진다.
    Released는 non-sky mean + 0.01*sky mean, rel은 valid 전체 mean이다.
    Paper는 valid pixels에 대한 weighted sum / count, rel에도 sky weight를 적용한다.
    """
    if not valid.any():
        raise ValueError('All-invalid batch: cannot train')
    denominator = (1-t[:, None, None, None]).clamp_min(t_eps)
    if prediction_type=='clean':
        predicted_clean = prediction
        predicted_v = (prediction-noisy)/denominator
    elif prediction_type=='velocity':
        predicted_v = prediction
        predicted_clean = noisy+denominator*prediction
    else:
        raise ValueError(prediction_type)
    if recipe=='released':
        target_v = (clean-noisy)/denominator
    elif recipe=='paper':
        if noise is None: raise ValueError('Paper velocity target needs the noise sample')
        target_v = clean-noise
    else:
        raise ValueError(recipe)
    squared = (predicted_v-target_v).square()
    relative = rel_point_loss(predicted_clean.float(), clean.float())
    if recipe=='released':
        if sky_weight>0:
            flow = region_mean(squared, (valid & ~sky)[:, None]) + sky_weight*region_mean(squared, (valid & sky)[:, None])
        else:
            flow = region_mean(squared, valid[:, None])
        rel = relative[valid].mean()
    else:
        weight = torch.where(sky, sky_weight, 1.)*valid
        # Paper Eq.(5) uses squared L2 norm (sum over XYZ, not channel mean).
        # Interpret M as each image's valid-pixel count, then take the batch expectation.
        count=valid.flatten(1).sum(1)
        flow=((squared.sum(1)*weight).flatten(1).sum(1)/count.clamp_min(1))[count>0].mean()
        rel=((relative*weight).flatten(1).sum(1)/count.clamp_min(1))[count>0].mean()
    total = flow+relative_weight*rel
    return total, dict(flow=flow, relative=rel, clean=predicted_clean)


@torch.no_grad()
def update_ema(ema, model, decay):
    for dst, src in zip(ema.parameters(), model.parameters()):
        if src.requires_grad: dst.lerp_(src, 1-decay)


@torch.no_grad()
def validate(model, ema, sample, args, device, out, index):
    L01, L03 = lesson(1), lesson(3)
    devices = [device.index] if device.type=='cuda' else []
    old = model.training
    with torch.random.fork_rng(devices=devices):
        model.eval(); ema.eval()
        g = torch.Generator().manual_seed(91)
        noise = torch.randn(sample['points'].shape, generator=g).to(device)*args.noise_scale
        t = torch.linspace(0, .85, len(noise), device=device)
        noisy = noisy_pair(sample['points'], sample['valid'], t, noise)
        rows, previews = {}, []
        for name, network in [('model', model), ('ema', ema)]:
            cond = L01.condition_tokens(network, sample['rgb'])
            prediction = network(noisy, t, sample['rgb']*2-1, cached_y_emb=cond)
            value, _ = pointdit_loss(prediction, sample['points'], noisy, t, sample['valid'], sample['sky'],
                recipe=args.recipe, prediction_type=args.prediction, noise=noise)
            generated, _, _ = L03.sample(network, sample['rgb'], args.preview_steps, initial_std=0.,
                prediction_type=args.prediction, keep=False)
            err = (generated-sample['points']).square().mean(1)[sample['valid'] & ~sample['sky']].mean()
            rows[name+'_loss'], rows[name+'_xyz_mse'] = float(value), float(err)
            previews.append(generated)
        if not all(math.isfinite(v) for v in rows.values()):
            raise FloatingPointError('Nonfinite validation')
        # 한 JPG/PNG에 고정 frame의 EMA 추론 결과. GT는 panel/metric에만 쓰고 sample에는 주지 않는다.
        finish(panels(sample['rgb'][0], sample['points'][0], previews[1][0], sample['valid'][0],
            f'Step {index} | EMA inference from zeros | {args.encoder} / {args.prediction} / {args.recipe}'), out/f'{index}.png')
        model.train(old)
    return rows


def plot_losses(history, out):
    plt=pyplot(); fig, axes=plt.subplots(1, 2, figsize=(10, 4))
    train=[r for r in history if r['kind']=='train']; val=[r for r in history if r['kind']=='validation']
    for name in ['loss', 'flow', 'relative']:
        axes[0].plot([r['step'] for r in train], [r[name] for r in train], label=name)
    for name in ['model_xyz_mse', 'ema_xyz_mse']:
        axes[1].plot([r['step'] for r in val], [r[name] for r in val], label=name)
    axes[0].set_title('Training objectives (relative shown unweighted)')
    axes[1].set_title('Heldout XYZ error, generated from zeros')
    for ax in axes: ax.set_xlabel('optimizer step'); ax.legend()
    finish(fig, out/'loss.png')


def inspect_objective(out):
    setup()
    t=sample_time(100000, 'cpu')
    time=torch.linspace(0, .999, 1000)
    denominator=(1-time).clamp_min(.05)
    plt=pyplot(); fig, axes=plt.subplots(1,3,figsize=(13,4))
    axes[0].hist(t.numpy(), bins=50, density=True)
    axes[0].set(title='Logit-normal + explicit t=0 samples',xlabel='t (noise=0 -> clean=1)',ylabel='density')
    axes[1].plot(time,denominator.reciprocal().square());axes[1].set_yscale('log')
    axes[1].set(title='Released v-loss = weighted x-MSE',xlabel='t',ylabel='1 / max(1-t,0.05)^2')
    axes[2].plot(time,(1-time)/denominator,label='released GT / (x-noise)')
    axes[2].plot(time,torch.ones_like(time),'--',label='paper GT / (x-noise)')
    axes[2].set(title='GT definitions differ near t=1',xlabel='t',ylim=(0,1.1));axes[2].legend()
    finish(fig,out/'objective.png')
    write_json(out/'objective.json',dict(observed_exact_zero_fraction=float((t==0).float().mean()),
        note='x-prediction is the output parameterization; the released velocity loss weights clean MSE by the clamped denominator squared'))


def train(a):
    device=setup(a.seed, a.device); L00, L01=lesson(0), lesson(1)
    if min(a.steps, a.batch_size, a.val_every, a.preview_steps)<1 or a.lr<=0 or a.noise_scale<=0 or not 0<=a.ema_decay<1:
        raise ValueError('Invalid training settings')
    train_paths=sorted((a.data/'train').glob('*.npz')); val_paths=sorted((a.data/'validation').glob('*.npz'))
    if not train_paths or not val_paths: raise FileNotFoundError('Run 00 first, or provide data/train + data/validation NPZ files')
    signature=hashlib.sha256(''.join(sha256(p) for p in train_paths+val_paths).encode()).hexdigest()
    first=L00.read_sample(train_paths[0]); size=first['rgb'].shape[-1]
    if first['rgb'].shape[-2]!=size: raise ValueError('Training batch uses square crops; update K when cropping your dataset')
    model=L01.build_model(a.profile, size, a.encoder, a.dino_weights, a.dino_repo).to(device)
    ema=deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, betas=(.9, .95), weight_decay=0.)
    config={k:getattr(a, k) for k in ['profile', 'encoder', 'batch_size', 'lr', 'warmup', 'ema_decay', 'noise_scale', 'prediction', 'recipe', 'precision', 'seed']}
    out=a.out or (a.resume.parent if a.resume else ROOT/'outputs/02')
    if not a.resume and (out/'latest.pt').exists(): raise FileExistsError('Use --resume or a new --out')
    if a.precision=='bf16' and (device.type!='cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 requires supported CUDA; use --precision fp32')
    begin, history, best=0, [], float('inf')
    if a.resume:
        ck=load(a.resume)
        if ck['format']!='pointdit_lesson_v1' or ck['train_config']!=config or ck['data_signature']!=signature or ck['model_config']!=model.lesson_config:
            raise ValueError('Resume settings/data/encoder differ from checkpoint')
        if (out/'latest.pt').is_file() and load(out/'latest.pt')['step']>ck['step']:
            raise ValueError('A newer checkpoint exists in --out; choose a new output folder')
        L01.load_trainable_state(model, ck['model']); L01.load_trainable_state(ema, ck['ema'])
        optimizer.load_state_dict(ck['optimizer']); restore_rng(ck['rng'], device)
        begin, history, best=ck['step'], ck['history'], ck['best']
    if begin>=a.steps: raise ValueError('--steps must exceed the resumed step (cumulative target)')
    out.mkdir(parents=True, exist_ok=True)
    validation=L00.batch(val_paths[:4], device)
    if begin==0: history.append(dict(kind='validation', step=0, **validate(model, ema, validation, a, device, out, 0)))
    model.train()
    progress=trange(begin+1, a.steps+1, desc=f'PointDiT {a.encoder} / {a.prediction}')
    for i in progress:
        ids=torch.randint(len(train_paths), (a.batch_size,)).tolist()
        batch=L00.batch([train_paths[j] for j in ids], device)
        t=sample_time(a.batch_size, device)
        noise=torch.randn_like(batch['points'])*a.noise_scale
        noisy=noisy_pair(batch['points'], batch['valid'], t, noise)
        context=torch.autocast('cuda', dtype=torch.bfloat16) if a.precision=='bf16' else nullcontext()
        with context:
            cond=L01.condition_tokens(model, batch['rgb'])
            prediction=model(noisy, t, batch['rgb']*2-1, cached_y_emb=cond)
        loss, values=pointdit_loss(prediction.float(), batch['points'], noisy, t, batch['valid'], batch['sky'],
            recipe=a.recipe, prediction_type=a.prediction, noise=noise)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        for group in optimizer.param_groups: group['lr']=a.lr*min(1., i/max(1, a.warmup))
        optimizer.step(); update_ema(ema, model, a.ema_decay)
        history.append(dict(kind='train', step=i, loss=float(loss.detach()), flow=float(values['flow'].detach()),
                            relative=float(values['relative'].detach()), grad_norm=float(norm)))
        if i%10==0: progress.set_postfix(loss=f'{float(loss.detach()):.4f}')
        if i%a.val_every==0 or i==a.steps:
            metrics=validate(model, ema, validation, a, device, out, i)
            history.append(dict(kind='validation', step=i, **metrics))
            improved=metrics['ema_xyz_mse']<best; best=min(best, metrics['ema_xyz_mse'])
            ck=dict(format='pointdit_lesson_v1', step=i, model=L01.state_without_frozen_encoder(model),
                ema=L01.state_without_frozen_encoder(ema), optimizer=optimizer.state_dict(), rng=rng_state(device),
                model_config=model.lesson_config, train_config=config, history=history, best=best,
                data_signature=signature, data=str(a.data.resolve()))
            save(out/'latest.pt', ck)
            if improved: save(out/'best.pt', ck)
            write_json(out/'history.json', history); plot_losses(history, out)
    print(out/'latest.pt')


def main():
    p=parser(__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'data/practice')
    p.add_argument('--inspect-objective', action='store_true')
    p.add_argument('--out', type=Path); p.add_argument('--resume', type=Path)
    p.add_argument('--profile', choices=['B','mini','smoke'], default='mini')
    p.add_argument('--encoder', choices=['dino','rgb'], default='dino')
    p.add_argument('--dino-weights', type=Path, default=DINO_WEIGHTS)
    p.add_argument('--dino-repo', type=Path, default=DINO_REPO)
    p.add_argument('--steps', type=int, default=2000); p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-4); p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--ema-decay', type=float, default=.99)
    p.add_argument('--noise-scale', type=float, default=1.)
    p.add_argument('--recipe', choices=['released','paper'], default='released')
    p.add_argument('--prediction', choices=['clean','velocity'], default='clean')
    p.add_argument('--val-every', type=int, default=100); p.add_argument('--preview-steps', type=int, default=3)
    p.add_argument('--precision', choices=['fp32','bf16'], default='fp32')
    p.add_argument('--device', default='cpu'); p.add_argument('--seed', type=int, default=7)
    a=p.parse_args()
    if a.inspect_objective: inspect_objective(a.out or ROOT/'outputs/02')
    else: train(a)


if __name__=='__main__':
    main()
