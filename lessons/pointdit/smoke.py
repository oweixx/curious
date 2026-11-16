"""수식·공식 forward·실제 optimizer·resume·샘플링을 검사한다. 새 lesson 단계는 아니다.

repo root (현재 비어 있는 GPU로 CUDA_VISIBLE_DEVICES 변경 가능):
    CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/smoke.py --device cuda
가중치 확보 후 frozen DINO 경로도 검사:
    CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/smoke.py --device cuda --with-dino --out .cache/smoke_dino
기존 실행의 수식/gradient 검사만 다시:
    CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/smoke.py --device cuda --reuse

기본 smoke는 explicit RGB ablation이다. 이를 pretrained DINO 또는 논문 성능 재현으로 기록하지 않는다.
"""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch
from torch import nn

from assets import DINO_WEIGHTS, hf_to_meta, meta_encoder
from common import ROOT, lesson, load, parser, setup, write_json
from upstream.denoiser import Denoiser


def contracts(out, device):
    L00, L01, L02, L03, L04=[lesson(i) for i in range(5)]
    sample=L00.read_sample(out/'data/train/0000.npz')
    # Depth+K -> point -> projection must recover the original pixel grid.
    xyz=sample['raw']; K=sample['K']; h,w=xyz.shape[-2:]
    projected=K@xyz.reshape(3,-1); xy=(projected[:2]/projected[2:]).reshape(2,h,w)
    yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    valid=sample['valid'] & ~sample['sky']
    projection_error=float((xy-torch.stack([xx,yy]))[:,valid].abs().max())
    assert projection_error<1e-4
    normalized,_,_,_=L00.normalize_points(2.4*xyz+torch.tensor([1.,-2.,.8])[:,None,None],valid)
    assert torch.allclose(normalized[:,valid],sample['points'][:,valid],atol=2e-6)
    assert torch.allclose(sample['points'][:,sample['sky']].norm(dim=0),torch.full_like(sample['points'][0,sample['sky']],3),atol=2e-6)
    # Public unpatchify must restore pixel placement, including non-square maps.
    model=L01.build_model('smoke',32,'rgb').to(device)
    image=torch.randn(2,3,32,48,device=device)
    patches=image.reshape(2,3,2,16,3,16).permute(0,2,4,3,5,1).reshape(2,6,16*16*3)
    assert torch.equal(model.unpatchify(patches,16,2,3),image)
    # x-pred -> velocity -> one Euler step equals clean network output at t=0.
    rgb=torch.rand_like(image)
    model.eval()
    result,trace,_=L03.sample(model,rgb,1,initial_std=1.)
    with torch.no_grad():
        direct=model(trace[0],torch.zeros(2,device=device),rgb*2-1)
    assert torch.allclose(result,direct,atol=2e-6)
    # Original denoiser.py exact parity including clamped GT velocity and sky reduction.
    class Fixed(nn.Module):
        def forward(self,z,t,labels): return z*.3+labels*.1
    args=SimpleNamespace(noise_fill_invalid=True,exclude_invalid_gt=True,center_shift_point=True,
                         sky_loss_weight=.01,force_zero_t_ratio=.1)
    ref=Denoiser.__new__(Denoiser); nn.Module.__init__(ref)
    ref.net=Fixed(); ref.args=args; ref.P_mean=-.8; ref.P_std=.8; ref.t_eps=.05; ref.noise_scale=1.
    clean=torch.randn(2,3,8,8,device=device); labels=torch.randn_like(clean)
    mask=torch.rand(2,8,8,device=device)>.15; sky=(torch.rand(2,8,8,device=device)>.8)&mask
    torch.manual_seed(33)
    ref_loss,io=ref(clean,labels,return_model_input_output=True,force_zero_t=True,args=args,valid_mask=mask,sky_mask=sky)
    loss,_=L02.pointdit_loss(io['output'],clean,io['input'],io['t'].flatten(),mask,sky,relative_weight=0.)
    loss_error=float((loss-ref_loss).abs()); assert loss_error<1e-6
    t_high=clean.new_full((2,),.99); noise=torch.randn_like(clean)
    z=L02.noisy_pair(clean,mask,t_high,noise)
    paper,_=L02.pointdit_loss(clean,clean,z,t_high,mask,sky,recipe='paper',noise=noise,relative_weight=0.)
    sq=((clean-z)/.05-(clean-noise)).square().sum(1)
    weights=torch.where(sky,.01,1.)*mask
    expected=((sq*weights).flatten(1).sum(1)/mask.flatten(1).sum(1)).mean()
    assert torch.allclose(paper,expected,atol=1e-6)
    # Invalid pixels must have zero loss gradient; an all-invalid batch is rejected.
    pred=io['output'].detach().clone().requires_grad_(True)
    value,_=L02.pointdit_loss(pred,clean,io['input'],io['t'].flatten(),mask,sky)
    value.backward(); assert torch.count_nonzero(pred.grad.masked_select(~mask[:,None].expand_as(pred)))==0
    try: L02.pointdit_loss(pred,clean,io['input'],io['t'].flatten(),torch.zeros_like(mask),sky)
    except ValueError: pass
    else: raise AssertionError('All-invalid batch must fail')
    # Resize must preserve a non-square frame; restoration removes padding.
    resized,geometry=L04.resize_preserving_frame(torch.rand(3,37,83),64)
    assert resized.shape[-2]%16==0 and resized.shape[-1]%16==0
    assert abs(geometry['resized'][0]/37-geometry['resized'][1]/83)<.04
    assert L04.restore_resolution(resized,geometry).shape==(1,3,37,83)
    ck=load(out/'02/latest.pt'); resumed=load(out/'resume/latest.pt')
    differences={name:max(float((ck[name][k]-resumed[name][k]).abs().max()) for k in ck[name]) for name in ['model','ema']}
    assert max(differences.values())<3e-6, differences
    network,_=L01.from_lesson_checkpoint(out/'02/latest.pt',device)
    data=L00.batch([out/'data/validation/0000.npz'],device)
    p=network(torch.randn_like(data['points']),torch.full((1,),.3,device=device),data['rgb']*2-1)
    p.square().mean().backward()
    grads=[p.grad for p in network.blocks.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads)>0
    if ck['model_config']['encoder']=='dino':
        assert all(p.grad is None and not p.requires_grad for p in network.y_embedder.parameters())
    before={k:v.detach().clone() for k,v in network.state_dict().items()}
    one,_,cost=L03.sample(network,data['rgb'],3,1.,seed=19)
    two,_,_=L03.sample(network,data['rgb'],3,1.,seed=19)
    assert torch.equal(one,two)
    assert all(torch.equal(v,network.state_dict()[k]) for k,v in before.items())
    assert cost['image_encoder_calls']==1 and cost['denoiser_nfe']==3
    reference_sampler=Denoiser.__new__(Denoiser); nn.Module.__init__(reference_sampler)
    reference_sampler.net=network; reference_sampler.steps=3
    reference_sampler.args=SimpleNamespace(generate_noise_scale=0.,sample_t_eps=0.)
    official_sample=reference_sampler.generate(data['rgb']*2-1)
    local_sample,_,_=L03.sample(network,data['rgb'],3,0.)
    sampler_error=float((official_sample-local_sample).abs().max())
    assert sampler_error<2e-6
    report=dict(projection_max_error=projection_error,scale_shift_invariance=True,sky_radius=True,
        rectangular_patch_roundtrip=True,one_step_identity=True,released_velocity_loss_parity_error=loss_error,
        paper_equation_reduction=True,masked_gradients=True,all_invalid_guard=True,preserve_image_frame=True,transformer_gradients=True,
        resume_max_abs_difference=differences,deterministic_sampling=True,inference_weights_unchanged=True,
        released_sampler_parity_max_error=sampler_error,
        tested_encoder=ck['model_config']['encoder'],pretrained_pointdit_inference='not part of the small training smoke')
    write_json(out/'smoke_report.json',report)
    return report


def main():
    p=parser(__doc__);p.add_argument('--out',type=Path,default=ROOT/'.cache/smoke')
    p.add_argument('--device',default='cpu');p.add_argument('--with-dino',action='store_true');p.add_argument('--reuse',action='store_true')
    a=p.parse_args();a.out=a.out.resolve()
    if a.with_dino and not DINO_WEIGHTS.is_file(): raise FileNotFoundError('Real pretrained DINO weights required for --with-dino')
    def run(i,args):
        subprocess.run([sys.executable,str(next(ROOT.glob(f'{i:02d}_*.py'))),*map(str,args)],check=True)
    if not a.reuse:
        run(0,['--size',32,'--train-scenes',8,'--val-scenes',4,'--data',a.out/'data','--out',a.out/'00'])
        encoder='dino' if a.with_dino else 'rgb'
        run(1,['--encoder',encoder,'--profile','smoke','--sample',a.out/'data/train/0000.npz','--device',a.device,'--out',a.out/'01'])
        settings=['--data',a.out/'data','--encoder',encoder,'--profile','smoke','--batch-size',2,
                  '--val-every',4,'--preview-steps',2,'--warmup',0,'--device',a.device]
        run(2,[*settings,'--steps',8,'--out',a.out/'02'])
        run(2,[*settings,'--steps',4,'--out',a.out/'resume'])
        run(2,[*settings,'--steps',8,'--resume',a.out/'resume/latest.pt'])
        run(3,['--checkpoint',a.out/'02/latest.pt','--device',a.device,'--out',a.out/'03'])
    print(contracts(a.out,setup(device=a.device)))
    print(a.out/'smoke_report.json')


if __name__=='__main__':
    main()
