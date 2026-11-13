r"""03 — camera-triplet rectified flow fine-tuning (Eq.5, Table7).

실행 위치 / 환경:
    conda activate elite
    cd /media/vilab/mh/curious/lessons/vdm

준비물: 01의 checkpoints/01/best.pt + 02의 stage별 cache/index.json.
아래 명령들은 모두 직접 학습한 01 checkpoint를 사용한다.
042 한 사람 학습이며 validation은 같은 사람의 다른 시간 구간이다.

1. Stage1: 288x512, 77프레임, 기본 5000 steps:
    CUDA_VISIBLE_DEVICES=4 python 03_train_avatar.py \
        --cache data/cache_stage1 --disentangler checkpoints/01/best.pt \
        --out checkpoints/03_stage1 --val-every 100

2. Stage1 학습 재개 (optimizer/RNG 포함; 같은 data/config/source 필요):
    CUDA_VISIBLE_DEVICES=4 python 03_train_avatar.py \
        --cache data/cache_stage1 --disentangler checkpoints/01/best.pt \
        --out checkpoints/03_stage1 --val-every 100 \
        --resume checkpoints/03_stage1/latest.pt

3. Stage2: 먼저 02에서 data/cache_stage2 생성. stage1 best에서 시작, 기본 250 steps:
    CUDA_VISIBLE_DEVICES=4 python 03_train_avatar.py --stage 2 \
        --cache data/cache_stage2 --disentangler checkpoints/01/best.pt \
        --adapter checkpoints/03_stage1/best.pt \
        --out checkpoints/03_stage2 --val-every 100
4. Stage2 재개 (--adapter 없이 optimizer/RNG까지 복원):
    CUDA_VISIBLE_DEVICES=4 python 03_train_avatar.py --stage 2 \
        --cache data/cache_stage2 --disentangler checkpoints/01/best.pt \
        --resume checkpoints/03_stage2/latest.pt \
        --out checkpoints/03_stage2 --val-every 100

짧은 실행 확인용 (실제 데이터 cache 필요; 본 학습과 별도 출력 폴더 사용):
    CUDA_VISIBLE_DEVICES=4 python 03_train_avatar.py \
        --cache data/cache_stage1 --disentangler checkpoints/01/best.pt \
        --steps 2 --val-every 1 --no-val-render --out checkpoints/03_quickcheck

출력: --out 아래 best.pt/latest.pt, training.jsonl, validation/metrics.jsonl,
validation/0.jpg,100.jpg,... (마지막 step도 저장).
--no-val-render는 생성 JPG만 생략하며 고정 flow validation은 수행한다.
--steps는 재개 시에도 누적 목표 step 수다. --adapter는 가중치만 초기화하는
warm start이고 --resume과 함께 쓸 수 없다. 기본 --mode paper는 expression
head도 학습한다. 비교용 --mode release는 이 head를 고정한다.
01의 --protocol single-subject는 데이터 검증 설정이고, 여기 --mode paper는
학습할 parameter 선택이므로 함께 사용해도 된다. 서로 다른 설정이다.
다음 04에서는 data/triplets/042_free_val과 이 단계의 best.pt를 사용한다.
도움말: python 03_train_avatar.py --help

x_sigma=(1-sigma)*x0+sigma*noise; velocity=noise-x0.
reference는 별도 latent slot0이고 loss에서 제외한다. 77 target frames ->20 slots.
오직 adapters를 optimize하지만 frozen DiT를 통과하는 backward는 유지한다.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm.auto import tqdm
from common import ROOT, COMMIT, config, lesson, load, save, log, seed_all, rng_state, restore_rng, digest


def flow_pair(x0, reference, sigma, noise):
    """학습되는 velocity에는 reference가 포함되지 않는다."""
    mixed=(1-sigma)*x0+sigma*noise
    return torch.cat((reference,mixed),dim=2), noise-x0


def weighted_flow_loss(prediction, target, sigma, bell_width=.25):
    # Bell-shaped만 명시, 정확한 함수/width는 미공개. Gaussian bell은 이 lesson의 선택.
    weight=torch.exp(-.5*((sigma.float()-.5)/bell_width).square())
    return weight*(prediction[:,:,1:].float()-target.float()).square().mean()


def to_device(blob, device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in blob.items()}


def conditions(blob, drop=False):
    """Joint identity/expression null + optional proxy null. Skeleton stays available.

    blur strength는 release 사용값, dropout 확률은 공개되지 않아 config에 명시.
    weak identity일 때 reference/CLIP/y를 함께 바꿔 조건 불일치를 막는다.
    """
    c={k:blob[k] for k in ("context","y","clip_feature","pose_latents","graft_latents","expr_codes")}
    ref=blob["first_frame_latents"]
    if drop:
        choice=config()["training"]["implementation_choices"]
        weak=torch.rand(()).item()<choice["weak_identity_dropout"]
        if weak:
            c["y"]=blob["y_weak"]; c["clip_feature"]=blob["clip_feature_weak"]
            ref=blob["first_frame_latents_weak"]
        if weak or torch.rand(()).item()<choice["expression_dropout"]:
            c["expr_codes"]=torch.zeros_like(c["expr_codes"])
        if torch.rand(()).item()<choice["proxy_dropout"]: c["graft_latents"]=None
    return c,ref


def predict_velocity(pipe,latent,sigma,c):
    # Official model_fn has no @no_grad; do not call pipe.__call__ during training.
    with torch.autocast(device_type=latent.device.type,dtype=torch.bfloat16,enabled=latent.device.type=="cuda"):
        return pipe.model_fn(dit=pipe.dit,animate_adapter=pipe.animate_adapter,
                             latents=latent,timestep=(sigma.reshape(1)*1000).to(latent.dtype),**c)


@torch.no_grad()
def sample_latents(pipe,blob,seed=1234,steps=20,guidance=1.5,ablation="none"):
    """20-step Euler. Reference slot remains clean, matching our training contract.

    Public release's full sampler leaves slot0 on its noisy trajectory. This function
    explicitly follows the paper's concatenated clean-reference interpretation.
    """
    c,ref=conditions(blob)
    if ablation=="no-expression": c["expr_codes"]=torch.zeros_like(c["expr_codes"])
    elif ablation=="no-proxy": c["graft_latents"]=None
    elif ablation=="freeze-expression": c["expr_codes"]=c["expr_codes"][:,:1].expand_as(c["expr_codes"])
    elif ablation!="none": raise ValueError(ablation)
    generator=torch.Generator(device=ref.device).manual_seed(seed)
    # shape from pose, never need target RGB at inference.
    x=torch.randn(blob["pose_latents"].shape,device=ref.device,dtype=ref.dtype,generator=generator)
    grid=torch.linspace(1,0,steps+1,device=ref.device); grid=5*grid/(1+4*grid)
    for i in range(steps):
        sigma=grid[i]
        pos=predict_velocity(pipe,torch.cat((ref,x),2),sigma,c)[:,:,1:]
        if guidance!=1:
            base={**c,"expr_codes":torch.zeros_like(c["expr_codes"]),"y":blob["y_weak"],"clip_feature":blob["clip_feature_weak"]}
            neg=predict_velocity(pipe,torch.cat((blob["first_frame_latents_weak"],x),2),sigma,base)[:,:,1:]
            pos=neg+guidance*(pos-neg)
        x=(x.float()+(grid[i+1]-sigma)*pos.float()).to(x.dtype)
    return x


@torch.no_grad()
def validate(pipe,blob,step,out,seed=1234,render=True):
    state=rng_state(); training=pipe.animate_adapter.training; pipe.animate_adapter.eval()
    try:
        c,ref=conditions(blob); x0=blob["x0"]
        gen=torch.Generator(device=x0.device).manual_seed(seed)
        noise=torch.randn(x0.shape,device=x0.device,dtype=x0.dtype,generator=gen)
        values=[]
        for value in (.25,.5,.75):
            sigma=x0.new_tensor(value); latent,target=flow_pair(x0,ref,sigma,noise)
            pred=predict_velocity(pipe,latent,sigma,c)
            values.append(float(weighted_flow_loss(pred,target,sigma,
                config()["training"]["implementation_choices"]["bell_width"])))
        metrics={"step":step,"flow_mse":sum(values)/len(values),"sigma_losses":values,"seed":seed}
        if not np.isfinite(values).all(): raise FloatingPointError("validation nonfinite")
        if render:
            latent=sample_latents(pipe,blob,seed)
            # fixed held-out target is decoded only for visualization, never generation conditioning.
            pred=pipe.vae.decode(latent,device=pipe.device,tiled=True).float().cpu()
            gt=pipe.vae.decode(x0,device=pipe.device,tiled=True).float().cpu()
            indices=[0,pred.shape[2]//2,pred.shape[2]-1]
            w,h=180,320; canvas=Image.new("RGB",(w*3,h*2+40),"#222222")
            for j,t in enumerate(indices):
                for row,v in enumerate((gt,pred)):
                    rgb=((v[0,:,t].clamp(-1,1)+1)*127.5).byte().permute(1,2,0).numpy()
                    canvas.paste(Image.fromarray(rgb).resize((w,h)),(w*j,40+h*row))
            ImageDraw.Draw(canvas).text((5,5),f"step {step}: held-out target VAE (top) / generated (bottom)",fill="white")
            dest=Path(out)/"validation"; dest.mkdir(parents=True,exist_ok=True)
            canvas.save(dest/f"{step}.jpg")
            metrics["decoded_l1"]=float((pred-gt).abs().mean()/2)
        log(Path(out)/"validation/metrics.jsonl",metrics)
        return metrics
    finally:
        pipe.animate_adapter.train(training); restore_rng(state)


def train(args):
    cfg=config(); settings=cfg["training"]; seed_all(cfg["seed"])
    cache=Path(args.cache)
    if not (cache/"index.json").is_file():
        raise FileNotFoundError(f"{cache/'index.json'} 없음. 02 cache로 stage{args.stage} 입력을 먼저 준비하세요")
    index=json.loads((cache/"index.json").read_text())
    if index["upstream"]!=COMMIT: raise ValueError("upstream mismatch")
    if index["weak_identity_blur"]!=settings["implementation_choices"]["weak_identity_blur"]:
        raise ValueError("weak reference preprocessing mismatch: cache를 다시 준비하세요")
    if index["size"]!=settings[f"stage{args.stage}_size"]: raise ValueError("stage/resolution cache mismatch")
    rows=index["records"]; training=[r for r in rows if r["split"]=="train"]; val=[r for r in rows if r["split"]=="validation"]
    if not training or not val: raise ValueError("train+held-out validation 필요")
    for row in rows:
        if row["row"]["frames"]!=77: raise ValueError("paper training: 77 supervised frames 필요")
    fingerprints={r["file"]:digest(cache/r["file"]) for r in rows}
    fingerprints["config"]=digest(ROOT/"vdm_config.json")
    fingerprints["disentangler"]=digest(args.disentangler)
    fingerprints["lesson_sources"]={p.name:digest(p) for p in ROOT.glob("*.py")}
    out=Path(args.out)
    if args.resume and args.adapter: raise ValueError("resume와 weight-only warm start를 함께 지정하지 않는다")
    if args.stage==2 and not (args.adapter or args.resume): raise ValueError("stage2는 stage1 adapter가 필요")
    if (out/"latest.pt").exists() and not args.resume: raise FileExistsError("--resume 또는 새 --out 필요")
    connect=lesson(2); pipe=connect.build_pipeline(args.disentangler,args.adapter,args.device)
    groups=connect.parameter_groups(pipe,args.mode); connect.checkpoint_blocks(pipe)
    optimizer=torch.optim.AdamW(groups,weight_decay=settings["weight_decay"],betas=tuple(settings["implementation_choices"]["adam_betas"]))
    parameters=[p for g in groups for p in g["params"]]
    for g in groups: print(f"{g['name']}: {sum(p.numel() for p in g['params']):,} parameters, lr={g['lr']:g}")
    start=0; best=float("inf")
    if args.resume:
        ck=load(args.resume)
        if ck["fingerprints"]!=fingerprints or ck["mode"]!=args.mode or ck["stage"]!=args.stage:
            raise ValueError("resume data/config/mode/stage mismatch")
        pipe.animate_adapter.load_state_dict(ck["adapter"],strict=True)
        optimizer.load_state_dict(ck["optimizer"]); start=ck["step"]; best=ck["best"]; restore_rng(ck["rng"])
    val_blob=to_device(load(cache/val[0]["file"]),args.device)
    # Inference encoders frozen; training dropout modules only in adapter.
    pipe.eval(); pipe.animate_adapter.train()
    validate(pipe,val_blob,start,out,render=not args.no_val_render)
    total=args.steps or settings[f"stage{args.stage}_steps"]
    bar=tqdm(range(start+1,total+1),desc=f"VDA stage {args.stage}")
    for step in bar:
        row=training[int(torch.randint(len(training),()))]
        blob=to_device(load(cache/row["file"]),args.device); c,ref=conditions(blob,drop=True)
        x0=blob["x0"]; sigma=torch.rand((),device=x0.device)
        noise=torch.randn_like(x0); latent,target=flow_pair(x0,ref,sigma,noise)
        optimizer.zero_grad(set_to_none=True)
        pred=predict_velocity(pipe,latent,sigma,c)
        loss=weighted_flow_loss(pred,target,sigma,settings["implementation_choices"]["bell_width"])
        if not torch.isfinite(loss): raise FloatingPointError("flow loss nonfinite; last/best checkpoint retained")
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(parameters,settings["grad_clip"],error_if_nonfinite=True)
        optimizer.step()
        metric={"step":step,"loss":float(loss),"grad_norm":float(norm),"sigma":float(sigma),"clip":row["file"]}
        log(out/"training.jsonl",metric); bar.set_postfix(loss=f"{float(loss):.5f}",grad=f"{float(norm):.3f}")
        improved=False
        if step%args.val_every==0 or step==total:
            report=validate(pipe,val_blob,step,out,render=not args.no_val_render)
            improved=report["flow_mse"]<best; best=min(best,report["flow_mse"])
            if step>=500 and report["flow_mse"]>max(10*best,1.):
                raise RuntimeError("validation collapse detected; inspect metrics/best.pt before restarting")
        if step%settings["save_every"]==0 or step==total or improved:
            ck={"adapter":pipe.animate_adapter.state_dict(),"optimizer":optimizer.state_dict(),"step":step,"best":best,
                "rng":rng_state(),"fingerprints":fingerprints,"mode":args.mode,"stage":args.stage,"upstream":COMMIT,
                "config":cfg,"protocol":"clean-reference-euler"}
            save(out/"latest.pt",ck)
            if improved: save(out/"best.pt",ck)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache",required=True); p.add_argument("--disentangler",required=True)
    p.add_argument("--out",default=str(ROOT/"checkpoints/03_stage1")); p.add_argument("--device",default="cuda")
    p.add_argument("--stage",type=int,choices=[1,2],default=1); p.add_argument("--mode",choices=["paper","release"],default="paper")
    p.add_argument("--steps",type=int); p.add_argument("--resume"); p.add_argument("--adapter")
    p.add_argument("--val-every",type=int,default=100); p.add_argument("--no-val-render",action="store_true")
    return p


def main():
    p=build_parser(); a=p.parse_args()
    if a.val_every<=0: p.error("val-every > 0")
    if a.steps is not None and a.steps<1: p.error("steps >= 1")
    train(a)


if __name__=="__main__": main()
