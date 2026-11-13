"""01 — Appendix H의 X-NeMo disentangler를 처음부터 학습한다.

읽는 순서: supervised_contrastive -> cycle_loss -> balanced_indices -> train.
공식 XNemoDisentangler의 MLP를 그대로 사용하되, @no_grad 추론 메서드를
학습에 호출하지 않는다. expression/camera encoder와 decoder를 직접 호출한다.
512->512 expression, 512->32 camera, concat->512 reconstruction.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from common import ROOT, config, upstream, seed_all, load, save, log, rng_state, restore_rng, digest


def make_model():
    upstream()
    from vda.disentangler import XNemoDisentangler
    return XNemoDisentangler(config()["disentangler"])


def supervised_contrastive(z, labels, temperature=.07):
    """Eq.14: 자기 자신 제외, positive별 log-prob 평균, positive 없는 anchor 제외.

    normalize는 이 loss 내부에서만 한다. decoder에 넘기는 latent norm은 보존.
    """
    z = F.normalize(z.float(), dim=-1)
    logits = z @ z.T / temperature
    diagonal = torch.eye(len(z),device=z.device,dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    valid = positive.any(1)
    if not valid.any():
        raise ValueError("contrastive batch에 positive pair가 없다")
    logits = logits.masked_fill(diagonal, -torch.inf)
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(logp.masked_fill(~positive, 0).sum(1) / positive.sum(1).clamp_min(1))[valid].mean()


def cycle_loss(model, z):
    """Eq.18의 deployed ex+cam 구성. 양쪽 factor에도 gradient를 전달한다.

    같은 a,b pair에 대해 두 swap 방향을 명시적으로 계산한다 (Eq.18).
    """
    ex, cam = model.expr_encoder(z), model.cam_encoder(z)
    perm = torch.randperm(len(z), device=z.device)
    swapped = model.decoder(torch.cat((ex, cam[perm]), -1))
    reverse = model.decoder(torch.cat((ex[perm], cam), -1))
    return .25 * (F.l1_loss(model.expr_encoder(swapped), ex)
                  + F.l1_loss(model.cam_encoder(swapped), cam[perm])
                  + F.l1_loss(model.expr_encoder(reverse), ex[perm])
                  + F.l1_loss(model.cam_encoder(reverse), cam))


class BalancedSampler:
    """class index를 한 번 만들어 매 step 전체 label scan을 피한다."""
    def __init__(self, labels, classes, samples):
        values=labels.cpu().numpy()
        order=np.argsort(values,kind="stable")
        boundaries=np.flatnonzero(np.diff(values[order]))+1
        self.groups=[torch.from_numpy(g.copy()) for g in np.split(order,boundaries) if len(g)>=samples]
        self.classes,self.samples=classes,samples
        if len(self.groups)<classes:
            raise ValueError(f"{classes} classes x {samples} samples 필요; 가능한 class={len(self.groups)}")

    def draw(self):
        choice=torch.randperm(len(self.groups))[:self.classes]
        return torch.cat([self.groups[i][torch.randperm(len(self.groups[i]))[:self.samples]] for i in choice])


def balanced_indices(labels, classes, samples):
    return BalancedSampler(labels,classes,samples).draw()


def objective(model, z, ex_labels, cam_labels, epoch, cfg, samplers=None):
    device = next(model.parameters()).device
    rec_idx = torch.randperm(len(z))[:cfg["reconstruction_batch"]]
    rec = z[rec_idx].to(device)
    e, c = model.expr_encoder(rec), model.cam_encoder(rec)
    terms = {"reconstruction": F.l1_loss(model.decoder(torch.cat((e,c),-1)),rec)}
    for name, labels, batch, head in (("expression",ex_labels,cfg["expression_batch"],model.expr_encoder),
                                      ("camera",cam_labels,cfg["camera_batch"],model.cam_encoder)):
        idx = samplers[name].draw() if samplers else balanced_indices(labels,*batch)
        terms[name] = supervised_contrastive(head(z[idx].to(device)),labels[idx].to(device),cfg["temperature"])
    terms["cycle"] = cycle_loss(model,rec) if epoch >= cfg["cycle_start_epoch"] else rec.new_zeros(())
    # Eq.19 has one cycle multiplier. Released checkpoint's cycle_contrastive_*
    # weights describe separate contrastive terms, not a second L1 multiplier.
    total = terms["reconstruction"] + terms["expression"] + terms["camera"]
    total = total + cfg["cycle_weight"] * terms["cycle"]
    return total, terms


@torch.no_grad()
def evaluate(model,z,labels):
    """Held-out reconstruction + same-frame retrieval proxy. 공식 DISTRACT는 아니다."""
    model.eval(); device=next(model.parameters()).device
    chunks=[]; errors=[]
    for x in z.split(256):
        x=x.to(device); e=model.expr_encoder(x); c=model.cam_encoder(x)
        errors.append((model.decoder(torch.cat((e,c),-1))-x).abs().sum().item())
        chunks.append(e.cpu())
    e=torch.cat(chunks)
    result={"reconstruction":sum(errors)/z.numel()}
    # chunked query avoids N x N allocation for a large code cache.
    # Full reconstruction validation; bounded, deterministic retrieval subset of complete classes.
    selected=labels.unique()[:256]
    indices=torch.where(torch.isin(labels,selected))[0][:4096]
    labels=labels[indices]
    result["retrieval_samples"]=len(indices)
    for name,features in (("raw",z[indices]),("expression",e[indices])):
        f=F.normalize(features,dim=-1); correct=0; total=0
        for start in range(0,len(f),256):
            sim=f[start:start+256]@f.T
            idx=torch.arange(start,start+len(sim))
            sim[torch.arange(len(sim)),idx]=-torch.inf
            positive=labels[idx,None].eq(labels[None,:]); positive[torch.arange(len(sim)),idx]=False
            valid=positive.any(1)
            correct+=((labels[sim.argmax(1)]==labels[idx])&valid).sum().item(); total+=valid.sum().item()
        result[f"{name}_retrieval_r1"]=correct/total if total else None
    return result


def pack(manifest, output, pairs=None):
    """JSON list: codes .npy, identity, sequence, camera, start, split.

    Optional verified_pairs JSON: [[[identity,sequence,frame],[...]],...].
    Human-verified cross-identity expression classes are joined transitively.
    """
    path=Path(manifest).resolve(); rows=json.loads(path.read_text())
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x: parent[x]=find(parent[x])
        return parent[x]
    if pairs:
        for a,b in json.loads(Path(pairs).read_text()):
            parent[find(tuple(a))]=find(tuple(b))
    zs=[]; expressions=[]; cameras=[]; identities=[]; splits=[]
    for row in rows:
        z=np.load(path.parent/row["codes"],allow_pickle=False)
        if z.ndim!=2 or z.shape[1]!=512 or not np.isfinite(z).all(): raise ValueError("RAW X-NeMo [N,512] 필요")
        zs.append(z)
        for t in range(len(z)):
            expressions.append(find((str(row["identity"]),str(row["sequence"]),int(row.get("start",0))+t)))
            cameras.append(str(row["camera"])); identities.append(str(row["identity"])); splits.append(row["split"])
    identity_splits={}; expression_splits={}
    for ident,e,s in zip(identities,expressions,splits):
        if s not in ("train","validation"): raise ValueError("unknown split")
        identity_splits.setdefault(ident,set()).add(s)
        expression_splits.setdefault(e,set()).add(s)
    for seen in identity_splits.values():
        if len(seen)!=1:
            raise ValueError("disentangler validation은 identity 단위로 분리해야 한다")
    mapping={v:i for i,v in enumerate(sorted(set(expressions)))}
    # Verified cross-id pairs cannot join training and validation classes.
    if any(len(s)>1 for s in expression_splits.values()): raise ValueError("expression class split leakage")
    cm={v:i for i,v in enumerate(sorted(set(cameras)))}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    np.savez(output,codes=np.concatenate(zs).astype(np.float32),expression=np.array([mapping[e] for e in expressions]),
             camera=np.array([cm[c] for c in cameras]),identity=np.array(identities),split=np.array(splits))


def train(args):
    cfg=config()["disentangler"]; seed_all(config()["seed"])
    d=np.load(args.codes,allow_pickle=False)
    if set(d["split"])!={"train","validation"}: raise ValueError("train+validation 필요")
    ti=d["split"]=="train"; vi=~ti
    if set(d["identity"][ti]) & set(d["identity"][vi]): raise ValueError("identity leakage")
    z=torch.from_numpy(d["codes"][ti]); ex=torch.from_numpy(d["expression"][ti]); cam=torch.from_numpy(d["camera"][ti])
    samplers={"expression":BalancedSampler(ex,*cfg["expression_batch"]),
              "camera":BalancedSampler(cam,*cfg["camera_batch"])}
    vz=torch.from_numpy(d["codes"][vi]); vl=torch.from_numpy(d["expression"][vi])
    model=make_model().to(args.device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["lr"],weight_decay=cfg["weight_decay"])
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=200,gamma=.5)
    out=Path(args.out); start=0; best=float("inf"); fingerprint=digest(args.codes)
    if args.resume:
        ck=load(args.resume)
        if ck["cfg"]!=cfg or ck["codes_sha256"]!=fingerprint: raise ValueError("resume config/data mismatch")
        model.load_state_dict(ck["model_state"]); opt.load_state_dict(ck["optimizer"]); scheduler.load_state_dict(ck["scheduler"])
        start=ck["epoch"]; best=ck["best"]; restore_rng(ck["rng"])
    elif (out/"latest.pt").exists(): raise FileExistsError("--resume 또는 새로운 --out 사용")
    steps=math.ceil(len(z)/cfg["reconstruction_batch"])
    for epoch in range(start,args.epochs or cfg["epochs"]):
        model.train(); bar=tqdm(range(steps),desc=f"disentangle epoch {epoch+1}")
        for _ in bar:
            opt.zero_grad(set_to_none=True)
            loss,terms=objective(model,z,ex,cam,epoch,cfg,samplers)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite disentangler loss")
            loss.backward()
            # No extra clipping in paper recipe; reject invalid gradients instead.
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError("gradient")
            opt.step(); bar.set_postfix(loss=f"{loss.item():.4f}",rec=f"{terms['reconstruction'].item():.4f}")
        scheduler.step(); metrics=evaluate(model,vz,vl); improved=metrics["reconstruction"]<best
        best=min(best,metrics["reconstruction"])
        ck={"cfg":cfg,"model_state":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":scheduler.state_dict(),
            "epoch":epoch+1,"best":best,"rng":rng_state(),"codes_sha256":fingerprint}
        save(out/"latest.pt",ck)
        if improved: save(out/"best.pt",ck)
        log(out/"validation.jsonl",{"epoch":epoch+1,**metrics})


def main():
    p=argparse.ArgumentParser(description=__doc__); s=p.add_subparsers(dest="stage",required=True)
    a=s.add_parser("pack"); a.add_argument("--manifest",required=True); a.add_argument("--out",required=True); a.add_argument("--verified-pairs")
    a=s.add_parser("train"); a.add_argument("--codes",required=True); a.add_argument("--out",default=str(ROOT/"checkpoints/01"))
    a.add_argument("--device",default="cuda"); a.add_argument("--epochs",type=int); a.add_argument("--resume")
    a=p.parse_args()
    if a.stage=="pack": pack(a.manifest,a.out,a.verified_pairs)
    else: train(a)


if __name__=="__main__": main()
