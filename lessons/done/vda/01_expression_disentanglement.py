r"""01 — Appendix H의 X-NeMo disentangler를 처음부터 학습한다.

실행 위치 / 환경:
    conda activate elite
    cd /media/vilab/mh/curious/lessons/done/vda

현재 경로: 00의 042_FREE(train, frame0..76), 042_FREE_val(validation,240..316).
각 구간의 동기화된 camera 222200037/222200042/222200036을 모두 사용한다.
00 triplet의 driver code 한 벌만으로는 시점 간 표정 positive를 만들 수 없다.

1. 모든 camera의 raw code 추출 (FLAME/MHR tracking을 다시 하지 않는다):
    CUDA_VISIBLE_DEVICES=4 python 01_expression_disentanglement.py prepare \
        --train-work data/preprocessed/042_FREE \
        --validation-work data/preprocessed/042_FREE_val --out data/01
    # 이미 추출한 code는 입력/가중치 hash가 같으면 재사용한다.

2. 같은 사람의 비중첩 시간 구간으로 pack하고 검사:
    python 01_expression_disentanglement.py pack \
        --manifest data/01/codes.json --out data/01/codes.npz \
        --protocol single-subject
    python 01_expression_disentanglement.py inspect \
        --codes data/01/codes.npz --protocol single-subject
    # codes.npz가 이미 있으면 pack을 반복하지 않고 inspect/train으로 진행.

3. 공식 구조를 random initialization부터 학습 (기본 1000 epochs):
    CUDA_VISIBLE_DEVICES=4 python 01_expression_disentanglement.py train \
        --codes data/01/codes.npz --protocol single-subject --out checkpoints/01

4. 중단된 학습 재개 (epochs는 추가 횟수가 아니라 누적 목표):
    CUDA_VISIBLE_DEVICES=4 python 01_expression_disentanglement.py train \
        --codes data/01/codes.npz --protocol single-subject --out checkpoints/01 \
        --resume checkpoints/01/latest.pt

출력: checkpoints/01/{best.pt,latest.pt,validation.jsonl}.
5. 학습 결과 시각화 (재학습 없이 CPU에서 실행):
    python 01_expression_disentanglement.py visualize \
        --codes data/01/codes.npz --checkpoint checkpoints/01/best.pt \
        --manifest data/01/codes.json --work data/preprocessed/042_FREE_val \
        --out outputs/01
    # learning_curves.png / frame_matching.png / retrieval.jpg / summary.json
    # 이미지들은 실제 관측 RGB의 검색 비교다. 이 network는 RGB/Avatar를 생성하지 않는다.

다음: 02 audit/03/04 모두 --disentangler checkpoints/01/best.pt로 연결한다.
single-subject는 모델과 loss를 유지하면서 expression batch=32x3, camera=3x16으로
현재 관측 수에 맞춘다. 검증은 같은 사람의 다른 시간이며 unseen-identity 성능이 아니다.
여러 사람/충분한 camera를 확보하면 별도 manifest를 pack/train --protocol paper로
실행한다. paper는 identity 분리와 원래 batch(32x4,8x16)를 요구한다.
도움말: python 01_expression_disentanglement.py train --help

읽는 순서: prepare/pack -> supervised_contrastive -> cycle_loss -> objective -> train.
공식 XNemoDisentangler의 MLP를 그대로 사용하되, @no_grad 추론 메서드를
학습에 호출하지 않는다. expression/camera encoder와 decoder를 직접 호출한다.
512->512 expression, 512->32 camera, concat->512 reconstruction.
"""
import argparse
import hashlib
import json
import math
import os
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


def validate_splits(identities, sequences, frames, cameras, splits, protocol):
    """paper: unseen identity. single-subject: one identity, disjoint time intervals.

    카메라가 달라도 동일 시간의 표정이 두 split에 들어가면 시간 누출이다.
    sample key에는 camera를, split 검사 key에는 camera를 넣지 않는다.
    """
    if protocol not in ("paper", "single-subject"):
        raise ValueError(f"unknown protocol: {protocol}")
    if set(splits) != {"train", "validation"}:
        raise ValueError("train + validation 구간이 모두 필요합니다")
    identity_splits, time_splits, samples = {}, {}, set()
    for identity, sequence, frame, camera, split in zip(identities,sequences,frames,cameras,splits):
        key = (str(identity),str(sequence),int(frame))
        sample = (*key,str(camera))
        if sample in samples: raise ValueError(f"중복 camera/frame: {sample}")
        samples.add(sample)
        identity_splits.setdefault(str(identity),set()).add(str(split))
        time_splits.setdefault(key,set()).add(str(split))
    if any(len(s)>1 for s in time_splits.values()):
        raise ValueError("train/validation 시간 구간이 겹칩니다 (다른 camera도 같은 시간으로 검사)")
    if protocol == "paper" and any(len(s)>1 for s in identity_splits.values()):
        raise ValueError("paper protocol은 identity 분리 필요; 042 실습은 --protocol single-subject")
    if protocol == "single-subject" and len(identity_splits)!=1:
        raise ValueError("single-subject protocol에는 한 사람의 데이터만 사용하세요")


def prepare(args):
    """00의 동기화 RGB로 각 camera의 raw X-NeMo code를 추출한다. FLAME 재실행 없음."""
    from PIL import Image
    from preprocess import run_info, image_files, write_json
    upstream()
    from vda.preprocess.xnemo_codes import XNemoCodeExtractor
    out=Path(args.out).resolve(); rows=[]; extractor=None
    works=[(Path(args.train_work).resolve(),"train"),
           (Path(args.validation_work).resolve(),"validation")]
    # GPU에 올리기 전에 split과 timeline부터 검사한다.
    sample_info={k:[] for k in ("identities","sequences","frames","cameras","splits")}
    for work,split in works:
        info=run_info(work)
        if info["split"]!=split or info["fps"]!=24:
            raise ValueError(f"{work}: {split}의 24fps preprocessing work 필요")
        for camera in info["cameras"]:
            for t in range(info["frames"]):
                for key,value in zip(sample_info,(info["identity"],info["sequence"],info["start"]+t,camera,split)):
                    sample_info[key].append(value)
    validate_splits(**sample_info,protocol="single-subject")
    weights=ROOT/"weights/third_party/xnemo_motion_encoder.pth"
    weights_hash=digest(weights)
    for work,split in works:
        info=run_info(work)
        for camera in info["cameras"]:
            files=image_files(work,camera)
            if len(files)!=info["frames"]: raise ValueError(f"{work}/{camera}: frame 수 불일치")
            frame_hash=hashlib.sha256()
            for f in files: frame_hash.update(bytes.fromhex(digest(f)))
            signature={"run_sha256":digest(work/"run.json"),"frames_sha256":frame_hash.hexdigest(),
                       "weights_sha256":weights_hash,"upstream":config()["upstream_commit"]}
            name=f"{info['identity']}_{info['sequence'].lower()}_{info['start']:06d}_cam{camera}"
            dest=out/"codes"/f"{name}.npy"; metadata=dest.with_suffix(".json")
            if dest.exists():
                if not metadata.exists(): raise ValueError(f"출처 없는 기존 code: {dest}; 새 --out 사용")
                recorded=json.loads(metadata.read_text())
                if recorded["source"]!=signature or recorded["codes_sha256"]!=digest(dest):
                    raise ValueError(f"code cache의 입력/가중치 변경: {dest}; 새 --out 사용")
                codes=np.load(dest,allow_pickle=False)
                print(f"재사용: {dest}")
            else:
                if extractor is None: extractor=XNemoCodeExtractor(device=args.device)
                fallback_before=extractor.n_fallback
                codes=np.stack([extractor.encode(np.asarray(Image.open(f).convert("RGB")))
                                for f in tqdm(files,desc=f"{split} / {camera}")])
                if extractor.n_fallback!=fallback_before:
                    raise ValueError(f"{camera}: face 검출 실패. fallback code를 학습용으로 저장하지 않습니다")
                if codes.shape!=(info["frames"],512) or not np.isfinite(codes).all():
                    raise ValueError(f"{camera}: finite raw [frames,512] code 필요")
                dest.parent.mkdir(parents=True,exist_ok=True)
                temporary=dest.with_suffix(".partial.npy"); np.save(temporary,codes); temporary.replace(dest)
                write_json(metadata,{"source":signature,"codes_sha256":digest(dest)})
            if codes.shape!=(info["frames"],512) or not np.isfinite(codes).all():
                raise ValueError(f"{dest}: 잘못된 code cache")
            rows.append(dict(codes=os.path.relpath(dest,out),identity=info["identity"],
                             sequence=info["sequence"],camera=camera,start=info["start"],split=split))
    write_json(out/"codes.json",rows)
    print(f"{len(rows)} camera clips -> {out/'codes.json'}; 다음은 pack --protocol single-subject")


def pack(manifest, output, pairs=None, protocol="paper"):
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
    zs=[]; expressions=[]; cameras=[]; identities=[]; splits=[]; sequences=[]; frames=[]
    for row in rows:
        z=np.load(path.parent/row["codes"],allow_pickle=False)
        if z.ndim!=2 or z.shape[1]!=512 or len(z)==0 or not np.isfinite(z).all(): raise ValueError("RAW X-NeMo [N,512] 필요")
        if int(row.get("start",0))<0: raise ValueError("start >= 0 필요")
        zs.append(z)
        for t in range(len(z)):
            expressions.append(find((str(row["identity"]),str(row["sequence"]),int(row.get("start",0))+t)))
            cameras.append(str(row["camera"])); identities.append(str(row["identity"])); splits.append(row["split"])
            sequences.append(str(row["sequence"])); frames.append(int(row.get("start",0))+t)
    validate_splits(identities,sequences,frames,cameras,splits,protocol)
    expression_splits={}
    for e,s in zip(expressions,splits):
        expression_splits.setdefault(e,set()).add(s)
    mapping={v:i for i,v in enumerate(sorted(set(expressions)))}
    # Verified cross-id pairs cannot join training and validation classes.
    if any(len(s)>1 for s in expression_splits.values()): raise ValueError("expression class split leakage")
    cm={v:i for i,v in enumerate(sorted(set(cameras)))}
    arrays=dict(codes=np.concatenate(zs).astype(np.float32),expression=np.array([mapping[e] for e in expressions]),
                camera=np.array([cm[c] for c in cameras]),identity=np.array(identities),split=np.array(splits),
                sequence=np.array(sequences),frame=np.array(frames),protocol=np.array(protocol))
    settings,report=dataset_settings(arrays,protocol)
    output=Path(output)
    if output.exists(): raise FileExistsError(f"기존 packed data를 보존합니다: {output}; 새 --out 사용")
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(".partial.npz"); np.savez(temporary,**arrays); temporary.replace(output)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    print(f"packed -> {output}")


def dataset_settings(data, protocol):
    """모델/loss는 고정. 한 사람 실습에서만 실제 view 수에 맞게 batch를 구성한다."""
    saved=str(data["protocol"]) if "protocol" in data else "paper"
    if saved!=protocol: raise ValueError(f"packed protocol={saved}; --protocol {saved} 사용")
    if "sequence" in data and "frame" in data:
        validate_splits(data["identity"],data["sequence"],data["frame"],data["camera"],data["split"],protocol)
    elif protocol!="paper":
        raise ValueError("시간 정보가 없습니다. pack을 다시 실행하세요")
    if set(data["split"])!={"train","validation"}: raise ValueError("train + validation 필요")
    ti=data["split"]=="train"; vi=~ti
    if protocol=="paper" and set(data["identity"][ti]) & set(data["identity"][vi]):
        raise ValueError("identity leakage")
    cfg=config()["disentangler"]
    if protocol=="single-subject":
        views=len(np.unique(data["camera"][ti]))
        _,counts=np.unique(data["expression"][ti],return_counts=True)
        if views<2 or counts.min()<2: raise ValueError("표정마다 서로 다른 실제 camera가 2개 이상 필요")
        if set(data["camera"][ti])!=set(data["camera"][vi]):
            raise ValueError("single-subject validation은 train과 같은 camera들의 다른 시간 구간을 사용")
        for split_mask in (ti,vi):
            for expr in np.unique(data["expression"][split_mask]):
                cam=data["camera"][split_mask & (data["expression"]==expr)]
                if len(cam)!=views or len(np.unique(cam))!=views:
                    raise ValueError("single-subject의 각 표정 frame은 모든 camera에서 한 번씩 관측돼야 합니다")
        cfg["expression_batch"]=[min(32,len(counts)),min(4,int(counts.min()))]
        _,camera_counts=np.unique(data["camera"][ti],return_counts=True)
        cfg["camera_batch"]=[min(8,views),min(16,int(camera_counts.min()))]
        if min(*cfg["expression_batch"],*cfg["camera_batch"])<2:
            raise ValueError("contrastive 학습은 class/sample이 각각 2개 이상 필요")
    for name in ("expression","camera"):
        BalancedSampler(torch.from_numpy(data[name][ti]),*cfg[f"{name}_batch"])
    report=dict(protocol=protocol,train_samples=int(ti.sum()),validation_samples=int(vi.sum()),
                cameras=len(np.unique(data["camera"][ti])),expression_batch=cfg["expression_batch"],
                camera_batch=cfg["camera_batch"],validation="unseen identity" if protocol=="paper" else "same identity, disjoint time")
    return cfg,report


def train(args):
    seed_all(config()["seed"])
    d=np.load(args.codes,allow_pickle=False)
    protocol=getattr(args,"protocol","paper")
    cfg,report=dataset_settings(d,protocol)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    ti=d["split"]=="train"; vi=~ti
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
        if ck["cfg"]!=cfg or ck["codes_sha256"]!=fingerprint or ck.get("protocol","paper")!=protocol:
            raise ValueError("resume config/data/protocol mismatch")
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
            "epoch":epoch+1,"best":best,"rng":rng_state(),"codes_sha256":fingerprint,"protocol":protocol,
            "dataset":report}
        save(out/"latest.pt",ck)
        if improved: save(out/"best.pt",ck)
        log(out/"validation.jsonl",{"epoch":epoch+1,**metrics})


@torch.no_grad()
def visualize(args):
    """학습 곡선 + 다른 시점에서 동일 시간의 표정을 찾는지 관찰한다.

    학습 때의 retrieval은 자신만 제외한 전체 gallery, 아래 RGB 비교는 같은
    camera 전체를 제외한 gallery다. 두 지표를 별도로 기록하고 혼동하지 않는다.
    정확히 같은 frame을 정답으로 세므로 인접 frame의 유사한 표정도 오답이다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
    checkpoint=load(args.checkpoint)
    if checkpoint["codes_sha256"]!=digest(args.codes):
        raise ValueError("checkpoint 학습에 사용한 codes.npz를 지정하세요")
    with np.load(args.codes,allow_pickle=False) as loaded:
        data={k:loaded[k] for k in loaded.files}
    if "frame" not in data: raise ValueError("시각화에는 시간 정보를 포함한 pack이 필요합니다")
    work=Path(args.work); info=json.loads((work/"run.json").read_text())
    manifest=Path(args.manifest); rows=json.loads(manifest.read_text())
    names=sorted({str(row["camera"]) for row in rows})
    # manifest와 pack 순서/실제 code를 대조하여 다른 RGB를 잘못 붙이지 않는다.
    cursor=0
    for row in rows:
        raw=np.load(manifest.parent/row["codes"],allow_pickle=False); n=len(raw)
        sl=slice(cursor,cursor+n)
        matches=(np.array_equal(raw,data["codes"][sl])
                 and np.all(data["camera"][sl]==names.index(str(row["camera"]))))
        if not matches: raise ValueError("manifest와 packed code의 순서/내용 불일치")
        cursor+=n
    if cursor!=len(data["codes"]): raise ValueError("manifest/pack 길이 불일치")
    keep=((data["identity"]==info["identity"]) & (data["sequence"]==info["sequence"])
          & (data["split"]==info["split"]) & (data["frame"]>=info["start"])
          & (data["frame"]<info["start"]+info["frames"]))
    frames=data["frame"][keep]; cameras=data["camera"][keep]; labels=data["expression"][keep]
    if len(frames)<2: raise ValueError("work에 대응하는 packed samples가 없습니다")
    model=make_model().cpu().eval(); model.load_state_dict(checkpoint["model_state"],strict=True)
    z=torch.from_numpy(data["codes"][keep]); encoded=model.expr_encoder(z)
    similarity={name:(F.normalize(x,dim=-1)@F.normalize(x,dim=-1).T).numpy()
                for name,x in (("raw",z),("expression",encoded))}
    other=cameras[:,None]!=cameras[None,:]
    if not other.any(1).all(): raise ValueError("각 query에 다른 camera의 gallery가 필요합니다")
    nearest={name:np.where(other,sim,-np.inf).argmax(1) for name,sim in similarity.items()}
    cross={name:dict(exact_frame_r1=float(np.mean(labels[idx]==labels)),
                     median_frame_error=float(np.median(np.abs(frames[idx]-frames))))
           for name,idx in nearest.items()}
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    log_path=Path(args.checkpoint).parent/"validation.jsonl"
    if log_path.is_file():
        history=[json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        epochs=[r["epoch"] for r in history]
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout="constrained")
        axes[0].plot(epochs,[r["reconstruction"] for r in history],color="tab:blue")
        axes[0].set(title="Validation: raw-code reconstruction",ylabel="L1 in 512D code space (lower is better)")
        for name,color in (("raw","gray"),("expression","tab:orange")):
            axes[1].plot(epochs,[100*r[f"{name}_retrieval_r1"] if r[f"{name}_retrieval_r1"] is not None
                                else np.nan for r in history],label=name,color=color)
        axes[1].set(title="Logged retrieval: all views, self excluded",ylabel="Exact same-frame top-1 (%)")
        axes[1].legend()
        for ax in axes:
            ax.axvline(checkpoint["epoch"],color="tab:red",linestyle="--",alpha=.7)
            ax.set_xlabel("Epoch"); ax.grid(alpha=.2)
        fig.suptitle(f"{info['identity']} | dashed line: selected checkpoint epoch {checkpoint['epoch']}")
        fig.savefig(out/"learning_curves.png",dpi=150); plt.close(fig)
    query_camera=names.index(info["driver_camera"]); gallery_camera=names.index(info["reference_camera"])
    query=np.flatnonzero(cameras==query_camera); gallery=np.flatnonzero(cameras==gallery_camera)
    query=query[np.argsort(frames[query])]; gallery=gallery[np.argsort(frames[gallery])]
    if not len(query) or not np.array_equal(frames[query],frames[gallery]):
        raise ValueError("driver/reference의 동기화 frame들이 필요합니다")
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout="constrained")
    for ax,(name,sim) in zip(axes,similarity.items()):
        values=sim[np.ix_(query,gallery)]
        im=ax.imshow(values,origin="upper",aspect="equal",cmap="viridis")
        ax.plot(np.arange(len(query)),np.arange(len(query)),"w--",linewidth=.8)
        ax.scatter(values.argmax(1),np.arange(len(query)),s=5,c="red",label="Top-1 match")
        ax.set(title=name,xlabel="Gallery frame (local index)",ylabel="Query frame (local index)")
        fig.colorbar(im,ax=ax,label="Cosine similarity (panel-specific scale)")
    fig.suptitle(f"{info['driver_camera']} -> {info['reference_camera']} | same time = diagonal; top-1 = red")
    fig.savefig(out/"frame_matching.png",dpi=150);plt.close(fig)
    # 시간 구간에서 균등하게 선택한 frame을 보여주며 성공 여부로 선별하지 않는다.
    selected=query[np.linspace(0,len(query)-1,min(5,len(query))).round().astype(int)]
    cell_w,cell_h=230,360; canvas=Image.new("RGB",(4*cell_w,80+len(selected)*cell_h),"#181c22")
    draw=ImageDraw.Draw(canvas)
    try: font=ImageFont.truetype("DejaVuSans.ttf",15)
    except OSError: font=ImageFont.load_default()
    draw.text((10,8),f"Observed RGB retrieval | epoch {checkpoint['epoch']} | other-camera gallery",font=font,fill="white")
    draw.text((10,30),"Green: exact same frame. Orange: different frame. These are observed photos, not generated images.",font=font,fill="white")
    for col,title in enumerate(("Query","Same-time reference","Raw top-1","Trained top-1")):
        draw.text((col*cell_w+6,57),title,font=font,fill="white")
    for row,q in enumerate(selected):
        correct=gallery[frames[gallery]==frames[q]][0]
        indices=(q,correct,nearest["raw"][q],nearest["expression"][q])
        for col,index in enumerate(indices):
            camera=names[cameras[index]]; local=int(frames[index]-info["start"])
            path=work/"frames"/camera/f"{local:05d}.jpg"
            with Image.open(path) as source:
                im=source.convert("RGB");im.thumbnail((cell_w-10,cell_h-45))
            x=col*cell_w;y=80+row*cell_h
            canvas.paste(im,(x+(cell_w-im.width)//2,y+38))
            color="#5ed18d" if frames[index]==frames[q] else "#ffb056"
            draw.rectangle((x+2,y+2,x+cell_w-3,y+cell_h-3),outline=color,width=2)
            draw.text((x+7,y+5),f"cam {camera}",font=font,fill="white")
            draw.text((x+7,y+21),f"frame {frames[index]} | delta {frames[index]-frames[q]:+d}",font=font,fill=color)
    canvas.save(out/"retrieval.jpg",quality=94)
    report=dict(checkpoint=str(Path(args.checkpoint).resolve()),epoch=checkpoint["epoch"],
                split=info["split"],samples=len(frames),all_view_retrieval=evaluate(model,z,torch.from_numpy(labels)),
                other_camera_retrieval=cross,query_camera=info["driver_camera"],
                notes=["Reconstruction is latent-code L1, not RGB loss.",
                       "Exact frame retrieval treats adjacent frames as incorrect, even for similar expressions.",
                       "Same-subject time holdout does not test unseen identities."])
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2));print(f"visualization -> {out}")


def build_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    s=p.add_subparsers(dest="stage",required=True)
    a=s.add_parser("prepare",help="동기화된 preprocessing work의 모든 camera에서 raw codes 추출")
    a.add_argument("--train-work",default=str(ROOT/"data/preprocessed/042_FREE"))
    a.add_argument("--validation-work",default=str(ROOT/"data/preprocessed/042_FREE_val"))
    a.add_argument("--out",default=str(ROOT/"data/01")); a.add_argument("--device",default="cuda")
    a=s.add_parser("pack"); a.add_argument("--manifest",required=True); a.add_argument("--out",required=True); a.add_argument("--verified-pairs")
    a.add_argument("--protocol",choices=["paper","single-subject"],default="paper")
    a=s.add_parser("inspect",help="모델 로딩 없이 dataset/split/batch 검사")
    a.add_argument("--codes",required=True); a.add_argument("--protocol",choices=["paper","single-subject"],default="paper")
    a=s.add_parser("train"); a.add_argument("--codes",required=True); a.add_argument("--out",default=str(ROOT/"checkpoints/01"))
    a.add_argument("--device",default="cuda"); a.add_argument("--epochs",type=int); a.add_argument("--resume")
    a.add_argument("--protocol",choices=["paper","single-subject"],default="paper")
    a=s.add_parser("visualize",help="저장된 checkpoint와 validation log로 결과 시각화 (재학습 없음)")
    a.add_argument("--codes",required=True); a.add_argument("--checkpoint",required=True)
    a.add_argument("--manifest",default=str(ROOT/"data/01/codes.json"))
    a.add_argument("--work",default=str(ROOT/"data/preprocessed/042_FREE_val"))
    a.add_argument("--out",default=str(ROOT/"outputs/01"))
    return p


def main():
    p=build_parser(); a=p.parse_args()
    if a.stage=="prepare": prepare(a)
    elif a.stage=="pack": pack(a.manifest,a.out,a.verified_pairs,a.protocol)
    elif a.stage=="visualize": visualize(a)
    elif a.stage=="inspect":
        with np.load(a.codes,allow_pickle=False) as data:
            _,report=dataset_settings(data,a.protocol)
        print(json.dumps(report,indent=2,ensure_ascii=False))
    else:
        if a.epochs is not None and a.epochs<1: p.error("epochs >= 1")
        train(a)


if __name__=="__main__": main()
