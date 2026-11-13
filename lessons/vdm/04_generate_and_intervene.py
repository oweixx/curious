"""04 — 학습한 Avatar로 생성하고 expression / proxy 제어를 개입 실험한다.

같은 reference와 seed를 유지하고 하나의 조건만 바꿔 원인을 관찰한다.
출력은 video이며 Gaussian/mesh asset이 아니다. 3DGS consistency probe는 별도 평가.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from common import ROOT, lesson, config, digest


@torch.no_grad()
def generate(args):
    connect=lesson(2); training=lesson(3)
    pipe=connect.build_pipeline(args.disentangler,args.adapter,args.device); pipe.eval(); pipe.requires_grad_(False)
    cond=Path(args.cond); codes=np.load(cond/"codes.npy",allow_pickle=False)
    n=min(len(codes),77); n=n-(n-1)%4
    if n<5: raise ValueError("최소5frames 필요")
    protocol=args.protocol or ("release" if str(args.adapter).endswith(".safetensors") else "paper")
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    from vda.generate import generate_full,save_outputs
    if protocol=="release":
        if args.ablation!="none": raise ValueError("release sampler는 원본 재현용. 개입 실험은 --protocol paper")
        # Translator's published fp32 weights must not take a bf16 round trip.
        for name,module in pipe.animate_adapter.named_children():
            if name!="translator": module.bfloat16()
        generate_full(pipe,str(cond/"ref.png"),str(cond/"pose_skel.mp4"),str(cond/"pose_proxy.mp4"),
                      str(cond/"codes.npy"),str(out/"release.mp4"),width=args.size[0],height=args.size[1],
                      num_frames=n+4,seed=args.seed)
    else:
        # encode_triplet's inference path never reads target video.
        blob=connect.encode_triplet(pipe,{"conditioning":str(cond),"frames":n},tuple(args.size),include_target=False)
        blob=training.to_device(blob,args.device)
        variants=["none","no-expression","freeze-expression","no-proxy"] if args.ablation=="all" else [args.ablation]
        for variant in variants:
            latent=training.sample_latents(pipe,blob,args.seed,steps=20,guidance=args.guidance,ablation=variant)
            video=pipe.vae.decode(latent,device=pipe.device,tiled=True)
            frames=pipe.vae_output_to_video(video)
            save_outputs(frames,str(out/f"{variant}.mp4"),24,False)
    (out/"run.json").write_text(json.dumps({"adapter":str(Path(args.adapter).resolve()),"adapter_sha256":digest(args.adapter),
        "conditioning":str(cond.resolve()),"protocol":protocol,"seed":args.seed,"size":args.size,"frames":n,
        "ablation":args.ablation,"note":"No explicit 3D asset; no claim of 3DGS consistency evaluation"},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cond",required=True); p.add_argument("--adapter",required=True); p.add_argument("--disentangler",required=True)
    p.add_argument("--out",default=str(ROOT/"outputs/04")); p.add_argument("--device",default="cuda")
    p.add_argument("--size",type=int,nargs=2,default=[288,512],metavar=("W","H")); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--protocol",choices=["paper","release"])
    p.add_argument("--ablation",choices=["none","no-expression","freeze-expression","no-proxy","all"],default="none")
    p.add_argument("--guidance",type=float,default=1.5)
    generate(p.parse_args())


if __name__=="__main__": main()
