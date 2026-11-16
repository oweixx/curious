r"""04 — 학습한 Avatar로 생성하고 expression / proxy 제어를 개입 실험한다.

실행 위치 / 환경:
    conda activate elite
    cd /media/vilab/mh/curious/lessons/done/vda

현재 실습 경로: 01에서 학습한 disentangler + 03에서 학습한 adapter.
00의 validation 조건(data/triplets/042_free_val)을 그대로 사용한다.
이 파일은 학습을 하지 않으며, 아래 명령은 해당 03 학습을 마친 뒤 실행한다.

1. Stage1 학습 결과부터 확인 (Stage2가 아직 없어도 실행 가능):
    CUDA_VISIBLE_DEVICES=4 python 04_generate_and_intervene.py \
        --cond data/triplets/042_free_val --adapter checkpoints/03_stage1/best.pt \
        --disentangler checkpoints/01/best.pt \
        --protocol paper --size 288 512 --out outputs/04_stage1

2. Stage2 학습 결과 확인:
    CUDA_VISIBLE_DEVICES=4 python 04_generate_and_intervene.py \
        --cond data/triplets/042_free_val --adapter checkpoints/03_stage2/best.pt \
        --disentangler checkpoints/01/best.pt \
        --protocol paper --size 576 1024 --out outputs/04_stage2

3. 같은 reference/seed로 expression/proxy 조건을 바꿔 4개 영상 비교:
    CUDA_VISIBLE_DEVICES=4 python 04_generate_and_intervene.py \
        --cond data/triplets/042_free_val --adapter checkpoints/03_stage2/best.pt \
        --disentangler checkpoints/01/best.pt \
        --protocol paper --size 576 1024 --seed 42 \
        --ablation all --out outputs/04_ablation

별도 참고: 저자 공개 결과 확인 (우리 01/03 학습 결과와 구분):
    CUDA_VISIBLE_DEVICES=4 python 04_generate_and_intervene.py \
        --cond .cache/upstream/examples/teaser_swimmer \
        --adapter weights/vda_full.safetensors \
        --disentangler weights/xnemo_disentangler.pt \
        --protocol release --size 288 512 --out outputs/04_public

--cond 필수 파일: ref.png, codes.npy, pose_proxy.mp4, pose_skel.mp4.
Target RGB는 추론에 필요하지 않다. --disentangler는 03과 동일한 파일을 사용한다.
--size는 width height 순서다. 01의 single-subject와 여기 sampler의 paper는
서로 다른 설정이다. paper sampler 선택이 unseen-identity 검증을 뜻하지 않는다.
--protocol release는 공개 sampler이며 ablation을 지원하지 않는다.
--protocol paper는 lesson의 clean-reference sampler이며 20-step Euler를 쓴다.
출력: release.mp4 또는 none.mp4, --ablation all이면 추가로
no-expression.mp4/freeze-expression.mp4/no-proxy.mp4, 실행 정보 run.json.
새 결과를 보존하려면 --out을 구분한다. 도움말:
    python 04_generate_and_intervene.py --help

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
    cond=Path(args.cond)
    missing=[str(cond/name) for name in ("ref.png","codes.npy","pose_proxy.mp4","pose_skel.mp4")
             if not (cond/name).is_file()]
    if missing:
        raise FileNotFoundError("00 build-triplet의 conditioning 파일이 필요합니다: "+", ".join(missing))
    codes=np.load(cond/"codes.npy",allow_pickle=False)
    if codes.ndim!=2 or codes.shape[1]!=512 or not np.isfinite(codes).all():
        raise ValueError("codes.npy는 finite raw [T,512] X-NeMo code여야 합니다")
    n=min(len(codes),77); n=n-(n-1)%4
    if n<5: raise ValueError("최소5frames 필요")
    protocol=args.protocol or ("release" if str(args.adapter).endswith(".safetensors") else "paper")
    if protocol=="release" and args.ablation!="none":
        raise ValueError("release sampler는 원본 재현용. 개입 실험은 --protocol paper")
    pipe=connect.build_pipeline(args.disentangler,args.adapter,args.device); pipe.eval(); pipe.requires_grad_(False)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    from vda.generate import generate_full,save_outputs
    if protocol=="release":
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


def build_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cond",required=True); p.add_argument("--adapter",required=True); p.add_argument("--disentangler",required=True)
    p.add_argument("--out",default=str(ROOT/"outputs/04")); p.add_argument("--device",default="cuda")
    p.add_argument("--size",type=int,nargs=2,default=[288,512],metavar=("W","H")); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--protocol",choices=["paper","release"])
    p.add_argument("--ablation",choices=["none","no-expression","freeze-expression","no-proxy","all"],default="none")
    p.add_argument("--guidance",type=float,default=1.5)
    return p


def main():
    generate(build_parser().parse_args())


if __name__=="__main__": main()
