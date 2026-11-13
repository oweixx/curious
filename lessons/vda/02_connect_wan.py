r"""02 — 공개 WAN-Animate 모듈 연결, trainable boundary, frozen encoder cache.

실행 위치 / 환경:
    conda activate elite
    cd /media/vilab/mh/curious/lessons/vdm

01 학습 완료 후: 준비된 WAN base + 직접 학습한 disentangler로 학습 대상 확인.
실제 14B 모델을 GPU에 로드하지만 optimizer 업데이트는 수행하지 않는다.
    CUDA_VISIBLE_DEVICES=4 python 02_connect_wan.py audit \
        --disentangler checkpoints/01/best.pt --mode paper

00에서 완성한 data/triplets.jsonl은 042_free_a(train)와 042_free_val(validation).
각각 77프레임, 9:16 crop이다. --size는 width height 순서다.
Stage1 학습에 사용할 VAE/CLIP cache:
    CUDA_VISIBLE_DEVICES=4 python 02_connect_wan.py cache \
        --manifest data/triplets.jsonl --size 288 512 --out data/cache_stage1

Stage1을 끝내고 Stage2로 갈 때 같은 입력으로 고해상도 cache 생성:
    CUDA_VISIBLE_DEVICES=4 python 02_connect_wan.py cache \
        --manifest data/triplets.jsonl --size 576 1024 --out data/cache_stage2

출력: data/cache_stage1/ 또는 data/cache_stage2/의 index.json과 clip별 .pt.
이미 index.json이 있는 cache는 덮어쓰지 않으므로 새 --out을 지정한다.
index.json이 이미 있으면 해당 cache로 03을 진행한다.
Cache 생성은 disentangler를 거치지 않고 raw code를 보존한다.
공개 모델만 별도로 확인할 때는 audit의 --disentangler를
weights/xnemo_disentangler.pt로 명시적으로 바꾼다. 이것은 01 학습 완료를 뜻하지 않는다.
도움말: python 02_connect_wan.py cache --help

읽는 순서: build_pipeline -> parameter_groups -> encode_triplet.
14B DiT를 작은 대체 네트워크로 바꾸지 않는다. frozen DiT도 backward graph에
남겨야 앞/중간에 삽입된 adapter가 학습된다. no_grad는 VAE/CLIP cache에만 사용.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm
from common import ROOT, upstream, config, read_manifest, load, save, digest


def base_pipeline(device="cuda", encoders_only=False):
    upstream()
    from vda.model import ANIMATE_ID
    from vda.paths import weights_dir
    from vda.third_party.diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    base=weights_dir()/"base"
    # The converted DiffSynth repository can be unavailable; official WAN .pth has
    # the same state-dict converter. Prefer safetensors when both are present.
    patterns=[("Wan2.1_VAE.safetensors","Wan2.1_VAE.pth"),
              ("models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors",
               "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")]
    if not encoders_only: patterns.insert(0,("diffusion_pytorch_model*.safetensors",))
    configs=[]
    for alternatives in patterns:
        files=[]
        for pattern in alternatives:
            files=sorted(base.rglob(pattern))
            if files: break
        if not files: raise FileNotFoundError(f"{base}: {alternatives}; 00 download -- base 먼저 실행")
        configs.append(ModelConfig(path=[str(f) for f in files] if len(files)>1 else str(files[0])))
    # Cached constant text embeddings: no tokenizer download and no UMT5 allocation.
    pipe=WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16,device=device,
                                         model_configs=configs,tokenizer_config=None)
    from vda.model import _install_text_cache, DEFAULT_PROMPT
    _install_text_cache(pipe,str(weights_dir()/"text_emb_cache.pt"),[DEFAULT_PROMPT])
    pipe.vda_prompts=(DEFAULT_PROMPT,DEFAULT_PROMPT)
    pipe.requires_grad_(False); pipe.eval()
    return pipe


def build_pipeline(disentangler, adapter=None, device="cuda"):
    """base WAN weights + pretrained disentangler + new translator/proxy branch.

    adapter=None: VDA adapter 학습 전 초기화 (stock face fusers/encoder 유지).
    translator의 merged MLP 초기화 recipe는 미공개이므로 PyTorch 초기화 사용.
    --adapter: 공개 또는 우리 adapter 가중치로 명시적 warm start.
    """
    if not Path(disentangler).is_file():
        raise FileNotFoundError(f"disentangler가 없습니다: {disentangler}. 01 train 완료 후 best.pt를 지정하세요")
    if adapter and not Path(adapter).is_file():
        raise FileNotFoundError(f"adapter가 없습니다: {adapter}. 03 학습 결과 또는 공개 checkpoint를 지정하세요")
    pipe=base_pipeline(device)
    from vda.third_party.diffsynth.models.wan_video_animate_adapter_xnemo import WanAnimateAdapterXNemo, build_translator_head
    ad=pipe.animate_adapter
    if ad is None: raise RuntimeError("WAN-Animate adapter가 없는 backbone")
    ad.__class__=WanAnimateAdapterXNemo
    ad.translator=build_translator_head(disentangler,device)
    p=ad.pose_patch_embedding
    ad.graft_patch_embedding=nn.Conv3d(p.in_channels,p.out_channels,p.kernel_size,p.stride).to(device)
    nn.init.zeros_(ad.graft_patch_embedding.weight); nn.init.zeros_(ad.graft_patch_embedding.bias)
    ad.float()  # trainable weights와 Adam moments는 fp32, 연산은 autocast bf16.
    if adapter:
        if str(adapter).endswith(".safetensors"):
            from safetensors.torch import load_file
            state=load_file(adapter)
            # Release checkpoints retain unused gates from an earlier experiment.
            # The official loader drops exactly these; fusers use ungated residuals.
            state={k:v for k,v in state.items() if not k.startswith("gates.")}
        else:
            state=load(adapter)["adapter"]
        # Released adapter can omit frozen motion/pre weights, but no learned branch may be missing.
        missing,unexpected=ad.load_state_dict(state,strict=False)
        required=("translator.f.","face_adapter.","face_encoder.","pose_patch_embedding.","graft_patch_embedding.")
        if unexpected or any(k.startswith(required) for k in missing):
            raise ValueError(f"adapter mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    return pipe


def parameter_groups(pipe, mode="paper"):
    """Table7: main DiT/VAE/CLIP/text/motion encoder frozen, adapters만 update.

    paper: D_XN.expr_encoder도 train. release: public pre-frozen behavior.
    두 모드를 암묵적으로 혼합하지 않는다.
    """
    cfg=config()["training"]; pipe.requires_grad_(False)
    ad=pipe.animate_adapter
    groups=[("face_fuser",ad.face_adapter,cfg["lr_face_fuser"]),
            ("face_encoder",ad.face_encoder,cfg["lr_face_encoder"]),
            ("translator",ad.translator.f,cfg["lr_translator"]),
            ("skeleton",ad.pose_patch_embedding,cfg["lr_pose"]),
            ("proxy",ad.graft_patch_embedding,cfg["lr_pose"])]
    if mode=="paper": groups.append(("expression",ad.translator.pre,cfg["lr_expression"]))
    elif mode!="release": raise ValueError(mode)
    out=[]
    for name,module,lr in groups:
        module.requires_grad_(True)
        out.append({"name":name,"params":list(module.parameters()),"lr":lr})
    return out


def checkpoint_blocks(pipe):
    """Frozen transformer도 gradient 전달은 필요. inference offload hook은 train에 쓰지 않는다."""
    for block in list(pipe.dit.blocks)+list(pipe.animate_adapter.face_adapter.fuser_blocks):
        if getattr(block,"_lesson_checkpointed",False): continue
        original=block.forward
        def forward(*args,_original=original,**kwargs):
            if torch.is_grad_enabled():
                return checkpoint(_original,*args,use_reentrant=False,**kwargs)
            return _original(*args,**kwargs)
        block.forward=forward; block._lesson_checkpointed=True


def load_images(path,size,n):
    """이미 9:16으로 crop된 clip만 resize. 시간축을 임의로 pad하지 않는다."""
    import cv2
    cap=cv2.VideoCapture(str(path)); result=[]
    while len(result)<n:
        ok,frame=cap.read()
        if not ok: break
        h,w=frame.shape[:2]
        if abs(w/h-9/16)>.005: raise ValueError(f"{path}: crop 먼저 필요")
        result.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize(size))
    cap.release()
    if len(result)!=n: raise ValueError(f"{path}: {n} frames 필요")
    return result


@torch.no_grad()
def encode_triplet(pipe,row,size,include_target=True):
    """77 real frames -> 20 latent slots; independent reference -> slot0; total21.

    num_frames=81은 81개의 target을 뜻하지 않는다. OOT reference에 1 latent slot
    (=4 frame bookkeeping)을 배정한 API 길이이다.
    """
    from vda.model import DEFAULT_PROMPT
    cond=Path(row["conditioning"]); w,h=size; n=row["frames"]
    ref=Image.open(cond/"ref.png").convert("RGB")
    if abs(ref.width/ref.height-9/16)>.005: raise ValueError("reference crop required")
    ref=ref.resize(size)
    pose=load_images(cond/"pose_skel.mp4",size,n); proxy=load_images(cond/"pose_proxy.mp4",size,n)
    target=load_images(row["target"],size,n) if include_target else None
    codes=torch.from_numpy(np.load(cond/"codes.npy",allow_pickle=False)).float()
    if not include_target: codes=codes[:n]
    if codes.shape!=(n,512): raise ValueError("RAW X-NeMo length/dimension mismatch")
    shared=dict(input_image=ref,seed=42,rand_device="cpu",height=h,width=w,num_frames=n+4,
                cfg_scale=1.,cfg_scale_id=2.,id_weak_image=ref.filter(ImageFilter.GaussianBlur(
                    config()["training"]["implementation_choices"]["weak_identity_blur"])),
                tiled=True,tile_size=(30,52),tile_stride=(15,26),animate_pose_video=pose,
                animate_graft_video=proxy,expr_codes=codes,
                animate_inpaint_video=[Image.new("RGB",size,(128,128,128))]*n,
                animate_mask_video=[Image.new("RGB",size,"white")]*n)
    positive={"prompt":DEFAULT_PROMPT}; negative={"negative_prompt":DEFAULT_PROMPT}
    for unit in pipe.units:
        shared,positive,negative=pipe.unit_runner(unit,pipe,shared,positive,negative)
    keys=("y","pose_latents","graft_latents","clip_feature","first_frame_latents",
          "y_weak","clip_feature_weak","first_frame_latents_weak")
    cached={k:shared[k].detach().cpu() for k in keys}
    cached.update(context=positive["context"].cpu(),expr_codes=positive["expr_codes"].cpu())
    if include_target:
        x0=pipe.vae.encode(pipe.preprocess_video(target),device=pipe.device,tiled=True).to(pipe.torch_dtype)
        cached["x0"]=x0.cpu()
        if cached["y"].shape[2]!=x0.shape[2]+1: raise ValueError("OOT reference alignment failed")
    return cached


def cache_manifest(args):
    rows=read_manifest(args.manifest)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if (out/"index.json").exists(): raise FileExistsError("새 cache 경로를 지정하세요")
    pipe=base_pipeline(args.device,encoders_only=True)
    import json
    from common import COMMIT
    records=[]
    for row in tqdm(rows,desc="VAE/CLIP cache"):
        blob=encode_triplet(pipe,row,tuple(args.size))
        fingerprints={str(p):digest(p) for p in [Path(row["target"]),*[Path(row["conditioning"])/n for n in ("ref.png","codes.npy","pose_proxy.mp4","pose_skel.mp4")]]}
        save(out/f"{row['id']}.pt",{**blob,"row":row,"source_sha256":fingerprints})
        records.append({"file":f"{row['id']}.pt","split":row["split"],"row":row})
    (out/"index.json").write_text(json.dumps({"upstream":COMMIT,"size":args.size,"records":records,
        "weak_identity_blur":config()["training"]["implementation_choices"]["weak_identity_blur"]},indent=2))


def build_parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    s=p.add_subparsers(dest="stage",required=True)
    a=s.add_parser("cache"); a.add_argument("--manifest",required=True); a.add_argument("--out",required=True)
    a.add_argument("--size",type=int,nargs=2,default=[288,512],metavar=("W","H")); a.add_argument("--device",default="cuda")
    a=s.add_parser("audit"); a.add_argument("--disentangler",required=True); a.add_argument("--adapter")
    a.add_argument("--device",default="cuda"); a.add_argument("--mode",choices=["paper","release"],default="paper")
    return p


def main():
    a=build_parser().parse_args()
    if a.stage=="cache": cache_manifest(a)
    else:
        pipe=build_pipeline(a.disentangler,a.adapter,a.device)
        groups=parameter_groups(pipe,a.mode)
        for group in groups: print(group["name"],sum(x.numel() for x in group["params"]),group["lr"])
        print("DiT trainable:",sum(p.numel() for p in pipe.dit.parameters() if p.requires_grad))


if __name__=="__main__": main()
