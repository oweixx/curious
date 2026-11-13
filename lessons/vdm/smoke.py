"""Small checks, not an avatar quality/convergence claim. CUDA_VISIBLE_DEVICES=4 python smoke.py --device cuda"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from common import ROOT, UPSTREAM, upstream, lesson, seed_all, save


def standalone(name,path):
    # Load the exact dependency-free upstream module, without importing its model downloader.
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--device",default="cpu"); args=p.parse_args()
    torch.set_num_threads(4); seed_all(42); upstream()
    zero,one,three=lesson(0),lesson(1),lesson(3)
    report={"device":args.device,"scope":"module/math smoke only; no pretrained WAN/video quality test"}
    out=ROOT/".cache/smoke"; out.mkdir(parents=True,exist_ok=True)
    model=one.make_model().to(args.device)
    # Synthetic factors ONLY test loss/optimization plumbing, never reported as NeRSemble results.
    ex=torch.arange(32).repeat_interleave(16); cam=torch.arange(16).repeat(32)
    z=torch.randn(32,512)[ex]+.4*torch.randn(16,512)[cam]
    cfg=one.config()["disentangler"]
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    before=one.evaluate(model,z[:128],ex[:128])["reconstruction"]
    for step in range(12):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss,terms=one.objective(model,z,ex,cam,40,cfg)
        loss.backward()
        for name,module in (("expr",model.expr_encoder),("cam",model.cam_encoder),("entangler",model.decoder)):
            norm=sum(float(p.grad.float().norm()) for p in module.parameters() if p.grad is not None)
            assert np.isfinite(norm) and norm>0,name
        optimizer.step()
    after=one.evaluate(model,z[:128],ex[:128])["reconstruction"]
    assert after<before,(before,after)
    save(out/"disentangler.pt",{"cfg":model.cfg,"model_state":model.state_dict()})
    from vda.disentangler import XNemoDisentangler
    restored=XNemoDisentangler.load(out/"disentangler.pt",args.device)
    x=z[:4].to(args.device)
    assert torch.equal(restored.expr_encoder(x),model.expr_encoder(x))
    report["disentangler"]={"synthetic_reconstruction_before":before,"after_12_updates":after,"public_checkpoint_roundtrip":True}
    # Local ray color changes only with part's relative rotation. Global rigid rotation cancels.
    K=np.array([[10,0,3],[0,10,3],[0,0,1.]])
    R=np.array([[0.,0,1],[0,1,0],[-1,0,0]])
    c2w=np.eye(4); gray=np.full((6,6),.4)
    a=zero.ray_colors(K,c2w,np.eye(3),gray)
    c2w[:3,:3]=R
    b=zero.ray_colors(K,c2w,R,gray)
    assert np.allclose(a,b)
    assert not np.allclose(a,zero.ray_colors(K,np.eye(4),R,gray))
    report["proxy_coordinate_invariance"]=True
    x0=torch.randn(1,16,3,4,4); ref=torch.randn(1,16,1,4,4); noise=torch.randn_like(x0)
    latent,velocity=three.flow_pair(x0,ref,torch.tensor(.6),noise)
    assert torch.equal(latent[:,:,:1],ref)
    pred=torch.cat((torch.full_like(ref,1000),velocity),2).requires_grad_()
    loss=three.weighted_flow_loss(pred,velocity,torch.tensor(.6)); loss.backward()
    assert loss.item()==0 and pred.grad[:,:,:1].abs().sum()==0
    report["reference_excluded_from_flow_loss"]=True
    # Use actual published full-width FaceEncoder + one full-width fuser. No downloaded weights.
    mod=standalone("official_adapter",UPSTREAM/"vda/third_party/diffsynth/models/wan_video_animate_adapter.py")
    with torch.device("meta"):
        all_modules=mod.WanAnimateAdapter()
    report["upstream_parameters"]={n:sum(p.numel() for p in getattr(all_modules,n).parameters()) for n in ("face_adapter","face_encoder","pose_patch_embedding")}
    if args.device.startswith("cuda"):
        encoder=mod.FaceEncoder(512,5120,4).to(args.device)
        fuser=mod.FaceBlock(5120,40).to(args.device)
        dit=standalone("official_dit",UPSTREAM/"vda/third_party/diffsynth/models/wan_video_dit.py")
        # One actual-width frozen DiT block: verify gradients traverse it into the adapter.
        block=dit.DiTBlock(False,5120,40,13824).to(device=args.device,dtype=torch.bfloat16)
        block.requires_grad_(False)
        from torch.utils.checkpoint import checkpoint
        codes=torch.randn(1,77,512,device=args.device,requires_grad=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            motion=encoder(codes)
            assert motion.shape==(1,20,5,5120)
            motion=torch.cat((torch.zeros_like(motion[:,:1]),motion),1)
            tokens=torch.randn(1,21*4,5120,device=args.device,requires_grad=True)
            fused=fuser(tokens,motion)
            context=torch.randn(1,8,5120,device=args.device,dtype=torch.bfloat16)
            t_mod=torch.zeros(1,6,5120,device=args.device,dtype=torch.bfloat16)
            freqs=dit.precompute_freqs_cis(128,end=84).to(args.device)[:,None,:]
            result=checkpoint(block,fused,context,t_mod,freqs,use_reentrant=False)
            result.float().square().mean().backward()
        assert codes.grad is not None and torch.isfinite(codes.grad).all() and codes.grad.norm()>0
        assert all(p.grad is None for p in block.parameters())
        report["frozen_dit_backward"]={"blocks_executed":1,"blocks_full_model":40,"frozen_parameter_gradients":False,"checkpointed":True}
        report["full_width_face_path"]={"motion_shape":list(motion.shape),"gradient_norm_raw_code":float(codes.grad.norm()),"fuser_count_executed":1,"fuser_count_model":8}
        report["peak_allocated_gib"]=torch.cuda.max_memory_allocated()/2**30
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)); print(json.dumps(report,indent=2))


if __name__=="__main__": main()
