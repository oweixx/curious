"""Smoke — 주요 수식/모듈과 gradient 경로를 빠르게 확인한다.

실행 위치 / 환경:
    conda activate elite
    cd /media/vilab/mh/curious/lessons/done/vda

GPU 4에서 실제 폭의 face encoder/fuser와 frozen DiT block backward까지 검사:
    CUDA_VISIBLE_DEVICES=4 python smoke.py --device cuda

GPU 없이 수식/disentangler 등 CPU 검사만 실행 (CUDA block 검사는 생략):
    CUDA_VISIBLE_DEVICES="" python smoke.py --device cpu

00 전처리의 시간축/triangulation/카메라 수식만 검사 (모델/가중치 불필요):
    python smoke.py --preprocess-only

pixel3DMM의 z=0 투영/gradient 회귀 검사 (GPU, 모델 학습은 실행하지 않음):
    CUDA_VISIBLE_DEVICES=4 python smoke.py --device cuda --tracking-projection

준비물: 00 setup으로 받은 upstream source와 import 가능한 dependencies.
출력: .cache/smoke/report.json과 테스트용 disentangler.pt (재실행 시 갱신).
합성 데이터/일부 block 검사이며 전체 pretrained 14B 학습이나 품질 검증이 아니다.
전체 모델 사전 검증 범위와 기록은 GUIDE.md의 검증 범위 참고.
"""
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


def preprocessing_smoke():
    """Calibrated multi-view math and invalid-observation handling; synthetic test, not a fit result."""
    from preprocess import sample_indices, triangulate, project, write_json
    # Common sequence clock must remain identical across independently prepared clips.
    full=sample_indices("73/1",0,350)
    assert np.array_equal(sample_indices("73/1",240,77),full[240:317])
    assert np.array_equal(sample_indices("73/1",0,5),[0,3,6,9,12])
    assert sample_indices("73/1",0,77)[-1]==231
    K=np.array([[800.,0,550],[0,800.,802],[0,0,1.]])
    cameras=np.repeat(np.eye(4)[None],4,0)
    cameras[:,0,3]=[-.6,-.2,.3,.7]
    points=np.array([[.1,.2,3.],[.3,-.1,2.7],[-.4,.2,3.1]])
    obs=np.stack([project(points,K,c)[0] for c in cameras])
    P=np.stack([K@c[:3] for c in cameras])
    obs[3,0]+=[130,-90]  # One bad detector result must not drag the reconstruction.
    obs[1,1]=np.nan
    xyz,inliers=triangulate(obs,P,threshold=3.)
    assert np.allclose(xyz,points,atol=1e-8)
    assert not inliers[3,0] and not inliers[1,1]
    invalid=obs.copy(); invalid[1:]=np.nan
    bad,mask=triangulate(invalid,P)
    assert np.isnan(bad).all() and not mask.any()
    behind=points.copy(); behind[:,2]*=-1
    q=np.einsum('vij,nj->vni',P,np.c_[behind,np.ones(len(behind))])
    bad,mask=triangulate(q[...,:2]/q[...,2:],P)
    assert np.isnan(bad).all() and not mask.any()
    # Native camera projection and RGB warp must share the exact crop matrix.
    s=1280/1604; left=(round(1100*s)-720)/2
    A=np.array([[s,0,-left],[0,s,0],[0,0,1.]])
    native=project(points,K,cameras[0])[0]
    warped=(np.c_[native,np.ones(len(native))]@A.T)[:,:2]
    assert np.allclose(warped,project(points,A@K,cameras[0])[0])
    report=dict(scope="synthetic preprocessing math only",time_alignment=True,
                multiview_outlier_rejection=True,missing_and_behind_camera_rejection=True,
                camera_crop_equivalence=True)
    from preprocess import SAM, mhr_rotations
    if (SAM/"sam_3d_body/models/modules/mhr_utils.py").is_file():
        rot=mhr_rotations()
        angular=torch.randn(5,133)*.2
        continuous=rot.compact_model_params_to_cont_body(angular).requires_grad_()
        decoded=rot.compact_cont_to_model_params_body(continuous)
        assert torch.allclose(decoded,angular,atol=1e-6)
        decoded.square().mean().backward()
        assert torch.isfinite(continuous.grad).all() and continuous.grad.norm()>0
        report["official_mhr_rotation_roundtrip_and_gradient"]=True
    write_json(ROOT/".cache/preprocess_smoke/math_report.json",report)
    print(json.dumps(report,indent=2))


def tracking_projection_smoke():
    """A masked-out vertex at the camera plane must not corrupt other gradients."""
    import os, sys
    from preprocess import stable_tracker_projection, child_env, write_json
    upstream()
    from vda import paths
    os.environ.update(paths.pixel3dmm_env())
    if not os.environ.get("CUDA_HOME") and (Path(sys.prefix)/"bin/nvcc").is_file():
        os.environ["CUDA_HOME"] = sys.prefix
    os.environ["PATH"] = str(Path(sys.prefix)/"bin")+os.pathsep+os.environ["PATH"]
    for path in child_env()["PYTHONPATH"].split(os.pathsep):
        if path and path not in sys.path: sys.path.insert(0,path)
    from pixel3dmm.tracking import tracker
    original = getattr(tracker.project_points_screen_space,"_torchdynamo_orig_callable",
                       tracker.project_points_screen_space)
    R={0:torch.eye(3,device="cuda")[None]}
    t={0:torch.tensor([[0.,0.,-1.]],device="cuda")}
    focal=torch.tensor([[3.5]],device="cuda",requires_grad=True)
    principal=torch.zeros(1,2,device="cuda",requires_grad=True)
    points=torch.tensor([[[.02,.03,0.],[.04,-.02,.4],[.01,.05,1.2]]],device="cuda",requires_grad=True)
    before=original(points,focal,principal,R,t,size=256)
    after=stable_tracker_projection(points,focal,principal,R,t,size=256)
    assert torch.equal(before,after)
    before_grad=torch.autograd.grad(before.sum(),(points,focal,principal))
    after_grad=torch.autograd.grad(after.sum(),(points,focal,principal))
    assert all(torch.equal(a,b) for a,b in zip(before_grad,after_grad))
    points=torch.tensor([[[.1,.2,1.],[.03,.02,.5],[.1,.2,1.-1e-7]]],device="cuda",requires_grad=True)
    before=original(points,focal,principal,R,t,size=256)
    old_grads=torch.autograd.grad(before[:,1,:2].square().mean(),(points,focal,principal))
    assert any(not g.isfinite().all() for g in old_grads),"expected original excluded-vertex regression"
    after=stable_tracker_projection(points,focal,principal,R,t,size=256)
    grads=torch.autograd.grad(after[:,1,:2].square().mean(),(points,focal,principal))
    assert after.isfinite().all() and all(g.isfinite().all() for g in grads)
    assert torch.count_nonzero(grads[0][:,[0,2]])==0
    report=dict(original_backward_failure_reproduced=True,
                unchanged_forward_and_gradients_away_from_plane=True,
                finite_projection_at_and_near_camera_plane=True,excluded_vertex_gradient_safe=True)
    write_json(ROOT/".cache/preprocess_smoke/projection_report.json",report)
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--device",default="cpu")
    checks=p.add_mutually_exclusive_group()
    checks.add_argument("--preprocess-only",action="store_true")
    checks.add_argument("--tracking-projection",action="store_true"); args=p.parse_args()
    if args.tracking_projection:
        if args.device!="cuda": p.error("--tracking-projection은 --device cuda와 CUDA_VISIBLE_DEVICES를 사용하세요")
        tracking_projection_smoke(); return
    if args.preprocess_only:
        preprocessing_smoke(); return
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
