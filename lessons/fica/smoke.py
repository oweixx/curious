"""GPU 0에서 00~04의 실제 자료/외부 가중치/학습/렌더링 경계를 검사한다.

CUDA_VISIBLE_DEVICES=0 python smoke.py
CUDA_VISIBLE_DEVICES=0 python smoke.py --stage generation

Full-data quality benchmark가 아니다. Sapiens-1B/SDXL/CLIP은 실제 크기/가중치,
DiT/refiner/UPM은 명시적으로 작은 profile이다. 경로는 .cache/smoke/에 격리한다.
"""

import argparse
import gc
import importlib.util
import json
import os
import shutil
from pathlib import Path
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location("fica_00", Path(__file__).with_name("00_prepare_nersemble.py"))
L00 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L00)


def clear():
    gc.collect()
    torch.cuda.empty_cache()


def finite(value):
    if not torch.isfinite(value).all():
        raise AssertionError("NaN/Inf")


def stage_data(config, path, device):
    if not (path / "manifest.json").exists():
        L00.prepare_dataset(config, path, [0,32], device)
    data = L00.PreparedData(path)
    row = data.samples("train")[0]
    sample, identity = data.sample(row,device), data.identity(row["identity"],device)
    assert len(data.samples("train")) == 1 and len(data.samples("validation")) == 1
    assert sample["labels"]["valid"].any()
    assert sample["partial_gt"][3].sum()>0
    bridge = L00.MeshBridge(identity,device)
    reconstructed = bridge(identity["xyz"])
    error = (bridge.vertices_to_uv(reconstructed)-identity["xyz"])[:,identity["valid"]].square().mean().sqrt()
    assert float(error)<.005, float(error)
    # UV-to-mesh가 differentiable하고 mask를 포함한 실제 데이터인지 검사한다.
    xyz=identity["xyz"].clone().requires_grad_(True)
    (bridge(xyz).square().mean()).backward()
    assert xyz.grad is not None and xyz.grad.abs().sum()>0
    return {"train":1,"validation":1,"valid_uv_texels":int(identity["valid"].sum()),
            "uv_roundtrip_rmse_world_units":float(error),"scope":data.manifest["scope"],"real_data":True}


def stage_partial(config,data,device):
    module=L00.lesson(1)
    for task in ("uv","xyz","normal"):
        module.train_task(config,data,task,device,steps=2,save=False,val_every=1)
        out,_=module.run_paths(config,task)
        assert all((out/"validation"/f"{step}.jpg").is_file() for step in (0,1,2))
        clear()
    module.compose_predictions(data,device)
    row=data.samples("train")[0]
    path=data.path/"partial_pred"/f"{Path(row['file']).stem}.pt"
    partial=L00.load_tensor(path)["partial"]
    finite(partial)
    assert partial.shape==(10,64,64)
    # 미학습 UV head가 망가져 visibility=0일 수도 있다. GT fallback으로 성공시키지 않는다.
    return {"tasks":["uv","xyz","normal"],"steps_each":2,"actual_encoder":"Sapiens-1B, 40 layers, 1536 channels",
            "pretrained_strict_load":True,"partial_shape":list(partial.shape),"observed_predicted_texels":int(partial[3].sum()),
            "validation_steps_each":[0,1,2],"input_transform":module.INPUT_TRANSFORM,
            "fallback_to_gt":False,"quality":"two-step training is not a converged predictor"}




def stage_repair(config, _data, device):
    """실제 50K UV로 초기화하고 별도 cache에서 mask/joint/resume/추론을 검사한다.

    Sapiens는 실제 1B, 입력만 smoke의 128x96이다. 사용자 학습 자료/가중치/
    partial_pred는 읽기만 한다. 큰 임시 checkpoint는 검사 후 삭제한다.
    """
    import nvdiffrast.torch as dr
    module = L00.lesson(1)
    context = dr.RasterizeCudaContext(device=device)
    # UV row 방향, source 좌표, bilinear sampling을 알려진 identity warp로 검사한다.
    y,x=torch.meshgrid(torch.arange(32,device=device),torch.arange(32,device=device),indexing="ij")
    u,v=(x+.5)/32,1-(y+.5)/32
    uv=torch.stack((u,v)); rgb=torch.stack((u,1-v,u*0+.25))
    normal=torch.zeros_like(rgb); normal[2]=1
    valid=torch.ones(1,32,32,device=device,dtype=torch.bool)
    partial,coverage,_=module.resample_partial_uv(rgb,uv,rgb,normal,valid,32,context,max_uv_edge=.1)
    assert coverage[:,2:-2,2:-2].all()
    warp_error=float((partial[:3,2:-2,2:-2]-rgb[:,2:-2,2:-2]).abs().max())
    assert warp_error<1e-6
    assert torch.allclose(partial[4:7,2:-2,2:-2],normal[:,2:-2,2:-2])
    empty,_,_=module.resample_partial_uv(rgb,uv,rgb,normal,valid&False,32,context,max_uv_edge=.1)
    assert not empty.any()
    seam=uv.clone(); seam[0,:,:16]*=.5; seam[0,:,16:]=.5+seam[0,:,16:]*.5
    _,seam_coverage,stats=module.resample_partial_uv(rgb,seam,rgb,normal,valid,32,context,max_uv_edge=.1)
    assert stats["rejected_seam_cells"]>0 and not seam_coverage[:,:,10:22].any()
    source=L00.ROOT/"data/prepared"
    parent=L00.ROOT/"checkpoints/01/crop/uv.pt"
    if not parent.exists():
        raise FileNotFoundError("repair smoke는 기존 checkpoints/01/crop/uv.pt와 XYZ/normal export가 필요하다.")
    destination=L00.CACHE/"smoke/repair_data"
    destination.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source/"manifest.json",destination/"manifest.json")
    for name in ("frames","identities"):
        link=destination/name
        if not link.exists():
            link.symlink_to((source/name).resolve(),target_is_directory=True)
    (destination/"predictions").mkdir(exist_ok=True)
    for name in ("xyz","normal"):
        link=destination/"predictions"/name
        if not link.exists():
            link.symlink_to((source/"predictions"/name).resolve(),target_is_directory=True)
    data=L00.PreparedData(destination)
    run=f"repair_smoke_{time.time_ns()}"
    module.train_uv_repair(config,data,device,steps=2,init_from=parent,run_name=run,val_every=1,val_frames=1)
    out,path=module.run_paths(config,"uv",run)
    assert all((out/"validation"/f"{step}.jpg").exists() for step in (0,1,2))
    log=[json.loads(line) for line in (out/"training.jsonl").read_text().splitlines()]
    assert [r["warmup"] for r in log]==[True,False]
    assert log[0]["gradient"]["tensors"]==4 and log[1]["gradient"]["tensors"]>4
    clone,metadata=module.load_predictor(path,"uv",config,data,device)
    assert isinstance(clone,module.UVValidityPredictor) and metadata["source_step"]==2
    row=data.samples("validation")[0]
    sample=module.prepare_input(data.sample(row,device),clone.input_size)
    clone.eval()
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        predicted,logit=clone.predict(sample["rgb"][None])
    assert predicted.dtype==torch.float32 and logit.dtype==torch.float32
    y,x,h,w=[sample["crop"][k] for k in ("top","left","height","width")]
    saved=L00.load_tensor(data.path/"predictions"/run/"uv"/f"{Path(row['file']).stem}.pt",device)
    # AMP encoder의 CUDA 연산은 bitwise 재현을 보장하지 않는다. Weight는 정확히,
    # 예측은 작은 수치 오차 허용 범위에서 검사한다.
    reload_error=float((saved[:,y:y+h,x:x+w]-predicted[0]).abs().mean())
    assert reload_error<1e-4
    state=torch.load(path,map_location="cpu",mmap=True,weights_only=True)
    for key in ("decode_head.conv_seg.weight","validity_head.0.weight","validity_head.2.weight"):
        assert torch.equal(clone.state_dict()[key].cpu(),state["model"][key])
    del state
    del clone,predicted,logit,saved
    clear()
    # 동일 config/data/code에서 optimizer와 step을 실제로 복원해 한 step 더 학습한다.
    module.train_uv_repair(config,data,device,steps=3,resume=True,run_name=run,val_every=1,val_frames=1)
    # 비교 이미지도 smoke 경로에 쓴다. Disk manifest의 fingerprint는 그대로 유지한다.
    data.manifest["config"]["profile"]="smoke"
    module.compose_uv_repair(data,device,run,config["uv_repair"])
    first=data.manifest["samples"][0]
    output=data.path/"partial_pred"/f"{Path(first['file']).stem}.pt"
    result=L00.load_tensor(output)
    finite(result["partial"])
    uv_size=data.manifest["config"]["uv_size"]
    assert result["partial"].shape==(10,uv_size,uv_size) and not result["gt_used_for_prediction"]
    # 3-step의 새 mask는 아직 비어 있다. GT 독립성 검사는 test cache의 confidence만
    # 1로 설정해 nonempty 출력에서 수행한다. 실제 학습 품질로 사용하지 않는다.
    probability_path=data.path/"predictions"/run/"validity"/f"{Path(first['file']).stem}.pt"
    probability=L00.load_tensor(probability_path)
    probability.fill_(1.)
    L00.save_tensor(probability_path,probability)
    original_rows=data.manifest["samples"]
    data.manifest["samples"]=[first]
    module.compose_uv_repair(data,device,run,config["uv_repair"])
    before=L00.load_tensor(output)
    assert before["partial"][3].sum()>0
    original_sample=data.sample
    def poisoned_labels(row,device="cpu"):
        sample=original_sample(row,device)
        return dict(sample,labels={k:torch.zeros_like(v) for k,v in sample["labels"].items()},partial_gt=torch.zeros_like(sample["partial_gt"]))
    data.sample=poisoned_labels
    module.compose_uv_repair(data,device,run,config["uv_repair"])
    after=L00.load_tensor(output)
    assert torch.equal(before["partial"],after["partial"])
    data.sample=original_sample
    data.manifest["samples"]=original_rows
    # 02가 이전 partial의 latent를 재사용하지 못하는지도 확인한다.
    cache_path=data.path/"latents"/f"{Path(first['file']).stem}.pt"
    L00.save_tensor(cache_path,{"partial_source":"predicted","partial_sha256":L00.sha256(output)})
    L00.lesson(2).read_cache(data,first,device)
    L00.save_tensor(cache_path,{"partial_source":"predicted","partial_sha256":"stale"})
    try:
        L00.lesson(2).read_cache(data,first,device)
    except ValueError:
        stale_rejected=True
    else:
        raise AssertionError("stale partial cache를 거부하지 않았다.")
    cache_path.unlink()
    path.unlink()  # 이 테스트가 만든 약 13GB checkpoint만 제거한다.
    return {"parent_checkpoint":str(parent),"actual_encoder":"Sapiens-1B","input_size":[128,96],
            "identity_resampling_max_error":warp_error,"seam_not_bridged":True,
            "warmup_gradient_tensors":log[0]["gradient"]["tensors"],
            "joint_gradient_tensors":log[1]["gradient"]["tensors"],
            "checkpoint_reload":True,"reload_uv_mean_difference":reload_error,"optimizer_resume_to_step":3,"fp32_decoder":True,
            "gt_independence":True,"gt_independence_nonempty_fixture":True,"stale_latent_rejected":stale_rejected,
            "validation_steps":[0,1,2,3],"output":str(out),
            "quality":"functional smoke only; three repair steps do not train the new foreground head"}


def stage_generation(config,data,device):
    module=L00.lesson(2)
    module.build_cache(data,device,"predicted")
    clear()
    model=module.train(config,data,device,4,val_every=2,val_sample_steps=4)
    validation=L00.work_root(config)/"outputs/02/validation"
    assert all((validation/f"{step}.jpg").is_file() for step in (0,2,4))
    validation_log=[json.loads(line) for line in (validation/"metrics.jsonl").read_text().splitlines()][-3:]
    assert [row["step"] for row in validation_log]==[0,2,4]
    assert all(row["frames"]==[data.samples("validation")[0]["file"]] for row in validation_log)
    assert all(row["seed"]==config["seed"] and row["sample_steps"]==4 for row in validation_log)
    # 평가가 학습 RNG와 mode를 바꾸지 않고, 같은 weight/noise로 같은 지표를 내는지 확인한다.
    import nvdiffrast.torch as dr
    with torch.random.fork_rng(devices=[device]):
        conditioners=module.FrozenConditioners(data,device,load_clip=False)
    context=dr.RasterizeCudaContext(device=device)
    model.train()
    model.zero_grad(set_to_none=True)
    rng_cpu,rng_cuda=torch.get_rng_state().clone(),torch.cuda.get_rng_state(device).clone()
    repeated=module.validate(config,data,model,device,conditioners,context,module.validation_rows(data,1),
                             L00.work_root(config)/"outputs/02/repeat_check",4,4)
    assert model.training and all(p.grad is None for p in model.parameters())
    assert torch.equal(rng_cpu,torch.get_rng_state()) and torch.equal(rng_cuda,torch.cuda.get_rng_state(device))
    for name in ("flow_loss","texture_l1","xyz_rmse"):
        assert abs(repeated[name]-validation_log[-1][name])<1e-6,(name,repeated[name],validation_log[-1][name])
    model.eval()
    del conditioners,context
    cache=module.read_cache(data,data.samples("train")[0],device)
    with torch.no_grad():
        latent=torch.randn(1,*cache["texture"].shape,device=device)
        t=torch.tensor([.5],device=device)
        full=model(latent,t,cache["clip"][None],cache["control"][None],torch.tensor([0],device=device))
        other=model(latent,t,cache["clip"][None],torch.zeros_like(cache["control"][None]),torch.tensor([0],device=device))
        control_effect=float((full-other).abs().max())
        assert control_effect>0, "ControlMLP가 출력에 영향을 주지 않는다."
        clone=module.UVFlowModel(config).to(device).eval()
        L00.restore(L00.work_root(config)/"checkpoints/02/latest.pt",clone,config,data)
        restored=clone(latent,t,cache["clip"][None],cache["control"][None],torch.tensor([0],device=device))
        assert torch.allclose(full,restored,atol=1e-6)
    module.generate(config,data,model,device,steps=4)
    result={"train_steps":4,"sample_steps":4,"actual_external_weights":["SDXL VAE","CLIP ViT-L/14"],
            "control_effect_max":control_effect,"checkpoint_roundtrip":True,"predicted_partial_used":True,
            "parameters":sum(p.numel() for p in model.parameters()),"profile":"small DiT; full paper depth stays in fica_config.json",
            "validation_steps":[0,2,4],"fixed_heldout_frame_and_seed":True,
            "validation_rng_and_mode_preserved":True,"validation_deterministic":True}
    del model,clone
    clear()
    return result


def stage_refinement(config,data,device):
    module=L00.lesson(3)
    module.cache_features(data,device)
    clear()
    model=module.train(config,data,device,2)
    module.predict(config,data,model,device)
    row=data.samples("train")[0]
    initial=module.read_initial(data,row,device)
    refined=L00.load_tensor(data.path/"refined"/f"{Path(row['file']).stem}.pt",device)
    finite(refined["xyz"]);finite(refined["texture"])
    change=float((initial["texture"]-refined["texture"]).abs().mean())
    assert change>0
    clone=module.RefinementNet(config).to(device)
    L00.restore(L00.work_root(config)/"checkpoints/03/latest.pt",clone,config,data)
    result={"train_steps":2,"module":"actual Diffusers UNet2DConditionModel","conditioner":"actual frozen Sapiens-1B",
            "mean_texture_change":change,"checkpoint_reload":True,"renderer":"nvdiffrast CUDA; original FLAME topology"}
    del model,clone
    clear()
    return result


def stage_avatar(config,data,device):
    module=L00.lesson(4)
    model=module.train(config,data,device,2,perceptual=True)
    module.render_sequence(config,data,model,device,"refined")
    first_packet=L00.load_tensor(L00.work_root(config)/"outputs/04/refined/000000.pt")
    second_packet=L00.load_tensor(L00.work_root(config)/"outputs/04/refined/000001.pt")
    assert first_packet["reference_frame"]==second_packet["reference_frame"]
    row=data.samples("train")[0]
    sample=data.sample(row,device)
    identity=data.identity(row["identity"],device)
    bridge=L00.MeshBridge(identity,device)
    mean,std=L00.lesson(3).stats(data,device)
    texture,xyz=module.identity_input(data,row,device,"refined")
    with torch.no_grad():
        bias=model.encode_identity(texture,xyz,mean,std)
        driver=module.driving(sample)
        first=model(texture,xyz,mean,std,*driver,biases=bias)
        second=model(texture,xyz,mean,std,driver[0]+.25,*driver[1:],biases=bias)
        changed=float((first["guide"]-second["guide"]).abs().max())
        assert changed>0,"expression이 guide geometry에 영향을 주지 않는다."
        gs=module.decode(first,texture,bridge,sample,config)
        assert gs["scale"].shape[1]==3
        assert torch.all(gs["scale"]>0)
        assert torch.allclose(gs["rotation"].norm(dim=-1),torch.ones(len(gs["rotation"]),device=device),atol=1e-5)
        rendered=module.render(gs,sample,torch.ones(3,device=device))
        finite(rendered["rgb"]);finite(rendered["depth"])
        assert rendered["alpha"].max()>.01
        clone=module.UniversalPrior(config).to(device)
        L00.restore(L00.work_root(config)/"checkpoints/04/latest.pt",clone,config,data)
    result={"train_steps":2,"renderer":"Graphdeco differentiable 3DGS CUDA","scale_channels":3,
            "gaussians":len(gs["position"]),"id_bias_shapes":[list(x.shape) for x in bias["geometry"]],
            "expression_geometry_effect_max":changed,"checkpoint_reload":True,"lpips":True,
            "geometry_anchor":"learned guide; GT posed mesh is supervision only", "end_to_end_refined_uv_to_3dgs":True,
            "fixed_single_portrait_for_animation":True}
    del model,clone,bridge
    clear()
    return result


def stage_full(config,data,device):
    """본 크기의 model도 실제 forward/backward한다. 최적화/품질 검증은 작은 연결 smoke와 구분.

    Full UV512, Sapiens 입력1024x768, 28-block DiT, full U-Net, full UPM.
    Large checkpoint/optimizer는 만들지 않는다. 출력은 scalar report만 남긴다.
    """
    full=L00.read_config()
    path=L00.CACHE/"full_shape_data"
    if not (path/"manifest.json").exists():
        L00.prepare_dataset(full,path,[0,32],device)
    full_data=L00.PreparedData(path)
    row=full_data.samples("train")[0]
    sample=full_data.sample(row,device)
    identity=full_data.identity(row["identity"],device)
    results={}
    print("FULL Sapiens-1B: 1024x768 forward/backward",flush=True)
    m1=L00.lesson(1)
    predictor=m1.DensePredictor("uv",device,full["sapiens"]["input_size"]).train()
    mean,std=L00.lesson(3).stats(full_data,device)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        prediction=predictor(sample["rgb"][None])
        loss=m1.prediction_loss("uv",prediction,sample,mean,std)
    loss.backward()
    results["sapiens"]={"parameters":sum(p.numel() for p in predictor.parameters()),"loss":float(loss.detach()),"gradient":L00.gradients(predictor)}
    del predictor,prediction,loss
    clear()
    m2=L00.lesson(2)
    print("FULL 28-block DiT/ControlMLP: UV512 forward/backward",flush=True)
    conditioners=m2.FrozenConditioners(full_data,device)
    small=data.samples("train")[0]
    partial=L00.load_tensor(data.path/"partial_pred"/f"{Path(small['file']).stem}.pt",device)["partial"]
    partial=F.interpolate(partial[None],size=(512,512),mode="nearest")[0]
    cache=conditioners.encode_sample(sample,identity,partial)
    cache={k:v.to(device) for k,v in cache.items()}
    del conditioners
    clear()
    flow=m2.UVFlowModel(full).to(device).train()
    with torch.autocast("cuda",dtype=torch.bfloat16):
        loss=m2.flow_loss(flow,cache)
    loss.backward()
    results["diffusion"]={"parameters":sum(p.numel() for p in flow.parameters()),"depth":28,"loss":float(loss.detach()),"gradient":L00.gradients(flow)}
    del flow,loss,cache
    clear()
    print("FULL refinement U-Net: UV512 forward/backward",flush=True)
    import nvdiffrast.torch as dr
    context=dr.RasterizeCudaContext(device=device)
    bridge=full_data.bridge(row["identity"],device)
    # Shape/gradient contract는 실제 target UV로 검사한다. End-to-end 생성은 앞의 small smoke.
    initial={"texture":identity["texture"],"xyz":identity["xyz"]}
    encoder=m1.pretrained_encoder(device)
    reference,features,_=L00.lesson(3).feature_pair(encoder,sample,initial,bridge,context)
    del encoder
    clear()
    refiner=L00.lesson(3).RefinementNet(full).to(device).train()
    with torch.autocast("cuda",dtype=torch.bfloat16):
        texture,xyz=refiner(initial["texture"][None],initial["xyz"][None],reference,features,mean,std)
    loss,terms,_=L00.lesson(3).refinement_loss(full,texture.float(),xyz.float(),initial,identity,sample,bridge,context)
    loss.backward()
    results["refinement"]={"parameters":sum(p.numel() for p in refiner.parameters()),"loss":float(loss.detach()),"gradient":L00.gradients(refiner)}
    del refiner,texture,xyz,loss,terms,reference,features
    clear()
    print("FULL UPM: UV512 real 3DGS forward/backward",flush=True)
    m4=L00.lesson(4)
    avatar=m4.UniversalPrior(full).to(device).train()
    with torch.autocast("cuda",dtype=torch.bfloat16):
        raw=avatar(identity["texture"][None],identity["xyz"][None],mean,std,*m4.driving(sample))
    gs=m4.decode(raw,identity["texture"][None],bridge,sample,full)
    rendered=m4.render(gs,sample,torch.ones(3,device=device))
    loss,terms=m4.avatar_loss(full,rendered,gs,sample,bridge)
    loss.backward()
    results["upm"]={"parameters":sum(p.numel() for p in avatar.parameters()),"gaussians":len(gs["position"]),"loss":float(loss.detach()),"gradient":L00.gradients(avatar)}
    del avatar,raw,gs,rendered,loss,terms
    clear()
    return {"models":results,"uv_size":512,"scope":"full-shape forward/backward; not full training/quality"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("all","data","partial","generation","refinement","avatar","full","repair"),default="all")
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="0":
        raise ValueError("이 smoke는 physical GPU 0만 사용한다. CUDA_VISIBLE_DEVICES=0으로 실행하세요.")
    config=L00.read_config(smoke=True)
    device=L00.setup_device("cuda:0",config["seed"])
    root=L00.work_root(config)
    root.mkdir(parents=True,exist_ok=True)
    report_path=root/"report.json"
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    report.update(physical_gpu=0,device=torch.cuda.get_device_name(device),config=config,scope="functional smoke, not trained avatar quality")
    functions={"data":stage_data,"partial":stage_partial,"generation":stage_generation,"refinement":stage_refinement,"avatar":stage_avatar,"full":stage_full,"repair":stage_repair}
    for name,function in functions.items():
        if args.stage not in ("all",name) or (name in ("full","repair") and args.stage!=name):
            continue
        start=time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        print(f"SMOKE {name}: start",flush=True)
        try:
            value=function(config,root/"data",device) if name=="data" else function(config,None,device) if name=="repair" else function(config,L00.PreparedData(root/"data"),device)
            value.update(status="passed",seconds=time.monotonic()-start,gpu_peak_gb=torch.cuda.max_memory_allocated(device)/1024**3)
            report[name]=value
            print(f"SMOKE {name}: passed",flush=True)
        except Exception as error:
            report[name]={"status":"failed","error":repr(error)}
            L00.save_json(report_path,report)
            raise
        L00.save_json(report_path,report)
        clear()
    print("Report:",report_path,flush=True)


if __name__=="__main__":
    main()
