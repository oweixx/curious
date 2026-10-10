"""08 — 실제 한 사람의 영상에 network를 학습한다.

main의 학습 loop를 읽는다:
 frame sample -> UV/driving -> network -> Gaussian decode -> 2DGS -> loss
 -> zero_grad -> backward -> finite gradient/clip -> optimizer.step -> scheduler.step.
FLAME tracking/UV lookup은 고정하고 network의 encoder/U-Net/두 head는 모두 학습한다.

CUDA_VISIBLE_DEVICES=2 python 08_train_avatar.py --experiment avatar
CUDA_VISIBLE_DEVICES=2 python 08_train_avatar.py --experiment avatar --resume
학습 설정은 avatar_config.json. 새 설정은 새 experiment에 저장한다.
preview는 고정 frame을 step 0부터 주기적으로 eval해 같은 조건의 변화를 저장한다.
preview/0.jpg, 100.jpg, ...: frame별 reference | render | absolute error x4.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

spec=importlib.util.spec_from_file_location("elite_03_entry",Path(__file__).with_name("03_inspect_tracking.py"))
L03=importlib.util.module_from_spec(spec)
spec.loader.exec_module(L03)


def experiment_dir(scene,name):
    if not name or Path(name).name!=name or name in (".",".."):
        raise ValueError("experiment는 경로가 아닌 이름이어야 한다.")
    return L03.output_dir(8,scene["sequence"],scene["run"])/name


def code_hashes():
    sha=L03.lesson(2).sha256
    return {p.name:sha(p) for p in sorted(L03.ROOT.glob("0[3-8]_*.py"))}


def package_versions():
    import importlib.metadata
    names=("torch","torchvision","numpy","roma","lpips","diff-surfel-rasterization","nvdiffrast","pytorch3d")
    result={}
    for name in names:
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]="not registered"
    result.update(cuda_runtime=torch.version.cuda,device=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")
    return result


def validate_config(config,uv):
    if config["format"]!=L03.lesson(6).MODEL_FORMAT:
        raise ValueError("checkpoint/config format 불일치")
    if config["uv_size"]!=uv["uv_size"]:
        raise ValueError("04 uv-size와 avatar_config.json이 다르다.")
    network=config["network"]
    if len(network["channel_mult"])!=6 or any(not isinstance(x,int) or x<1 for x in network["channel_mult"]):
        raise ValueError("이번 network의 channel_mult는 6개의 양의 정수다.")
    if network["embedding"]<4 or network["embedding"]%4 or network["model_channels"]<4 or network["num_blocks"]<1:
        raise ValueError("embedding/model_channels/num_blocks 설정을 확인하세요.")
    size=config["uv_size"]
    if size<64 or size & (size-1):
        raise ValueError("UV size는 64 이상 2의 거듭제곱이다.")
    if config["image_max_side"]!=0 and config["image_max_side"]<64:
        raise ValueError("image_max_side는 native를 뜻하는 0 또는 64 이상이다.")
    t=config["training"]
    for key in ("steps","batch_size","log_every","validate_every","checkpoint_every","validation_frames"):
        if not isinstance(t[key],int) or t[key]<1:
            raise ValueError(f"training.{key}는 양의 정수")
    if t["lr"]<=0 or t["grad_clip"]<=0 or t["weight_decay"]<0 or not 0<=t["warmup_steps"]<t["steps"]:
        raise ValueError("optimizer/warmup 설정을 확인하세요.")
    if t["background"] not in ("white","random"):
        raise ValueError("background는 white 또는 random")
    if type(t.get("preview_every",0)) is not int or t.get("preview_every",0)<0:
        raise ValueError("preview_every는 0(끄기) 또는 양의 정수")
    frames=t.get("preview_frames",[])
    if not isinstance(frames,list) or any(type(i) is not int or i<0 for i in frames) or len(frames)!=len(set(frames)):
        raise ValueError("preview_frames는 중복 없는 frame ID 목록. []이면 첫 train/validation frame")
    if config["loss"]["lpips"]<=0:
        raise ValueError("최종 perceptual 평가를 위해 LPIPS weight>0을 사용하세요.")
    if any(value<0 for value in config["loss"].values()):
        raise ValueError("loss 설정은 음수가 될 수 없다.")
    if config["loss"]["lpips_max_side"]<16:
        raise ValueError("LPIPS max side는 16 이상이다.")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    # CUDA splatting의 atomic accumulation까지 bitwise deterministic하다는 주장은 하지 않는다.


def random_state():
    np_state=np.random.get_state()
    return {"python":random.getstate(),"numpy_name":np_state[0],
            "numpy_keys":torch.from_numpy(np_state[1].astype(np.int64)),
            "numpy_position":np_state[2],"numpy_has_gaussian":np_state[3],"numpy_cached":np_state[4],
            "torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}


def restore_random_state(state):
    random.setstate(state["python"])
    np.random.set_state((state["numpy_name"],state["numpy_keys"].cpu().numpy().astype(np.uint32),
                         state["numpy_position"],state["numpy_has_gaussian"],state["numpy_cached"]))
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def forward_frame_batch(model,surface,scene,ids,device):
    network=L03.lesson(6)
    parameters=network.parameter_batch(scene,ids,device)
    geometry_map,appearance_map=model(surface.uv_input(len(ids)),parameters)
    points,rotations=surface.posed_surface(ids)
    return surface.decode(geometry_map,appearance_map,points,rotations)


def render_batch(gaussians,dataset,device,background):
    K,w2c=dataset.K.to(device),dataset.w2c.to(device)
    return [L03.lesson(5).render(gaussians,i,K,w2c,dataset.h,dataset.w,background)
            for i in range(gaussians["position"].shape[0])]


def image_metrics(prediction,target,alpha,loss):
    error=(prediction-target).square()
    mse=error.mean().clamp_min(1e-10)
    foreground_mse=(error*alpha).sum()/(alpha.sum().clamp_min(1)*3)
    return {"psnr":float(-10*torch.log10(mse)),
            "foreground_psnr":float(-10*torch.log10(foreground_mse.clamp_min(1e-10))),
            "ssim":float(L03.lesson(7).ssim(prediction[None],target[None])),
            "lpips":float(loss.perceptual_loss(prediction[None],target[None])),
            "l1":float((prediction-target).abs().mean())}


@torch.no_grad()
def evaluate(model,surface,dataset,loss,ids,device,destination=None,panels=None):
    was_training=model.training
    model.eval()
    background=torch.ones(3,device=device)
    rows=[]
    for slot,index in enumerate(ids):
        targets=dataset.batch([index],device)
        gaussians=forward_frame_batch(model,surface,dataset.scene,[index],device)
        rendered=render_batch(gaussians,dataset,device,background)[0]
        target=L03.lesson(7).composite(targets["rgb"],targets["alpha"],background)[0]
        row=image_metrics(rendered["rgb"],target,targets["alpha"][0],loss)
        mask_a,mask_b=rendered["alpha"]>.5,targets["alpha"][0]>.5
        row.update(frame_id=int(index),alpha_iou=float((mask_a&mask_b).sum()/(mask_a|mask_b).sum().clamp_min(1)))
        rows.append(row)
        if (destination or panels is not None) and slot<4:
            # Reference | render | absolute error. 각 panel은 같은 image/camera다.
            image=torch.cat((target,rendered["rgb"],(target-rendered["rgb"]).abs()*4),-1)
            if destination:
                L03.save_image(Path(destination)/f"{index:06d}.jpg",image)
            if panels is not None:
                # Preview는 중간 파일 없이 이 CPU tensor들을 모아 이미지 하나만 저장한다.
                panels.append(image.cpu())
    model.train(was_training)
    return {"mean":{key:sum(r[key] for r in rows)/len(rows) for key in rows[0] if key!="frame_id"},"frames":rows}


def gradient_report(model):
    report={}
    for name,module in (("driving",model.driving),("unet",model.unet),
                        ("geometry_head",model.geometry_head),("appearance_head",model.appearance_head)):
        gradients=[p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None]
        report[name]=float(torch.stack(gradients).sum().sqrt()) if gradients else 0.
    return report


def preview_frames(scene,training):
    """빈 설정에서는 첫 train/validation frame을 고른다. 학습 sampling과 별개다."""
    if not training.get("preview_every",0):
        return []
    ids=training.get("preview_frames",[]) or [scene["split"]["train"][0],scene["split"]["validation"][0]]
    if any(i>=scene["manifest"]["num_frames"] for i in ids):
        raise ValueError("preview_frames에 sequence 범위 밖 frame ID가 있다.")
    return list(ids)


def save_preview(model,surface,dataset,loss,ids,device,out,completed):
    """고정 frame/camera/흰 배경으로 비교한다. evaluate가 eval/no_grad 후 원래 모드를 복구한다.

    모든 고정 frame을 세로로 이어 preview/<step>.jpg 한 장만 저장한다.
    Preview metric은 best.pt 선택에 사용하지 않는다. train frame과 validation frame을 구분한다.
    """
    from PIL import Image,ImageDraw,ImageFont
    directory=out/"preview"
    directory.mkdir(parents=True,exist_ok=True)
    tensors=[]
    report=evaluate(model,surface,dataset,loss,ids,device,panels=tensors)
    membership={i:name for name in ("train","validation","gap") for i in dataset.scene["split"][name]}
    panels=[Image.fromarray((tensor.clamp(0,1).permute(1,2,0).numpy()*255).round().astype(np.uint8))
            for tensor in tensors]
    header=32
    montage=Image.new("RGB",(panels[0].width,sum(image.height+header for image in panels)),"white")
    draw=ImageDraw.Draw(montage)
    try:font=ImageFont.truetype("DejaVuSans.ttf",18)
    except OSError:font=ImageFont.load_default()
    y=0
    for i,image in zip(ids,panels):
        draw.text((8,y+5),f"step {completed} | frame {i} ({membership[i]}) | reference / render / error x4",fill="black",font=font)
        montage.paste(image,(0,y+header))
        y+=image.height+header
    path=directory/f"{completed}.jpg"
    temporary=path.with_suffix(".tmp")
    montage.save(temporary,format="JPEG",quality=92)
    temporary.replace(path)
    print(f"preview {completed}: fixed frames={ids} foreground PSNR={report['mean']['foreground_psnr']:.3f} | {path}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    L03.add_arguments(parser)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--experiment",default="avatar")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("실제 2DGS CUDA backend를 사용한다. GPU를 노출한 환경에서 실행하세요.")
    device=torch.device(args.device)
    torch.cuda.set_device(device)
    scene=L03.read_scene(args.sequence,args.run)
    uv=L03.lesson(5).read_uv(scene)
    config=L03.lesson(6).load_config(args.config)
    validate_config(config,uv)
    fixed_preview=preview_frames(scene,config["training"])
    if len(fixed_preview)>4:
        raise ValueError("preview_frames는 4개 이하를 사용하세요. 전체 평가는 validation/10을 사용한다.")
    out=experiment_dir(scene,args.experiment)
    if args.resume and not (out/"latest.pt").is_file():
        raise FileNotFoundError("resume할 latest.pt가 없다.")
    if not args.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError("기존 experiment가 있다. --resume 또는 새 --experiment 사용")
    out.mkdir(parents=True,exist_ok=True)
    seed_all(config["seed"])
    dataset=L03.lesson(7).VideoFrames(scene,config["image_max_side"])
    surface=L03.lesson(5).SurfaceMap(scene,uv,device)
    model=L03.lesson(6).Mesh2Gaussian(config).to(device)
    loss=L03.lesson(7).AvatarLoss(config,uv,device)
    training=config["training"]
    optimizer=torch.optim.AdamW(model.parameters(),lr=training["lr"],weight_decay=training["weight_decay"])
    def rate(step):
        if step<training["warmup_steps"]:
            return (step+1)/max(1,training["warmup_steps"])
        progress=(step-training["warmup_steps"])/max(1,training["steps"]-training["warmup_steps"])
        return .1+.9*.5*(1+math.cos(math.pi*min(1.,progress)))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    start,best=0,-math.inf
    hashes=code_hashes()
    if args.resume:
        checkpoint=L03.read_tensor(out/"latest.pt")
        if checkpoint["config"]!=config or checkpoint["data_fingerprint"]!=dataset.fingerprint or checkpoint["code_hashes"]!=hashes:
            raise ValueError("resume의 config/data/code가 다르다. 변경 실험은 새 experiment로 진행하세요.")
        model.load_state_dict(checkpoint["model"],strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start,best=checkpoint["step"],checkpoint["best_foreground_psnr"]
        restore_random_state(checkpoint["random_state"])
    metadata={"config":config,"data_fingerprint":dataset.fingerprint,"code_hashes":hashes,
              "versions":package_versions(),"scope":"single-person scratch; no pretrained MGPM/enhancer/generative supervision",
              "sequence":args.sequence,"tracking_run":args.run,"experiment":args.experiment,
              "preview_frame_ids":fixed_preview}
    L03.save_json(out/"experiment.json",metadata)
    train_ids=torch.tensor(scene["split"]["train"])
    validation=scene["split"]["validation"]
    slots=np.unique(np.linspace(0,len(validation)-1,min(len(validation),training["validation_frames"])).astype(int))
    validation=[validation[i] for i in slots]
    def checkpoint_state(completed):
        return dict(metadata,step=completed,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),random_state=random_state(),best_foreground_psnr=best)
    model.train()
    clock=time.monotonic()
    if fixed_preview and not args.resume:
        save_preview(model,surface,dataset,loss,fixed_preview,device,out,completed=0)
    for step in range(start,training["steps"]):
        # 공유 UV에 대해 다른 driving/target을 sample한다. Frame ID 자체는 network에 넣지 않는다.
        ids=train_ids[torch.randint(len(train_ids),(training["batch_size"],))].tolist()
        targets=dataset.batch(ids,device)
        background=torch.ones(3,device=device) if training["background"]=="white" else torch.rand(3,device=device)
        gaussians=forward_frame_batch(model,surface,scene,ids,device)
        rendered=render_batch(gaussians,dataset,device,background)
        total,raw,weighted=loss(rendered,targets,gaussians,background,step)
        if not torch.isfinite(total):
            raise FloatingPointError(f"step {step}: loss NaN/Inf")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),training["grad_clip"],error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        completed=step+1
        if step==start or completed%training["log_every"]==0:
            record={"step":completed,"seconds":time.monotonic()-clock,"frame_ids":ids,
                    "total":float(total.detach()),"lr":optimizer.param_groups[0]["lr"],
                    "raw":{k:float(v.detach()) for k,v in raw.items()},
                    "weighted":{k:float(v.detach()) for k,v in weighted.items()},
                    "gradient_norm_before_clip":float(grad_norm),"gradient_by_module_after_clip":gradient_report(model),
                    "visible_gaussians":int((rendered[0]["radii"]>0).sum()),
                    "gpu_peak_gb":torch.cuda.max_memory_allocated(device)/1024**3}
            with (out/"training.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(record)+"\n")
            print(f"step {completed:6d} loss={record['total']:.5f} photo={record['raw']['photo']:.5f} "
                  f"alpha={record['raw']['alpha']:.5f} gradient={float(grad_norm):.4f}",flush=True)
        if fixed_preview and (completed%training["preview_every"]==0 or completed==training["steps"]):
            save_preview(model,surface,dataset,loss,fixed_preview,device,out,completed)
        if completed%training["validate_every"]==0 or completed==training["steps"]:
            report=evaluate(model,surface,dataset,loss,validation,device,out/"validation"/f"{completed:06d}")
            L03.save_json(out/"validation"/f"{completed:06d}.json",report)
            score=report["mean"]["foreground_psnr"]
            if score>best:
                best=score
                L03.save_tensor(out/"best.pt",checkpoint_state(completed))
            print(f"validation {completed}: foreground PSNR={score:.3f} SSIM={report['mean']['ssim']:.4f}",flush=True)
        if completed%training["checkpoint_every"]==0 or completed==training["steps"]:
            L03.save_tensor(out/"latest.pt",checkpoint_state(completed))
    print("08 완료:",out,"| 이후 09 render와 10 전체 평가를 실행하세요.")


if __name__=="__main__":
    main()
