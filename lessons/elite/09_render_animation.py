"""09 — Native 학습 checkpoint로 sequence/다른 driving scene을 실제 렌더링한다.

CUDA_VISIBLE_DEVICES=2 python 09_render_animation.py --experiment avatar
CUDA_VISIBLE_DEVICES=2 python 09_render_animation.py --experiment avatar --driver-scene /다른/03/scene.pt
다른 driver의 shape/appearance는 사용하지 않는다. Target identity의 UV를 유지한다.
Driver translation의 평균을 target train 평균으로 옮겨 camera depth 차이를 줄인다.
Retargeting은 완전한 motion normalization이 아니며 미관측 표정의 품질은 결과로 확인한다.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

spec=importlib.util.spec_from_file_location("elite_03_entry",Path(__file__).with_name("03_inspect_tracking.py"))
L03=importlib.util.module_from_spec(spec)
spec.loader.exec_module(L03)


def load_avatar(sequence,run,experiment,checkpoint_name,device):
    scene=L03.read_scene(sequence,run)
    path=L03.lesson(8).experiment_dir(scene,experiment)/checkpoint_name
    checkpoint=L03.read_tensor(path)
    if checkpoint["sequence"]!=sequence or checkpoint["tracking_run"]!=run:
        raise ValueError("checkpoint의 identity/tracking run이 다르다.")
    if checkpoint["code_hashes"]!=L03.lesson(8).code_hashes():
        raise ValueError("03~08 code가 checkpoint 생성 이후 바뀌었다. 기존 코드/실험을 보존해 실행하세요.")
    config=checkpoint["config"]
    dataset=L03.lesson(7).VideoFrames(scene,config["image_max_side"])
    if dataset.fingerprint!=checkpoint["data_fingerprint"]:
        raise ValueError("checkpoint와 dataset/UV/head mask가 다르다.")
    uv=L03.lesson(5).read_uv(scene)
    model=L03.lesson(6).Mesh2Gaussian(config).to(device).eval()
    model.load_state_dict(checkpoint["model"],strict=True)
    surface=L03.lesson(5).SurfaceMap(scene,uv,device)
    return scene,checkpoint,model,surface,dataset


def make_video(directory,pattern,destination,fps):
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("PNG는 저장됐다. mp4 작성에는 FFmpeg가 필요하다.")
    # Image sequence의 입력 framerate를 지정하므로 01의 -fps_mode에 의존하지 않는다.
    # Pixel size가 홀수이면 encode에만 우측/하단 1px padding. 학습 camera는 바꾸지 않는다.
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-n",
        "-framerate",str(fps),"-start_number","0","-i",str(directory/pattern),
        "-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2","-c:v","libx264","-crf","18",
        "-pix_fmt","yuv420p",str(destination)],check=True)


def dense_map(values,indices,size):
    output=values.new_zeros(size*size,values.shape[-1])
    output[indices]=values
    return output.reshape(size,size,-1).permute(2,0,1)


def save_uv_maps(out,gaussians,index=0):
    """실제 값은 tensor로 저장한다. Displacement 시각화는 0을 회색으로 표시한다."""
    indices,size=gaussians["indices"],gaussians["uv_size"]
    for key in ("color","coarse","fine","displacement","scale","opacity","position","rotation"):
        value=dense_map(gaussians[key][index],indices,size)
        L03.save_tensor(out/f"{key}.pt",value.cpu())
        if key=="color" or key=="opacity":image=value
        elif key in ("coarse","fine","displacement"):image=.5+value/(.6 if key!="fine" else .2)
        elif key=="scale":image=torch.cat((value,value[:1]),0)/.003
        else:continue
        L03.save_image(out/f"{key}.png",image)


def retarget_driver(target,driver,surface):
    if driver.get("version")!=target["version"] or driver.get("vhap_revision")!=target["vhap_revision"]:
        raise ValueError("동일한 lesson03/VHAP format의 driver scene이 필요하다.")
    if not torch.equal(driver["faces"],target["faces"]):
        raise ValueError("driver의 teeth/mesh topology가 다르다.")
    p=dict(target["parameters"])
    for key in ("expr","rotation","translation","neck_pose","jaw_pose","eyes_pose"):
        value=driver["parameters"][key].clone()
        if not torch.isfinite(value).all():
            raise ValueError(f"driver {key} NaN/Inf")
        p[key]=value
    target_mean=target["parameters"]["translation"][target["split"]["train"]].mean(0)
    driver_mean=p["translation"].mean(0)
    p["translation"]=p["translation"]-driver_mean+target_mean
    driving=dict(target,parameters=p,manifest=driver["manifest"])
    for key in ("expr","rotation","translation","neck_pose","jaw_pose","eyes_pose"):
        surface.geometry.params[key]=p[key].to(surface.geometry.device)
    return driving,{"driver_mean_translation":driver_mean.tolist(),"target_mean_translation":target_mean.tolist(),
                   "identity":"target UV/encoding only; driver shape/static_offset not used"}


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    L03.add_arguments(parser)
    parser.add_argument("--experiment",default="avatar")
    parser.add_argument("--checkpoint",choices=("best.pt","latest.pt"),default="best.pt")
    parser.add_argument("--driver-scene",type=Path)
    parser.add_argument("--name",default="reconstruction",help="09 출력을 구분하는 이름")
    parser.add_argument("--max-frames",type=int,default=0,help="0이면 전체; 처음 확인할 때만 제한")
    parser.add_argument("--save-aux-every",type=int,default=25,help="alpha/normal/float depth/UV dump 간격")
    args=parser.parse_args()
    if args.max_frames<0 or args.save_aux_every<1:
        parser.error("max-frames>=0, save-aux-every>=1 필요")
    device=torch.device(args.device)
    torch.cuda.set_device(device)
    scene,checkpoint,model,surface,dataset=load_avatar(args.sequence,args.run,args.experiment,args.checkpoint,device)
    # 이름은 experiment_dir와 같은 규칙으로 검증한다.
    L03.lesson(8).experiment_dir(scene,args.name)
    out=L03.output_dir(9,args.sequence,args.run)/args.experiment/args.name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("09 출력이 이미 있다. --name으로 새 폴더를 지정하세요.")
    out.mkdir(parents=True,exist_ok=True)
    driving=scene
    driver_info=None
    if args.driver_scene:
        driver=L03.read_tensor(args.driver_scene)
        driving,driver_info=retarget_driver(scene,driver,surface)
    count=driving["manifest"]["num_frames"]
    if args.max_frames:
        count=min(count,args.max_frames)
    background=torch.ones(3,device=device)
    with torch.no_grad():
        for i in range(count):
            gaussians=L03.lesson(8).forward_frame_batch(model,surface,driving,[i],device)
            rendered=L03.lesson(8).render_batch(gaussians,dataset,device,background)[0]
            L03.save_image(out/"rgb"/f"{i:06d}.png",rendered["rgb"])
            if not args.driver_scene:
                target=dataset.batch([i],device)
                expected=L03.lesson(7).composite(target["rgb"],target["alpha"],background)[0]
                comparison=torch.cat((expected,rendered["rgb"],(expected-rendered["rgb"]).abs()*4),-1)
                L03.save_image(out/"comparison"/f"{i:06d}.png",comparison)
            if i%args.save_aux_every==0:
                L03.save_image(out/"alpha"/f"{i:06d}.png",rendered["alpha"])
                normal=torch.nn.functional.normalize(rendered["normal"],dim=0)
                L03.save_image(out/"normal"/f"{i:06d}.png",normal*.5+.5)
                # Depth는 표시용 변환 없이 float32로 저장한다.
                depth_dir=out/"depth"
                depth_dir.mkdir(exist_ok=True)
                np.save(depth_dir/f"{i:06d}.npy",rendered["depth"].cpu().numpy())
                save_uv_maps(out/"uv"/f"{i:06d}",gaussians)
            if i%25==0:
                print(f"09 frame {i}/{count}",flush=True)
    fps=driving["manifest"]["fps"]
    make_video(out/"rgb","%06d.png",out/"rgb.mp4",fps)
    if not args.driver_scene:
        make_video(out/"comparison","%06d.png",out/"comparison.mp4",fps)
    L03.save_json(out/"render.json",{"checkpoint":args.checkpoint,"checkpoint_step":checkpoint["step"],
        "frames":count,"fps":fps,"driver_scene":str(args.driver_scene) if args.driver_scene else None,
        "retargeting":driver_info,"columns":"reference | rendered | absolute error x4" if not args.driver_scene else "target identity + driver motion"})
    print("09 완료:",out)


if __name__=="__main__":
    main()
