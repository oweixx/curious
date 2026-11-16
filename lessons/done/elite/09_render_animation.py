"""09 — Native 학습 checkpoint로 sequence/다른 driving scene을 실제 렌더링한다.

CUDA_VISIBLE_DEVICES=2 python 09_render_animation.py --experiment avatar
CUDA_VISIBLE_DEVICES=2 python 09_render_animation.py --experiment avatar --driver-scene /다른/03/scene.pt
다른 driver의 shape/appearance는 사용하지 않는다. Target identity의 UV를 유지한다.
Driver translation의 평균을 target train 평균으로 옮겨 camera depth 차이를 줄인다.
Retargeting은 완전한 motion normalization이 아니며 미관측 표정의 품질은 결과로 확인한다.

읽는 순서: main -> UVVideoPanels -> depth_image -> make_video.
모든 frame을 RGB/alpha/depth/normal/UV 영상으로 저장한다. UV 주소는 고정되고 값이 바뀐다.
Raw 16채널은 표시할 때만 tanh를 적용한다. 이 표시값을 SurfaceMap.decode에 넣지 않는다.
Float depth와 UV tensor dump만 --save-aux-every 간격으로 저장한다.
"""

import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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
        "-vf","pad=ceil(iw/2)*2:ceil(ih/2)*2","-filter_threads","1",
        "-c:v","libx264","-threads","4","-crf","18",
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
    # 학습 출력도 보존한다. 위의 8개 속성은 decode 이후의 값이며 raw 13+3채널과 다르다.
    L03.save_tensor(out/"network_output.pt",{
        "geometry_map":gaussians["geometry_map"][index].cpu(),
        "appearance_map":gaussians["appearance_map"][index].cpu(),
        "valid":dense_map(torch.ones(len(indices),1,device=indices.device),indices,size)[0].bool().cpu(),
    })


def position_display_range(surface,count):
    """전체 frame의 posed base mesh에서 한 번만 구한 XYZ 표시 범위.

    Network를 두 번 실행할 필요 없이 FLAME만 작은 batch로 평가한다.
    3cm 여유를 추가한다. 범위를 넘는 큰 displacement는 영상에서 clip되지만 tensor는 보존한다.
    """
    lower=torch.full((3,),float("inf"),device=surface.geometry.device)
    upper=-lower
    with torch.no_grad():
        for start in range(0,count,16):
            vertices=surface.geometry.forward(list(range(start,min(start+16,count))))
            lower=torch.minimum(lower,vertices.amin(dim=(0,1)))
            upper=torch.maximum(upper,vertices.amax(dim=(0,1)))
    if not torch.isfinite(torch.cat((lower,upper))).all():
        raise ValueError("FLAME XYZ 표시 범위가 NaN/Inf다.")
    return (lower-.03).cpu(),(upper+.03).cpu()


class UVVideoPanels:
    """UV 격자에 raw 출력과 실제 Gaussian 값을 나란히 배치한다.

    Frame마다 min/max를 다시 맞추면 값이 변해도 같은 색처럼 보일 수 있다.
    여기서는 모든 frame에 같은 변환/범위를 사용한다. 빈 UV texel은 짙은 회색이다.
    """
    RAW_NAMES=("coarse T", "coarse B", "coarse N", "scale axis 1", "scale axis 2",
               "quat x residual", "quat y residual", "quat z residual", "quat w residual",
               "fine T", "fine B", "fine N", "opacity logit",
               "appearance R", "appearance G", "appearance B")

    def __init__(self,surface,count,tile_size):
        self.tile_size=tile_size
        self.lower,self.upper=position_display_range(surface,count)
        try:
            self.font=ImageFont.truetype("DejaVuSans.ttf",14)
        except OSError:
            self.font=ImageFont.load_default()

    def montage(self,images,names,columns,title,valid):
        # 512 UV tensor 자체는 바꾸지 않고, 영상 패널만 작게 표시한다.
        pixels=torch.stack([image.expand(3,-1,-1) for image in images])
        pixels=torch.where(valid[None,None],pixels,pixels.new_tensor(.08))
        pixels=torch.nn.functional.interpolate(pixels,size=(self.tile_size,self.tile_size),mode="nearest")
        pixels=(pixels.clamp(0,1).permute(0,2,3,1).numpy()*255).round().astype(np.uint8)
        header,label=32,26
        panel_h=self.tile_size+label
        canvas=Image.new("RGB",(columns*self.tile_size,header+((len(images)+columns-1)//columns)*panel_h),(20,20,20))
        draw=ImageDraw.Draw(canvas)
        draw.text((8,8),title,font=self.font,fill="white")
        for i,(array,name) in enumerate(zip(pixels,names)):
            x,y=(i%columns)*self.tile_size,header+(i//columns)*panel_h
            draw.text((x+5,y+5),name,font=self.font,fill="white")
            canvas.paste(Image.fromarray(array),(x,y+label))
        return canvas

    def save(self,out,gaussians,frame_id):
        indices=gaussians["indices"].cpu()
        size=gaussians["uv_size"]
        valid=torch.zeros(size*size,dtype=torch.bool)
        valid[indices]=True
        valid=valid.reshape(size,size)
        raw=torch.cat((gaussians["geometry_map"][0],gaussians["appearance_map"][0])).cpu()
        if not torch.isfinite(raw[:,valid]).all():
            raise ValueError(f"frame {frame_id}: network UV map NaN/Inf")
        # 고정된 부호색: 음수=파랑, 0=흰색, 양수=빨강. tanh는 표시용 압축이다.
        shown=raw.tanh()
        raw_images=[torch.stack((1+value.clamp(max=0),1-value.abs(),1-value.clamp(min=0))) for value in shown]
        raw_panel=self.montage(raw_images,self.RAW_NAMES,4,
            f"Frame {frame_id} | raw 13+3: tanh display, blue(-) white(0) red(+)",valid)

        maps={key:dense_map(gaussians[key][0].cpu(),indices,size) for key in
              ("color","position","displacement","scale","opacity","fine")}
        q=gaussians["rotation"][0].cpu()
        # SurfaceMap의 world quaternion은 wxyz, quaternion_matrix 입력은 xyzw.
        normals=L03.lesson(5).quaternion_matrix(q[:,[1,2,3,0]])[:,:,2]
        normal_map=dense_map(normals,indices,size)
        position=(maps["position"]-self.lower[:,None,None])/(self.upper-self.lower)[:,None,None]
        scale_max=.1*np.exp(-3.5)  # 05 scale 식의 상한, FLAME world unit.
        decoded_images=[maps["color"],position,.5+maps["displacement"]/.6,
                        .5+normal_map*.5,maps["scale"][:1]/scale_max,
                        maps["scale"][1:]/scale_max,maps["opacity"],.5+maps["fine"]/.2]
        names=("Color RGB [0,1]","World XYZ (fixed range)","Local TBN delta [-0.3,0.3]",
               "World normal [-1,1]","Scale axis 1 [0,0.00302]","Scale axis 2 [0,0.00302]",
               "Opacity [0,1]","Fine TBN delta [-0.1,0.1]")
        decoded_panel=self.montage(decoded_images,names,4,
            f"Frame {frame_id} | decoded Gaussian UV | fixed display ranges",valid)
        for key,panel in (("gaussian_uv_raw",raw_panel),("gaussian_uv_decoded",decoded_panel)):
            directory=out/key
            directory.mkdir(exist_ok=True)
            panel.save(directory/f"{frame_id:06d}.png")

    def metadata(self):
        return {"raw_channels":list(self.RAW_NAMES),"raw_display":"tanh(raw): blue=-1, white=0, red=+1; display only",
                "position_min":self.lower.tolist(),"position_max":self.upper.tolist(),
                "position_range":"all-frame posed base mesh bounds + 0.03; out-of-range decoded positions display-clipped",
                "local_displacement_range":[-.3,.3],"fine_displacement_range":[-.1,.1],
                "scale_range":[0.,float(.1*np.exp(-3.5))],"tile_size":self.tile_size,
                "invalid_uv":"dark gray","normal_space":"world","uv_addresses":"fixed across frames"}


def cache_depth(path,rendered):
    """임시 depth cache는 float32. 배경/불확실한 edge만 NaN으로 표시한다.

    Renderer 원본 depth는 main에서 따로 보존한다. Cache는 영상 생성 후 자동 삭제한다.
    각 frame의 foreground 1~99 percentile 범위를 모아 영상 전체의 고정 범위를 만든다.
    """
    depth=rendered["depth"].detach().cpu().numpy()
    alpha=rendered["alpha"].detach().cpu().numpy()
    valid=np.isfinite(depth)&(depth>0)&(alpha>.05)
    values=depth[valid]
    np.save(path,np.where(valid,depth,np.nan).astype(np.float32))
    return np.quantile(values,[.01,.99]) if values.size else None


def depth_image(depth,near,far):
    """Camera depth: 가까움=밝음, 멂=어두움. 배경=검정.

    표시 범위만 clip한다. 저장된 raw .npy의 실제 거리에는 정규화를 적용하지 않는다.
    """
    valid=np.isfinite(depth)
    gray=.15+.85*(1-np.clip((np.nan_to_num(depth,nan=far)-near)/(far-near),0,1))
    return torch.from_numpy(np.where(valid,gray,0).astype(np.float32))


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
    parser.add_argument("--save-aux-every",type=int,default=25,help="float depth/UV tensor dump 간격; 영상은 항상 모든 frame")
    parser.add_argument("--depth-range",type=float,nargs=2,metavar=("NEAR","FAR"),
                        help="depth 영상의 고정 거리 범위; 기본은 전체 frame foreground의 1~99%% 범위를 모아 결정")
    parser.add_argument("--uv-video-size",type=int,default=256,help="UV 영상의 패널 하나 크기; 실제 UV 해상도는 유지")
    args=parser.parse_args()
    if args.max_frames<0 or args.save_aux_every<1:
        parser.error("max-frames>=0, save-aux-every>=1 필요")
    if not 64<=args.uv_video_size<=512:
        parser.error("uv-video-size는 64~512 필요")
    if args.depth_range and (not np.isfinite(args.depth_range).all() or not 0<args.depth_range[0]<args.depth_range[1]):
        parser.error("depth-range는 유한한 0<NEAR<FAR 필요")
    if shutil.which("ffmpeg") is None:
        parser.error("영상 생성에 FFmpeg가 필요하다.")
    # 많은 CPU core가 있는 서버에서도 작은 UV 표시 연산에 수십 thread를 동원하지 않는다.
    # OMP_NUM_THREADS로 더 작은 값을 지정했다면 그 설정을 유지한다. 08 학습에는 적용하지 않는다.
    torch.set_num_threads(min(torch.get_num_threads(),8))
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
    if count<1:
        raise ValueError("렌더링할 frame이 없다.")
    background=torch.ones(3,device=device)
    uv_panels=UVVideoPanels(surface,count,args.uv_video_size)
    # Depth 범위를 전체 영상에서 결정하기 위해 depth만 잠깐 disk에 보관한다.
    # RGB/UV/normal은 한 번만 계산한다. 재추론이나 모든 frame의 GPU tensor 보관은 필요 없다.
    depth_lower,depth_upper=float("inf"),-float("inf")
    with tempfile.TemporaryDirectory(prefix=".depth_cache_",dir=out) as temporary:
        cache=Path(temporary)
        with torch.no_grad():
            for i in range(count):
                gaussians=L03.lesson(8).forward_frame_batch(model,surface,driving,[i],device)
                rendered=L03.lesson(8).render_batch(gaussians,dataset,device,background)[0]
                L03.save_image(out/"rgb"/f"{i:06d}.png",rendered["rgb"])
                L03.save_image(out/"alpha"/f"{i:06d}.png",rendered["alpha"])
                normal=torch.nn.functional.normalize(rendered["normal"],dim=0)
                # World normal을 RGB로 표시한다. 배경을 방향 벡터로 오해하지 않게 검정 처리한다.
                normal_image=torch.where(rendered["alpha"]>.05,normal*.5+.5,torch.zeros_like(normal))
                L03.save_image(out/"normal"/f"{i:06d}.png",normal_image)
                uv_panels.save(out,gaussians,i)
                limits=cache_depth(cache/f"{i:06d}.npy",rendered)
                if limits is not None:
                    depth_lower=min(depth_lower,float(limits[0]))
                    depth_upper=max(depth_upper,float(limits[1]))
                if not args.driver_scene:
                    target=dataset.batch([i],device)
                    expected=L03.lesson(7).composite(target["rgb"],target["alpha"],background)[0]
                    comparison=torch.cat((expected,rendered["rgb"],(expected-rendered["rgb"]).abs()*4),-1)
                    L03.save_image(out/"comparison"/f"{i:06d}.png",comparison)
                if i%args.save_aux_every==0:
                    # 원본 depth: 배경과 alpha가 작은 pixel을 포함하며 표시용 masking/clip이 없다.
                    depth_dir=out/"depth"
                    depth_dir.mkdir(exist_ok=True)
                    np.save(depth_dir/f"{i:06d}.npy",rendered["depth"].cpu().numpy())
                    save_uv_maps(out/"uv"/f"{i:06d}",gaussians)
                if i%25==0 or i==count-1:
                    print(f"09 frame {i+1}/{count}",flush=True)
        if args.depth_range:
            depth_lower,depth_upper=args.depth_range
        elif not np.isfinite([depth_lower,depth_upper]).all():
            # 모든 frame에 foreground가 없으면 depth 영상은 검정이고 범위는 정의되지 않는다.
            depth_lower,depth_upper=0.,1.
        elif depth_upper-depth_lower<1e-6:
            depth_upper=depth_lower+1e-6
        print(f"09 depth 고정 표시 범위: {depth_lower:.6g} ~ {depth_upper:.6g}",flush=True)
        for i in range(count):
            depth=np.load(cache/f"{i:06d}.npy",allow_pickle=False)
            L03.save_image(out/"depth"/f"{i:06d}.png",depth_image(depth,depth_lower,depth_upper))
    fps=driving["manifest"]["fps"]
    videos=["rgb","alpha","depth","normal","gaussian_uv_raw","gaussian_uv_decoded"]
    if not args.driver_scene:
        videos.append("comparison")
    for key in videos:
        print(f"09 video: {key}.mp4",flush=True)
        make_video(out/key,"%06d.png",out/f"{key}.mp4",fps)
    L03.save_json(out/"render.json",{"checkpoint":args.checkpoint,"checkpoint_step":checkpoint["step"],
        "frames":count,"fps":fps,"driver_scene":str(args.driver_scene) if args.driver_scene else None,
        "videos":[f"{key}.mp4" for key in videos],"save_aux_every":args.save_aux_every,
        "uv_display":uv_panels.metadata(),"normal_space":"world; RGB=(normalize(normal)+1)/2; alpha<=0.05 is black",
        "depth_display":{"range":[depth_lower,depth_upper],"quantity":"median camera depth; FLAME world units",
                         "range_source":"manual" if args.depth_range else "envelope of each frame foreground 1st/99th percentile",
                         "foreground":"finite depth>0 and alpha>0.05","brightness":"near=1, far=0.15, background=0"},
        "retargeting":driver_info,"columns":"reference | rendered | absolute error x4" if not args.driver_scene else "target identity + driver motion"})
    print("09 완료:",out)


if __name__=="__main__":
    main()
