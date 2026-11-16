"""07 — 관측 RGB/alpha를 학습 target으로 만들고 각 loss의 의미를 확인한다.

RGB는 scene-lighting을 포함한다. RVM alpha * 03 head cutoff를 foreground로 사용한다.
UV 입력은 한 identity에 공유되지만 driving parameter와 RGB target은 frame마다 다르다.
Loss: composited RGB L1, SSIM, LPIPS, alpha, 2DGS normal/distortion, UV regularization.
UV texture가 보이지 않는 영역은 color supervision으로 쓰지 않는다.

python 07_dataset_and_losses.py
CPU에서 split/입력 무결성, 실제 target을 확인한다. LPIPS weight는 08 실행 시 준비된다.
"""

import argparse
import importlib.util
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

spec=importlib.util.spec_from_file_location("elite_03_entry",Path(__file__).with_name("03_inspect_tracking.py"))
L03=importlib.util.module_from_spec(spec)
spec.loader.exec_module(L03)


class VideoFrames:
    """CPU 이미지 읽기를 담당한다. FLAME과 CUDA 계산은 학습 loop에서 수행한다."""
    def __init__(self,scene,max_side=1024,verify=True):
        self.scene=scene
        self.sequence=Path(scene["paths"]["sequence"])
        self.keep_dir=L03.output_dir(3,scene["sequence"],scene["run"])/"head_masks"
        h,w=scene["manifest"]["height"],scene["manifest"]["width"]
        ratio=min(1.,max_side/max(h,w)) if max_side>0 else 1.
        self.h,self.w=max(2,round(h*ratio)),max(2,round(w*ratio))
        self.K=scene["K"].clone()
        # 정수 반올림을 반영해 fx/cx와 fy/cy를 각각 변환한다.
        self.K[0]*=self.w/w
        self.K[1]*=self.h/h
        self.w2c=scene["w2c"]
        self.head_hashes=[]
        hash_file=L03.lesson(2).sha256
        for i,record in enumerate(scene["manifest"]["frames"]):
            paths=[self.sequence/record["image"],self.sequence/record["alpha_png"],self.keep_dir/f"{i:06d}.png"]
            if not all(p.is_file() for p in paths):
                raise FileNotFoundError(f"frame {i} RGB/alpha/head mask 누락")
            if verify:
                if hash_file(paths[0])!=scene["image_hashes"][i] or hash_file(paths[1])!=scene["alpha_hashes"][i]:
                    raise ValueError(f"03 이후 입력 frame {i}가 바뀌었다.")
            self.head_hashes.append(hash_file(paths[2]))
        self.fingerprint={"scene":hash_file(L03.scene_path(scene["sequence"],scene["run"])),
                          "uv":hash_file(L03.output_dir(4,scene["sequence"],scene["run"])/"uv.pt"),
                          "head_masks":self.head_hashes,"image_height":self.h,"image_width":self.w}

    def frame(self,index):
        record=self.scene["manifest"]["frames"][index]
        def read(path,mode):
            with Image.open(path) as image:
                array=np.array(image.convert(mode),copy=True)
            tensor=torch.from_numpy(array).float()/255
            return tensor.permute(2,0,1) if mode=="RGB" else tensor[None]
        rgb=read(self.sequence/record["image"],"RGB")
        alpha=read(self.sequence/record["alpha_png"],"L")
        keep=read(self.keep_dir/f"{index:06d}.png","L")
        expected=(self.scene["manifest"]["height"],self.scene["manifest"]["width"])
        if rgb.shape[-2:]!=expected or alpha.shape[-2:]!=expected or keep.shape[-2:]!=expected:
            raise ValueError(f"frame {index}: 이미지 크기가 manifest와 다르다.")
        if rgb.shape[-2:]!=(self.h,self.w):
            rgb=F.interpolate(rgb[None],size=(self.h,self.w),mode="bilinear",align_corners=False,antialias=True)[0]
            alpha=F.interpolate((alpha*keep)[None],size=(self.h,self.w),mode="bilinear",align_corners=False,antialias=True)[0]
        else:alpha=alpha*keep
        return {"frame_id":index,"rgb":rgb,"alpha":alpha.clamp(0,1)}

    def batch(self,ids,device):
        rows=[self.frame(int(i)) for i in ids]
        return {"rgb":torch.stack([r["rgb"] for r in rows]).to(device),
                "alpha":torch.stack([r["alpha"] for r in rows]).to(device)}


def composite(rgb,alpha,background):
    # RGB는 straight color다. Renderer 출력에는 배경이 합성됐으므로 alpha를 다시 곱하지 않는다.
    return rgb*alpha+background[None,:,None,None]*(1-alpha)


def ssim(prediction,target):
    """11x11 Gaussian local SSIM, 입력 [0,1]. 학습과 평가에 같은 정의를 사용한다."""
    x= torch.arange(11,device=prediction.device,dtype=prediction.dtype)-5
    kernel=torch.exp(-x.square()/(2*1.5**2))
    kernel/=kernel.sum()
    window=(kernel[:,None]*kernel[None,:])[None,None].expand(3,1,11,11)
    def blur(value):
        return F.conv2d(value,window,padding=5,groups=3)
    mu1,mu2=blur(prediction),blur(target)
    variance1=blur(prediction.square())-mu1.square()
    variance2=blur(target.square())-mu2.square()
    covariance=blur(prediction*target)-mu1*mu2
    numerator=(2*mu1*mu2+.01**2)*(2*covariance+.03**2)
    denominator=(mu1.square()+mu2.square()+.01**2)*(variance1+variance2+.03**2)
    return (numerator/denominator.clamp_min(1e-8)).mean()


def uv_tv(value,valid,face):
    """UV seam을 가로질러 정규화하지 않는다. 인접 face 관계는 별도 확장할 수 있다.

    여기서는 같은 face의 인접 texel 사이에만 TV를 적용한다. 배경과 다른 island는 제외한다.
    """
    horizontal=valid[:,:-1]&valid[:,1:]&(face[:,:-1]==face[:,1:])
    vertical=valid[:-1]&valid[1:]&(face[:-1]==face[1:])
    dx=(value[:,:,:,1:]-value[:,:,:,:-1]).abs()*horizontal[None,None]
    dy=(value[:,:,1:]-value[:,:,:-1]).abs()*vertical[None,None]
    count=(horizontal.sum()+vertical.sum()).clamp_min(1)*value.shape[0]*value.shape[1]
    return (dx.sum()+dy.sum())/count


class AvatarLoss(nn.Module):
    def __init__(self,config,uv,device):
        super().__init__()
        self.config=config
        self.weights=config["loss"]
        self.register_buffer("valid",uv["valid"].to(device))
        self.register_buffer("face",uv["face"].to(device))
        self.register_buffer("texture",uv["texture"].to(device))
        # 실제 관측 confidence. Nearest fill 영역은 0으로 유지한다.
        weight=uv["observation_weight"]
        self.register_buffer("observed",(weight/weight.max().clamp_min(1e-8)).clamp(0,1).to(device))
        self.perceptual=None
        if self.weights["lpips"]>0:
            import lpips
            self.perceptual=lpips.LPIPS(net="vgg").to(device).eval().requires_grad_(False)

    def perceptual_loss(self,prediction,target):
        if self.perceptual is None:
            return prediction.sum()*0
        side=self.weights["lpips_max_side"]
        ratio=min(1.,side/max(prediction.shape[-2:]))
        if ratio<1:
            size=tuple(max(16,round(x*ratio)) for x in prediction.shape[-2:])
            prediction=F.interpolate(prediction,size=size,mode="bilinear",align_corners=False,antialias=True)
            target=F.interpolate(target,size=size,mode="bilinear",align_corners=False,antialias=True)
        # Frozen weight여도 prediction으로 gradient가 흘러야 하므로 no_grad를 쓰지 않는다.
        return self.perceptual(prediction*2-1,target*2-1).mean()

    def forward(self,rendered,target,gaussians,background,step):
        predicted=torch.stack([r["rgb"] for r in rendered])
        alpha=torch.stack([r["alpha"] for r in rendered])
        expected=composite(target["rgb"],target["alpha"],background)
        normal=torch.stack([r["normal"] for r in rendered])
        # Depth에서 만든 surface normal은 pseudo target이다. 초기 geometry loss는 천천히 켠다.
        surface=torch.stack([r["surface_normal"] for r in rendered]).detach()
        normal_unit=F.normalize(normal,dim=1,eps=1e-6)
        surface_unit=F.normalize(surface,dim=1,eps=1e-6)
        valid_surface=(surface.norm(dim=1,keepdim=True)>.5).float()*target["alpha"]*alpha.detach()
        normal_error=(1-(normal_unit*surface_unit).sum(1,keepdim=True)).clamp(0,2)
        b,_,u,_=gaussians["geometry_map"].shape
        disp=gaussians["geometry_map"][:,:3].tanh()*.2+gaussians["geometry_map"][:,9:12].tanh()*.1
        color=(gaussians["appearance_map"]+.5).clamp(0,1)
        uv_error=(color-self.texture[None]).abs()*self.observed[None,None]
        terms={"photo":F.l1_loss(predicted,expected),"alpha":F.l1_loss(alpha,target["alpha"]),
               "ssim":1-ssim(predicted,expected),"lpips":self.perceptual_loss(predicted,expected),
               "normal":(normal_error*valid_surface).sum()/valid_surface.sum().clamp_min(1),
               "distortion":torch.stack([r["distortion"] for r in rendered]).mean(),
               "tv_color":uv_tv(color,self.valid,self.face),
               "tv_displacement":uv_tv(disp,self.valid,self.face),
               "displacement_l2":gaussians["displacement"].square().mean(),
               "uv_color":uv_error.sum()/(self.observed.sum().clamp_min(1)*b*3)}
        ramp=min(1.,max(0.,(step-self.weights["geometry_start"])/max(1,self.weights["geometry_ramp"])))
        weighted={name:value*self.weights[name]*(ramp if name in ("normal","distortion") else 1.) for name,value in terms.items()}
        total=sum(weighted.values())
        return total,terms,weighted


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    L03.add_arguments(parser)
    parser.add_argument("--config",type=Path)
    args=parser.parse_args()
    scene=L03.read_scene(args.sequence,args.run)
    config=L03.lesson(6).load_config(args.config)
    dataset=VideoFrames(scene,config["image_max_side"])
    out=L03.output_dir(7,args.sequence,args.run)
    images=[]
    for name in ("train","validation"):
        i=scene["split"][name][0]
        row=dataset.frame(i)
        target=composite(row["rgb"][None],row["alpha"][None],torch.ones(3))[0]
        L03.save_image(out/f"{name}_{i:06d}_target.png",target)
        L03.save_image(out/f"{name}_{i:06d}_head_alpha.png",row["alpha"])
        images.append(target)
    L03.save_json(out/"dataset.json",{"fingerprint":dataset.fingerprint,"split":scene["split"],
        "K":dataset.K.tolist(),"w2c_opengl":dataset.w2c.tolist(),"loss_weights":config["loss"],
        "evaluation_scope":"held-out Gaussian RGB; all-frame tracking shared"})
    print("07 완료:",out,"| train",len(scene["split"]["train"]),"validation",len(scene["split"]["validation"]))
    print("Target은 white background + 실제 head alpha. Head/neck cutoff를 확인하세요.")


if __name__=="__main__":
    main()
