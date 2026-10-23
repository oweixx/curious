"""04 — Identity hypernetwork + driving decoder로 실제 3D Gaussian avatar를 학습한다.

읽는 순서: IdentityEncoder -> GaussianDecoder -> UniversalPrior -> decode -> render.
Identity bias pyramid는 매 decoder layer에 더한다(FiCA 3.3 / URAvatar Fig.3).
Private expression encoder 대신 FLAME100->256 adapter를 학습한다. Geometry branch는
표정의 guide mesh를 학습하며 GT posed mesh를 Gaussian anchor로 몰래 넣지 않는다.
Guide mesh supervision/normal/Laplacian을 사용하여 RGB만 맞추는 퇴화를 점검한다.

CUDA_VISIBLE_DEVICES=0 python 04_gaussian_avatar.py --stage train
CUDA_VISIBLE_DEVICES=0 python 04_gaussian_avatar.py --stage train --val-every 100 --val-frames 1
CUDA_VISIBLE_DEVICES=0 python 04_gaussian_avatar.py --stage render --identity-source refined
FiCA의 RGB UPM을 대상으로 하며 URAvatar의 조명/반사/별도 안구 모듈을 주장하지 않는다.
"""

import importlib.util
import math
from pathlib import Path
import subprocess
import time

from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

spec = importlib.util.spec_from_file_location("fica_00", Path(__file__).with_name("00_prepare_nersemble.py"))
L00 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L00)


class IdentityEncoder(nn.Module):
    """UV texture+geometry -> decoder의 각 해상도에서 쓸 untied bias maps.

    U-Net skip은 identity encoder 내부에 있다. ID bias를 1D embedding 하나로 대체하지 않는다.
    공개 안 된 채널 수는 config에서 명시한다. 해상도 4->U의 모든 단계를 유지한다.
    """
    def __init__(self, channels):
        super().__init__()
        widths = list(reversed(channels))
        self.stem = nn.Conv2d(6, widths[0], 3, padding=1)
        self.down = nn.ModuleList([nn.Sequential(nn.Conv2d(a,b,4,2,1),nn.LeakyReLU(.2)) for a,b in zip(widths,widths[1:])])
        self.up = nn.ModuleList()
        self.fuse = nn.ModuleList()
        incoming = widths[-1]
        for width in channels:
            self.up.append(nn.ConvTranspose2d(incoming, width, 4, 2, 1))
            self.fuse.append(nn.Conv2d(width*2, width, 3, padding=1))
            incoming = width
        self.bias_geo = nn.ModuleList([nn.Conv2d(c,c,1) for c in channels])
        self.bias_app = nn.ModuleList([nn.Conv2d(c,c,1) for c in channels])

    def forward(self, uv):
        features = [F.leaky_relu(self.stem(uv), .2)]
        for block in self.down:
            features.append(block(features[-1]))
        current = F.adaptive_avg_pool2d(features[-1], (4,4))
        pyramid = []
        for up, fuse, skip in zip(self.up, self.fuse, reversed(features)):
            current = F.leaky_relu(up(current), .2)
            skip = F.interpolate(skip, size=current.shape[-2:], mode="bilinear", align_corners=False)
            current = F.leaky_relu(fuse(torch.cat((current,skip),1)), .2)
            pyramid.append(current)
        return {"geometry": [layer(x) for layer,x in zip(self.bias_geo,pyramid)],
                "appearance": [layer(x) for layer,x in zip(self.bias_app,pyramid)]}


class GaussianDecoder(nn.Module):
    def __init__(self, channels, driving_dim, output_channels):
        super().__init__()
        self.seed = nn.Sequential(nn.Linear(driving_dim,256),nn.LeakyReLU(.2),nn.Linear(256,channels[0]*4*4))
        self.blocks = nn.ModuleList()
        incoming = channels[0]
        for width in channels:
            self.blocks.append(nn.ConvTranspose2d(incoming,width,4,2,1))
            incoming = width
        self.output = nn.Conv2d(channels[-1],output_channels,1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.seed_channels = channels[0]

    def forward(self, driving, biases):
        x = self.seed(driving).reshape(-1,self.seed_channels,4,4)
        for block,bias in zip(self.blocks,biases):
            x = F.leaky_relu(block(x)+bias,.2)
        return self.output(x)


class UniversalPrior(nn.Module):
    def __init__(self, config):
        super().__init__()
        settings = config["upm"]
        channels, expression_dim = settings["channels"], settings["expression_dim"]
        if 4*2**len(channels) != config["uv_size"]:
            raise ValueError("UPM channels 수는 4->UV size의 upsample 수여야 한다.")
        self.identity_encoder = IdentityEncoder(channels)
        self.expression_adapter = nn.Sequential(nn.Linear(100,256),nn.LeakyReLU(.2),nn.Linear(256,expression_dim))
        # Neck/gaze는 geometry, view는 appearance에만 전달한다.
        self.geometry_decoder = GaussianDecoder(channels, expression_dim+3+6,13)
        self.appearance_decoder = GaussianDecoder(channels, expression_dim+3+6+3,4)
        self.max_offset = settings["max_offset"]

    def encode_identity(self, texture, xyz, mean, std):
        return self.identity_encoder(torch.cat((texture*2-1,(xyz-mean)/std),1))

    def forward(self, texture, xyz, mean, std, expression, neck, gaze, view, biases=None):
        biases = self.encode_identity(texture,xyz,mean,std) if biases is None else biases
        e = self.expression_adapter(expression)
        driving = torch.cat((e,neck,gaze),-1)
        geometry = self.geometry_decoder(driving,biases["geometry"])
        appearance = self.appearance_decoder(torch.cat((driving,view),-1),biases["appearance"])
        # Guide offsets와 Gaussian residual을 분리해 guide를 직접 감독한다.
        return {"guide":xyz+.2*geometry[:,:3].tanh(), "offset":self.max_offset*geometry[:,3:6].tanh(),
                "quaternion_residual":geometry[:,6:10], "scale_raw":geometry[:,10:13],
                "color_offset":appearance[:,:3], "opacity_logit":appearance[:,3:4], "biases":biases}


def driving(sample):
    q = L00.elite(5).axis_angle_to_quaternion(sample["eyes_pose"].reshape(2,3))
    rotations = L00.elite(5).quaternion_matrix(q)
    gaze = rotations[:,:,2].reshape(1,6)
    camera = torch.linalg.inv(sample["w2c"])[:3,3]
    view = F.normalize(camera-sample["global_translation"],dim=0)
    # View direction을 head-local 좌표계로 옮겨 global pose에 대한 appearance를 학습한다.
    view = (view @ sample["global_rotation"])[None]
    return sample["expression"][None],sample["neck_pose"][None],gaze,view


def decode(raw, texture, bridge, sample, config):
    """고정 parameter 해석. 3축 scale, wxyz rotation, sigmoid opacity.

    Gaussians는 learned guide에 배치한다. Camera/root pose만 외부 tracking에서 얻는다.
    """
    l05 = L00.elite(5)
    guide = bridge(raw["guide"][0].float())
    points = bridge.vertices_to_uv(guide).permute(1,2,0).reshape(-1,3)[bridge.indices]
    # UV triangle의 T/B/N. Canonical guide의 deformation에 따라 방향이 갱신된다.
    frames, bad = L00.elite(4).face_tbn(guide[None],bridge.vt,bridge.faces,bridge.ft)
    basis = frames[0,bridge.face[bridge.valid]]
    def take(value):
        return value[0].float().permute(1,2,0).reshape(-1,value.shape[1])[bridge.indices]
    position = points+(basis@take(raw["offset"])[...,None]).squeeze(-1)
    residual = take(raw["quaternion_residual"])
    identity_q = torch.zeros_like(residual);identity_q[:,3]=1
    local_q = F.normalize(identity_q+residual,dim=-1)
    local_q = torch.where((local_q.norm(dim=-1)<1e-6)[:,None],identity_q,local_q)
    rotation = sample["global_rotation"][None]@basis@l05.quaternion_matrix(local_q)
    scale = (.0006*512/config["uv_size"])*torch.exp(take(raw["scale_raw"]).clamp(-6,3))
    # 초기 normal 방향의 얇은 3D ellipsoid. 2DGS의 2축 scale과 다르다.
    scale = scale*scale.new_tensor([1.,1.,.2])
    color = (take(texture)+take(raw["color_offset"])).clamp(0,1)
    return {"position":position@sample["global_rotation"].T+sample["global_translation"],
            "rotation":l05.matrix_quaternion(rotation),"scale":scale,
            "opacity":take(raw["opacity_logit"]).sigmoid(),"color":color,
            "normal":rotation[:,:,2],"guide_vertices":guide,"offset":take(raw["offset"])}


def render(gaussians, sample, background):
    """Graphdeco의 실제 3DGS backend. 2DGS rasterizer를 3DGS로 부르지 않는다."""
    from diff_gaussian_rasterization import GaussianRasterizer,GaussianRasterizationSettings
    h,w = sample["rgb"].shape[-2:]
    view,projection,center = L00.elite(5).camera_matrices(sample["K"],sample["w2c"],h,w)
    def raster(colors,bg):
        settings=GaussianRasterizationSettings(image_height=h,image_width=w,
            tanfovx=float(w/(2*sample["K"][0,0])),tanfovy=float(h/(2*sample["K"][1,1])),bg=bg,
            scale_modifier=1.,viewmatrix=view,projmatrix=projection,sh_degree=0,campos=center,prefiltered=False,debug=False)
        points=gaussians["position"].contiguous()
        screen=torch.zeros_like(points,requires_grad=torch.is_grad_enabled())
        result=GaussianRasterizer(settings)(means3D=points,means2D=screen,shs=None,
            colors_precomp=colors.contiguous(),opacities=gaussians["opacity"].contiguous(),
            scales=gaussians["scale"].contiguous(),rotations=gaussians["rotation"].contiguous(),cov3D_precomp=None)
        return result[0],result[1]
    rgb,radii=raster(gaussians["color"],background)
    alpha,_=raster(torch.ones_like(gaussians["color"]),torch.zeros_like(background))
    normal,_=raster(gaussians["normal"]*.5+.5,torch.zeros_like(background))
    normal=F.normalize(normal/alpha.clamp_min(1e-6)*2-1,dim=0)
    camera=gaussians["position"]@sample["w2c"][:3,:3].T+sample["w2c"][:3,3]
    depth,_=raster((-camera[:,2:3]).expand(-1,3),torch.zeros_like(background))
    depth=depth[:1]/alpha[:1].clamp_min(1e-6)
    return {"rgb":rgb,"alpha":alpha[:1],"normal":normal,"depth":depth,"radii":radii}


def target_rgb(sample):
    """이미 흰 배경으로 합성된 자료에 alpha를 두 번 적용하지 않는다."""
    return sample["rgb"] if sample.get("rgb_composited_white",False) else sample["rgb"]*sample["alpha"]+1-sample["alpha"]


def avatar_loss(config, rendered, gaussians, sample, bridge, perceptual=None):
    settings=config["upm"]
    target_alpha=sample["alpha"]
    expected=target_rgb(sample)
    guide=gaussians["guide_vertices"]
    target=sample["local_vertices"]
    # Guide geometry와 실제 Gaussian 방향을 모두 감독한다. RGB만 좋아지는 퇴화를 점검한다.
    guide_normal=L00.vertex_normals(guide,bridge.faces)
    target_normal=L00.vertex_normals(target,bridge.faces)
    geometry_scale=target.std(dim=0).clamp_min(.01).detach()
    geometry=((guide-target)/geometry_scale).square().mean()
    normal=(1-(guide_normal*target_normal).sum(-1)).mean()
    valid=sample["labels"]["valid"].float()*rendered["alpha"].detach()
    normal=normal+((1-(rendered["normal"]*sample["labels"]["normal_world"]).sum(0,keepdim=True))*valid).sum()/valid.sum().clamp_min(1)
    laplacian=((L00.mesh_laplacian(guide,bridge.faces)-L00.mesh_laplacian(target,bridge.faces))/geometry_scale).square().mean()
    lpips=rendered["rgb"].sum()*0 if perceptual is None else perceptual(rendered["rgb"][None]*2-1,expected[None]*2-1).mean()
    terms={"photo":F.l1_loss(rendered["rgb"],expected),"mask":F.l1_loss(rendered["alpha"],target_alpha),
           "lpips":lpips,"geometry":geometry,"normal":normal,"laplacian":laplacian,"offset":gaussians["offset"].square().mean()}
    return sum(value*settings[name] for name,value in terms.items()),terms


def identity_input(data,row,device,source="target"):
    identity=data.identity(row["identity"],device)
    if source=="target":
        return identity["texture"][None],identity["xyz"][None]
    if source not in ("generated","refined"):
        raise ValueError(source)
    value=L00.load_tensor(data.path/source/f"{Path(row['file']).stem}.pt",device)
    return value["texture"][None],value["xyz"][None]


def validation_setup(data,device,limit=1,sources=("target","refined"),portrait_frame=None):
    """고정 held-out driving frames + train split의 한 portrait를 선택한다.

    Target은 UPM 자체를, refined는 03까지 연결한 조건을 평가한다. UV는 한 번만
    읽어 고정하므로 학습 중 03 export가 바뀌어도 validation 조건은 변하지 않는다.
    """
    if limit < 0 or not sources or len(set(sources))!=len(sources) or any(s not in ("target","generated","refined") for s in sources):
        raise ValueError("val-frames >= 0, 중복 없는 val-sources가 필요하다.")
    rows=data.samples("validation")
    if not rows:
        raise ValueError("validation split이 없다. Train frame으로 대체하지 않는다.")
    rows=rows[:limit] if limit else rows
    portraits={}
    for row in data.samples("train"):
        if row["identity"] not in portraits and (portrait_frame is None or row["frame"]==portrait_frame):
            portraits[row["identity"]]=row
    inputs={source:{} for source in sources}
    for key in dict.fromkeys(row["identity"] for row in rows):
        if key not in portraits:
            raise ValueError(f"identity {key}에 train split의 고정 portrait가 없다.")
        reference=portraits[key]
        for source in sources:
            path=(data.path/"identities"/f"{key}.pt" if source=="target" else
                  data.path/source/f"{Path(reference['file']).stem}.pt")
            if not path.is_file():
                raise FileNotFoundError(f"validation {source} UV가 없다: {path}. --val-sources target으로 UPM만 평가할 수 있다.")
            texture,xyz=identity_input(data,reference,device,source)
            inputs[source][key]={"texture":texture.detach(),"xyz":xyz.detach(),"portrait":reference["file"],
                                "path":str(path),"sha256":L00.sha256(path)}
    return rows,inputs


def validation_manifest(inputs):
    return {source:{key:{name:value for name,value in entry.items() if name not in ("texture","xyz")}
                    for key,entry in group.items()} for source,group in inputs.items()}


def save_validation_panel(path,images,titles,step,row,metrics,manifest,frame_count):
    """0/100/200.jpg 한 장에 모든 조건을 비교한다. 사진 종횡비는 letterbox로 보존한다."""
    side,label,columns=260,28,4
    header=76+len(metrics)*20
    canvas=Image.new("RGB",(columns*side,header+math.ceil(len(images)/columns)*(side+label)),(24,24,24))
    draw=ImageDraw.Draw(canvas)
    draw.text((10,8),f"04 | optimizer step {step} | held-out {Path(row['file']).stem}",fill="white")
    reference=next(iter(manifest.values()))[row["identity"]]["portrait"]
    draw.text((10,27),f"Fixed train portrait: {Path(reference).stem} | {frame_count} validation frame(s)",fill="white")
    draw.text((10,46),"Same first held-out frame is displayed; metrics are means over selected frames.",fill="white")
    for index,(source,values) in enumerate(metrics.items()):
        draw.text((10,66+index*20),f"{source}: loss={values['loss']:.4f} | photo={values['photo']:.5f} | "
                  f"PSNR={values['psnr']:.2f} | mask={values['mask']:.5f} | geometry={values['geometry']:.5f}",fill="white")
    for index,(value,title) in enumerate(zip(images,titles)):
        value=value.detach().float().cpu().expand(3,-1,-1)
        array=(value.clamp(0,1).permute(1,2,0).numpy()*255).round().astype("uint8")
        tile=Image.fromarray(array)
        scale=min(side/tile.width,side/tile.height)
        tile=tile.resize((max(1,round(tile.width*scale)),max(1,round(tile.height*scale))),Image.Resampling.BILINEAR)
        x,y=index%columns*side,header+index//columns*(side+label)
        draw.text((x+5,y+6),title,fill="white")
        canvas.paste(tile,(x+(side-tile.width)//2,y+label+(side-tile.height)//2))
    path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path,quality=95)


@torch.no_grad()
def validate(config,data,model,device,rows,inputs,out,step,perceptual=None):
    """고정 identity UV로 held-out 표정/카메라를 평가한다. 학습 RNG/모드를 복원한다."""
    if not rows:
        raise ValueError("validation frame이 없다.")
    mean,std=L00.lesson(3).stats(data,device)
    totals={source:{} for source in inputs}
    biases={}
    preview_images,preview_titles=[],[]
    depth_range=None
    was_training=model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=[device]):
            for index,row in enumerate(rows):
                sample=data.sample(row,device)
                bridge=data.bridge(row["identity"],device)
                expected=target_rgb(sample)
                if index==0:
                    # 같은 driving frame의 tracked mesh로 표시 범위만 정한다.
                    # 매번 예측 depth의 min/max를 쓰면 학습 중 변화를 비교할 수 없다.
                    camera=sample["posed_vertices"]@sample["w2c"][:3,:3].T+sample["w2c"][:3,3]
                    depth=-camera[:,2]
                    near,far=float(depth.min()),float(depth.max())
                    margin=max((far-near)*.1,.01)
                    depth_range=(near-margin,far+margin)
                for source,group in inputs.items():
                    entry=group[row["identity"]]
                    texture,xyz=entry["texture"],entry["xyz"]
                    token=(source,row["identity"])
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        if token not in biases:
                            biases[token]=model.encode_identity(texture,xyz,mean,std)
                        raw=model(texture,xyz,mean,std,*driving(sample),biases=biases[token])
                    gs=decode(raw,texture,bridge,sample,config)
                    rendered=render(gs,sample,torch.ones(3,device=device))
                    loss,terms=avatar_loss(config,rendered,gs,sample,bridge,perceptual)
                    if not all(torch.isfinite(value).all() for value in (loss,rendered["rgb"],rendered["alpha"],rendered["normal"],rendered["depth"])):
                        raise FloatingPointError(f"04 validation NaN/Inf: {source} {row['file']}")
                    mse=F.mse_loss(rendered["rgb"],expected)
                    values={"loss":float(loss),**{name:float(value) for name,value in terms.items()},
                            "psnr":float(-10*mse.clamp_min(1e-10).log10()),"gaussians":len(gs["position"]),
                            "visible_gaussians":int((rendered["radii"]>0).sum())}
                    for name,value in values.items():
                        totals[source][name]=totals[source].get(name,0.)+value
                    if index==0:
                        error=((rendered["rgb"]-expected).abs().mean(0,keepdim=True)/.25).clamp(0,1)
                        heatmap=torch.cat((error,1-(2*error-1).abs(),1-error),0)
                        alpha=rendered["alpha"].clamp(0,1)
                        normal=(rendered["normal"]*.5+.5)*alpha+1-alpha
                        ratio=((rendered["depth"]-depth_range[0])/(depth_range[1]-depth_range[0])).clamp(0,1)
                        depth_color=torch.cat((ratio,1-(2*ratio-1).abs(),1-ratio),0)*alpha+1-alpha
                        preview_images.extend([expected,rendered["rgb"],heatmap,texture[0],sample["alpha"],alpha,normal,depth_color])
                        preview_titles.extend([f"{source}: reference",f"{source}: Gaussian RGB",f"{source}: error (red >= 0.25)",f"{source}: fixed portrait UV",
                            f"{source}: reference alpha",f"{source}: Gaussian alpha",f"{source}: world normal",f"{source}: depth [{depth_range[0]:.3f}, {depth_range[1]:.3f}]"])
        metrics={source:{name:value/len(rows) for name,value in values.items()} for source,values in totals.items()}
        manifest=validation_manifest(inputs)
        directory=out/"validation"
        save_validation_panel(directory/f"{step}.jpg",preview_images,preview_titles,step,rows[0],metrics,manifest,len(rows))
        L00.append_log(directory/"metrics.jsonl",{"step":step,"metrics":metrics,"frames":[row["file"] for row in rows],
            "preview_frame":rows[0]["file"],"fixed_identity_inputs":manifest,"lpips_enabled":perceptual is not None,
            "depth_display_range":depth_range,"scope":data.manifest["scope"],
            "geometry_target":"tracking-derived FLAME local vertices; supervision/metrics only",
            "photo_psnr_target":"full image with white matting composite; per-frame metrics averaged"})
        return metrics
    finally:
        model.train(was_training)


def train(config,data,device,steps,perceptual=True,resume=False,val_every=100,val_frames=1,
          val_sources=("target","refined"),portrait_frame=None):
    if steps<1 or val_every<1:
        raise ValueError("steps/val-every는 1 이상이어야 한다.")
    rows,inputs=validation_setup(data,device,val_frames,val_sources,portrait_frame)
    out=L00.work_root(config)/"outputs/04"
    path=L00.work_root(config)/"checkpoints/04/latest.pt"
    if path.exists() and not resume and config.get("profile")!="smoke":
        raise FileExistsError(f"{path}가 있다. --resume로 이어서 학습하세요.")
    model=UniversalPrior(config).to(device).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config["upm"]["lr"])
    loss_network=None
    if perceptual:
        import lpips
        loss_network=lpips.LPIPS(net="vgg").to(device).eval().requires_grad_(False)
    mean,std=L00.lesson(3).stats(data,device)
    bridges={}
    start=L00.training_start(path,model,optimizer,config,data,resume)
    if start>=steps:
        raise ValueError(f"이미 {start} step이다. --steps는 추가 횟수가 아닌 총 목표이므로 더 큰 값을 지정하세요.")
    validation_info={"frames":[row["file"] for row in rows],"inputs":validation_manifest(inputs),"lpips_enabled":bool(perceptual)}
    if resume:
        state=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
        if state["extra"].get("validation")!=validation_info:
            raise ValueError("resume 중 validation frame/portrait/UV가 바뀌었다. 같은 조건을 사용하세요.")
        del state
    accumulation=config["training"]["grad_accumulation"]
    with tqdm(total=steps,initial=start,desc="04 Gaussian avatar",unit="step",dynamic_ncols=True,mininterval=1) as bar:
        recent=validate(config,data,model,device,rows,inputs,out,start,loss_network)
        for step in range(start,steps):
            started=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            mean_loss=0.;mean_terms={}
            for micro in range(accumulation):
                row=data.training_row()
                if row["identity"] not in bridges:
                    bridges[row["identity"]]=data.bridge(row["identity"],device)
                bridge=bridges[row["identity"]]
                sample=data.sample(row,device)
                texture,xyz=identity_input(data,row,device,"target")
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    raw=model(texture,xyz,mean,std,*driving(sample))
                gaussians=decode(raw,texture,bridge,sample,config)
                rendered=render(gaussians,sample,torch.ones(3,device=device))
                loss,terms=avatar_loss(config,rendered,gaussians,sample,bridge,loss_network)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"04 training loss NaN/Inf at step {step+1}")
                (loss/accumulation).backward()
                values=torch.stack([loss.detach(),*[value.detach() for value in terms.values()]]).cpu().tolist()
                mean_loss+=values[0]/accumulation
                for name,value in zip(terms,values[1:]):
                    mean_terms[name]=mean_terms.get(name,0.)+value/accumulation
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config["training"]["clip_grad"],error_if_nonfinite=True)
            report={"norm":float(norm),"tensors":sum(p.grad is not None for p in model.parameters())}
            optimizer.step()
            torch.cuda.synchronize(device)
            L00.append_log(out/"training.jsonl",{"step":step+1,"loss":mean_loss,"terms":mean_terms,
                "gradient":report,"step_seconds":time.perf_counter()-started})
            if (step+1)%val_every==0 or step==steps-1:
                recent=validate(config,data,model,device,rows,inputs,out,step+1,loss_network)
            display={"loss":f"{mean_loss:.5f}"}
            display.update({f"val_{source}":f"{values['photo']:.5f}" for source,values in recent.items()})
            bar.set_postfix(display,refresh=False)
            bar.update(1)
            if (step+1)%config["training"]["checkpoint_every"]==0 or step==steps-1:
                L00.checkpoint(path,model,optimizer,config,data,step+1,extra={"validation":validation_info})
    return model.eval()


@torch.no_grad()
def render_sequence(config,data,model,device,source="refined",portrait_frame=None):
    mean,std=L00.lesson(3).stats(data,device)
    bridges={}
    references={}
    for row in data.manifest["samples"]:
        if row["identity"] not in references and (portrait_frame is None or row["frame"]==portrait_frame):
            references[row["identity"]]=row
    current_identity=None
    out=L00.work_root(config)/"outputs/04"/source
    for i,row in enumerate(tqdm(data.manifest["samples"],desc=f"04 Render {source}",unit="frame",dynamic_ncols=True)):
        if row["identity"] not in bridges:
            bridges[row["identity"]]=data.bridge(row["identity"],device)
        sample=data.sample(row,device)
        if row["identity"] not in references:
            raise ValueError(f"identity {row['identity']}에 portrait frame이 없다.")
        reference=references[row["identity"]]
        if current_identity!=row["identity"]:
            # 한 portrait의 canonical identity UV/bias를 고정하고 driving만 바꾼다.
            texture,xyz=identity_input(data,reference,device,source)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                biases=model.encode_identity(texture,xyz,mean,std)
            current_identity=row["identity"]
        with torch.autocast("cuda",dtype=torch.bfloat16):
            raw=model(texture,xyz,mean,std,*driving(sample),biases=biases)
        gs=decode(raw,texture,bridges[row["identity"]],sample,config)
        rendered=render(gs,sample,torch.ones(3,device=device))
        L00.save_image(out/"rgb"/f"{i:06d}.png",rendered["rgb"])
        L00.save_panels(out/f"{i:06d}.jpg",[sample["rgb"],rendered["rgb"],rendered["normal"]*.5+.5,rendered["alpha"]],
                        ["Reference","Generated avatar","Normal","Alpha"])
        L00.save_tensor(out/f"{i:06d}.pt",dict({k:v.cpu() for k,v in gs.items()},reference_frame=reference["file"],driver_frame=row["file"]))
    L00.save_json(out/"render.json",{"identity_source":source,"fixed_portraits":{k:v["file"] for k,v in references.items()},
        "frames":len(data.manifest["samples"]),"scope":"one fixed portrait per identity; frame driving/camera changes"})


def main():
    parser=L00.arguments(__doc__)
    parser.add_argument("--stage",choices=("train","render","validate"),default="train")
    parser.add_argument("--identity-source",choices=("target","generated","refined"),default="refined")
    parser.add_argument("--portrait-frame",type=int,help="identity마다 고정할 reference frame; 기본은 첫 frame")
    parser.add_argument("--val-every",type=int,default=100,help="고정 validation 주기. 기본 100 optimizer steps")
    parser.add_argument("--val-frames",type=int,default=1,help="고정 held-out frame 수. 기본 1, 0이면 전체")
    parser.add_argument("--val-sources",nargs="+",choices=("target","generated","refined"),default=["target","refined"],
                        help="validation의 identity UV 조건. 기본 target refined")
    args=parser.parse_args()
    config=L00.read_config(args.config,args.smoke)
    device=L00.setup_device(args.device,config["seed"])
    data=L00.PreparedData(args.data)
    if args.stage=="train":
        train(config,data,device,args.steps or config["training"]["steps"],resume=args.resume,
              val_every=args.val_every,val_frames=args.val_frames,val_sources=args.val_sources,portrait_frame=args.portrait_frame)
    else:
        model=UniversalPrior(config).to(device).eval()
        state=L00.restore(L00.work_root(config)/"checkpoints/04/latest.pt",model,config,data)
        step=state["step"]
        del state
        if args.stage=="render":
            render_sequence(config,data,model,device,args.identity_source,args.portrait_frame)
        else:
            import lpips
            perceptual=lpips.LPIPS(net="vgg").to(device).eval().requires_grad_(False)
            rows,inputs=validation_setup(data,device,args.val_frames,args.val_sources,args.portrait_frame)
            out=L00.work_root(config)/"outputs/04"
            validate(config,data,model,device,rows,inputs,out,step,perceptual)
            print("04 validation:",out/"validation"/f"{step}.jpg")


if __name__=="__main__":
    main()
