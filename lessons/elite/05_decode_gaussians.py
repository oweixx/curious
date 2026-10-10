"""05 — UV parameter map -> 3D surface Gaussian -> 실제 differentiable 2DGS.

읽는 순서: quaternion 함수 -> SurfaceMap.decode -> render -> main.
Geometry 13채널: coarse xyz(3), scale(2), local quaternion(4), fine xyz(3), opacity(1).
Appearance 3채널: RGB. 즉 전체 16채널이다.
xyz_world = xyz_posed_uv + R_TBN @ (coarse + fine).
R_world = R_TBN @ R_local; CUDA renderer의 quaternion 순서는 wxyz다.

CUDA_VISIBLE_DEVICES=2 python 05_decode_gaussians.py --frame 0
이 lesson의 render는 network 학습 전의 실제 초기화 상태이며 학습 결과가 아니다.
결과에 raw 16채널 Map, decoded UV Map 시각화와 실제 수치 tensor도 저장한다.
"""

import argparse
import importlib.util
import math
from pathlib import Path

spec = importlib.util.spec_from_file_location("elite_03_entry", Path(__file__).with_name("03_inspect_tracking.py"))
L03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L03)


def axis_angle_to_quaternion(vector):
    """rotation vector radians -> xyzw. sinc로 angle=0에서 gradient를 유지한다."""
    import torch
    angle = vector.norm(dim=-1, keepdim=True)
    xyz = vector * (.5*torch.sinc(angle/(2*math.pi)))
    return torch.cat((xyz, torch.cos(angle*.5)), -1)


def quaternion_matrix(q):
    """단위 xyzw quaternion -> 회전 matrix. q와 -q는 동일한 회전이다."""
    import torch
    q = torch.nn.functional.normalize(q, dim=-1)
    x,y,z,w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1],3,3)


def matrix_quaternion(matrix):
    """회전 matrix -> renderer용 wxyz. 가장 안정적인 component를 기준으로 선택한다."""
    import torch
    r = matrix
    a,b,c = r[...,0,0], r[...,1,1], r[...,2,2]
    squared = torch.stack((1+a+b+c, 1+a-b-c, 1-a+b-c, 1-a-b+c), -1).clamp_min(0)
    # sqrt(0)의 무한 gradient를 피한다. 선택되는 component는 정상 회전에서 >=1이다.
    root = squared.clamp_min(1e-8).sqrt()
    candidates = torch.stack((
        torch.stack((squared[...,0], r[...,2,1]-r[...,1,2], r[...,0,2]-r[...,2,0], r[...,1,0]-r[...,0,1]),-1),
        torch.stack((r[...,2,1]-r[...,1,2], squared[...,1], r[...,1,0]+r[...,0,1], r[...,0,2]+r[...,2,0]),-1),
        torch.stack((r[...,0,2]-r[...,2,0], r[...,1,0]+r[...,0,1], squared[...,2], r[...,2,1]+r[...,1,2]),-1),
        torch.stack((r[...,1,0]-r[...,0,1], r[...,2,0]+r[...,0,2], r[...,2,1]+r[...,1,2], squared[...,3]),-1),
    ), -2)/(2*root[..., :,None])
    index = squared.argmax(-1)
    q = candidates.gather(-2, index[...,None,None].expand(*index.shape,1,4)).squeeze(-2)
    q = torch.nn.functional.normalize(q, dim=-1)
    return torch.where(q[..., :1]<0, -q, q)


def initial_scale_logit(uv_size):
    """공개 decoding 식의 역함수. 첫 render에서 splat이 지나치게 작아지지 않게 한다."""
    scale_max = .1*math.exp(-3.5)
    target = min(.0006*512/uv_size, scale_max*.8)
    # scale = scale_max / (1 + exp(1.5 - raw))
    return 1.5-math.log(scale_max/target-1)


def read_uv(scene):
    path = L03.output_dir(4,scene["sequence"],scene["run"])/"uv.pt"
    uv = L03.read_tensor(path)
    if uv["scene_sha256"] != L03.lesson(2).sha256(L03.scene_path(scene["sequence"],scene["run"])):
        raise ValueError("04 UV와 03 scene이 다르다.")
    return uv


class SurfaceMap:
    def __init__(self, scene, uv, device):
        self.L04 = L03.lesson(4)
        self.geometry = L03.FrameGeometry(scene, device)
        self.uv = uv
        self.lookup = {k:uv[k].to(device) for k in ("face","bary","valid")}
        self.indices = self.lookup["valid"].reshape(-1).nonzero().squeeze(1)
        self.texture = uv["texture"].to(device)
        self.xyz = uv["normalized_xyz"].to(device)

    def uv_input(self, batch_size):
        import torch
        # RGB [-1,1], XYZ는 04에 저장된 single-identity normalization을 그대로 사용.
        return torch.cat((self.texture*2-1,self.xyz),0)[None].expand(batch_size,-1,-1,-1)

    def posed_surface(self, ids):
        import torch
        with torch.no_grad():
            vertices = self.geometry.forward(ids)
            points = self.L04.interpolate_vertices(vertices,self.geometry.faces,self.lookup)
            frame, bad = self.L04.face_tbn(vertices,self.geometry.vt,self.geometry.faces,self.geometry.ft)
            rotations = frame[:, self.lookup["face"].clamp_min(0)]
        return points, rotations

    def decode(self, geometry_map, appearance_map, points, rotations):
        """UV parameter Map을 Gaussian으로 변환하는 고정된 미분 가능 계산이다.

        여기에는 학습할 weight가 없다. 08에서는 network가 만든 Map이 입력되고,
        render loss의 gradient가 이 계산을 통과해 network까지 돌아간다.
        """
        import torch
        import torch.nn.functional as F
        b, c, h, w = geometry_map.shape
        if c!=13 or appearance_map.shape!=(b,3,h,w):
            raise ValueError("geometry는 Bx13xUxU, appearance는 Bx3xUxU여야 한다.")
        if h!=self.uv["uv_size"] or w!=h:
            raise ValueError("network output과 04 UV lookup의 해상도가 다르다.")
        def flat(value):
            return value.reshape(b,h*w,*value.shape[3:])[:,self.indices]
        
        geo = flat(geometry_map.permute(0,2,3,1))
        app = flat(appearance_map.permute(0,2,3,1))
        
        base, frame = flat(points), flat(rotations)
        coarse = torch.tanh(geo[..., :3])*.2
        fine = torch.tanh(geo[..., 9:12])*.1
        displacement = coarse+fine
        # ELITE 공개 decoding의 두 축 scale 식. 단위는 FLAME world unit.
        scales = .1*torch.exp(-3.5-F.softplus(1.5-geo[...,3:5]))
        identity = torch.zeros_like(geo[...,5:9])
        identity[...,3]=1
        local_q = F.normalize(identity+geo[...,5:9],dim=-1)
        # 예측 residual이 정확히 -identity이면 norm=0 -> identity fallback.
        local_q = torch.where((local_q.norm(dim=-1)<1e-6)[...,None],identity,local_q)
        rotation = frame @ quaternion_matrix(local_q)
        position = base + (frame @ displacement[...,None]).squeeze(-1)
        return {"position":position,"rotation":matrix_quaternion(rotation),"scale":scales,
                "opacity":torch.sigmoid(geo[...,12:13]), "color":(app+.5).clamp(0,1),
                "coarse":coarse,"fine":fine,"displacement":displacement,"base":base,
                "geometry_map":geometry_map,"appearance_map":appearance_map,
                "indices":self.indices,"uv_size":h}


def camera_matrices(K, w2c_gl, height, width):
    """Tracker OpenGL(x right,y up,z backward) -> 2DGS OpenCV(x right,y down,z forward).

    Rasterizer에는 transpose한 matrix를 전달한다. full_proj = view.T @ projection.T.
    이 lesson에서는 principal point가 이미지 중심인 monocular camera를 사용한다.
    """
    import torch
    flip = torch.diag(K.new_tensor([1.,-1.,-1.,1.]))
    view = flip @ w2c_gl
    projection = torch.zeros(4,4,device=K.device)
    near, far = .01, 10.
    projection[0,0], projection[1,1] = 2*K[0,0]/width, 2*K[1,1]/height
    projection[0,2], projection[1,2] = 2*K[0,2]/width-1, 2*K[1,2]/height-1
    projection[2,2], projection[2,3], projection[3,2] = far/(far-near), -far*near/(far-near), 1
    return view.T.contiguous(), (projection@view).T.contiguous(), torch.linalg.inv(view)[:3,3].contiguous()


def depth_normal(depth, K, w2c_gl):
    """Median camera-depth -> world-space surface normal. 배경 depth=0은 loss에서 제외한다."""
    import torch
    import torch.nn.functional as F
    h,w = depth.shape[-2:]
    y,x = torch.meshgrid(torch.arange(h,device=depth.device)+.5,torch.arange(w,device=depth.device)+.5,indexing="ij")
    z = depth[0]
    cam = torch.stack(((x-K[0,2])/K[0,0]*z, -(y-K[1,2])/K[1,1]*z, -z),-1)
    world = (cam-w2c_gl[:3,3]) @ w2c_gl[:3,:3]
    dy = world[2:,1:-1]-world[:-2,1:-1]
    dx = world[1:-1,2:]-world[1:-1,:-2]
    normals = F.normalize(torch.cross(dy,dx,dim=-1),dim=-1)
    return F.pad(normals.permute(2,0,1),[1,1,1,1])


def render(gaussians, index, K, w2c, h, w, background):
    """실제 2DGS CUDA backend. 위치/크기/회전/색/opacity에 대한 gradient를 제공한다."""
    import torch
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    view, projection, center = camera_matrices(K,w2c,h,w)
    settings = GaussianRasterizationSettings(image_height=h,image_width=w,
        tanfovx=float(w/(2*K[0,0])),tanfovy=float(h/(2*K[1,1])),bg=background,
        scale_modifier=1.,viewmatrix=view,projmatrix=projection,sh_degree=0,campos=center,prefiltered=False,debug=False)
    means = gaussians["position"][index].contiguous()
    screen = torch.zeros_like(means,requires_grad=torch.is_grad_enabled())
    # 원본 backend는 optional empty tensor를 현재 CUDA device에 만든다.
    with torch.cuda.device(means.device):
        rgb,radii,maps = GaussianRasterizer(raster_settings=settings)(
            means3D=means, means2D=screen, shs=None, colors_precomp=gaussians["color"][index].contiguous(),
            opacities=gaussians["opacity"][index].contiguous(), scales=gaussians["scale"][index].contiguous(),
            rotations=gaussians["rotation"][index].contiguous(), cov3D_precomp=None)
    alpha = maps[1:2]
    cv_rotation = view.T[:3,:3]
    normal = (maps[2:5].permute(1,2,0) @ cv_rotation).permute(2,0,1)
    expected = maps[0:1]/alpha.clamp_min(1e-8)
    median = torch.nan_to_num(maps[5:6])
    return {"rgb":rgb,"alpha":alpha,"normal":normal,"depth":median,
            "expected_depth":torch.nan_to_num(expected),"distortion":maps[6:7],
            "surface_normal":depth_normal(median,K,w2c),"radii":radii,"screen":screen}


def save_gaussian_maps(out, frame_id, surface, decoded):
    """Raw Map과 실제 Gaussian 파라미터를 UV 격자에서 보여준다.

    Raw는 activation 전 값이다. 예: opacity raw=0 -> sigmoid -> opacity=0.5.
    Decode가 추린 Gaussian list를 원래 texel 주소에 다시 배치한다.
    표시용 정규화는 PNG에만 적용하고 실제 수치는 .pt에 저장한다.
    """
    import numpy as np
    import torch
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    size = surface.uv["uv_size"]
    valid = surface.lookup["valid"].detach().cpu()
    indices = surface.indices.detach().cpu()

    def to_uv(values):
        # [N,C] -> [C,H,W]. Invalid 값은 0이며 valid mask와 함께 해석한다.
        values = values.detach().cpu()
        grid = values.new_zeros(size*size, values.shape[-1])
        grid[indices] = values
        return grid.reshape(size, size, -1).permute(2, 0, 1).contiguous()

    geometry = decoded["geometry_map"][0].detach().cpu()
    appearance = decoded["appearance_map"][0].detach().cpu()
    raw = torch.cat((geometry, appearance), dim=0)
    names = ["coarse T raw", "coarse B raw", "coarse N raw",
             "scale axis 1 raw", "scale axis 2 raw",
             "local quaternion x residual", "local quaternion y residual",
             "local quaternion z residual", "local quaternion w residual",
             "fine T raw", "fine B raw", "fine N raw", "opacity logit",
             "appearance R residual", "appearance G residual", "appearance B residual"]
    maps = {key:to_uv(decoded[key][0]) for key in
            ("base", "position", "coarse", "fine", "displacement", "scale", "opacity", "color")}
    # 저장한 world quaternion은 wxyz다. 기존 matrix 함수에는 xyzw로 바꿔 전달한다.
    world_q = decoded["rotation"][0].detach().cpu()
    maps["normal"] = to_uv(quaternion_matrix(world_q[:, [1, 2, 3, 0]])[:, :, 2])
    xyz_min = decoded["position"][0].detach().cpu().amin(dim=0)
    xyz_max = decoded["position"][0].detach().cpu().amax(dim=0)
    L03.save_tensor(out/f"{frame_id:06d}_initial_uv_maps.pt", {
        "geometry_map":geometry, "appearance_map":appearance,
        "channel_names":names, "valid":valid, "decoded":maps,
        "xyz_display_min":xyz_min, "xyz_display_max":xyz_max,
        "convention":"row=0 -> v=1; local displacement axes are T,B,N; decoded scale is in FLAME world units",
    })

    # 16개의 raw 채널을 따로 표시한다. 표시 색은 값의 크기를 나타낸다.
    figure = Figure(figsize=(15, 13), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(4, 4).reshape(-1)
    invalid = ~valid.numpy()
    for i, axis in enumerate(axes):
        selected = raw[i][valid]
        limit = max(1., float(selected.abs().max()))
        image = axis.imshow(np.ma.array(raw[i].numpy(), mask=invalid),
                            cmap="coolwarm", vmin=-limit, vmax=limit, origin="upper")
        axis.set_facecolor("#202020")
        axis.set_title(f"{i:02d}: {names[i]}\nmin={float(selected.min()):.4g}, max={float(selected.max()):.4g}", fontsize=9)
        axis.set_axis_off()
        figure.colorbar(image, ax=axis, shrink=.65)
    figure.suptitle("Initial Gaussian UV Map: raw 13 geometry + 3 appearance channels")
    figure.savefig(out/f"{frame_id:06d}_initial_uv_raw_channels.png", dpi=120)

    def rgb_display(value):
        # 아래 변환은 표시용이다. maps에 저장된 실제 값은 바꾸지 않는다.
        return (value.clamp(0, 1)*valid[None]).permute(1, 2, 0).numpy()

    scale_max = .1*math.exp(-3.5)
    xyz_display = (maps["position"]-xyz_min[:, None, None])/(xyz_max-xyz_min).clamp_min(1e-8)[:, None, None]
    panels = [
        ("Gaussian RGB", rgb_display(maps["color"]), None, 0, 1),
        ("World XYZ (display normalized)", rgb_display(xyz_display), None, 0, 1),
        ("Local displacement T/B/N (0 = gray; +/-0.3)", rgb_display(maps["displacement"]/.6+.5), None, 0, 1),
        ("World Gaussian normal XYZ (-1..1)", rgb_display(maps["normal"]*.5+.5), None, 0, 1),
        ("Scale axis 1 (FLAME world units)", maps["scale"][0].numpy(), "viridis", 0, scale_max),
        ("Scale axis 2 (FLAME world units)", maps["scale"][1].numpy(), "viridis", 0, scale_max),
        ("Opacity (0..1)", maps["opacity"][0].numpy(), "gray", 0, 1),
        ("Valid UV texels", valid.numpy(), "gray", 0, 1),
    ]
    figure = Figure(figsize=(16, 8), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 4).reshape(-1)
    for axis, (title, value, cmap, low, high) in zip(axes, panels):
        if cmap is None:
            axis.imshow(value, origin="upper")
        else:
            mask = np.zeros_like(invalid) if title == "Valid UV texels" else invalid
            image = axis.imshow(np.ma.array(value, mask=mask), cmap=cmap, vmin=low, vmax=high, origin="upper")
            figure.colorbar(image, ax=axis, shrink=.7)
        axis.set_facecolor("#202020")
        axis.set_title(title, fontsize=10)
        axis.set_axis_off()
    figure.suptitle("Decoded Gaussian UV Maps: initialization, before training")
    figure.savefig(out/f"{frame_id:06d}_initial_uv_decoded.png", dpi=120)


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    L03.add_arguments(parser)
    parser.add_argument("--frame",type=int,default=0)
    parser.add_argument("--check-backward",action="store_true",help="실제 RGB/alpha와 loss를 계산해 decoding->CUDA backward를 확인")
    args = parser.parse_args()
    import torch
    torch.cuda.set_device(torch.device(args.device))
    scene = L03.read_scene(args.sequence,args.run)
    if not 0<=args.frame<scene["manifest"]["num_frames"]:
        parser.error("frame 범위를 확인하세요.")
    uv = read_uv(scene)
    size = uv["uv_size"]
    
    surface = SurfaceMap(scene,uv,args.device)
    geometry_map = torch.zeros(1,13,size,size,device=args.device)
    geometry_map[:,3:5] = initial_scale_logit(size)
    appearance = uv["texture"][None].to(args.device)-.5
    points, rotations = surface.posed_surface([args.frame])
    
    
    with torch.no_grad():
        decoded = surface.decode(geometry_map,appearance,points,rotations)
        h,w = scene["manifest"]["height"],scene["manifest"]["width"]
        image = render(decoded,0,scene["K"].to(args.device),scene["w2c"].to(args.device),h,w,torch.ones(3,device=args.device))
        
        
    out = L03.output_dir(5,args.sequence,args.run)
    L03.save_image(out/f"{args.frame:06d}_initial_rgb.png",image["rgb"])
    L03.save_image(out/f"{args.frame:06d}_initial_alpha.png",image["alpha"])
    L03.save_image(out/f"{args.frame:06d}_initial_normal.png",image["normal"]*.5+.5)
    L03.save_tensor(out/f"{args.frame:06d}_initial_gaussians.pt",{k:decoded[k].cpu() for k in ("position","rotation","scale","color","opacity")})
    save_gaussian_maps(out, args.frame, surface, decoded)
    if args.check_backward:
        dataset=L03.lesson(7).VideoFrames(scene,max_side=max(h,w))
        target=dataset.batch([args.frame],args.device)
        geometry_map.requires_grad_(True)
        appearance.requires_grad_(True)
        decoded=surface.decode(geometry_map,appearance,points,rotations)
        rendered=render(decoded,0,scene["K"].to(args.device),scene["w2c"].to(args.device),h,w,torch.ones(3,device=args.device))
        expected=L03.lesson(7).composite(target["rgb"],target["alpha"],torch.ones(3,device=args.device))[0]
        actual_loss=(rendered["rgb"]-expected).abs().mean()+(rendered["alpha"]-target["alpha"][0]).abs().mean()
        actual_loss.backward()
        report={}
        for name,tensor in (("geometry",geometry_map),("appearance",appearance)):
            if tensor.grad is None or not torch.isfinite(tensor.grad).all() or tensor.grad.abs().sum()==0:
                raise RuntimeError(f"{name} map으로 돌아오는 finite nonzero gradient가 없다.")
            report[name]=tensor.grad.abs().mean((0,2,3)).detach().cpu().tolist()
        L03.save_json(out/"backward_check.json",{"loss":float(actual_loss.detach()),"mean_abs_gradient_per_channel":report})
        print("실제 CUDA backward 확인:",report)
    print("05 초기화 render:",out,"| Gaussian 수",len(surface.indices))
    print("학습 전이다. 08에서 network를 학습한 뒤 비교한다.")


if __name__=="__main__":
    main()
