"""02 — 동일한 canonical Avatar에서 '값을 학습'하는 것과 '함수를 학습'하는 것을 비교.

repo root에서:
    CUDA_VISIBLE_DEVICES=7 python lessons/online_avatar/02_avatar_and_updates.py --steps 400
lesson 폴더에서:
    CUDA_VISIBLE_DEVICES=7 python 02_avatar_and_updates.py --uv-size 128 --steps 400

출력: data/042_FREE/initial.pt, initialization/{0,100,...}.jpg, canonical.jpg.
읽는 순서: CoordinateField -> Avatar.raw -> Surface.gaussians -> bootstrap.
여기서는 처음 prefix 시각만 사용해 작은 coordinate MLP를 warm-start한다.
다인물로 사전학습한 MGPM/VGGT를 재현한다고 주장하지 않는다.
이후 03에서 동일한 출력으로 frozen / texel parameter / low-rank adapter를 시작한다.
RGB supervision은 관측 색이다. 단순 조명 모형을 붙였다고 true albedo가 되지는 않는다.
"""
import argparse
import math

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import trange

from common import (UPSTREAM, Stream, arguments, cpu_state, elite, finite_step, load_pt,
                    panels, project, require_new, save_pt, setup, sha, timer)


def read_uv_obj(path):
    vt, faces, ft = [], [], []
    for line in path.read_text().splitlines():
        bits = line.split()
        if not bits:
            continue
        if bits[0] == "vt":
            vt.append([float(v) for v in bits[1:3]])
        elif bits[0] == "f":
            if len(bits) != 4:
                raise ValueError("삼각형 UV template 필요")
            corners = [x.split("/") for x in bits[1:]]
            faces.append([int(x[0])-1 for x in corners])
            ft.append([int(x[1])-1 for x in corners])
    return torch.tensor(vt), torch.tensor(faces), torch.tensor(ft)


def positional_features(xyz):
    """이미지가 아닌 canonical 표면 주소를 network에 넣는다. 시간/표정은 넣지 않는다.

    같은 표면 주소의 identity 정보는 frame 간 공유되어야 한다.
    expression은 이 함수가 아닌 FLAME deformation 경로에 들어간다.
    """
    x = (xyz-xyz.mean(0))/xyz.std(0).clamp_min(.01)
    values = [x]
    for frequency in (1., 2., 4., 8.):
        values.extend([torch.sin(math.pi*frequency*x), torch.cos(math.pi*frequency*x)])
    return torch.cat(values, -1)


class CoordinateField(nn.Module):
    """prefix로 학습하는 작은 canonical field: position feature -> Gaussian raw 6 channels."""
    def __init__(self, width=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(27, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.head = nn.Linear(width, 6)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        return self.head(self.trunk(x))


class Avatar(nn.Module):
    """무엇을 optimizer에 넣느냐가 세 경우의 차이다. decode/render는 전부 동일하다.

    frozen:         raw_i = f_theta(x_i)                    (theta 고정)
    representation: raw_i = f_theta(x_i) + delta_i          (texel별 delta 학습)
    adapter:        raw_i = W h_i + b + B(A h_i)            (A,B weight 학습)

    A는 random, B는 zero로 초기화한다. 둘 다 zero면 두 행렬의 gradient가 모두 0이다.
    이 adapter는 LoRA 형태의 마지막 Linear 갱신이며 ZipMap의 KV fast-weight layer가 아니다.
    """
    def __init__(self, initial, mode="frozen", rank=4):
        super().__init__()
        self.mode = mode
        self.field = CoordinateField(initial["width"])
        self.field.load_state_dict(initial["field"])
        self.register_buffer("features", initial["features"].clone())
        self.field.requires_grad_(mode == "base")
        if mode == "representation":
            self.delta = nn.Parameter(torch.zeros(len(self.features), 6))
        if mode == "adapter":
            self.A = nn.Linear(initial["width"], rank, bias=False)
            self.B = nn.Linear(rank, 6, bias=False)
            nn.init.normal_(self.A.weight, std=.02)
            nn.init.zeros_(self.B.weight)

    def raw(self):
        h = self.field.trunk(self.features)
        result = self.field.head(h)
        if self.mode == "representation":
            result = result + self.delta
        elif self.mode == "adapter":
            result = result + self.B(self.A(h))
        return result

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]


class Lighting(nn.Module):
    """현재 시각의 저차 조명 근사. 2개 camera에 공유하며 frame마다 초기화해 다시 fit.

    exp(.35*tanh([1,nx,ny,nz] @ L))는 항상 양수이고 과도한 색 보정을 제한한다.
    DC와 texture는 여전히 모호하다. prefix 첫 frame L=0, 크기 prior로 gauge를 고정한다.
    그림자/정반사/카메라별 response를 모두 설명하는 물리 renderer는 아니다.
    """
    def __init__(self):
        super().__init__()
        self.coefficients = nn.Parameter(torch.zeros(4, 3))

    def forward(self, normals):
        basis = torch.cat([torch.ones_like(normals[:, :1]), normals], -1)
        return torch.exp(.35*torch.tanh(basis @ self.coefficients))


class Surface:
    """Gaussian은 UV texel에 붙고, FLAME pose/expression에 따라 이동한다.

    Persistent: canonical local XYZ residual(3), appearance residual(3).
    Transient: 01의 FLAME vertices/R/t + 현재 lighting.
    2D splat scale/orientation/opacity는 표면으로부터 정하며 이번 실험에서는 고정한다.
    """
    def __init__(self, stream, tracking, data):
        self.stream, self.tracking, self.data = stream, tracking, data
        self.device = stream.device
        self.L04, self.L05 = elite(4), elite(5)
        self.faces = tracking["faces"].to(self.device)
        self.head_faces = self.faces[data["head_faces"].to(self.device)]
        self.vt, self.ft = data["vt"].to(self.device), data["ft"].to(self.device)
        self.face = data["face"].to(self.device)
        self.bary = data["bary"].to(self.device)
        self.scale = data["splat_scale"].to(self.device)
        self.base_logit = torch.logit(data["base_rgb"].to(self.device).clamp(.01, .99))
        self.indices = data["indices"].to(self.device)
        self.size = data["uv_size"]
        self._posed, self._roi = {}, {}

    def uv(self, values):
        if values.ndim == 1:
            values = values[:, None]
        image = values.new_zeros(self.size*self.size, values.shape[-1])
        image[self.indices] = values
        return image.reshape(self.size, self.size, -1).permute(2, 0, 1)

    def posed(self, t):
        if t not in self._posed:
            row = self.tracking["frames"][t]
            v = row["local_vertices"].to(self.device)
            R, tr = row["R"].to(self.device), row["translation"].to(self.device)
            s = self.tracking["scale"].to(self.device)
            p = (v[self.faces[self.face]]*self.bary[..., None]).sum(1)
            tbn, _ = self.L04.face_tbn(v[None], self.vt, self.faces, self.ft)
            frame = R[None] @ tbn[0, self.face]
            self._posed[t] = dict(position=s*(p @ R.T)+tr, rotation=frame,
                                  vertices=s*(v @ R.T)+tr, scale=s)
        return self._posed[t]

    def decode(self, raw):
        # tanh로 최대 5 mm: coarse geometry 오류를 appearance로 무제한 우회하지 않게 한다.
        offset = .005*torch.tanh(raw[:, :3])
        color = torch.sigmoid(self.base_logit+raw[:, 3:])
        return offset, color

    def gaussians(self, raw, t, light):
        p = self.posed(t)
        delta, albedo_proxy = self.decode(raw)
        position = p["position"] + p["scale"]*(p["rotation"] @ delta[..., None]).squeeze(-1)
        color = (albedo_proxy*light(p["rotation"][..., 2])).clamp(0, 1)
        return dict(position=position[None], rotation=self.L05.matrix_quaternion(p["rotation"])[None],
                    scale=(self.scale*p["scale"])[None, :, None].expand(1, -1, 2).contiguous(),
                    opacity=torch.full_like(position[None, :, :1], .95), color=color[None])

    def render(self, raw, t, camera, light):
        # ELITE renderer contract는 OpenGL w2c; 데이터 calibration은 OpenCV.
        flip = torch.diag(raw.new_tensor([1., -1., -1., 1.]))
        return self.L05.render(self.gaussians(raw, t, light), 0, self.stream.K[camera],
                               flip @ self.stream.w2c[camera], self.stream.size, self.stream.size,
                               raw.new_zeros(3))

    def roi(self, t, camera):
        """Tracking mesh로 정한 고정 평가 영역. 학습된 alpha로 평가 영역을 줄이지 않는다.

        GT segmentation이 없으므로 얼굴 표면 내부만 사용. hair/background 수치는 아니다.
        """
        key = (t, camera)
        if key not in self._roi:
            import nvdiffrast.torch as dr
            if not hasattr(self, "context"):
                self.context = dr.RasterizeCudaContext()
            v = self.posed(t)["vertices"][None]
            flip = torch.diag(v.new_tensor([1., -1., -1., 1.]))
            _, mask = self.L04.image_depth(v, self.head_faces, self.stream.K[camera],
                                          flip @ self.stream.w2c[camera], self.stream.size,
                                          self.stream.size, self.context)
            mask = -F.max_pool2d(-mask[None], 5, stride=1, padding=2)[0]
            if mask.sum() < 64:
                raise ValueError("ROI가 비어 있다. camera projection/01 tracking부터 확인하세요.")
            self._roi[key] = mask.detach()
        return self._roi[key]


def photo_loss(surface, raw, t, light, cameras=None, pixel_weights=None):
    """입력 2개만 읽는 self-supervision. ROI의 모든 픽셀을 세므로 alpha 축소로 피할 수 없다."""
    losses = []
    for c in cameras or surface.stream.inputs:
        target = surface.stream.rgb(t, c)
        out = surface.render(raw, t, c, light)
        mask = surface.roi(t, c)
        if pixel_weights is not None:
            mask = mask*pixel_weights[c]
        residual = (out["rgb"]-target).abs()
        losses.append((residual*mask).sum()/(3*mask.sum()) + .05*((out["alpha"]-1).abs()*mask).sum()/mask.sum())
    return torch.stack(losses).mean()


def regularizer(raw, data):
    # UV에서 맞닿았지만 실제 3D에서 먼 seam을 TV로 이어 붙이지 않는다.
    pairs = data["neighbors"].to(raw.device)
    smooth = (raw[pairs[:, 0]]-raw[pairs[:, 1]]).square().mean() if len(pairs) else raw.sum()*0
    return .002*raw[:, :3].square().mean() + .0002*raw[:, 3:].square().mean() + .001*smooth


def build_surface(stream, tracking, uv_size):
    import nvdiffrast.torch as dr
    L04 = elite(4)
    device = stream.device
    obj = UPSTREAM / "third_party/pixel3dmm/assets/head_template.obj"
    vt, faces, ft = read_uv_obj(obj)
    if not torch.equal(faces, tracking["faces"]):
        raise ValueError("FLAMEとUV templateのface順序が異なる。vertex数だけで対応させないこと。")
    vt, faces, ft = vt.to(device), faces.to(device), ft.to(device)
    neutral = tracking["neutral"].to(device)
    lm = tracking["neutral_landmarks"].to(device)
    up = F.normalize(lm[27]-lm[8], dim=0)
    keep = ((neutral-(lm[8]-.01*up)) @ up) > 0
    head_faces = keep[faces].all(-1)
    context = dr.RasterizeCudaContext()
    lookup = L04.uv_lookup(vt, ft, uv_size, context)
    valid = lookup["valid"] & head_faces[lookup["face"].clamp_min(0)]
    ids = valid.reshape(-1).nonzero().flatten()
    face, bary = lookup["face"].reshape(-1)[ids], lookup["bary"].reshape(-1, 3)[ids]
    xyz = (neutral[faces[face]]*bary[..., None]).sum(1)
    a, b, c = neutral[faces].unbind(1)
    area3 = torch.cross(b-a, c-a, dim=-1).norm(dim=-1)/2
    a, b, c = vt[ft].unbind(1)
    area2 = ((b-a)[:, 0]*(c-a)[:, 1]-(b-a)[:, 1]*(c-a)[:, 0]).abs()/2
    sigma = (.8*torch.sqrt(area3[face]/area2[face].clamp_min(1e-10))/uv_size).clamp(.0003, .008)
    # local smoothness edge 목록: 빈 texel/island 간 연결을 차단한다.
    grid = torch.full((uv_size, uv_size), -1, device=device, dtype=torch.long)
    grid.reshape(-1)[ids] = torch.arange(len(ids), device=device)
    pairs = torch.cat([torch.stack([grid[:, :-1].reshape(-1), grid[:, 1:].reshape(-1)], -1),
                       torch.stack([grid[:-1].reshape(-1), grid[1:].reshape(-1)], -1)])
    pairs = pairs[(pairs >= 0).all(-1)]
    pairs = pairs[(xyz[pairs[:, 0]]-xyz[pairs[:, 1]]).norm(dim=-1) < .01]
    data = dict(vt=vt.cpu(), ft=ft.cpu(), head_faces=head_faces.cpu(), indices=ids.cpu(),
                face=face.cpu(), bary=bary.cpu(), uv_size=uv_size, xyz=xyz.cpu(),
                splat_scale=sigma.cpu(), base_rgb=torch.full((len(ids), 3), .5), neighbors=pairs.cpu())
    surface = Surface(stream, tracking, data)
    accum, weight = torch.zeros_like(xyz), torch.zeros_like(xyz[:, :1])
    # 오직 prefix 입력 RGB로 texture를 초기화한다. all-frame texture는 causal 실험에서 누수다.
    with torch.no_grad():
        for t in range(stream.prefix):
            posed = surface.posed(t)
            for camera in stream.inputs:
                p, z = project(posed["position"], stream.K[camera], stream.w2c[camera])
                flip = torch.diag(xyz.new_tensor([1., -1., -1., 1.]))
                depth, mask = L04.image_depth(posed["vertices"][None], surface.head_faces,
                    stream.K[camera], flip@stream.w2c[camera], stream.size, stream.size, context)
                grid_sample = ((p+.5)*2/stream.size-1)[None, None]
                sample = lambda x: F.grid_sample(x[None], grid_sample, align_corners=False)[0, :, 0].T
                observed = sample(stream.rgb(t, camera))
                center = torch.linalg.inv(stream.w2c[camera])[:3, 3]
                ray = F.normalize(center-posed["position"], dim=-1)
                cosine = (ray*posed["rotation"][..., 2]).sum(-1).clamp_min(0)
                visible = ((sample(depth)[:, 0]-z).abs() < .003) & (sample(mask)[:, 0] > .9)
                visible &= (p >= 0).all(-1) & (p < stream.size-1).all(-1) & (z > 0)
                w = (cosine*visible)[:, None]
                accum += observed*w
                weight += w
    data["base_rgb"] = torch.where(weight > 0, accum/weight.clamp_min(1e-6), .5).cpu()
    data["coverage"] = weight[:, 0].cpu()
    return data


def canonical_panels(path, surface, raw, title="canonical"):
    offset, color = surface.decode(raw)
    panels(path, [surface.uv(color), surface.uv((offset/.005+1)/2),
                  surface.uv(surface.data["coverage"].to(raw.device).clamp(0, 1))],
           [f"{title}: appearance", "local XYZ delta [-5,+5] mm", "prefix coverage (fixed)"])


def load_components(work, device):
    stream = Stream(work, device)
    tracking, initial = load_pt(work / "tracking.pt"), load_pt(work / "initial.pt")
    stream.check_artifact(tracking)
    stream.check_artifact(initial)
    if initial["tracking_sha256"] != sha(work / "tracking.pt"):
        raise ValueError("01을 다시 실행했다면 02도 다시 실행해야 한다.")
    return stream, tracking, initial, Surface(stream, tracking, initial["surface"])


def bootstrap(a):
    device = setup(a.device)
    stream = Stream(a.work, device)
    tracking = load_pt(a.work / "tracking.pt")
    stream.check_artifact(tracking)
    require_new(a.work / "initial.pt", a.overwrite)
    if a.steps < 1 or a.uv_size < 32 or a.save_every < 1:
        raise ValueError("steps/save-every 양수, uv-size>=32 필요")
    data = build_surface(stream, tracking, a.uv_size)
    initial = dict(width=64, features=positional_features(data["xyz"]), field=cpu_state(CoordinateField()))
    model = Avatar(initial, "base").to(device)
    surface = Surface(stream, tracking, data)
    lights = [Lighting().to(device) for _ in range(stream.prefix)]
    # 첫 frame의 조명을 1로 고정해 canonical appearance의 기준을 하나 정한다.
    lights[0].requires_grad_(False)
    params = model.trainable() + [p for light in lights[1:] for p in light.parameters()]
    optimizer = torch.optim.Adam(params, lr=.002)
    losses = []
    start = timer()
    for step in trange(a.steps+1, desc="prefix-only canonical field"):
        if step % a.save_every == 0 or step == a.steps:
            with torch.no_grad():
                c = stream.inputs[0]
                output = surface.render(model.raw(), 0, c, lights[0])
                panels(a.work / "initialization" / f"{step}.jpg",
                       [stream.rgb(0, c), output["rgb"], surface.roi(0, c), output["alpha"]],
                       ["prefix RGB", f"warm start {step}", "fixed surface ROI", "render alpha"])
        if step == a.steps:
            break
        t = step % stream.prefix
        raw = model.raw()
        loss = photo_loss(surface, raw, t, lights[t]) + regularizer(raw, data)
        loss = loss + .002*lights[t].coefficients.square().mean()
        finite_step(loss, params, optimizer)
        losses.append(float(loss.detach()))
    initial.update(field=cpu_state(model.field), surface=data, stream_sha256=stream.fingerprint,
                   tracking_sha256=sha(a.work / "tracking.pt"), steps=a.steps, losses=losses,
                   seconds=timer()-start, lights=[cpu_state(l) for l in lights], rank=4,
                   initialization="single-person prefix warm start; NOT a multi-identity pretrained reconstructor")
    save_pt(a.work / "initial.pt", initial)
    canonical_panels(a.work / "initialization/canonical.jpg", surface, model.raw())
    print(f"{len(surface.indices)} Gaussians | prefix={stream.prefix} | {a.work / 'initial.pt'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    arguments(p)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--uv-size", type=int, default=128)
    p.add_argument("--save-every", type=int, default=100)
    bootstrap(p.parse_args())


if __name__ == "__main__":
    main()
