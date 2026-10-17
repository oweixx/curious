"""00 — NeRSemble 관측과 canonical mesh를 같은 좌표계의 학습 자료로 만든다.

읽는 순서: prepare_dataset -> rasterize_labels -> splat_partial_uv -> MeshBridge.
공유 함수는 입출력/좌표계/학습 기록만 담당한다. 각 Network는 01~04에 있다.

python 00_prepare_nersemble.py --stage prepare --frames 0,32
python 00_prepare_nersemble.py --stage assets
CUDA_VISIBLE_DEVICES=0 python 00_prepare_nersemble.py --stage prepare --device cuda:0

기존 ELITE scene은 단일 카메라 debug 자료다. Multi-view는 같은 world-space tracking의
scene과 views manifest를 제공한다. 다른 좌표계의 monocular tracking에 calibration만
붙이지 않는다. Train/held-out camera를 texture 생성 전에 나눈다.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import base64
import urllib.request

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
os.environ.setdefault("TORCH_HOME", str(CACHE / "torch"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT.parent / "elite/.cache/torch_extensions"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))

ASSETS = {
    "sapiens_encoder": ("facebook/sapiens-pretrain-1b", "3ea7a6f421806a1ae3d5b4dd3e33e4d9c19a3e28", "sapiens_1b_epoch_173_clean.pth"),
    "sapiens_normal": ("facebook/sapiens-normal-1b", "9bb2ddb21fa13f98535cea48a6398522a2cadd6e", "sapiens_1b_normal_render_people_epoch_115.pth"),
    "vae": ("stabilityai/stable-diffusion-xl-base-1.0", "462165984030d82259a11f4367a4eed129e94a7b", "vae"),
    "clip": ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41", "."),
}
SOURCE_REVISIONS = {"pippo": "6f80b22f32f58f7ae08f7b20a4901a3c66809368", "sapiens": "2cb07227a740cf09896309ea3a3b8fa44429865c"}
SOURCE_FILES = {
    "pippo": ["latent_diffusion/models/control_mlp.py", "latent_diffusion/models/dit.py", "latent_diffusion/blocks/norm.py",
              "latent_diffusion/blocks/patch.py", "latent_diffusion/blocks/timestep.py", "latent_diffusion/blocks/siren.py", "LICENSE"],
    "sapiens": ["pretrain/mmpretrain/models/backbones/vision_transformer.py", "seg/mmseg/models/decode_heads/vit_depth_head.py",
                "seg/mmseg/models/decode_heads/vit_normal_head.py", "seg/configs/sapiens_depth/render_people/sapiens_1b_render_people-1024x768.py",
                "seg/configs/sapiens_normal/normal_render_people/sapiens_1b_normal_render_people-1024x768.py", "LICENSE"],
}


def lesson(number):
    name = f"fica_lesson_{number:02d}"
    if name not in sys.modules:
        path, = ROOT.glob(f"{number:02d}_*.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def elite(number):
    name = f"fica_elite_{number:02d}"
    if name not in sys.modules:
        path, = (ROOT.parent / "elite").glob(f"{number:02d}_*.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def read_config(path=None, smoke=False):
    config = json.loads(Path(path or ROOT / "fica_config.json").read_text())
    if config["format"] != "fica_nersemble_v1":
        raise ValueError("다른 lesson/checkpoint format이다.")
    if smoke:
        # Smoke의 모델 축소는 명시적이다. 실제 foundation weights는 축소/대체하지 않는다.
        config["uv_size"], config["image_size"] = 64, 128
        config["diffusion"].update(width=128, depth=2, heads=4)
        config["refinement"].update(channels=[32, 64], layers_per_block=1)
        config["upm"]["channels"] = [64, 32, 16, 16]
        config["training"].update(steps=2, preview_every=1, checkpoint_every=2)
        config["sapiens"].update(input_size=[128,96],augmentation=False)
    if config["training"]["batch_size"]!=1 or config["training"]["grad_accumulation"]<1:
        raise ValueError("가변 camera 해상도 때문에 microbatch=1을 사용한다. effective batch는 grad_accumulation으로 늘리세요.")
    config["profile"] = "smoke" if smoke else "full"
    return config


def work_root(config):
    """Smoke가 본 학습 checkpoint/출력을 덮어쓰지 않도록 경로를 분리한다."""
    return CACHE / "smoke" if config.get("profile") == "smoke" else ROOT


def setup_device(name="cuda:0", seed=42):
    device = torch.device(name)
    if device.type != "cuda":
        raise ValueError("실제 mesh/Gaussian CUDA renderer가 필요하다. CUDA_VISIBLE_DEVICES=0을 사용하세요.")
    torch.cuda.set_device(device)
    # Conda Python을 절대 경로로 실행해도 해당 환경의 ninja/nvcc를 찾도록 한다.
    environment_bin = Path(sys.executable).resolve().parent
    os.environ["PATH"] = str(environment_bin) + os.pathsep + os.environ.get("PATH", "")
    if (environment_bin / "nvcc").is_file():
        os.environ.setdefault("CUDA_HOME", str(environment_bin.parent))
    for key, value in (("CC", "/usr/bin/gcc-11"), ("CXX", "/usr/bin/g++-11")):
        if Path(value).is_file():
            os.environ.setdefault(key, value)
    os.environ.setdefault("MAX_JOBS", "4")
    major, minor = torch.cuda.get_device_capability(device)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return device


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_hashes():
    return {p.name: sha256(p) for p in sorted(ROOT.glob("[0-4][0-9]_*.py"))}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def load_tensor(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)


def save_image(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if value.ndim == 4:
        value = value[0]
    array = (value.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array[..., 0] if array.shape[-1] == 1 else array).save(path)


def save_panels(path, images, labels):
    """각 milestone에서 같은 sample을 비교한다. 표시만 resize한다."""
    panels = []
    for value in images:
        value = value.detach().float().cpu()
        if value.ndim == 4:
            value = value[0]
        value = value.expand(3, -1, -1)
        value = F.interpolate(value[None], size=(256, 256), mode="bilinear", align_corners=False)[0]
        array = (value.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        panels.append(Image.fromarray(array))
    canvas = Image.new("RGB", (256 * len(panels), 282), "#161616")
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(zip(panels, labels)):
        canvas.paste(image, (256 * i, 26))
        draw.text((256 * i + 5, 7), label, fill="white")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def asset_path(key, download=False):
    """Revision을 고정한다. 공개 안 된 FiCA task weights를 있다고 가정하지 않는다."""
    from huggingface_hub import hf_hub_download, snapshot_download
    repo, revision, name = ASSETS[key]
    cache = str(CACHE / "huggingface")
    if key in ("vae", "clip"):
        patterns = ["vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"] if key == "vae" else ["config.json", "preprocessor_config.json", "model.safetensors"]
        # 이미 있는 SDXL cache를 재사용하여 수백 MB를 중복 저장하지 않는다.
        if key == "vae":
            shared = Path.home() / ".cache/huggingface/hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
            if all((shared / p).is_file() for p in patterns):
                return shared
        return Path(snapshot_download(repo, revision=revision, allow_patterns=patterns, cache_dir=cache, local_files_only=not download))
    return Path(hf_hub_download(repo, name, revision=revision, cache_dir=cache, local_files_only=not download))


def prepare_sources():
    """기반 코드도 commit 고정. 원본 LICENSE/출처와 파일 hash를 함께 보존한다."""
    manifest={"revisions":SOURCE_REVISIONS,"sha256":{}}
    for project,paths in SOURCE_FILES.items():
        for path in paths:
            destination=CACHE/"reference"/project/path
            if not destination.is_file():
                url=f"https://api.github.com/repos/facebookresearch/{project}/contents/{path}?ref={SOURCE_REVISIONS[project]}"
                request=urllib.request.Request(url,headers={"User-Agent":"curious-fica-lesson"})
                with urllib.request.urlopen(request,timeout=30) as response:
                    value=json.load(response)
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(base64.b64decode(value["content"]))
            manifest["sha256"][f"{project}/{path}"]=sha256(destination)
    save_json(CACHE/"reference/manifest.json",manifest)
    return manifest


def verify_sources():
    path=CACHE/"reference/manifest.json"
    if not path.is_file():
        raise FileNotFoundError("00 --stage assets로 source revision/hash manifest도 준비하세요.")
    manifest=json.loads(path.read_text())
    if manifest["revisions"]!=SOURCE_REVISIONS:
        raise ValueError("기반 코드 revision 불일치")
    for name,digest in manifest["sha256"].items():
        if sha256(CACHE/"reference"/name)!=digest:
            raise ValueError(f"기반 코드가 바뀌었다: {name}")


def resize_rgb_alpha(rgb, alpha, size):
    h, w = rgb.shape[-2:]
    ratio = size / max(h, w)
    shape = (max(16, round(h * ratio)), max(16, round(w * ratio)))
    return (F.interpolate(rgb[None], size=shape, mode="bilinear", align_corners=False)[0],
            F.interpolate(alpha[None], size=shape, mode="bilinear", align_corners=False)[0], shape)


def camera_clip(vertices, K, w2c, height, width):
    """OpenGL camera, row0=위, pixel center=(col+.5,row+.5)."""
    camera = vertices @ w2c[:3, :3].T + w2c[:3, 3]
    depth = -camera[..., 2]
    # clip.w=depth를 유지해야 perspective-correct attribute interpolation이 된다.
    x = 2 * K[0, 0] / width * camera[..., 0] + (2 * K[0, 2] / width - 1) * depth
    y = 2 * K[1, 1] / height * camera[..., 1] + (1 - 2 * K[1, 2] / height) * depth
    near, far = .01, 10.
    z = (far + near) / (far - near) * depth - 2 * far * near / (far - near)
    return torch.stack((x, y, z, depth), -1)


def vertex_normals(vertices, faces):
    triangle = vertices[faces]
    normal = torch.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0], dim=-1)
    result = torch.zeros_like(vertices)
    for corner in range(3):
        result.index_add_(0, faces[:, corner], normal)
    return F.normalize(result, dim=-1, eps=1e-8)


def pose_linear_maps(geometry, frame):
    """고정 tracking joint의 per-vertex LBS rotation. Canonical delta의 축을 보존한다."""
    parameters = geometry.params
    pose = torch.cat([parameters[key][frame] for key in ("rotation", "neck_pose", "jaw_pose", "eyes_pose")]).reshape(-1, 3)
    rotations = elite(5).quaternion_matrix(elite(5).axis_angle_to_quaternion(pose))
    global_rotations = []
    for index, parent in enumerate(geometry.model.parents.tolist()):
        global_rotations.append(rotations[index] if parent < 0 else global_rotations[parent] @ rotations[index])
    return torch.einsum("vj,jab->vab", geometry.model.lbs_weights, torch.stack(global_rotations))


def posed_from_canonical(vertices, sample):
    """추정된 canonical geometry를 같은 frame의 고정 FLAME LBS 좌표계로 운반한다.

    Private Codec rig 대신 쓰는 명시적 adapter다. Joint center는 tracking에서 고정한다.
    """
    delta = vertices - sample["canonical_vertices"]
    return sample["posed_vertices"] + (sample["pose_linear"] @ delta[..., None]).squeeze(-1)


def rasterize_labels(vertices, canonical, faces, vt, ft, K, w2c, height, width, context):
    """Image-space label 생성. 이 tracking-derived label은 pseudo GT다."""
    import nvdiffrast.torch as dr
    clip = camera_clip(vertices, K, w2c, height, width)
    # UV seam에서는 같은 3D vertex에 서로 다른 vt가 있다. face-corner를 분리한다.
    corners = faces.reshape(-1)
    tri = torch.arange(len(corners), device=vertices.device, dtype=torch.int32).reshape(-1, 3)
    rast, _ = dr.rasterize(context, clip[corners][None], tri, [height, width])
    normals = vertex_normals(vertices, faces)
    attributes = torch.cat((vt[ft.reshape(-1)], canonical[corners], normals[corners]), -1)
    image, _ = dr.interpolate(attributes[None].contiguous(), rast, tri)
    image = image[0].flip(0).permute(2, 0, 1)
    valid = (rast[0, ..., 3] > 0).flip(0)[None]
    normal_world = F.normalize(image[5:8], dim=0)
    normal_camera = torch.einsum("ab,bhw->ahw", w2c[:3,:3], normal_world)
    return {"uv": image[:2], "xyz": image[2:5], "normal": normal_camera,
            "normal_world": normal_world, "valid": valid}


def render_mesh(vertices, faces, vt, ft, texture, K, w2c, height, width, context):
    """Diff rasterizer: texture와 vertex 양쪽으로 gradient를 보존한다."""
    import nvdiffrast.torch as dr
    corners = faces.reshape(-1)
    tri = torch.arange(len(corners), device=vertices.device, dtype=torch.int32).reshape(-1, 3)
    clip = camera_clip(vertices, K, w2c, height, width)[corners][None].contiguous()
    rast, _ = dr.rasterize(context, clip, tri, [height, width])
    uv, _ = dr.interpolate(vt[ft.reshape(-1)][None].contiguous(), rast, tri)
    color = dr.texture(texture.flip(-2)[None].permute(0, 2, 3, 1).contiguous(), uv, filter_mode="linear", boundary_mode="clamp")
    alpha = (rast[..., 3:] > 0).float()
    color = color * alpha + 1 - alpha
    color = dr.antialias(color, rast, clip, tri)
    alpha = dr.antialias(alpha, rast, clip, tri)
    return {"rgb": color[0].flip(0).permute(2, 0, 1), "alpha": alpha[0].flip(0).permute(2, 0, 1)}


def splat_partial_uv(rgb, uv, xyz, normal, valid, size):
    """Image pixel -> UV texel, bilinear forward splat + observation weight.

    grid_sample은 UV→image 위치가 있을 때의 backward sampling이다. 여기에는
    image→UV 예측이 있으므로 scatter_add를 사용한다. 유효하지 않은 UV를 clip해 넣지 않는다.
    """
    mask = valid[0].bool() & torch.isfinite(uv).all(0) & (uv >= 0).all(0) & (uv <= 1).all(0)
    x = uv[0][mask] * size - .5
    y = (1 - uv[1][mask]) * size - .5
    features = torch.cat((rgb, normal, xyz), 0)[:, mask]
    grid = features.new_zeros(9, size * size)
    weight = features.new_zeros(1, size * size)
    for oy in (0, 1):
        for ox in (0, 1):
            col, row = x.floor().long() + ox, y.floor().long() + oy
            w = (1 - (x - col).abs()) * (1 - (y - row).abs())
            keep = (col >= 0) & (col < size) & (row >= 0) & (row < size)
            index = row[keep] * size + col[keep]
            grid.scatter_add_(1, index[None].expand(9, -1), features[:, keep] * w[keep])
            weight.scatter_add_(1, index[None], w[keep][None])
    grid = (grid / weight.clamp_min(1e-8)).reshape(9, size, size)
    mask = (weight.reshape(1, size, size) > 1e-6).float()
    return torch.cat((grid[:3], mask, F.normalize(grid[3:6], dim=0), grid[6:9]), 0), weight.reshape(1, size, size)


class MeshBridge(nn.Module):
    """Barycentric UV position map -> 원래 mesh vertex의 미분 가능한 least squares.

    G=A V를 역으로 푼다. UV seam의 여러 texel이 같은 vertex를 감독한다.
    (A^T A + ridge I)V=A^T G + ridge V_template. 낮은 UV 해상도의 rank 부족만
    template ridge로 안정화한다. full mesh를 UV grid mesh로 바꾸지 않는다.
    """
    def __init__(self, topology, device, ridge=1e-4):
        super().__init__()
        self.register_buffer("faces", topology["faces"].to(device).long())
        self.register_buffer("vt", topology["vt"].to(device))
        self.register_buffer("ft", topology["ft"].to(device).long())
        self.register_buffer("template", topology["template_vertices"].to(device))
        self.register_buffer("valid", topology["valid"].to(device))
        self.register_buffer("face", topology["face"].to(device))
        self.register_buffer("bary", topology["bary"].to(device))
        self.register_buffer("indices", self.valid.flatten().nonzero().squeeze(1))
        ids = self.faces[self.face[self.valid]]
        bary = self.bary[self.valid]
        self.register_buffer("corner_ids", ids)
        self.register_buffer("weights", bary)
        n = len(self.template)
        matrix = torch.zeros(n * n, device=device)
        for i in range(3):
            for j in range(3):
                matrix.index_add_(0, ids[:, i] * n + ids[:, j], bary[:, i] * bary[:, j])
        matrix = matrix.reshape(n, n)
        matrix.diagonal().add_(ridge)
        self.register_buffer("factor", torch.linalg.cholesky(matrix))
        self.ridge = ridge

    def vertices_to_uv(self, vertices):
        values = (vertices[self.corner_ids] * self.weights[..., None]).sum(1)
        out = vertices.new_zeros(self.valid.numel(), 3)
        out[self.indices] = values
        return out.reshape(*self.valid.shape, 3).permute(2, 0, 1)

    def forward(self, xyz):
        values = xyz.permute(1, 2, 0).reshape(-1, 3)[self.indices]
        rhs = self.template * self.ridge
        for corner in range(3):
            rhs = rhs.index_add(0, self.corner_ids[:, corner], values * self.weights[:, corner:corner+1])
        return torch.cholesky_solve(rhs, self.factor)


def mesh_laplacian(vertices, faces):
    edges = torch.cat((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = torch.cat((edges, edges.flip(1)))
    result = torch.zeros_like(vertices).index_add(0, edges[:, 0], vertices[edges[:, 1]])
    degree = vertices.new_zeros(len(vertices)).index_add(0, edges[:, 0], vertices.new_ones(len(edges)))
    return vertices - result / degree.clamp_min(1)[:, None]


def cotangent_laplacian_values(vertices,faces):
    """같은 topology의 geometry-dependent, row-L1 normalized cotangent matrix.

    단순한 vertex L2 대신 삼각형의 각도/형태가 변한 정도를 비교한다.
    Coalesce의 index 순서는 topology만으로 정해져 initial/current가 일치한다.
    가느다란 생성 삼각형의 cotangent가 무한히 커지지 않게 각 row를 L1 정규화한다.
    FiCA가 Laplacian의 종류/정규화를 공개하지 않았으므로 이 선택은 GUIDE에 명시한다.
    """
    triangle=vertices[faces]
    area=torch.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0],dim=-1).norm(dim=-1).clamp_min(1e-10)
    weights=[];indices=[]
    for corner in range(3):
        a,b=(corner+1)%3,(corner+2)%3
        cot=((triangle[:,a]-triangle[:,corner])*(triangle[:,b]-triangle[:,corner])).sum(-1)/area*.5
        for source,target in ((a,b),(b,a)):
            indices.extend([torch.stack((faces[:,source],faces[:,target])),torch.stack((faces[:,source],faces[:,source]))])
            weights.extend([-cot,cot])
    matrix=torch.sparse_coo_tensor(torch.cat(indices,1),torch.cat(weights),(len(vertices),len(vertices))).coalesce()
    row=matrix.indices()[0]
    values=matrix.values()
    row_norm=values.new_zeros(len(vertices)).index_add(0,row,values.abs()).clamp_min(1e-8)
    return values/row_norm[row]


class PreparedData:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest["format"] != "fica_data_v1":
            raise ValueError("00의 prepared data가 필요하다.")
        self.identities = {}
        self.bridges = {}
        self.training_groups={}
        for row in self.samples("train"):
            self.training_groups.setdefault(row["identity"],[]).append(row)

    def identity(self, key, device="cpu"):
        if key not in self.identities:
            self.identities[key] = load_tensor(self.path / "identities" / f"{key}.pt")
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in self.identities[key].items()}

    def samples(self, split="train"):
        return [r for r in self.manifest["samples"] if r["split"] == split]

    def sample(self, row, device="cpu"):
        return load_tensor(self.path / row["file"], device)  # nested labels도 같은 device로 읽는다.

    def training_row(self):
        if not self.training_groups:
            raise ValueError("train identity/frame이 없다.")
        group=self.training_groups[random.choice(list(self.training_groups))]
        return random.choices(group,weights=[r.get("sampling_weight",1.) for r in group],k=1)[0]

    def bridge(self,key,device):
        # 같은 topology/UV의 A^T A factor는 identity와 무관하다. 수백 개를 GPU에 중복 보관하지 않는다.
        identity=self.identity(key)
        digest=hashlib.sha256()
        for name in ("faces","vt","ft","template_vertices","valid","face","bary"):
            digest.update(identity[name].contiguous().numpy().tobytes())
        token=(str(device),digest.hexdigest())
        if token not in self.bridges:
            self.bridges[token]=MeshBridge(identity,device)
        return self.bridges[token]


class TrackedGeometry:
    """기존 asset/topology 검증을 재사용하면서 VHAP dynamic_offset도 보존한다."""
    def __init__(self,scene,device):
        base=elite(3).FrameGeometry(scene,device)
        self.__dict__.update(base.__dict__)

    def forward(self,ids,personal=False,neutral=False,shaped=False):
        index=torch.as_tensor(ids,device=self.device,dtype=torch.long)
        p=self.params;batch=len(index)
        def value(key):
            tensor=p[key][index]
            return torch.zeros_like(tensor) if neutral else tensor
        dynamic=p["dynamic_offset"][index] if personal and not neutral and "dynamic_offset" in p else None
        result=self.model(shape=p["shape"][None].expand(batch,-1) if personal else torch.zeros(batch,300,device=self.device),
            expr=value("expr"),rotation=value("rotation"),neck=value("neck_pose"),jaw=value("jaw_pose"),eyes=value("eyes_pose"),
            translation=value("translation"),static_offset=p["static_offset"] if personal else None,dynamic_offset=dynamic,
            return_landmarks=False,return_verts_cano=shaped)
        return result[1] if shaped else result


def prepare_dataset(config, destination, frames, device, views_manifest=None):
    """여러 scene/identity를 지원한다. Native multi-view camera override는 명시적으로 opt-in.

    views manifest: {world_space_tracking:true, cameras:[{camera_id,K,w2c_gl,
    images:[absolute paths],alpha:[absolute paths],split:train|validation}]}.
    Train UV asset은 train camera/frame만 누적한다. 보이지 않은 texture를 GT로 복사하지 않는다.
    """
    import nvdiffrast.torch as dr
    destination = Path(destination)
    if (destination / "manifest.json").exists():
        raise FileExistsError("기존 dataset을 덮어쓰지 않는다. 새 --data 경로를 사용하세요.")
    context = dr.RasterizeCudaContext(device=device)
    samples, identity_ids = [], []
    all_xyz = []
    view_counts=[]
    for source_index, source in enumerate(config["sources"]):
        source = Path(source)
        source = source if source.is_absolute() else ROOT / source
        scene = load_tensor(source)
        key = f"id_{source_index:03d}"
        geometry = TrackedGeometry(scene, device)
        faces, vt, ft = geometry.faces, geometry.vt, geometry.ft
        size = config["uv_size"]
        lookup = elite(4).uv_lookup(vt, ft, size, context)
        canonical = geometry.forward([0], personal=True, neutral=True, shaped=True)[0]
        template = geometry.forward([0], neutral=True, shaped=True)[0]
        target_xyz = elite(4).interpolate_vertices(canonical[None], faces, lookup)[0].permute(2, 0, 1)
        target_xyz = target_xyz * lookup["valid"][None]
        identity_split = config.get("identity_splits", {}).get(key)
        if identity_split in (None,"train"):
            all_xyz.append(target_xyz[:, lookup["valid"]].cpu())
        cameras = None
        manifests = config.get("views_manifests", [])
        camera_manifest = views_manifest or (manifests[source_index] if manifests else None)
        if camera_manifest:
            camera_manifest = Path(camera_manifest)
            camera_manifest = camera_manifest if camera_manifest.is_absolute() else ROOT / camera_manifest
            info = json.loads(camera_manifest.read_text())
            if not info.get("world_space_tracking"):
                raise ValueError("views manifest는 world-space multi-view tracking이어야 한다.")
            cameras = info["cameras"]
        if cameras is None:
            cameras = [{"camera_id": "tracked_camera", "K": scene["K"], "w2c_gl": scene["w2c"], "split": "frame_split"}]
        view_counts.append(len(cameras))
        color_sum = torch.zeros(3, size, size, device=device)
        color_weight = torch.zeros(1, size, size, device=device)
        frame_ids = list(range(scene["manifest"]["num_frames"])) if frames=="all" else frames or list(range(0, scene["manifest"]["num_frames"], 25))
        for frame in frame_ids:
            if not 0 <= frame < scene["manifest"]["num_frames"]:
                raise ValueError(f"잘못된 frame: {frame}")
            posed = geometry.forward([frame], personal=True)[0]
            p = geometry.params
            local_vertices = geometry.model(shape=p["shape"][None], expr=p["expr"][frame:frame+1],
                rotation=torch.zeros(1,3,device=device), neck=p["neck_pose"][frame:frame+1],
                jaw=p["jaw_pose"][frame:frame+1], eyes=p["eyes_pose"][frame:frame+1],
                translation=torch.zeros(1,3,device=device), static_offset=p["static_offset"],
                dynamic_offset=p["dynamic_offset"][frame:frame+1] if "dynamic_offset" in p else None, return_landmarks=False)[0]
            global_rotation = elite(5).quaternion_matrix(elite(5).axis_angle_to_quaternion(p["rotation"][frame]))
            global_translation = (posed - local_vertices @ global_rotation.T).mean(0)
            if not torch.allclose(posed, local_vertices @ global_rotation.T + global_translation, atol=1e-5):
                raise ValueError("FLAME root pose의 rigid decomposition 검증 실패")
            for camera in cameras:
                record = scene["manifest"]["frames"][frame]
                if "images" in camera:
                    if camera["images"][frame] is None:
                        continue
                    image_path, alpha_path = Path(camera["images"][frame]), Path(camera["alpha"][frame])
                else:
                    image_path = Path(scene["paths"]["sequence"]) / record["image"]
                    alpha_path = Path(scene["paths"]["sequence"]) / record["alpha_png"]
                rgb = torch.from_numpy(np.array(Image.open(image_path).convert("RGB"), copy=True)).permute(2, 0, 1).float() / 255
                alpha = torch.from_numpy(np.array(Image.open(alpha_path).convert("L"), copy=True))[None].float() / 255
                original_h, original_w = rgb.shape[-2:]
                rgb, alpha, (height, width) = resize_rgb_alpha(rgb, alpha, config["image_size"])
                rgb, alpha = rgb.to(device), alpha.to(device)
                K = torch.as_tensor(camera["K"], device=device).clone().float()
                K[0] *= width / original_w
                K[1] *= height / original_h
                w2c = torch.as_tensor(camera["w2c_gl"], device=device).float()
                labels = rasterize_labels(posed, canonical, faces, vt, ft, K, w2c, height, width, context)
                labels["valid"] &= alpha > .5
                partial, weight = splat_partial_uv(rgb, labels["uv"], labels["xyz"], labels["normal"], labels["valid"], size)
                split = camera["splits"][frame] if "splits" in camera else camera["split"]
                if split == "frame_split":
                    split = "train" if frame in scene["split"]["train"] else "validation"
                texture_support = split == "train"
                if identity_split in ("validation","test"):
                    split = identity_split
                if split not in ("train", "validation", "test"):
                    raise ValueError("camera split은 train/validation/test여야 한다.")
                if texture_support:
                    color_sum += partial[:3] * weight
                    color_weight += weight
                frame_name = f"{key}_{frame:06d}_{camera['camera_id']}"
                params = scene["parameters"]
                row = {"identity": key, "frame": frame, "camera": camera["camera_id"], "split": split, "file": f"frames/{frame_name}.pt"}
                to_camera=F.normalize(torch.linalg.inv(w2c)[:3,3]-posed.mean(0),dim=0)
                view_angle=torch.acos((global_rotation[:,2]*to_camera).sum().clamp(-1,1))
                row["sampling_weight"]=1+float(view_angle)/math.pi
                # 실제 FLAME의 해부학적 68 landmark embedding을 사용한다.
                # 이 FLAME asset은 70개를 담고 있으며 STAR와 대응하는 앞의 68개를 쓴다.
                landmark_faces=geometry.model.full_lmk_faces_idx.reshape(-1)[:68]
                landmark_bary=geometry.model.full_lmk_bary_coords.reshape(-1,3)[:68]
                landmarks=(posed[faces[landmark_faces]]*landmark_bary[...,None]).sum(1)
                keypoints, depth = elite(3).project(landmarks, K, w2c)
                keypoint_valid=depth>.01
                keypoint_source="tracking-derived anatomical landmarks"
                landmark_file=Path(scene["paths"]["sequence"])/"landmark2d/STAR/0.npz"
                if "images" not in camera and landmark_file.is_file():
                    measured=torch.from_numpy(np.load(landmark_file,allow_pickle=False)["face_landmark_2d"][frame]).to(device)
                    keypoints=measured[:,:2]*measured.new_tensor([width,height])
                    keypoint_valid &= measured[:,2]>.5
                    keypoint_source="STAR 2D detector"
                item = {"rgb": rgb, "alpha": alpha, "labels": labels, "partial_gt": partial,
                        "rgb_composited_white":scene.get("import_source",{}).get("white_background",False),
                        "K": K, "w2c": w2c, "posed_vertices": posed, "canonical_vertices": canonical,
                        "pose_linear": pose_linear_maps(geometry, frame),
                        "local_vertices": local_vertices, "global_rotation": global_rotation,
                        "global_translation": global_translation,
                        "expression": params["expr"][frame].to(device), "eyes_pose": params["eyes_pose"][frame].to(device),
                        "neck_pose": params["neck_pose"][frame].to(device), "landmark_faces":landmark_faces,"landmark_bary":landmark_bary,
                        "keypoints": keypoints, "keypoint_source":keypoint_source,
                        "keypoint_valid": keypoint_valid & (keypoints[:, 0] >= 0) & (keypoints[:, 0] < width) & (keypoints[:, 1] >= 0) & (keypoints[:, 1] < height)}
                def cpu(value):
                    return {k: cpu(v) for k, v in value.items()} if isinstance(value, dict) else value.detach().cpu() if isinstance(value, torch.Tensor) else value
                save_tensor(destination / row["file"], cpu(item))
                samples.append(row)
                if len(samples) == 1:
                    rendered = render_mesh(posed, faces, vt, ft, partial[:3], K, w2c, height, width, context)
                    save_panels(destination / "inspection.jpg", [rgb, rendered["rgb"], labels["normal"] * .5 + .5, partial[:3], partial[3:4]],
                                ["Reference", "Observed UV render", "Pseudo GT normal", "Partial UV RGB", "UV visibility"])
        observed = (color_weight > 0) & lookup["valid"][None]
        texture = color_sum / color_weight.clamp_min(1e-8)
        identity = {"faces": faces, "vt": vt, "ft": ft, "template_vertices": template,
                    "canonical_vertices": canonical, "xyz": target_xyz, "texture": texture,
                    "observed": observed, **{k: lookup[k] for k in ("face", "bary", "valid")},
                    "source": str(source.resolve()), "source_sha256": sha256(source),
                    "target_kind": "FLAME pseudo geometry; train-only observed UV texture"}
        save_tensor(destination / "identities" / f"{key}.pt", {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in identity.items()})
        identity_ids.append(key)
    if not any(r["split"] == "train" for r in samples):
        raise ValueError("train sample이 없다.")
    if not all_xyz:
        raise ValueError("normalization 통계를 구할 train identity가 없다.")
    xyz = torch.cat(all_xyz, 1)
    save_json(destination / "manifest.json", {"format": "fica_data_v1", "config": config, "samples": samples,
        "identities": identity_ids, "xyz_mean": xyz.mean(1).tolist(), "xyz_std": xyz.std(1).clamp_min(.01).tolist(),
        "scope": "single-identity debugging only" if len(identity_ids) == 1 else "multi-identity; configure identity-disjoint split before generalization evaluation",
        "multi_view": any(n>1 for n in view_counts),"cameras_per_identity":view_counts, "supervision": "tracking-derived, not ground-truth high-resolution geometry",
        "missing_texture": "masked, never filled and presented as GT", "code_hashes": code_hashes()})
    print("00 prepared:", destination, "| samples", len(samples), flush=True)


def gradients(model):
    values = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not values or not all(torch.isfinite(v).all() for v in values):
        raise FloatingPointError("gradient가 없거나 NaN/Inf다.")
    return {"tensors": len(values), "absolute_sum": float(sum(v.detach().abs().sum() for v in values))}


def append_log(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def checkpoint(path, model, optimizer, config, data, step, extra=None):
    save_tensor(path, {"format": "fica_checkpoint_v1", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "config": config, "step": step, "data_sha256": sha256(data.path / "manifest.json"),
        "code_hashes": code_hashes(), "extra": extra or {}, "external_revisions":ASSETS,
        "rng_torch":torch.get_rng_state(),"rng_cuda":torch.cuda.get_rng_state_all(),"rng_python":random.getstate()})


def restore(path, model, config, data, optimizer=None):
    state = load_tensor(path)
    if state["format"] != "fica_checkpoint_v1" or state["config"] != config:
        raise ValueError("checkpoint format/config 불일치")
    if state["data_sha256"] != sha256(data.path / "manifest.json"):
        raise ValueError("checkpoint dataset 불일치")
    if state["code_hashes"] != code_hashes():
        raise ValueError("checkpoint 이후 lesson 코드가 바뀌었다. 기존 실험을 보존하세요.")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_torch"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in state["rng_cuda"]])
        random.setstate(state["rng_python"])
    return state


def training_start(path, model, optimizer, config, data, resume):
    if resume:
        return restore(path,model,config,data,optimizer)["step"]
    if Path(path).exists() and config.get("profile")!="smoke":
        raise FileExistsError("이미 checkpoint가 있다. --resume 또는 별도 lesson 복사본/설정을 사용하세요.")
    return 0


def import_vhap_export(exported, destination, assets_scene):
    """VHAP NeRF export의 *일치하는* camera/FLAME parameters를 scene+views로 가져온다.

    Raw NeRSemble calibration과 export의 recentered parameters를 섞지 않는다.
    Export는 symlink/copy하지 않고 원래 RGB/alpha 경로를 참조한다. single/multi-view 지원.
    """
    exported,destination=Path(exported).resolve(),Path(destination).resolve()
    if (destination/"scene.pt").exists():
        raise FileExistsError("기존 import를 덮어쓰지 않는다.")
    prototype=load_tensor(assets_scene)
    records=[]
    for split in ("train","val","test"):
        path=exported/f"transforms_{split}.json"
        if not path.is_file():
            continue
        meta=json.loads(path.read_text())
        for frame in meta["frames"]:
            records.append(("validation" if split=="val" else split,dict(meta,**frame)))
    if not records:
        raise FileNotFoundError("VHAP export의 transforms_train/val/test.json이 필요하다.")
    timesteps=sorted(set(int(frame["timestep_index"]) for _,frame in records))
    frame_index={value:index for index,value in enumerate(timesteps)}
    per_frame={}
    cameras={}
    seen=set()
    for split,frame in records:
        tid=int(frame["timestep_index"]);camera=str(frame["camera_id"])
        if (tid,camera) in seen:
            raise ValueError("동일 frame/camera가 여러 split에 들어 있다.")
        seen.add((tid,camera))
        values=dict(np.load(exported/frame["flame_param_path"],allow_pickle=False))
        per_frame.setdefault(tid,values)
        K=[[frame["fl_x"],0,frame["cx"]],[0,frame["fl_y"],frame["cy"]],[0,0,1]]
        w2c=np.linalg.inv(np.asarray(frame["transform_matrix"],dtype=np.float32)).tolist()
        if camera not in cameras:
            cameras[camera]={"camera_id":camera,"K":K,"w2c_gl":w2c,"images":[None]*len(timesteps),
                             "alpha":[None]*len(timesteps),"splits":[None]*len(timesteps),"split":"train"}
        c=cameras[camera]
        if not np.allclose(c["K"],K) or not np.allclose(c["w2c_gl"],w2c):
            raise ValueError("이 importer는 고정 calibrated camera를 대상으로 한다.")
        index=frame_index[tid]
        c["images"][index]=str(exported/frame["file_path"])
        if "fg_mask_path" not in frame:
            raise ValueError("Foreground alpha가 포함된 VHAP export가 필요하다.")
        c["alpha"][index]=str(exported/frame["fg_mask_path"])
        c["splits"][index]=split
    parameters={}
    first=per_frame[timesteps[0]]
    for name in ("shape","static_offset"):
        if name not in first:
            raise ValueError(f"export {name} 누락")
        if not all(np.allclose(per_frame[t][name],first[name]) for t in timesteps):
            raise ValueError("같은 identity의 shape/static_offset이 frame마다 다르다.")
        parameters[name]=torch.from_numpy(first[name].copy()).float()
    for name in ("rotation","translation","neck_pose","jaw_pose","eyes_pose","expr"):
        parameters[name]=torch.from_numpy(np.concatenate([per_frame[t][name].reshape(1,-1) for t in timesteps],0)).float()
    if "dynamic_offset" in first:
        parameters["dynamic_offset"]=torch.from_numpy(np.concatenate([per_frame[t]["dynamic_offset"].reshape(1,-1,3) for t in timesteps],0)).float()
    frame_records=[]
    training=[];validation=[]
    for tid in timesteps:
        index=frame_index[tid]
        camera=next(c for c in cameras.values() if c["images"][index] is not None)
        frame_records.append({"frame_id":index,"image":camera["images"][index],"alpha_png":camera["alpha"][index]})
        (training if any(c["splits"][index]=="train" for c in cameras.values()) else validation).append(index)
    sample_image=Image.open(frame_records[0]["image"])
    scene=dict(prototype,version="fica_imported_vhap_export_v1",parameters=parameters,
               manifest={"frames":frame_records,"num_frames":len(timesteps),"height":sample_image.height,"width":sample_image.width,"fps":25.},
               split={"train":training,"validation":validation,"gap":[]})
    scene["paths"]=dict(prototype["paths"],sequence=str(exported))
    scene["import_source"]={"export":str(exported),"timestep_indices":timesteps,"coordinate_space":"matching VHAP NeRF export; not legacy raw scene","white_background":True}
    save_tensor(destination/"scene.pt",scene)
    save_json(destination/"views.json",{"world_space_tracking":True,"cameras":list(cameras.values()),"source":str(exported)})
    print("Imported:",destination/"scene.pt","| cameras",len(cameras),flush=True)


def arguments(description):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data/prepared")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true", help="명시적으로 작은 trainable network; foundation model은 실제 가중치")
    def positive_int(value):
        value=int(value)
        if value<1:
            raise argparse.ArgumentTypeError("1 이상의 정수가 필요하다.")
        return value
    parser.add_argument("--steps", type=positive_int)
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    parser = arguments(__doc__)
    parser.add_argument("--stage", choices=("assets", "prepare", "import-vhap"), default="prepare")
    parser.add_argument("--frames", default="", help="예: 0,32; all은 전체 frame; 비우면 25 frame 간격")
    parser.add_argument("--views-manifest", type=Path)
    parser.add_argument("--vhap-export", type=Path)
    parser.add_argument("--assets-scene", type=Path, default=ROOT.parent/"elite/outputs/03/person/full/scene.pt")
    args = parser.parse_args()
    config = read_config(args.config, args.smoke)
    if args.stage == "assets":
        prepare_sources()
        for name in ASSETS:
            print(name, asset_path(name, download=True), flush=True)
        save_json(CACHE / "assets.json", ASSETS)
        return
    if args.stage == "import-vhap":
        if args.vhap_export is None:
            parser.error("--vhap-export 필요")
        import_vhap_export(args.vhap_export,args.data,args.assets_scene)
        return
    device = setup_device(args.device, config["seed"])
    frames = "all" if args.frames.strip()=="all" else [int(x) for x in args.frames.split(",") if x] or None
    prepare_dataset(config, args.data, frames, device, args.views_manifest)


if __name__ == "__main__":
    main()
