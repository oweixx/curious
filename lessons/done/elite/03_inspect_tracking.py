"""03 — Tracking 결과와 avatar 학습의 좌표계를 연결한다.

읽는 순서: split_frames -> FrameGeometry -> project -> head_keep_mask -> main.
02의 raw tracking 공간을 끝까지 유지한다. NeRF export의 recentering과 섞지 않는다.
개인의 encoding mesh(shape/static offset 있음, LBS 이전 v_shaped)와 Gaussian 배치용
base mesh(shape=0, offset 없음, frame expression/pose 있음)를 명시적으로 구분한다.

CUDA_VISIBLE_DEVICES=2 python 03_inspect_tracking.py
결과: outputs/03/person/full/scene.pt, split.json, projection.jpg, head_masks/.
이후 lesson은 이 파일의 데이터 읽기/FLAME 연결 함수만 재사용한다.
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ROOT / ".cache" / "torch_extensions"))


def lesson(number):
    """숫자로 시작하는 lesson도 import 가능하게 한다. 계산은 각 번호 파일에 있다."""
    name = f"elite_lesson_{number:02d}"
    if name not in sys.modules:
        paths = list(ROOT.glob(f"{number:02d}_*.py"))
        if len(paths) != 1:
            raise RuntimeError(f"lesson {number}: 파일이 하나여야 한다: {paths}")
        spec = importlib.util.spec_from_file_location(name, paths[0])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def save_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_tensor(path, content):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(content, tmp)
    tmp.replace(path)


def read_tensor(path, device="cpu"):
    import torch
    return torch.load(path, map_location=device, weights_only=True)


def add_arguments(parser):
    parser.add_argument("--sequence", default="person")
    parser.add_argument("--run", default="full", help="02의 tracking run 이름")
    parser.add_argument("--device", default="cuda:0")


def output_dir(number, sequence, run):
    for value in (sequence, run):
        if not value or Path(value).name != value or value in (".", ".."):
            raise ValueError("sequence/run은 경로가 아닌 폴더 이름이어야 한다.")
    return ROOT / "outputs" / f"{number:02d}" / sequence / run


def scene_path(sequence, run):
    return output_dir(3, sequence, run) / "scene.pt"


def read_scene(sequence, run):
    """Tracking/입력의 변경을 감지한다. RGB 자체의 hash는 03에서 별도로 저장한다."""
    s = read_tensor(scene_path(sequence, run))
    tracking = lesson(2)
    for key in ("manifest", "tracking_parameters"):
        if tracking.sha256(Path(s["paths"][key])) != s["hashes"][key]:
            raise ValueError(f"03 이후 {key}가 변경됐다. 새 run으로 03부터 다시 준비하세요.")
    if tracking.VHAP_REVISION != s["vhap_revision"]:
        raise ValueError("VHAP source revision 불일치")
    return s


def save_image(path, tensor):
    """[C,H,W] float [0,1] -> PNG/JPEG. 시각화 값만 clip한다."""
    import numpy as np
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()*255).round().astype(np.uint8)
    Image.fromarray(array[..., 0] if array.shape[-1] == 1 else array).save(path)


def split_frames(count, block=40, fraction=.2, gap=2):
    """연속 block 끝의 frame을 validation에 두고 그 경계의 train frame을 제외한다.

    랜덤 홀짝 split은 거의 같은 인접 frame을 train/val에 넣는다. 여기서도 동일 영상의
    tracking을 공유하므로 평가 의미는 'held-out Gaussian RGB fitting'으로 한정한다.
    """
    if count < 13 or block < 10 or not 0 < fraction < .5 or gap < 0:
        raise ValueError("13개 이상 frame, block>=10, 0<val fraction<.5, gap>=0 필요")
    validation = set()
    for start in range(0, count, block):
        stop = min(count, start + block)
        if stop - start >= 10:
            validation.update(range(stop - max(2, math.ceil((stop-start)*fraction)), stop))
    excluded = {j for i in validation for j in range(max(0, i-gap), min(count, i+gap+1))}
    train = [i for i in range(count) if i not in excluded]
    if len(train) < 8 or len(validation) < 2:
        raise ValueError("split 이후 train/validation frame이 부족하다.")
    return {"train": train, "validation": sorted(validation),
            "gap": sorted(excluded - validation), "block": block,
            "validation_fraction": fraction, "gap_frames": gap}


class FrameGeometry:
    """외부 FLAME forward만 이용한다. 어떤 parameter를 넣는지는 이곳에 전부 보인다."""
    def __init__(self, scene, device):
        import torch
        from types import SimpleNamespace
        self.scene = scene
        self.device = torch.device(device)
        tracker = lesson(2)
        args = SimpleNamespace(vhap=Path(scene["paths"]["vhap"]))
        tracker.check_revision(args.vhap)
        # FLAME 원본은 relative asset path를 사용한다. 생성 시에만 원본 cwd가 필요하다.
        with tracker.vhap_context(args):
            from vhap.model.flame import FlameHead
            self.model = FlameHead(300, 100, add_teeth=True).to(device).eval()
        self.model.requires_grad_(False)
        self.params = {k: v.to(device) for k, v in scene["parameters"].items()}
        self.faces = self.model.faces.long()
        self.vt = self.model.verts_uvs
        self.ft = self.model.textures_idx.long()
        if not torch.equal(self.faces.cpu(), scene["faces"]):
            raise ValueError("FLAME/teeth topology가 03과 다르다.")
        for key in ("flame_model", "flame_masks"):
            if tracker.sha256(Path(scene["paths"][key])) != scene["hashes"][key]:
                raise ValueError(f"FLAME asset이 바뀌었다: {key}")

    def forward(self, ids, personal=False, neutral=False, root_center=False, shaped=False):
        import torch
        index = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        p, b = self.params, len(index)
        shape = p["shape"][None].expand(b, -1) if personal else torch.zeros(b, 300, device=self.device)
        def frame(name):
            value = p[name][index]
            return torch.zeros_like(value) if neutral else value
        result = self.model(
            shape=shape, expr=frame("expr"), rotation=frame("rotation"),
            neck=frame("neck_pose"), jaw=frame("jaw_pose"), eyes=frame("eyes_pose"),
            translation=frame("translation"),
            static_offset=p["static_offset"] if personal else None,
            zero_centered_at_root_node=root_center, return_landmarks=False, return_verts_cano=shaped,
        )
        # ELITE process_enc_flame은 forward의 두 번째 결과인 v_shaped를 선택한다.
        # root-centered 옵션은 첫 결과(posed vertices)에 적용되므로 v_shaped에는 영향이 없다.
        return result[1] if shaped else result


def project(vertices, K, w2c):
    """OpenGL world -> camera -> 위가 row=0인 pixel 좌표. depth=-z_camera.

    x_pixel = fx*x/depth + cx, y_pixel = cy - fy*y/depth.
    텍스처 sampling에서 pixel center는 (col+.5,row+.5)임에 주의한다.
    """
    import torch
    cam = vertices @ w2c[:3, :3].T + w2c[:3, 3]
    depth = -cam[..., 2]
    safe = depth.clamp_min(1e-8)
    pixels = torch.stack((K[0, 0]*cam[..., 0]/safe + K[0, 2],
                          K[1, 2] - K[1, 1]*cam[..., 1]/safe), dim=-1)
    return pixels, depth


def head_keep_mask(vertices, model, K, w2c, height, width):
    """VHAP export처럼 목 하단을 지나는 기울어진 선 아래를 억제한다.

    사람 전체의 RVM alpha를 head 학습에 바로 쓰면 torso까지 설명해야 한다.
    이 mask는 semantic segmentation이 아니다. Hair/ear를 FLAME 실루엣으로 잘라내지 않는다.
    """
    import torch
    import torchvision.transforms.functional as TF
    xy, _ = project(vertices[0], K, w2c)
    def point(region):
        return xy[model.mask.get_vid_by_region([region])].mean(0)
    left, right = point("neck_right_point"), point("neck_left_point")
    bottom = point("front_middle_bottom_point_boundary")
    dx = left[0] - right[0]
    if abs(float(dx)) < 1e-5:
        raise ValueError("목 기준점이 수직으로 겹친다. tracking overlay를 확인하세요.")
    slope = (left[1] - right[1]) / dx
    intercept = bottom[1] - slope * bottom[0]
    row, col = torch.meshgrid(torch.arange(height, device=xy.device)+.5,
                              torch.arange(width, device=xy.device)+.5, indexing="ij")
    mask = (row < slope*col + intercept).float()[None]
    kernel = max(3, int(.03*width)//2*2+1)
    return TF.gaussian_blur(mask, [kernel, kernel], [float(kernel), float(kernel)]).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    parser.add_argument("--split-block", type=int, default=40)
    parser.add_argument("--val-fraction", type=float, default=.2)
    parser.add_argument("--gap-frames", type=int, default=2)
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from types import SimpleNamespace
    tracker = lesson(2)
    sequence_dir, manifest = tracker.read_sequence(args)
    params, state = tracker.read_parameters(sequence_dir, manifest, output_dir(2, args.sequence, args.run))
    out = output_dir(3, args.sequence, args.run)
    if (out / "scene.pt").exists():
        raise FileExistsError("03이 이미 있다. 기존 결과를 보존하려면 새 tracking run을 사용하세요.")
    split = split_frames(manifest["num_frames"], args.split_block, args.val_fraction, args.gap_frames)
    # 02는 model path를 session에 기록한다. 실제 연결된 asset의 hash도 확인한다.
    vhap = ROOT / "external" / "vhap"
    tracker.check_revision(vhap)
    for name,expected_hash in state["assets"].items():
        if tracker.sha256(vhap/"asset"/"flame"/name)!=expected_hash:
            raise ValueError(f"02 tracking과 현재 VHAP asset이 다르다: {name}")
    h, w = manifest["height"], manifest["width"]
    focal = float(params["focal_length"].item()) * max(h, w)
    K = torch.tensor([[focal, 0., w/2], [0., focal, h/2], [0., 0., 1.]])
    w2c = torch.eye(4)
    w2c[2, 3] = -1
    keys = ("shape", "expr", "rotation", "translation", "neck_pose", "jaw_pose", "eyes_pose", "static_offset")
    paths = {"sequence": str(sequence_dir), "manifest": str(sequence_dir/"manifest.json"),
             "tracking_parameters": state["parameters_path"], "vhap": str(vhap),
             "flame_model": str((vhap/"asset/flame/flame2023.pkl").resolve()),
             "flame_masks": str((vhap/"asset/flame/FLAME_masks.pkl").resolve())}
    hashes = {k: tracker.sha256(Path(paths[k])) for k in ("manifest", "tracking_parameters", "flame_model", "flame_masks")}
    with tracker.vhap_context(SimpleNamespace(vhap=vhap)):
        from vhap.model.flame import FlameHead
        model = FlameHead(300, 100, add_teeth=True)
    scene = {"version": 1, "sequence": args.sequence, "run": args.run,
             "paths": paths, "hashes": hashes, "vhap_revision": tracker.VHAP_REVISION,
             "manifest": manifest, "split": split, "K": K, "w2c": w2c,
             "parameters": {k: torch.from_numpy(params[k]).float() for k in keys},
             "faces": model.faces.long(), "vt": model.verts_uvs, "ft": model.textures_idx.long()}
    del model
    geometry = FrameGeometry(scene, args.device)
    with torch.no_grad():
        scene["encoding_vertices"] = geometry.forward([0], personal=True, neutral=True, root_center=True, shaped=True).cpu()
        scene["base_vertices"] = geometry.forward([0], neutral=True).cpu()
        samples = set(np.linspace(0, manifest["num_frames"]-1, 6).astype(int).tolist())
        tiles = []
        image_hashes, alpha_hashes = [], []
        for i, record in enumerate(manifest["frames"]):
            vertices = geometry.forward([i], personal=True)
            keep = head_keep_mask(vertices, geometry.model, K.to(args.device), w2c.to(args.device), h, w)
            save_image(out/"head_masks"/f"{i:06d}.png", keep)
            image_hashes.append(tracker.sha256(sequence_dir/record["image"]))
            alpha_hashes.append(tracker.sha256(sequence_dir/record["alpha_png"]))
            if i in samples:
                image = Image.open(sequence_dir/record["image"]).convert("RGB")
                xy, depth = project(vertices[0], K.to(args.device), w2c.to(args.device))
                draw = ImageDraw.Draw(image)
                for x, y in xy[depth>0][::8].cpu().tolist():
                    draw.ellipse((x-1, y-1, x+1, y+1), fill=(0, 255, 0))
                image.thumbnail((240, 350))
                tiles.append(image)
            if i % 50 == 0:
                print(f"03 frame {i}/{manifest['num_frames']}: real FLAME + head cutoff")
    scene["image_hashes"], scene["alpha_hashes"] = image_hashes, alpha_hashes
    canvas = Image.new("RGB", (sum(im.width for im in tiles), max(im.height for im in tiles)), "white")
    cursor = 0
    for image in tiles:
        canvas.paste(image, (cursor, 0))
        cursor += image.width
    canvas.save(out/"projection.jpg")
    save_json(out/"split.json", split)
    save_tensor(out/"scene.pt", scene)
    print("03 완료:", out)
    print("개인 tracking mesh의 녹색 projection과 목 cutoff를 먼저 확인하세요.")


if __name__ == "__main__":
    main()
