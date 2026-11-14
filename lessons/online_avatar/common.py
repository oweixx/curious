"""경로, 입출력, 시각화만 공유한다. 학습 수식은 01~03에서 직접 읽는다.

별도 실행하지 않는다. 실행 명령은 각 번호 파일 맨 위에 있다.
기존 elite Conda 환경 사용. 데이터/가중치 다운로드나 환경 변경은 하지 않는다.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parent
LESSONS = ROOT.parent
DEFAULT_SOURCE = LESSONS / "vda/data/preprocessed/042_FREE"
UPSTREAM = LESSONS / "vda/.cache/upstream"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(LESSONS / "elite/.cache/torch_extensions"))


def module(path, name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        sys.modules[name] = value
        spec.loader.exec_module(value)
    return sys.modules[name]


def lesson(index):
    return module(next(ROOT.glob(f"{index:02d}_*.py")), f"online_avatar_{index}")


def elite(index):
    return module(next((LESSONS / "elite").glob(f"{index:02d}_*.py")), f"online_elite_{index}")


def setup(device="cuda:0", seed=7):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = torch.device(device)
    if d.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("01~04는 실제 FLAME/2DGS CUDA 연산이 필요하다. CUDA_VISIBLE_DEVICES=7 사용.")
    torch.cuda.set_device(d)
    env_bin = Path(sys.executable).resolve().parent
    os.environ["PATH"] = str(env_bin) + os.pathsep + os.environ.get("PATH", "")
    if (env_bin / "nvcc").is_file():
        os.environ.setdefault("CUDA_HOME", str(env_bin.parent))
    for key, path in (("CC", "/usr/bin/gcc-11"), ("CXX", "/usr/bin/g++-11")):
        if Path(path).exists():
            os.environ.setdefault(key, path)
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "%d.%d" % torch.cuda.get_device_capability(d))
    return d


def timer():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save_pt(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def load_pt(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def require_new(path, overwrite=False):
    if Path(path).exists() and not overwrite:
        raise FileExistsError(f"이미 결과가 있다: {path}. 새 --work/--out을 쓰거나 --overwrite를 지정하세요.")


def finite_step(loss, parameters, optimizer):
    params = list(parameters)
    if not torch.isfinite(loss):
        raise FloatingPointError("유한하지 않은 loss. 업데이트하지 않고 중단한다.")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [p.grad for p in params if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError("gradient가 없거나 NaN/Inf가 있다.")
    torch.nn.utils.clip_grad_norm_(params, 1.)
    optimizer.step()


def panels(path, values, labels, size=256):
    """표시만 letterbox 한다. 직사각형 이미지를 정사각형으로 찌그러뜨리지 않는다."""
    canvas = Image.new("RGB", (size * len(values), size + 30), "#171c26")
    draw = ImageDraw.Draw(canvas)
    for i, (value, label) in enumerate(zip(values, labels)):
        if isinstance(value, torch.Tensor):
            x = value.detach().float().cpu()
            if x.ndim == 2:
                x = x[None]
            x = x.expand(3, -1, -1) if x.shape[0] == 1 else x
            value = (x.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        im = value if isinstance(value, Image.Image) else Image.fromarray(value)
        im = im.convert("RGB")
        ratio = size/max(im.width, im.height)
        im = im.resize((round(im.width*ratio), round(im.height*ratio)), Image.Resampling.LANCZOS)
        canvas.paste(im, (size*i+(size-im.width)//2, 30+(size-im.height)//2))
        draw.text((size*i+6, 8), str(label), fill="white")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def figure(path, rows):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(6 * len(rows), 4), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = np.asarray(fig.subplots(1, len(rows), squeeze=False)).ravel()
    for ax, row in zip(axes, rows):
        for label, x, y in row["series"]:
            ax.plot(x, y, marker=".", label=label)
        ax.set(title=row["title"], xlabel=row.get("xlabel", "observed time index"), ylabel=row["ylabel"])
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)


def project(xyz, K, w2c):
    """OpenCV world -> pixels. +z는 카메라 앞, y는 아래 방향."""
    camera = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    q = camera @ K.T
    return q[..., :2] / q[..., 2:].clamp_min(1e-6), camera[..., 2]


class Stream:
    """Training API는 입력 카메라만 허용한다. 평가 RGB는 04에서 명시적으로 연다."""
    def __init__(self, work, device="cpu"):
        self.work = Path(work).resolve()
        path = self.work / "stream.json"
        self.info = json.loads(path.read_text())
        if self.info["format"] != "online_avatar_v1":
            raise ValueError("stream format 불일치")
        self.fingerprint = sha(path)
        self.device = torch.device(device)
        self.inputs = self.info["input_cameras"]
        self.heldout = self.info["heldout_camera"]
        assert len(self.inputs) == 2 and len(set(self.inputs + [self.heldout])) == 3
        self.count = len(self.info["frames"])
        self.prefix = self.info["prefix"]
        self.size = self.info["image_size"]
        self.source = Path(self.info["source"])
        if sha(self.work / "observations.pt") != self.info["observations_sha256"]:
            raise ValueError("00 이후 landmark tensor가 변경됐다. stream을 다시 준비하세요.")
        self.K = {c: torch.tensor(v["K"], device=self.device, dtype=torch.float32)
                  for c, v in self.info["cameras"].items()}
        self.w2c = {c: torch.tensor(v["w2c"], device=self.device, dtype=torch.float32)
                    for c, v in self.info["cameras"].items()}
        self.landmarks = load_pt(self.work / "observations.pt")
        self._cache = {}

    def rgb(self, t, camera, *, evaluation=False):
        if camera not in self.inputs and not evaluation:
            raise ValueError("평가 카메라는 학습/초기화/조명 fitting에서 읽을 수 없다.")
        key = (int(t), camera)
        if key not in self._cache:
            rec = self.info["frames"][t]
            path = self.source / "frames" / camera / f"{rec['index']:05d}.jpg"
            if camera in self.inputs and sha(path) != rec["input_sha256"][camera]:
                raise ValueError(f"00 이후 입력 RGB가 변경됐다: {path}")
            with Image.open(path) as im:
                im = im.convert("RGB").crop(self.info["cameras"][camera]["crop"])
                im = im.resize((self.size, self.size), Image.Resampling.LANCZOS)
                x = torch.from_numpy(np.array(im, copy=True)).permute(2, 0, 1).float()/255
            self._cache[key] = x.to(self.device)
        return self._cache[key]

    def check_artifact(self, value):
        if value["stream_sha256"] != self.fingerprint:
            raise ValueError("서로 다른 stream의 결과다. 01부터 같은 --work를 사용하세요.")


def arguments(parser):
    parser.add_argument("--work", type=Path, default=ROOT / "data/042_FREE")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
