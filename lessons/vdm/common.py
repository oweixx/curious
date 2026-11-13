"""경로/입출력만 공유한다. 모델과 학습 수식은 번호가 붙은 lesson에 둔다."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMMIT = "8ba8b8f3e43c98df2e99fe46906cf011b10d34df"
UPSTREAM = ROOT / ".cache/upstream"


def upstream():
    if not (UPSTREAM / "vda").is_dir():
        raise FileNotFoundError("먼저 python 00_data_and_proxy.py setup 실행")
    revision = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if revision != COMMIT:
        raise RuntimeError(f"공개 코드 revision 불일치: {revision} != {COMMIT}")
    sys.path.insert(0, str(UPSTREAM)) if str(UPSTREAM) not in sys.path else None
    # Optional lesson-local dependencies installed during preflight; no conda mutation.
    runtime=ROOT/".cache/runtime"
    if runtime.is_dir() and str(runtime) not in sys.path:
        sys.path.insert(0,str(runtime))
    os.environ.setdefault("VDA_WEIGHTS", str(ROOT / "weights"))
    os.environ.setdefault("VDA_CACHE", str(ROOT / ".cache/preprocess"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))
    return UPSTREAM


def lesson(number):
    path = next(ROOT.glob(f"{number:02d}_*.py"))
    spec = importlib.util.spec_from_file_location(f"vdm_{number:02d}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config():
    return json.loads((ROOT / "vdm_config.json").read_text())


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save(path, value):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary); temporary.replace(path)


def load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def read_manifest(path):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("빈 triplet manifest")
    names = set()
    for row in rows:
        if not row["id"] or int(row["start"])<0 or int(row["frames"])<5 or int(row["frames"])%4!=1:
            raise ValueError("nonempty id, start>=0, frames=4k+1 (>=5) 필요")
        if row["id"] in names:
            raise ValueError("중복 clip id")
        names.add(row["id"])
        if Path(row["id"]).name != row["id"]:
            raise ValueError("clip id에는 경로를 넣을 수 없다")
        if len(set(row["cameras"][k] for k in ("reference", "driver", "target"))) != 3:
            raise ValueError("reference/driver/target은 서로 다른 camera여야 한다")
        if row["split"] not in ("train", "validation"):
            raise ValueError("split은 train 또는 validation")
        for key in ("conditioning", "target"):
            row[key] = str((path.parent / row[key]).resolve())
    # 같은 identity/sequence의 중첩된 시간 구간이 두 split으로 갈라지면 leakage다.
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            if a["split"] != b["split"] and (a["identity"],a["sequence"]) == (b["identity"],b["sequence"]):
                if max(a["start"], b["start"]) < min(a["start"]+a["frames"],b["start"]+b["frames"]):
                    raise ValueError("train/validation의 동기화 시간 구간이 겹친다")
    return rows
