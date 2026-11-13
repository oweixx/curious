"""경로/입출력만 공유한다. 모델과 학습 수식은 번호가 붙은 lesson에 둔다.

실행 방법: 이 파일은 별도 CLI가 없는 공통 모듈이며 00~04와 smoke.py에서 import한다.
각 lesson 파일 맨 위의 실행 명령을 사용한다. python common.py만 실행하면
학습이나 출력 파일 생성은 일어나지 않는다.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMMIT = "8ba8b8f3e43c98df2e99fe46906cf011b10d34df"
UPSTREAM = ROOT / ".cache/upstream"


def run_preprocessing(command, cwd, env, log_path, echo=False):
    """Stop the entire preprocessing child group at the first printed NaN/Inf loss.

    The public pixel3DMM tracker otherwise keeps optimizing every remaining frame
    and only fails during final visualization. Keep its original log and inputs;
    never turn an invalid loss into zero or treat this as a completed stage.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    bad_loss = re.compile(rb"\bLoss(?:\s+for\s+camera)?(?:\s*[:=])?\s*[+-]?(?:nan|inf)\b", re.I)
    tail = b""
    env = dict(env, PYTHONUNBUFFERED="1")
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)

        def stop_children():
            # start_new_session gives ONLY this invocation its own process group.
            # pixel3DMM is a grandchild of the reference/driving CLI.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()

        try:
            while True:
                chunk = os.read(process.stdout.fileno(), 8192)
                if not chunk:
                    break
                log_file.write(chunk)
                log_file.flush()
                if echo:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                tail = tail[-4096:] + chunk
                match = bad_loss.search(tail)
                if match:
                    begin = max(tail.rfind(b"\r", 0, match.start()), tail.rfind(b"\n", 0, match.start())) + 1
                    detail = tail[begin:match.end()].decode("utf-8", errors="replace")
                    raise RuntimeError(f"pixel3DMM 수치 발산 감지: {detail}\n"
                        f"남은 tracking을 중단했습니다. 로그: {log_path}\n"
                        "실패한 tracking을 완료 결과로 사용하지 마세요. "
                        "로그의 첫 비정상 frame과 투영/gradient를 확인해야 합니다.")
            code = process.wait()
            if code:
                detail = "\n".join(tail.decode("utf-8", errors="replace").splitlines()[-12:])
                raise RuntimeError(f"전처리 실패 (exit={code}). 로그: {log_path}\n{detail}")
        except BaseException:
            stop_children()
            raise
        finally:
            process.stdout.close()


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
    # Keep model caches local without hiding an existing `hf auth login` token.
    # Point to the original credential file; never copy credentials into the repo.
    if "HF_HOME" not in os.environ:
        default_cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        login_token = default_cache / "huggingface/token"
        if login_token.is_file():
            os.environ.setdefault("HF_TOKEN_PATH", str(login_token))
        os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
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
