r"""학습 lesson 시작 전 NeRSemble 전처리. 00~04와 별개의 실행 파일이다.

    cd /media/vilab/mh/curious/lessons/vdm
    python preprocess.py check
    python preprocess.py assets
    python preprocess.py frames
    CUDA_VISIBLE_DEVICES=4 python preprocess.py landmarks
    CUDA_VISIBLE_DEVICES=4 python preprocess.py face
    CUDA_VISIBLE_DEVICES=4 python preprocess.py codes
    CUDA_VISIBLE_DEVICES=4 python preprocess.py body
    CUDA_VISIBLE_DEVICES=4 python preprocess.py align
    CUDA_VISIBLE_DEVICES=4 python preprocess.py export

기본 work=data/preprocessed/042_FREE, 처음 77 frames @24fps.
다른 구간: frames --work data/preprocessed/042_FREE_val --start 240 --split validation
이후 단계에도 같은 --work를 지정한다. start는 원본 sequence의 24fps 시계다.
전체 실행: CUDA_VISIBLE_DEVICES=4 python preprocess.py all
도움말: python preprocess.py --help
export는 검사된 state/spec/reference/target까지만 만든다. 이후 00을 실행한다:
    CUDA_VISIBLE_DEVICES=4 python 00_data_and_proxy.py build-triplet \
        --spec data/preprocessed/042_FREE/clip_spec.json --out data/triplets/042_free_a

읽는 순서:
  prepare_frames: 공통 시간 -> 원본 frame index -> 이미지와 K
  triangulate: 여러 camera의 2D landmark -> dataset-world 3D
  track_face / align_face: 원본 pixel3DMM -> FLAME -> calibrated world
  stable_tracker_projection: z=0 투영이 제외된 vertex를 통해 backward까지 오염시키는 문제 방지
  estimate_body / fit_body: 원본 SAM/MHR -> differentiable FK + pelvis anchor
  export: 검사된 state -> reference/target/clip_spec (proxy/triplet은 이후 00에서 생성)

pixel3DMM은 공개 loss/iteration 설정을 사용하되 카메라 평면 근처의 투영 나눗셈을
안정화한다. NaN/Inf gradient는 optimizer update 전에 검사한다.
공개 repo에는 저자의 training-data fitting 코드가 없다. 이 파일의 triangulation,
world alignment와 fitting optimizer는 논문을 참고한 lesson 구현이다. 공개 추론용
canonical body를 042의 몸처럼 재사용하지 않는다. 검출/정렬 실패는 숨기지 않고
중단하며, raw model output과 진단 이미지를 남긴다. SAM 가중치는 접근 승인 필요.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from fractions import Fraction
import hashlib
import importlib
import importlib.util
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from common import ROOT, UPSTREAM, upstream, run_preprocessing

P3 = UPSTREAM / "third_party/pixel3dmm"
PRE = P3 / "src/pixel3dmm/preprocessing"
SAM = ROOT / ".cache/sam-3d-body"
SAM_COMMIT = "b5c765a0d89d789985e186d396315e7590887b94"
SAM_WEIGHTS = ROOT / "weights/third_party/sam-3d-body-dinov3"
DEFAULT_WORK = ROOT / "data/preprocessed/042_FREE"
WIDTH, HEIGHT, FPS = 1100, 1604, 24


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def save_npz(path, **arrays):
    """An interrupted write must not look like a completed reusable stage."""
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.stem+".partial.npz")
    np.savez_compressed(temporary,**arrays)
    temporary.replace(path)


def file_hash(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def mhr_rotations(sam_root=SAM):
    """Load official DOF/rotation conversion without importing the entire SAM network.

    Three-DOF joints use continuous 6D; hinge DOFs use sin/cos. The last three
    slots of the compact body layout hold the global root Euler rotation.
    """
    path = sam_root / "sam_3d_body/models/modules/mhr_utils.py"
    require(path)
    spec = importlib.util.spec_from_file_location("sam_mhr_rotations",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require(*paths):
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("필요한 실제 파일이 없습니다:\n" + "\n".join(missing))


def run_info(work):
    require(work / "run.json")
    return json.loads((work / "run.json").read_text())


def image_files(work, camera):
    info = run_info(work)
    files = [work / "frames" / camera / f"{i:05d}.jpg" for i in range(info["frames"])]
    require(*files)
    return files


def child_env():
    upstream()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / ".cache/runtime"), str(UPSTREAM),
        str(P3 / "src"), str(PRE / "facer"), env.get("PYTHONPATH", "")])
    env["WANDB_MODE"] = "disabled"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("OMP_NUM_THREADS","4")
    env.setdefault("MKL_NUM_THREADS","4")
    return env


def logged(command, cwd, log, env=None):
    """긴 외부 실행의 상세 로그는 파일에, 터미널에는 단계와 결과만 표시."""
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"실행: {' '.join(map(str, command))}\n로그: {log}", flush=True)
    run_preprocessing(command, cwd, env or child_env(), log)


def stable_tracker_projection(points3d, focal_length, principal_point, R_base, t_base, size=512):
    """pixel3DMM projection with a finite denominator at the camera plane.

    Adapted from pixel3dmm/tracking/tracker.py::project_points_screen_space.
    A vertex excluded by UV correspondences still participates in the full batched
    projection. Dividing its coordinates by z=0 contaminates backward (0 * NaN),
    even when every loss is finite. Preserve the original projection and signed
    camera depth away from a 0.1 mm band. The renderer's near plane is 0.1 m, so
    this band lies outside its visible range. Do not replace invalid losses by zero.
    """
    import torch
    from pixel3dmm.tracking.tracker import get_intrinsics, get_extrinsics
    intrinsics = get_intrinsics(focal_length, principal_point, size=size)
    extrinsics = get_extrinsics(R_base, t_base).repeat(focal_length.shape[0],1,1)
    n = len(points3d)
    re = n if len(extrinsics)==1 else 1
    ri = n if len(intrinsics)==1 else 1
    camera = torch.bmm(torch.cat([points3d,torch.ones_like(points3d[...,:1])],-1),
                       extrinsics.permute(0,2,1).repeat(re,1,1))
    depth = -camera[...,[2]]
    denominator = torch.where(depth>=0, depth.clamp_min(1e-4), depth.clamp_max(-1e-4))
    normalized = camera[...,:3] / denominator
    screen = -torch.bmm(normalized,intrinsics.permute(0,2,1).repeat(ri,1,1))[...,:2]
    return torch.stack([size-1-screen[...,0],screen[...,1],camera[...,2]],-1)


def run_pixel3dmm_tracker(overrides):
    """Run the public dense tracker, changing only projection numerical safety.

    Called in a separate process via `preprocess.py _track key=value ...`.
    Keep the public source/weights, losses, iteration counts and learning rates.
    """
    upstream()
    if not os.environ.get("CUDA_HOME") and (Path(sys.prefix)/"bin/nvcc").is_file():
        os.environ["CUDA_HOME"] = sys.prefix
    os.environ["PATH"] = str(Path(sys.prefix)/"bin")+os.pathsep+os.environ["PATH"]
    os.environ.setdefault("MAX_JOBS","4")
    for path in child_env()["PYTHONPATH"].split(os.pathsep):
        if path and path not in sys.path: sys.path.insert(0,path)
    import torch
    from torch.optim.optimizer import register_optimizer_step_pre_hook
    from omegaconf import OmegaConf
    from pixel3dmm import env_paths
    from pixel3dmm.tracking import tracker as module
    module.project_points_screen_space = (
        torch.compile(stable_tracker_projection) if module.COMPILE else stable_tracker_projection)
    cfg = OmegaConf.merge(OmegaConf.load(f"{env_paths.CODE_BASE}/configs/tracking.yaml"),
                          OmegaConf.from_dotlist(overrides))

    def check_gradient(optimizer, args, kwargs):
        gradients = [(group.get("name",index),p.grad) for index,group in enumerate(optimizer.param_groups)
                     for p in group["params"] if p.grad is not None]
        if not gradients: return
        valid = torch.stack([torch.isfinite(g.coalesce().values() if g.is_sparse else g).all()
                             for _,g in gradients])
        if not valid.all().item():
            bad = [str(name) for (name,_),ok in zip(gradients,valid.tolist()) if not ok]
            raise RuntimeError(f"pixel3DMM nonfinite gradient: {bad}. optimizer update 전에 중단했습니다.")

    print("pixel3DMM: signed camera-depth floor=1e-4 m; finite-gradient guard enabled",flush=True)
    handle = register_optimizer_step_pre_hook(check_gradient)
    try:
        module.Tracker(cfg).run()
    finally:
        handle.remove()


def run_official_preprocessing(stage, arguments):
    """Keep reference/driving's public CLI, route its tracker through the same fix."""
    if stage not in ("reference","driving"):
        raise ValueError(stage)
    upstream()
    from vda import paths
    from vda.preprocess import reference
    original_runner = reference.pixel3dmm_runner

    def runner(work):
        original = original_runner(work)
        def run(script, *options):
            if script == "scripts/track.py":
                env = child_env()
                env.update(paths.pixel3dmm_env(work))
                subprocess.run([sys.executable,str(Path(__file__).resolve()),"_track",*options],
                               cwd=P3,env=env,check=True)
            else:
                original(script,*options)
        return run

    reference.pixel3dmm_runner = runner
    module = importlib.import_module(f"vda.preprocess.{stage}")
    sys.argv = [module.__file__,*arguments]
    module.main()


def clone_pinned(url, path, revision):
    if not (path / ".git").is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", url, str(path)], check=True)
    current = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if current != revision:
        if subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip():
            raise RuntimeError(f"수정된 source를 자동 checkout하지 않습니다: {path}")
        subprocess.run(["git", "-C", str(path), "checkout", revision], check=True)


def prepare_assets(a):
    """공개 source/weights만 준비. conda 패키지를 변경하거나 라이선스에 동의하지 않음."""
    upstream()
    from huggingface_hub import hf_hub_download, snapshot_download
    clone_pinned("https://github.com/FacePerceiver/facer.git", PRE / "facer",
                 "ddd35c76ff840174b8a5403ad1c1255e37b8782b")
    clone_pinned("https://github.com/jhb86253817/PIPNet.git", PRE / "PIPNet",
                 "b9eab58816437403a34aa5bc3adeafe5081fd36b")
    clone_pinned("https://github.com/Zielon/MICA.git", PRE / "MICA",
                 "af22e7a5810d474bc28a1433db533723d6bd2b07")
    clone_pinned("https://github.com/facebookresearch/sam-3d-body.git", a.sam_root, SAM_COMMIT)
    for source, destination in (("farl.py", "face_parsing/farl.py"), ("facer_transform.py", "transform.py")):
        shutil.copyfile(PRE / "replacement_code" / source, PRE / "facer/facer" / destination)
    model = Path(a.flame).resolve()
    require(model)
    link = PRE / "MICA/data/FLAME2020/generic_model.pkl"
    if not link.exists():
        link.symlink_to(os.path.relpath(model, link.parent))
    elif link.resolve() != model:
        raise ValueError(f"기존 FLAME 경로를 확인하세요: {link}")
    runtime = ROOT / ".cache/runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    for name, target in (("pixel3dmm", P3 / "src/pixel3dmm"), ("facer", PRE / "facer/facer")):
        if not (runtime / name).exists():
            (runtime / name).symlink_to(os.path.relpath(target, runtime))
    downloads = [
        ("SimonGiebenhain/Pixel3dmm", "uv.ckpt", P3 / "pretrained_weights/uv.ckpt", "8b6dfa2e9e42dbd7ee338fe0e2cbd856"),
        ("SimonGiebenhain/Pixel3dmm", "normals.ckpt", P3 / "pretrained_weights/normals.ckpt", "86e1dad3392a290d91fa3bbdc2ce03ac"),
        ("SimonGiebenhain/video-diffusion-avatars", "third_party/pipnet/epoch59.pth",
         PRE / "PIPNet/snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth", "af44570e01de8bd19ff63389ac67a965"),
    ]
    for repo, name, dest, md5 in downloads:
        if dest.exists() and file_hash(dest,"md5") == md5:
            continue
        source = Path(hf_hub_download(repo, name))
        if file_hash(source,"md5") != md5:
            raise ValueError(f"공식 installer checksum 불일치: {name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    # The official Torch downloader can stall on IPv6 release-assets connections on
    # this host. Prefetch the SAME official files with bounded IPv4 curl downloads.
    checkpoints = ROOT / ".cache/torch/hub/checkpoints"
    checkpoints.mkdir(parents=True,exist_ok=True)
    for url,sha in (
        ("https://github.com/elliottzheng/face-detection/releases/download/0.0.1/mobilenet0.25_Final.pth",
         "2979b33ffafda5d74b6948cd7a5b9a7a62f62b949cef24e95fd15d2883a65220"),
        ("https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.celebm.main_ema_181500_jit.pt",
         "bbc1f0e9f68c80eb83a0b23f33850d1e10f2ec1eda96884112d111c2c1f15c79")):
        dest = checkpoints / url.rsplit("/",1)[-1]
        if dest.exists() and file_hash(dest) == sha:
            continue
        temp = dest.with_suffix(dest.suffix+".part")
        subprocess.run(["curl","-4","-fL","--retry","2","--connect-timeout","15",
                        "--max-time","600",url,"-o",str(temp)],check=True)
        if file_hash(temp) != sha:
            raise ValueError(f"official model checksum 불일치: {dest.name}")
        temp.replace(dest)
    utils = PRE / "PIPNet/FaceBoxesV2/utils"
    if not list((utils / "nms").glob("cpu_nms*.so")):
        logged([sys.executable, "build.py", "build_ext", "--inplace"], utils,
               ROOT / ".cache/preprocess_setup_nms.log")
    if a.download_sam:
        try:
            snapshot_download("facebook/sam-3d-body-dinov3", local_dir=str(a.sam_weights))
        except Exception as exc:
            # HF 오류 메시지에 인증 정보가 섞이지 않도록 유형만 출력한다.
            raise RuntimeError(f"SAM 다운로드: {type(exc).__name__}. HF 모델 페이지 접근 승인 및 로그인 필요. "
                               f"가중치 저장 위치: {a.sam_weights}") from None
    check_assets(a)


def check_assets(a):
    import importlib.util
    upstream()
    files = {
        "FLAME2020": PRE / "MICA/data/FLAME2020/generic_model.pkl",
        "FLAME landmarks": PRE / "MICA/data/FLAME2020/landmark_embedding.npy",
        "pixel3DMM UV": P3 / "pretrained_weights/uv.ckpt",
        "pixel3DMM normals": P3 / "pretrained_weights/normals.ckpt",
        "PIPNet": PRE / "PIPNet/snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth",
        "X-NeMo": ROOT / "weights/third_party/xnemo_motion_encoder.pth",
        "SAM source": a.sam_root / "sam_3d_body/__init__.py",
        "SAM weights": a.sam_weights / "model.ckpt",
        "SAM config": a.sam_weights / "model_config.yaml",
        "MHR": a.sam_weights / "assets/mhr_model.pt",
    }
    for name, path in files.items():
        print(f"{'OK     ' if path.is_file() else 'MISSING'} {name}: {path}")
    modules = ("face_alignment", "mediapy", "pytorch_lightning", "dreifus", "pyvista", "facer", "tyro")
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        print("부족한 Python import: " + ", ".join(missing))
        print("의존성 목록: .cache/upstream/envs/requirements_preprocess.txt (기존 Torch 유지)")
    if not (a.sam_weights / "model.ckpt").is_file():
        print("SAM 접근 승인: https://huggingface.co/facebook/sam-3d-body-dinov3\n"
              "승인 및 HF 로그인 후: python preprocess.py assets --download-sam")
    return all(p.is_file() for p in files.values()) and not missing


def probe_video(path):
    result = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,start_time",
        "-of", "json", str(path)], text=True))["streams"][0]
    if Fraction(result["avg_frame_rate"]) != Fraction(result["r_frame_rate"]):
        raise ValueError(f"CFR 동기화 데이터만 지원합니다: {path}")
    return result


def sample_indices(source_fps, start, count):
    """24fps 공통 시각에서 가장 가까운 source frame. 모든 camera에 동일하게 적용."""
    fps = Fraction(source_fps)
    # float rounding 대신 정수 유리수로 half-up rounding.
    clock = np.arange(start, start + count, dtype=np.int64)
    num = clock * fps.numerator
    den = FPS * fps.denominator
    return (2 * num + den) // (2 * den)


def prepare_frames(a):
    import cv2
    from tqdm import tqdm
    if a.frames < 5 or a.frames % 4 != 1 or a.start < 0:
        raise ValueError("frames >=5, frames=4k+1, start>=0 필요")
    if len({a.reference_camera, a.driver_camera, a.target_camera}) != 3:
        raise ValueError("reference/driver/target은 서로 다른 camera여야 합니다")
    cameras = list(dict.fromkeys([a.reference_camera, a.driver_camera, a.target_camera, *a.extra_cameras]))
    subject = a.subject_dir.resolve()
    calibration = subject / "calibration/camera_params.json"
    require(calibration)
    cal = json.loads(calibration.read_text())
    videos = {c: subject / "sequences" / a.sequence / "images" / f"cam_{c}.mp4" for c in cameras}
    require(*videos.values())
    probes = {c: probe_video(p) for c, p in videos.items()}
    p0 = probes[cameras[0]]
    source_fps = p0["avg_frame_rate"]
    indices = sample_indices(source_fps, a.start, a.frames)
    for c, p in probes.items():
        if (p["width"], p["height"]) != (2200, 3208):
            raise ValueError(f"calibration 원본 해상도 2200x3208 필요: {c}")
        if p["avg_frame_rate"] != source_fps or float(p.get("start_time", 0)) != float(p0.get("start_time", 0)):
            raise ValueError(f"camera {c}: 시간축 불일치")
        if indices[-1] >= int(p["nb_frames"]):
            raise ValueError(f"요청 구간이 영상을 벗어납니다: {c}")
        if c not in cal["world_2_cam"]:
            raise ValueError(f"calibration에 없는 camera {c}")
    info = dict(version=1, identity=subject.name, sequence=a.sequence, split=a.split,
        start=a.start, frames=a.frames, fps=FPS, source_fps=source_fps,
        source_indices=indices.tolist(), timestamps=(np.arange(a.start, a.start+a.frames)/FPS).tolist(),
        cameras=cameras, reference_camera=a.reference_camera, driver_camera=a.driver_camera,
        target_camera=a.target_camera, face_camera=a.reference_camera,
        source_videos={c: str(p) for c, p in videos.items()},
        source_files={c: dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for c,p in videos.items()},
        calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
        image_size=[WIDTH, HEIGHT])
    work = a.work
    if (work / "run.json").exists() and run_info(work) != info:
        raise FileExistsError("기존 work와 입력/구간이 다릅니다. 새 --work 폴더를 사용하세요")
    write_json(work / "run.json", info)
    (work / "cameras").mkdir(exist_ok=True)
    K = np.diag([.5, .5, 1.]) @ np.asarray(cal["intrinsics"], np.float64)
    sheets = []
    for c in cameras:
        w2c = np.asarray(cal["world_2_cam"][c], np.float64)
        np.savez(work / "cameras" / f"{c}.npz", K=K, c2w=np.linalg.inv(w2c),
                 world_2_cam=w2c, image_size=[WIDTH, HEIGHT])
        dest = work / "frames" / c
        dest.mkdir(parents=True, exist_ok=True)
        done = dest / ".complete.json"
        if not done.exists():
            cap = cv2.VideoCapture(str(videos[c]))
            try:
                # Sequential decode gives deterministic frame numbering (no approximate seeking).
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
                wanted = {int(source): local for local, source in enumerate(indices)}
                bar = tqdm(total=a.frames, desc=f"frames {c}")
                for source in range(int(indices[0]), int(indices[-1]) + 1):
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f"video decode 실패: {c}, source frame {source}")
                    if source in wanted:
                        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                        if not cv2.imwrite(str(dest / f"{wanted[source]:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 97]):
                            raise OSError("JPEG 저장 실패")
                        bar.update()
                bar.close()
            finally:
                cap.release()
            write_json(done, {"frames": a.frames})
        files = image_files(work, c)
        sheets.append([Image.open(files[i]).copy() for i in (0, a.frames//2, a.frames-1)])
    contact_sheet(sheets, cameras, work / "frames.jpg")
    print(f"동기화 완료: {work / 'frames.jpg'}; source indices {indices[0]}..{indices[-1]}")


def contact_sheet(rows, labels, out):
    canvas = Image.new("RGB", (275*len(rows[0]), 425*len(rows)), "#181818")
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(rows):
        draw.text((4, r*425+4), labels[r], fill="white")
        for c, im in enumerate(row):
            canvas.paste(im.resize((275,401)), (c*275, r*425+24))
    canvas.save(out)


def project(points, K, w2c):
    cam = np.asarray(points) @ w2c[:3,:3].T + w2c[:3,3]
    pix = cam @ K.T
    return pix[...,:2] / np.maximum(pix[...,2:], 1e-8), cam[...,2]


def triangulate(xy, projections, threshold=8.):
    """Pair hypotheses + reprojection inlier selection + DLT refit.

    xy=[views,points,2], NaN=검출 실패. 얼굴 윤곽은 시점별 대응이 달라 inner51만 쓴다.
    Missing/degenerate/behind-camera points stay invalid; they are never filled with zeros as targets.
    """
    V, N, _ = xy.shape
    xyz = np.full((N,3), np.nan)
    inliers = np.zeros((V,N), bool)

    def solve(obs, Ps):
        A = np.stack([obs[:,0,None]*Ps[:,2]-Ps[:,0], obs[:,1,None]*Ps[:,2]-Ps[:,1]], 1).reshape(-1,4)
        _, s, vt = np.linalg.svd(A)
        if abs(vt[-1,3]) < 1e-9 or s[-2] < 1e-8:
            return None
        return vt[-1,:3] / vt[-1,3]

    for n in range(N):
        valid = np.isfinite(xy[:,n]).all(-1)
        ids = np.flatnonzero(valid)
        best = None
        for pair in combinations(ids, 2):
            pair = list(pair)
            X = solve(xy[pair,n], projections[pair])
            if X is None:
                continue
            q = projections @ np.r_[X,1.]
            error = np.linalg.norm(q[:,:2]/np.maximum(q[:,2:],1e-8)-xy[:,n],axis=-1)
            mask = valid & (q[:,2]>0) & (error<threshold)
            score = (int(mask.sum()), -float(error[mask].sum()))
            if mask.sum() >= 2 and (best is None or score > best[0]):
                best = score, mask
        if best is not None:
            X = solve(xy[best[1],n],projections[best[1]])
            if X is not None:
                q = projections @ np.r_[X,1.]
                error = np.linalg.norm(q[:,:2]/np.maximum(q[:,2:],1e-8)-xy[:,n],axis=-1)
                mask = valid & (q[:,2]>0) & (error<threshold)
                if mask.sum() >= 2:
                    xyz[n], inliers[:,n] = X, mask
    return xyz, inliers


def detect_landmarks(a):
    import face_alignment
    from tqdm import tqdm
    info = run_info(a.work)
    kind = getattr(face_alignment.LandmarksType, "TWO_D", None) or face_alignment.LandmarksType._2D
    detector = face_alignment.FaceAlignment(kind, device="cuda", flip_input=False)
    all_xy, projections = [], []
    for c in info["cameras"]:
        path = a.work / "landmarks" / f"{c}.npz"
        if path.exists():
            xy = load_npz(path)["xy"]
        else:
            points = []
            for f in tqdm(image_files(a.work,c), desc=f"landmarks {c}"):
                result = detector.get_landmarks_from_image(np.asarray(Image.open(f).convert("RGB")))
                points.append(max(result, key=lambda p: np.ptp(p[:,1])) if result else np.full((68,2), np.nan))
            xy = np.asarray(points, np.float32)
            path.parent.mkdir(exist_ok=True)
            save_npz(path, xy=xy)
        if xy.shape != (info["frames"],68,2):
            raise ValueError(f"landmark timeline 불일치: {path}")
        all_xy.append(xy)
        camera = load_npz(a.work / "cameras" / f"{c}.npz")
        projections.append(camera["K"] @ camera["world_2_cam"][:3])
    obs = np.stack(all_xy)[:,:,17:]  # Silhouette contour has no fixed multiview correspondence.
    P = np.stack(projections)
    fitted = [triangulate(obs[:,t],P,a.triangulation_threshold) for t in range(info["frames"])]
    xyz = np.stack([x[0] for x in fitted])
    inlier = np.stack([x[1] for x in fitted],axis=1)
    save_npz(a.work / "landmarks_world.npz", xyz=xyz, inliers=inlier, xy=obs, projections=P)
    good = np.isfinite(xyz).all(-1).sum(-1)
    write_json(a.work / "triangulation.json", dict(min_valid=int(good.min()),
        median_valid=float(np.median(good)), threshold_px=a.triangulation_threshold))
    if good.min() < 20:
        raise ValueError("frame당 20개 이상의 triangulated inner landmark 필요. --extra-cameras 또는 구간 변경")
    print(f"world landmarks: frame당 {good.min()}~{good.max()}/51 points")


def track_face(a):
    """공식 pixel3DMM의 UV/normal dense fitting을 그대로 실행한다."""
    import torch
    from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
    from vda.preprocess.reference import TRACK_SUFFIX
    info = run_info(a.work)
    dest = a.work / "mono_flame.npz"
    if dest.exists():
        if any(not np.isfinite(v).all() for v in load_npz(dest).values()):
            raise ValueError(f"기존 FLAME tracking에 NaN/Inf가 있습니다: {dest}")
        print(f"재사용: {dest}"); return
    require(P3 / "pretrained_weights/uv.ckpt", P3 / "pretrained_weights/normals.ckpt",
            PRE / "MICA/data/FLAME2020/generic_model.pkl")
    work = a.work / "pixel3dmm"
    # Source folder contains only images; the upstream crop script reads every entry.
    name = f"{info['identity']}_{info['sequence']}"
    src = work / "src" / name
    src.mkdir(parents=True, exist_ok=True)
    for f in image_files(a.work,info["face_camera"]):
        if not (src / f.name).exists():
            (src / f.name).symlink_to(os.path.relpath(f,src))
    from vda import paths
    env = child_env()
    env.update(paths.pixel3dmm_env(work))
    # The official tracker clamps batch size to T and forms temporal pairs. With a
    # 5-frame smoke clip, batch_size=8 becomes 5 and fails its even-batch assertion.
    tracker_batch = min(a.batch_size,info["frames"])
    tracker_batch -= tracker_batch % 2
    if tracker_batch < 2:
        raise ValueError("pixel3DMM temporal fitting에는 batch-size>=2 필요")
    stages = [
        ("crop", ["scripts/run_cropping.py", "--video_or_images_path", str(src)]),
        ("segmentation", ["scripts/run_facer_segmentation.py", "--video_name", name]),
        ("normals", ["scripts/network_inference.py", "model.prediction_type=normals", f"video_name={name}"]),
        ("uv", ["scripts/network_inference.py", "model.prediction_type=uv_map", f"video_name={name}"]),
        ("track", [str(Path(__file__).resolve()), "_track", f"video_name={name}",
                   f"batch_size={tracker_batch}", "cache_on_cpu=true"]),
    ]
    for stage, args in stages:
        marker = work / f".{stage}.complete"
        if not marker.exists():
            logged([sys.executable,*args], P3, a.work / "logs" / f"pixel3dmm_{stage}.log", env)
            folder = work / "preprocessed" / name
            # Several upstream scripts catch per-frame exceptions and still exit 0.
            # A completion marker therefore requires actual per-frame output files.
            expected = {"crop":[("cropped","jpg"),("PIPnet_landmarks","npy")],
                        "segmentation":[("seg_og","png")],"normals":[("p3dmm/normals","png")],
                        "uv":[("p3dmm/uv_map","png")]}.get(stage,[])
            for subdir,ext in expected:
                require(*[folder/subdir/f"{i:05d}.{ext}" for i in range(info["frames"])])
            if stage != "track": marker.touch()
    ck = work / "tracking" / f"{name}{TRACK_SUFFIX}" / "checkpoint"
    files = [ck / f"{i:05d}.frame" for i in range(info["frames"])]
    require(*files)
    result = {k: [] for k in ("expression","jaw","eyes","neck","eyelids","shape")}
    def aa(x):
        return matrix_to_axis_angle(rotation_6d_to_matrix(torch.as_tensor(x).reshape(-1,6))).numpy().reshape(-1)
    for i,f in enumerate(files):
        ckpt = torch.load(f,map_location="cpu",weights_only=False)
        fl = ckpt["flame"]
        # Pinned tracker checkpoints can contain an entire selected batch. The saved
        # frame_id selects a particular entry; silently taking [0] would repeat a frame.
        n = np.asarray(fl["exp"]).shape[0]
        if n != 1:
            raise ValueError(f"{f}: checkpoint batch={n}; per-frame export required, inspect tracker save_checkpoint")
        for key, source in (("expression","exp"),("shape","shape"),("eyelids","eyelids")):
            result[key].append(np.asarray(fl[source])[0].reshape(-1))
        for key in ("jaw","eyes","neck"):
            result[key].append(aa(np.asarray(fl[key])[0]))
    out = {k: np.asarray(v,np.float32) for k,v in result.items()}
    out["shape"] = np.median(out["shape"],axis=0)
    bad = [k for k,v in out.items() if not np.isfinite(v).all()]
    if bad:
        raise ValueError(f"FLAME tracking에 NaN/Inf: {bad}. 완료 결과로 저장하지 않습니다.")
    save_npz(dest,**out)
    (work / ".track.complete").touch()
    print(f"FLAME tracking 완료: {dest}")


def extract_codes(a):
    from tqdm import tqdm
    from vda.preprocess.xnemo_codes import XNemoCodeExtractor
    info = run_info(a.work)
    path = a.work / "codes_raw.npy"
    if path.exists():
        if np.load(path).shape != (info["frames"],512):
            raise ValueError("기존 code timeline 불일치")
        print(f"재사용: {path}"); return
    extractor = XNemoCodeExtractor()
    codes = np.stack([extractor.encode(np.asarray(Image.open(f).convert("RGB")))
                     for f in tqdm(image_files(a.work,info["driver_camera"]),desc="X-NeMo")])
    if extractor.n_fallback:
        raise ValueError(f"X-NeMo face 검출 실패 {extractor.n_fallback} frames. center crop fallback을 학습용으로 저장하지 않음")
    if codes.shape != (info["frames"],512) or not np.isfinite(codes).all():
        raise ValueError("잘못된 X-NeMo output")
    temporary = path.with_name("codes_raw.partial.npy")
    np.save(temporary,codes)
    temporary.replace(path)
    print(f"raw codes: {codes.shape}, norm={np.linalg.norm(codes,axis=1).mean():.2f}")


def estimate_body(a):
    """Known calibration / single person → official SAM3D, no guessed FOV or detector.

    mhr_model_params를 직접 저장한다. scale_params는 PCA 계수이므로 MHR에 그대로
    넣으면 안 된다. 공식 head에서 scale_mean + scale_params @ scale_comps를 거친
    최종 coefficient를 저장해 동일 FK를 재구성한다.
    """
    import torch
    from tqdm import tqdm
    info = run_info(a.work)
    out = a.work / "body_sam3d.npz"
    if out.exists():
        print(f"재사용: {out}"); return
    require(a.sam_root / "sam_3d_body/__init__.py", a.sam_weights / "model.ckpt",
            a.sam_weights / "model_config.yaml", a.sam_weights / "assets/mhr_model.pt")
    sys.path.insert(0,str(a.sam_root))
    # Use the official TorchScript MHR backend also used by the fitting stage.
    os.environ.setdefault("MOMENTUM_ENABLED","0")
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    camera = load_npz(a.work / "cameras" / f"{info['face_camera']}.npz")
    mhr = a.sam_weights / "assets/mhr_model.pt"
    model,cfg = load_sam_3d_body(str(a.sam_weights / "model.ckpt"),device="cuda",mhr_path=str(mhr))
    estimator = SAM3DBodyEstimator(model,cfg)
    raw = {k: [] for k in ("mhr_model_params","shape_params","expr_params","pred_cam_t","pred_keypoints_3d")}
    log = a.work / "logs/sam3d.log"
    log.parent.mkdir(exist_ok=True)
    K = torch.as_tensor(camera["K"],dtype=torch.float32,device="cuda")[None]
    with log.open("w") as f:
        for image in tqdm(image_files(a.work,info["face_camera"]),desc="SAM3D body"):
            with redirect_stdout(f), redirect_stderr(f):
                outputs = estimator.process_one_image(str(image),cam_int=K,inference_type="full")
            if len(outputs) != 1:
                raise ValueError(f"single-person body estimation 실패: {image}")
            for k in raw:
                raw[k].append(np.asarray(outputs[0][k],np.float32))
    result = {k: np.stack(v) for k,v in raw.items()}
    result.update(kp_map70=model.head_pose.keypoint_mapping[:70].detach().cpu().numpy(),
                  mhr_path=str(mhr.resolve()),camera=info["face_camera"],sam_commit=SAM_COMMIT)
    save_npz(out,**result)
    print(f"SAM raw parameters: {out}")


def flame_forward(shape, expression, jaw, eyes, neck, eyelids):
    """Differentiable official FLAME forward, with axis-angle poses as the external contract."""
    import torch
    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d
    from vda.preprocess.geometry import flame20
    n = len(expression)
    r6 = lambda x: matrix_to_rotation_6d(axis_angle_to_matrix(x))
    return flame20()(cameras=torch.eye(3,device=expression.device)[None].expand(n,-1,-1),
        shape_params=shape.reshape(1,-1).expand(n,-1),expression_params=expression,
        jaw_pose_params=r6(jaw),neck_pose_params=r6(neck),
        eye_pose_params=r6(eyes.reshape(-1,3)).reshape(n,12),eyelid_params=eyelids)[:2]


def flame_sequence(mono, device="cuda"):
    """FLAME vertices/landmarks remain in local metric coordinates. No tracker camera reused."""
    import torch
    vertices, landmarks = [], []
    tt = lambda x: torch.as_tensor(x,dtype=torch.float32,device=device)
    with torch.no_grad():
        for start in range(0,len(mono["expression"]),8):
            part = {k:v[start:start+8] for k,v in mono.items() if k != "shape"}
            v,lm = flame_forward(tt(mono["shape"]),**{k:tt(part[k]) for k in
                ("expression","jaw","eyes","neck","eyelids")})
            vertices.append(v); landmarks.append(lm)
    return torch.cat(vertices),torch.cat(landmarks)


def align_face(a):
    """Tracker의 monocular camera 대신 실제 calibrated views에 Sim(3)를 맞춘다.

    pixel3DMM을 초기값으로 사용하고 shared identity 및 frame별 expression/jaw를
    실제 multi-view 관측에 재최적화한다. world scale은 sequence 공유다.
    단안 fitting이 턱을 뒤집는 등 발산한 경우 해당 frame의 pose/expression을 중립
    초기값에서 다시 fit한다. 원본 mono_flame.npz는 보존하고 복구 내역을 기록한다.
    """
    import torch
    import torch.nn.functional as F
    from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
    from tqdm import trange
    from vda.preprocess import geometry as G
    info = run_info(a.work)
    mono = load_npz(a.work / "mono_flame.npz")
    lm = load_npz(a.work / "landmarks_world.npz")
    if len(mono["expression"]) != info["frames"]:
        raise ValueError("mono FLAME과 frames 길이 불일치")
    if any(not np.isfinite(x).all() for x in mono.values()):
        raise ValueError("pixel3DMM output에 nonfinite parameter가 있습니다")
    # A single-view local minimum is not trustworthy supervision. Preserve the raw
    # checkpoint and explicitly record neutral reinitialization of impossible poses.
    repaired = np.flatnonzero(np.linalg.norm(mono["jaw"],axis=-1)>1.)
    bad_eyes = np.linalg.norm(mono["eyes"].reshape(-1,2,3),axis=-1).max(-1)>1.2
    for key in ("jaw","expression","eyes","eyelids"):
        mono[key][repaired] = 0
    mono["eyes"][bad_eyes] = 0
    if len(repaired):
        print(f"단안 jaw 발산 {len(repaired)} frames: multi-view FLAME 재최적화 (원본 파일 보존)")
    v,local = flame_sequence(mono)
    xyz = lm["xyz"]
    valid = np.isfinite(xyz).all(-1)
    rotations,translations,scales = [],[],[]
    local_np = local.detach().cpu().numpy()[:,17:]
    for i,good in enumerate(valid):
        if good.sum() < 20:
            raise ValueError(f"frame {i}: triangulated inner landmarks 부족")
        R,t,s = G.sim3_align(local_np[i,good],xyz[i,good])
        rotations.append(R); translations.append(t); scales.append(s)
    scale = float(np.median(scales))
    if not .5 < scale < 2.:
        raise ValueError(f"FLAME / calibration 단위 또는 정렬 오류: initial scale={scale:.3f}")
    tt = lambda x: torch.as_tensor(x,dtype=torch.float32,device="cuda")
    shape = torch.nn.Parameter(tt(mono["shape"]))
    shape0 = shape.detach().clone()
    expr = torch.nn.Parameter(tt(mono["expression"]))
    jaw = torch.nn.Parameter(tt(mono["jaw"]))
    eyelids = torch.nn.Parameter(tt(mono["eyelids"]))
    eyes,neck = tt(mono["eyes"]),tt(mono["neck"])
    with torch.no_grad():
        jaw[:,0].clamp_(0,.8); jaw[:,1:].clamp_(-.15,.15); eyelids.clamp_(0,1)
    r6 = torch.nn.Parameter(matrix_to_rotation_6d(tt(np.stack(rotations))))
    trans = torch.nn.Parameter(tt(np.stack(translations)))
    log_s = torch.nn.Parameter(tt(np.log(scale)))
    initial_r6 = r6.detach().clone()
    obs = tt(np.nan_to_num(lm["xy"]))
    mask = torch.as_tensor(lm["inliers"],device="cuda")
    P = tt(lm["projections"])
    parameters = [r6,trans,log_s,shape,expr,jaw,eyelids]
    optimizer = torch.optim.Adam([{"params":[r6,shape,expr,eyelids],"lr":.005},
        {"params":[trans,log_s,jaw],"lr":.001}])
    def forward():
        verts,lmk = flame_forward(shape,expr,jaw,eyes,neck,eyelids)
        R = rotation_6d_to_matrix(r6)
        world = log_s.exp()*torch.einsum("tij,tnj->tni",R,lmk[:,17:])+trans[:,None]
        homogeneous = torch.cat([world,torch.ones_like(world[...,:1])],-1)
        cam = torch.einsum("vij,tnj->vtni",P,homogeneous)
        uv = cam[...,:2] / cam[...,2:].clamp_min(1e-6)
        return R,world,uv,cam[...,2],verts
    progress = trange(a.face_steps,desc="world FLAME alignment")
    for _ in progress:
        optimizer.zero_grad()
        R,world,uv,depth,verts = forward()
        reproj = F.smooth_l1_loss(uv[mask]/10.,obs[mask]/10.,beta=.5)
        temporal = (r6[1:]-r6[:-1]).square().mean() + 100*(trans[1:]-trans[:-1]).square().mean()
        temporal = temporal + .1*(expr[1:]-expr[:-1]).square().mean() + (jaw[1:]-jaw[:-1]).square().mean()
        prior = .005*(shape-shape0).square().mean()+.001*expr.square().mean()+.02*jaw.square().mean()
        loss = reproj + .03*temporal + .001*(r6-initial_r6).square().mean() + prior
        loss.backward()
        if any(x.grad is None or not torch.isfinite(x.grad).all() for x in parameters):
            raise ValueError("calibrated FLAME fitting gradient 오류")
        torch.nn.utils.clip_grad_norm_(parameters,10.)
        optimizer.step()
        with torch.no_grad():
            jaw[:,0].clamp_(0,.8); jaw[:,1:].clamp_(-.15,.15); eyelids.clamp_(0,1)
        progress.set_postfix(px=f"{(uv[mask]-obs[mask]).norm(dim=-1).median().item():.2f}")
    with torch.no_grad():
        R,world,uv,depth,v = forward()
        error = (uv[mask]-obs[mask]).norm(dim=-1).cpu().numpy()
        world_v = log_s.exp()*torch.einsum("tij,tnj->tni",R,v)+trans[:,None]
        result = {f"z_{k}":x.cpu().numpy() for k,x in
                  dict(expression=expr,jaw=jaw,eyes=eyes,neck=neck,eyelids=eyelids).items()}
        result.update(gtw_lm=world_v[:,G.LM].cpu().numpy(),face_landmarks=world.cpu().numpy(),
            face_verts=world_v.cpu().numpy(),shape=shape.cpu().numpy(),world_R=R.cpu().numpy(),
            world_t=trans.cpu().numpy(),world_scale=log_s.exp().cpu().numpy())
    save_npz(a.work / "face_world.npz",**result)
    passed = bool(np.median(error) < a.max_face_error and
                  np.percentile(error,95) < 2.5*a.max_face_error and (depth[mask]>0).all().item())
    report = dict(passed=passed,median_reprojection_px=float(np.median(error)),
        p95_reprojection_px=float(np.percentile(error,95)),scale=float(result["world_scale"]),
        max_face_error=a.max_face_error,steps=a.face_steps,
        reinitialized_pose_frames=repaired.tolist(),reset_eye_frames=np.flatnonzero(bad_eyes).tolist(),
        method="pixel3dmm_initialization+calibrated_multiview_FLAME_bundle_adjustment")
    write_json(a.work / "face_alignment.json",report)
    alignment_preview(a.work,result,None)
    if not passed:
        raise ValueError(f"face alignment 품질 검사 실패: {report}. face_alignment.jpg 확인")
    return result


def mhr_forward(mhr, shape, params, expression, km, c2w):
    """MHR centimetres -> SAM OpenCV metres -> dataset world. Differentiable throughout."""
    import torch
    v,jm = mhr(shape.contiguous(),params.contiguous(),expression.contiguous())
    flip = v.new_tensor([1.,-1.,-1.])
    verts = v / 100. * flip
    joints = jm[...,:3] / 100. * flip
    verts = verts @ c2w[:3,:3].T + c2w[:3,3]
    joints = joints @ c2w[:3,:3].T + c2w[:3,3]
    kp = torch.einsum("nv,tvj->tnj",km,torch.cat([verts,joints],1))
    return verts,kp,jm


def fit_body(a,face):
    """SAM identity는 고정하고 실제 MHR pose를 최적화한다. 초기 pelvis/shoulder 중앙값을 anchor로 쓴다.

    Eq.6: one trainable sequence-wide Sim(3), per-frame continuous joint rotations,
    fixed sequence shape/scales. Public SAM converters supply 6D rotations for
    three-DOF joints and sin/cos for hinge DOFs; MHR FK consumes decoded coefficients.
    Face correspondences (nose and eye centres) are a documented lesson choice.
    """
    import torch
    import torch.nn.functional as F
    from tqdm import trange
    from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
    from vda.preprocess import geometry as G
    info = run_info(a.work)
    raw = load_npz(a.work / "body_sam3d.npz")
    if len(raw["mhr_model_params"]) != info["frames"]:
        raise ValueError("SAM timeline mismatch")
    require(Path(str(raw["mhr_path"])))
    mhr = torch.jit.load(str(raw["mhr_path"]),map_location="cuda").eval()
    for p in mhr.parameters():
        p.requires_grad_(False)
    tt = lambda x: torch.as_tensor(x,dtype=torch.float32,device="cuda")
    initial = tt(raw["mhr_model_params"])
    if initial.shape[-1] != 204:
        raise ValueError(f"MHR coefficient schema: expected 136 pose + 68 scale, got {initial.shape}")
    count = len(initial)
    # The same person has one shape and bone scale, not a new identity each frame.
    shape = tt(np.median(raw["shape_params"],axis=0))[None].expand(count,-1)
    expr = tt(raw["expr_params"])
    scales = initial[:,136:].median(0).values[None].expand(count,-1).clone()
    camera = load_npz(a.work / "cameras" / f"{str(raw['camera'])}.npz")
    c2w,km = tt(camera["c2w"]),tt(raw["kp_map70"])
    with torch.no_grad():
        _,check_kp,_ = mhr_forward(mhr,tt(raw["shape_params"][:1]),initial[:1],expr[:1],km,
                                  torch.eye(4,device="cuda"))
        fk_error = float((check_kp-tt(raw["pred_keypoints_3d"][:1])).abs().max())
    if fk_error>1e-4:
        raise ValueError(f"SAM 출력과 MHR 재구성 불일치 ({fk_error:.6f} m). units/order/model version 확인")
    # One fixed pelvis translation / camera offset for the entire sequence. Keeping
    # SAM's noisy per-frame cam_t here would re-introduce the drift Eq.6 removes.
    cam_t_world = tt(np.median(raw["pred_cam_t"],axis=0)) @ c2w[:3,:3].T
    base_translation = initial[:,:3].median(0).values[None].expand(count,-1).clone()
    base_params = torch.cat([base_translation,initial[:,3:136],scales],1)
    with torch.no_grad():
        init_parts = [mhr_forward(mhr,shape[i:i+a.batch_size],base_params[i:i+a.batch_size],
                    expr[i:i+a.batch_size],km,c2w) for i in range(0,count,a.batch_size)]
        init_kp = torch.cat([x[1] for x in init_parts]) + cam_t_world
    # MHR nose / anatomical left eye / right eye ↔ FLAME iBUG68 tip / eye centres.
    full_lm = face["gtw_lm"]
    target_np = np.stack([full_lm[:,30],full_lm[:,42:48].mean(1),full_lm[:,36:42].mean(1)],1)
    R,t,s = G.sim3_align(init_kp[:,:3].cpu().numpy().reshape(-1,3),target_np.reshape(-1,3))
    if not .5 < s < 2:
        raise ValueError(f"MHR/FLAME initial alignment scale={s:.3f}: units/landmarks 확인")
    sim_r6 = torch.nn.Parameter(matrix_to_rotation_6d(tt(R)))
    sim_t = torch.nn.Parameter(tt(t))
    sim_log_s = torch.nn.Parameter(tt(np.log(s)))
    initial_world = s*(init_kp @ tt(R).T)+tt(t)
    pelvis_anchor = initial_world[:,[9,10]].mean(1).median(0).values.detach()
    shoulder_anchor = initial_world[:,[5,6]].median(0).values.detach()
    rotations = mhr_rotations(a.sam_root)
    compact = torch.cat([initial[:,6:136],initial[:,3:6]],1)
    pose = torch.nn.Parameter(rotations.compact_model_params_to_cont_body(compact))
    pose0 = pose.detach().clone()
    target = tt(target_np)
    parameters = [pose,sim_r6,sim_t,sim_log_s]
    optimizer = torch.optim.Adam([{"params":[pose,sim_r6],"lr":.002},
                                  {"params":[sim_t,sim_log_s],"lr":.001}])
    history = []
    def forward(start,end):
        compact = rotations.compact_cont_to_model_params_body(pose[start:end])
        params = torch.cat([base_translation[start:end],compact[:,130:],compact[:,:130],scales[start:end]],1)
        v,k,j = mhr_forward(mhr,shape[start:end],params,expr[start:end],km,c2w)
        align_R = rotation_6d_to_matrix(sim_r6)
        v = sim_log_s.exp()*((v+cam_t_world) @ align_R.T)+sim_t
        k = sim_log_s.exp()*((k+cam_t_world) @ align_R.T)+sim_t
        return v,k
    progress = trange(a.body_steps,desc="pelvis-anchored MHR fitting")
    for step in progress:
        optimizer.zero_grad()
        total,face_error = 0.,0.
        for start in range(0,count,a.batch_size):
            end = min(count,start+a.batch_size)
            _,kp = forward(start,end)
            pelvis = kp[:,[9,10]].mean(1)
            loss = F.mse_loss(kp[:,:3],target[start:end])
            loss = loss + F.mse_loss(pelvis,pelvis_anchor[None].expand_as(pelvis))
            loss = loss + .3*F.mse_loss(kp[:,[5,6]],shoulder_anchor[None].expand(end-start,-1,-1))
            weighted = loss * (end-start)/count
            weighted.backward()
            total += float(weighted.detach())
            face_error += float((kp[:,:3]-target[start:end]).detach().norm(dim=-1).sum())/(count*3)
        # Eq.6 prior/temporal losses live in the continuous representation, not Euler angles.
        diff = pose-pose0
        temporal = pose[1:]-pose[:-1]
        reg = .001*diff.square().mean()+.03*temporal.square().mean()
        reg.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError("MHR FK backward에 비정상 gradient")
        torch.nn.utils.clip_grad_norm_(parameters,1.)
        optimizer.step()
        if step%10 == 0 or step == a.body_steps-1:
            history.append(dict(step=step,loss=total+float(reg.detach()),face_error_m=face_error))
        progress.set_postfix(face_mm=f"{1000*face_error:.1f}")
    with torch.no_grad():
        chunks = [forward(i,min(count,i+a.batch_size)) for i in range(0,count,a.batch_size)]
        verts = torch.cat([x[0] for x in chunks]).cpu().numpy()
        kp = torch.cat([x[1] for x in chunks]).cpu().numpy()
    error = np.linalg.norm(kp[:,:3]-target_np,axis=-1)
    passed = bool(np.isfinite(verts).all() and np.median(error)<a.max_body_error
                  and np.percentile(error,95)<2.5*a.max_body_error)
    report = dict(passed=passed,median_face_error_m=float(np.median(error)),
        p95_face_error_m=float(np.percentile(error,95)),max_body_error=a.max_body_error,
        steps=a.body_steps,history=history,method="sam3d_mhr_fk_pelvis_anchored_lesson_reimplementation",
        face_correspondences="MHR nose/left_eye/right_eye to FLAME tip/eye centres",
        rotation_representation="SAM continuous 6D + hinge sincos; shared trainable Sim3")
    report["raw_fk_roundtrip_max_error_m"] = fk_error
    write_json(a.work / "body_alignment.json",report)
    save_npz(a.work / "body_world.npz",verts=verts,kp=kp,
        mhr_pose_continuous=pose.detach().cpu().numpy(),
        pelvis_anchor=pelvis_anchor.cpu().numpy(),shoulder_anchor=shoulder_anchor.cpu().numpy(),
        sim_R=rotation_6d_to_matrix(sim_r6).detach().cpu().numpy(),
        sim_t=sim_t.detach().cpu().numpy(),sim_s=sim_log_s.exp().detach().cpu().numpy())
    alignment_preview(a.work,face,dict(verts=verts,kp=kp))
    if not passed:
        raise ValueError("MHR alignment 검사 실패. body_alignment.json / alignment.jpg 확인")
    return dict(verts=verts,kp=kp)


def alignment_preview(work,face,body):
    """Real RGB + fitted mesh overlay, every view at first/middle/last frame."""
    from vda.preprocess import geometry as G
    info = run_info(work)
    faces = G.flame20().faces.detach().cpu().numpy().astype(np.int32)
    rows = []
    for c in info["cameras"]:
        camera = load_npz(work / "cameras" / f"{c}.npz")
        files = image_files(work,c)
        row = []
        for i in (0,info["frames"]//2,info["frames"]-1):
            # Render at quarter size with correspondingly scaled K.
            rgb = np.asarray(Image.open(files[i]).resize((275,401))).copy()
            K = np.diag([.25,.25,1.]) @ camera["K"]
            v = face["face_verts"][i]
            f = faces
            if body is not None:
                f = np.concatenate([faces,G.B_FACES_CUT+len(v)])
                v = np.concatenate([v,body["verts"][i,G.BIDX]])
            gray,mask = G.render_shaded(v,f,K,camera["c2w"],275,401)
            rgb[mask] = (.5*rgb[mask]+.5*gray[mask,None]).astype(np.uint8)
            row.append(Image.fromarray(rgb))
        rows.append(row)
    contact_sheet(rows,info["cameras"],work / ("alignment.jpg" if body is not None else "face_alignment.jpg"))


def align(a):
    face = align_face(a)
    # Face preview is useful even when the gated SAM weights are not available yet.
    require(a.work / "body_sam3d.npz")
    body = fit_body(a,face)
    require(a.work / "codes_raw.npy")
    codes = np.load(a.work / "codes_raw.npy")
    info = run_info(a.work)
    if codes.shape != (info["frames"],512) or not np.isfinite(codes).all():
        raise ValueError("X-NeMo timeline/values 불일치")
    state = {k:v for k,v in face.items() if k.startswith("z_") or k == "gtw_lm"}
    from vda.preprocess import geometry as G
    # Renderer faces index the compact, HEAD-REMOVED body vertex array. Passing
    # all 18439 MHR vertices with B_FACES_CUT produces a scrambled mesh.
    state.update(verts=body["verts"][:,G.BIDX],kp=body["kp"],codes=codes,
                 timeline=np.array([info["sequence"]]*info["frames"]),
                 sequence_frame=np.arange(info["start"],info["start"]+info["frames"]),
                 source_frame=np.array(info["source_indices"]),fps=FPS)
    save_npz(a.work / "aligned_state.npz",**state)
    write_json(a.work / "alignment_provenance.json",{name:file_hash(a.work/name) for name in
        ("run.json","face_world.npz","body_world.npz","codes_raw.npy","aligned_state.npz",
         "face_alignment.json","body_alignment.json")})
    print(f"aligned state 완료: {a.work / 'aligned_state.npz'}")


def export(a):
    """Only measured passing fits get a training spec. Rendering belongs to lesson 00."""
    info = run_info(a.work)
    require(a.work / "aligned_state.npz",a.work / "face_alignment.json",a.work / "body_alignment.json")
    require(a.work / "alignment_provenance.json")
    for name,expected in json.loads((a.work / "alignment_provenance.json").read_text()).items():
        if file_hash(a.work/name) != expected:
            raise ValueError(f"정렬 후 {name}이 변경됐습니다. python preprocess.py align을 다시 실행하세요")
    for name in ("face_alignment","body_alignment"):
        if not json.loads((a.work / f"{name}.json").read_text())["passed"]:
            raise ValueError(f"{name} 품질 검사를 통과하지 못했습니다")
    face = load_npz(a.work / "face_world.npz")
    from vda.preprocess.reference import crop_916
    ref = a.work / "reference"
    ref.mkdir(exist_ok=True)
    # Identity comes from the subject's real tracked face, with the official reference crop.
    np.save(ref / "flame_shape.npy",face["shape"])
    crop = crop_916(image_files(a.work,info["reference_camera"])[0],ref / "ref_916.png")
    write_json(ref / "crop.json",crop)
    target = a.work / "target_24fps.mp4"
    if not target.exists():
        temporary = a.work / "target_24fps.partial.mp4"
        temporary.unlink(missing_ok=True)
        # Target stays on the 1100x1604 calibrated canvas; builder applies the official 9:16 crop.
        subprocess.run(["ffmpeg","-nostdin","-n","-loglevel","error","-framerate",str(FPS),
            "-start_number","0","-i",str(a.work / "frames" / info["target_camera"] / "%05d.jpg"),
            "-frames:v",str(info["frames"]),"-c:v","libx264","-crf","15","-pix_fmt","yuv420p",str(temporary)],check=True)
        temporary.replace(target)
    clip_id = f"{info['identity']}_{info['sequence'].lower()}_{info['start']:06d}_{info['target_camera']}"
    spec = dict(id=clip_id,identity=info["identity"],sequence=info["sequence"],split=info["split"],
        start=0,sequence_start=info["start"],frames=info["frames"],fps=FPS,
        cameras={k:info[f"{k}_camera"] for k in ("reference","driver","target")},
        world_space="dataset",body_fitting="pelvis_anchored",state="aligned_state.npz",
        camera=f"cameras/{info['target_camera']}.npz",reference="reference",target_video=target.name,
        preprocessing="run.json",alignment_implementation="lesson_reimplementation")
    write_json(a.work / "clip_spec.json",spec)
    print(f"전처리 완료: {a.work / 'clip_spec.json'}")
    print("다음은 lesson 00에서 proxy/triplet 생성:\n"
          f"CUDA_VISIBLE_DEVICES=4 python 00_data_and_proxy.py build-triplet "
          f"--spec {a.work / 'clip_spec.json'} --out {ROOT / 'data/triplets' / clip_id}")


def add_arguments(p):
    p.add_argument("part",choices=("check","assets","frames","landmarks","face","codes","body","align-face","align","export","all"))
    p.add_argument("--work",type=Path,default=DEFAULT_WORK)
    p.add_argument("--subject-dir",type=Path,default=ROOT / "data/042")
    p.add_argument("--sequence",default="FREE")
    p.add_argument("--reference-camera",default="222200037")
    p.add_argument("--driver-camera",default="222200042")
    p.add_argument("--target-camera",default="222200036")
    p.add_argument("--extra-cameras",nargs="*",default=[])
    p.add_argument("--start",type=int,default=0,help="sequence 24fps clock의 시작 frame")
    p.add_argument("--frames",type=int,default=77)
    p.add_argument("--split",choices=("train","validation"),default="train")
    p.add_argument("--flame",default=str(ROOT / "data/assets/FLAME2020/generic_model.pkl"))
    p.add_argument("--sam-root",type=Path,default=SAM)
    p.add_argument("--sam-weights",type=Path,default=SAM_WEIGHTS)
    p.add_argument("--download-sam",action="store_true")
    p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--face-steps",type=int,default=300)
    p.add_argument("--body-steps",type=int,default=400)
    p.add_argument("--triangulation-threshold",type=float,default=8.)
    p.add_argument("--max-face-error",type=float,default=12.,help="native canvas median reprojection pixels")
    p.add_argument("--max-body-error",type=float,default=.03,help="median face alignment metres")


def dispatch(a):
    a.work = a.work.resolve()
    a.sam_root,a.sam_weights = a.sam_root.resolve(),a.sam_weights.resolve()
    if a.batch_size < 1 or a.face_steps < 1 or a.body_steps < 1:
        raise ValueError("batch-size 및 optimization steps는 양수여야 합니다")
    if a.part in ("frames","check"):
        return {"frames":prepare_frames,"check":check_assets}[a.part](a)
    upstream()
    # `python` may be invoked by absolute path without `conda activate` (smoke tests).
    # torch's JIT needs the toolkit before importing torch.utils.cpp_extension.
    if not os.environ.get("CUDA_HOME") and (Path(sys.prefix)/"bin/nvcc").is_file():
        os.environ["CUDA_HOME"] = sys.prefix
    os.environ["PATH"] = str(Path(sys.prefix)/"bin")+os.pathsep+os.environ.get("PATH","")
    os.environ.setdefault("MAX_JOBS","4")
    # Child processes need the same ignored runtime packages as this process.
    os.environ["PYTHONPATH"] = child_env()["PYTHONPATH"]
    funcs = dict(assets=prepare_assets,landmarks=detect_landmarks,face=track_face,codes=extract_codes,
                 body=estimate_body,align=align,export=export)
    funcs["align-face"] = align_face
    if a.part == "all":
        # No automatic downloads or environment changes inside a data-processing run.
        if not check_assets(a):
            raise RuntimeError("전체 실행에 필요한 자산/의존성이 부족합니다. 먼저 check 출력을 확인하세요. "
                               "SAM 준비 전에는 frames, landmarks, face, codes, align-face를 따로 실행할 수 있습니다.")
        for fn in (prepare_frames,detect_landmarks,track_face,extract_codes,estimate_body,align,export):
            fn(a)
    else:
        funcs[a.part](a)


def main():
    # Internal child entrypoints keep tracking/preprocessing outside lesson 00.
    if sys.argv[1:2] == ["_track"]:
        run_pixel3dmm_tracker(sys.argv[2:]); return
    if sys.argv[1:2] == ["_official"]:
        run_official_preprocessing(sys.argv[2],sys.argv[3:]); return
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args()
    try:
        dispatch(args)
    except (FileNotFoundError,ValueError,RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
