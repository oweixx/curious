"""02: 01의 RGB/alpha에 실제 VHAP photometric FLAME tracking을 수행한다.

읽는 순서:
  prepare_assets -> detect_landmarks -> tracking_config -> track
  -> read_parameters -> inspect_parameters -> preview -> export_dataset.

VHAP의 FLAME/LBS, loss, renderer, optimizer를 원본에서 가져온다.
이 파일에서는 입력 대응, fitting 설정, 실제 parameter 읽기와 projection을 배운다.
최적화 내부의 시작점은 external/vhap/vhap/model/tracker.py의 optimize/compute_energy다.

현재 활성화한 Conda 환경에서 실행. 기본 sequence=person, run=full.
  python 02_track_flame.py --stage prepare
  CUDA_VISIBLE_DEVICES=2 python 02_track_flame.py --stage check
  CUDA_VISIBLE_DEVICES=2 python 02_track_flame.py --stage landmarks
  CUDA_VISIBLE_DEVICES=2 python 02_track_flame.py --stage track
  python 02_track_flame.py --stage inspect
  CUDA_VISIBLE_DEVICES=2 python 02_track_flame.py --stage preview --preview-all

prepare는 원본 코드 다운로드와 자산 연결만 한다. package 설치는 GUIDE를 따른다.
check/landmarks/track/preview/export는 사용자가 실행하며 CUDA를 사용한다.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType
import urllib.request


LESSON_DIR = Path(__file__).resolve().parent
REPO_DIR = LESSON_DIR.parent.parent
VHAP_REVISION = "11eb930d7bd4473be6e7acd5d420a3fb14f3d6e0"
STAR_REVISION = "be3c8605efb03849c23093f12b9bb87908a7a4d6"
VHAP_URL = "https://github.com/Youwang-Kim/VHAP.git"
BUNDLED_ASSETS = (
    "head_template_mesh.obj", "landmark_embedding_with_eyes.npy",
    "tex_mean_painted.png", "uv_masks.npz",
)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sequence(args):
    sequence_dir = LESSON_DIR / "data" / "sequences" / args.sequence
    path = sequence_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"먼저 01을 완료하세요: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "matting_ready" or manifest["sequence"] != args.sequence:
        raise ValueError("같은 sequence의 01 matting 완료 결과가 필요하다.")
    count = manifest["num_frames"]
    if count < 2 or len(manifest["frames"]) != count:
        raise ValueError("manifest의 frame 수가 잘못됐다.")
    # VHAP은 filename의 정렬 순서로 timestep을 만든다. 누락/추가 frame을 허용하지 않는다.
    expected = [f"{index:06d}" for index in range(count)]
    images = sorted(p.stem for p in (sequence_dir / "images").glob("*.jpg"))
    alphas = sorted(p.stem for p in (sequence_dir / "alpha_maps").glob("*.jpg"))
    if images != expected or alphas != expected:
        raise ValueError("RGB/alpha JPEG의 번호가 연속하지 않거나 서로 다르다.")
    matches = sorted(path for path in sequence_dir.parent.glob(f"{args.sequence}*") if path.is_dir())
    if matches != [sequence_dir]:
        raise ValueError("VHAP은 sequence 이름을 prefix로 찾는다. 같은 prefix의 다른 폴더가 있어 모호하다.")
    for index, record in enumerate(manifest["frames"]):
        if (record["frame_id"] != index or record["image"] != f"images/{expected[index]}.jpg"
                or record.get("alpha_vhap") != f"alpha_maps/{expected[index]}.jpg"):
            raise ValueError(f"manifest의 frame 대응이 잘못됐다: {index}")
    return sequence_dir, manifest


def check_image_sizes(sequence_dir, manifest):
    from PIL import Image

    size = (manifest["width"], manifest["height"])
    for record in manifest["frames"]:
        for key in ("image", "alpha_vhap", "alpha_png"):
            with Image.open(sequence_dir / record[key]) as image:
                if image.size != size:
                    raise ValueError(f"RGB/alpha 해상도 불일치: {record[key]}")


def check_revision(vhap_dir):
    result = subprocess.run(
        ["git", "-C", str(vhap_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    if result.stdout.strip() != VHAP_REVISION:
        raise ValueError(f"이 lesson은 ELITE의 VHAP commit {VHAP_REVISION}을 기준으로 한다.")
    result = subprocess.run(
        ["git", "-C", str(vhap_dir), "diff", "--exit-code", "HEAD", "--", "vhap", "asset/flame"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise ValueError("VHAP 계산 코드/기본 자산이 수정되어 있다. 원본과 변경 내용을 먼저 확인하세요.")


def prepare_assets(args):
    read_sequence(args)
    if shutil.which("git") is None:
        raise RuntimeError("git 실행 파일이 필요하다.")
    if not args.vhap.exists():
        args.vhap.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".vhap_", dir=args.vhap.parent) as directory:
            checkout = Path(directory) / "vhap"
            subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
            subprocess.run(["git", "-C", str(checkout), "remote", "add", "origin", VHAP_URL], check=True)
            subprocess.run(["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", VHAP_REVISION], check=True)
            subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"], check=True)
            checkout.rename(args.vhap)
    check_revision(args.vhap)
    asset_dir = args.vhap / "asset" / "flame"
    for filename in BUNDLED_ASSETS:
        if not (asset_dir / filename).is_file():
            raise FileNotFoundError(f"VHAP에 포함된 자산이 없다: {filename}")
    # 라이선스 자산은 다운로드하지 않는다. 사용자가 보유한 원본을 symlink로 연결한다.
    for source, filename in ((args.flame_model, "flame2023.pkl"), (args.flame_masks, "FLAME_masks.pkl")):
        source = source.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"자산 경로를 지정하세요: {source}")
        target = asset_dir / filename
        if target.exists() or target.is_symlink():
            if target.resolve() != source:
                raise FileExistsError(f"기존 자산을 덮어쓰지 않는다: {target}. 경로를 확인하세요.")
        else:
            target.symlink_to(source)
        print(f"Asset: {target} -> {source}")
    print("VHAP 원본 준비 완료. GUIDE의 dependency 설치 후 --stage check를 실행하세요.")


@contextmanager
def vhap_context(args):
    """원본의 상대 asset 경로와 import 경로를 맞추고, cache를 lesson 안에 둔다."""
    check_revision(args.vhap)
    for filename in (*BUNDLED_ASSETS, "flame2023.pkl", "FLAME_masks.pkl"):
        if not (args.vhap / "asset" / "flame" / filename).is_file():
            raise FileNotFoundError(f"먼저 --stage prepare: {filename}")
    cache = LESSON_DIR / ".cache"
    os.environ["TORCH_HOME"] = str(cache / "torch")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(cache / "torch_extensions")
    os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(cache)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    cache.mkdir(exist_ok=True)
    previous = Path.cwd()
    sys.path.insert(0, str(args.vhap))
    os.chdir(args.vhap)
    try:
        yield
    finally:
        os.chdir(previous)
        sys.path.pop(0)


def check_runtime():
    """사용자가 실행하는 실제 CUDA/rasterizer backward 확인. Fitting은 하지 않는다."""
    import numpy as np
    import torch
    import torchvision
    from pytorch3d.structures import Meshes
    import chumpy  # FLAME pickle을 읽을 수 있는 NumPy/Python 조합인지도 확인한다.
    import nvdiffrast.torch as dr

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA가 필요하다. H200/PyTorch 설정은 GUIDE를 확인하세요.")
    print(f"Python {sys.version.split()[0]} | NumPy {np.__version__}")
    print(f"Torch {torch.__version__} / CUDA {torch.version.cuda} | torchvision {torchvision.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)} | visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    # PyTorch wheel의 runtime과 CUDA extension을 빌드하는 toolkit은 별도다.
    # cpp_extension은 import 시 CUDA_HOME을 찾으므로 shell에서 먼저 설정해야 한다.
    from torch.utils.cpp_extension import CUDA_HOME
    nvcc = Path(CUDA_HOME) / "bin" / "nvcc" if CUDA_HOME else None
    print(f"CUDA toolkit: {CUDA_HOME} | nvcc: {nvcc}")
    if nvcc is None or not nvcc.is_file():
        raise RuntimeError(
            "NVDiffrast JIT build에 필요한 CUDA toolkit/nvcc를 찾지 못했다.\n"
            f"torch.version.cuda={torch.version.cuda}는 PyTorch runtime의 버전이다.\n"
            "GUIDE의 'CUDA_HOME 오류' 항목처럼 toolkit을 준비하고, 실제 toolkit root를\n"
            "shell에서 CUDA_HOME으로 지정한 뒤 새 Python process로 실행하세요."
        )
    try:
        # 작은 삼각형의 rasterization -> interpolation -> antialias -> backward를 검증한다.
        vertices = torch.tensor([[[-.8, -.8, 0., 1.], [.8, -.8, 0., 1.], [0., .8, 0., 1.]]],
                                device="cuda", requires_grad=True)
        triangles = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        context = dr.RasterizeCudaContext()
        raster, _ = dr.rasterize(context, vertices, triangles, (32, 32))
        attributes = torch.eye(3, device="cuda")[None].contiguous()
        colors, _ = dr.interpolate(attributes, raster, triangles)
        colors = dr.antialias(colors, raster, vertices, triangles)
        colors.square().mean().backward()
        torch.cuda.synchronize()
        if not (raster[..., 3] > 0).any() or vertices.grad is None or not torch.isfinite(vertices.grad).all():
            raise RuntimeError("rasterization/gradient가 유효하지 않다.")
        # VHAP의 Laplacian regularization에서 사용하는 PyTorch3D 경로도 확인한다.
        mesh = Meshes(verts=[vertices[0, :, :3].detach()], faces=[triangles.long()])
        laplacian = mesh.laplacian_packed().to_dense()
        if not torch.isfinite(laplacian).all():
            raise RuntimeError("PyTorch3D Laplacian 계산 실패.")
    except (RuntimeError, OSError) as error:
        raise RuntimeError(
            f"CUDA/compiled extension 확인 실패: {error}\n"
            "GUIDE의 CUDA toolkit / PyTorch3D / nvdiffrast 설치 항목을 확인하세요."
        ) from None
    print("CUDA rasterization, backward, PyTorch3D Laplacian: OK")


def prepare_star_weights():
    """STAR의 공식 asset 경로만 lesson cache로 연결한다. Network 계산은 원본 그대로다."""
    import gdown

    cache = LESSON_DIR / ".cache" / "STAR"
    cache.mkdir(parents=True, exist_ok=True)
    predictor = cache / "shape_predictor_68_face_landmarks.dat"
    model = cache / "300W_STARLoss_NME_2_87.pkl"
    predictor_url = (
        "https://raw.githubusercontent.com/italojs/facial-landmarks-recognition/"
        "master/shape_predictor_68_face_landmarks.dat"
    )
    if not predictor.is_file():
        print("Downloading STAR face predictor:", predictor, flush=True)
        partial = predictor.with_suffix(".dat.partial")
        with urllib.request.urlopen(predictor_url, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(predictor)
    if not model.is_file():
        partial = model.with_suffix(".pkl.partial")
        result = gdown.download(id="1NFcZ9jzql_jnn3ulaSzUlyhS05HWB9n_", output=str(partial), quiet=False)
        if result is None:
            raise RuntimeError("STAR weight 다운로드 실패. GUIDE의 공식 경로에서 cache에 준비하세요.")
        partial.replace(model)
    # 원본 star.asset은 import 시 ~/.cache/STAR에 다운로드한다.
    # 같은 두 실제 파일 경로를 제공해서 여기서는 lesson의 cache를 사용한다.
    asset = ModuleType("star.asset")
    asset.CACHE_DIR = cache
    asset.predictor_path, asset.model_path = str(predictor), str(model)
    sys.modules["star.asset"] = asset
    return {"star_revision": STAR_REVISION, "predictor_sha256": sha256(predictor), "model_sha256": sha256(model)}


def landmark_path(sequence_dir):
    return sequence_dir / "landmark2d" / "STAR" / "0.npz"


def read_landmarks(sequence_dir, manifest, allow_missing=False):
    import numpy as np

    path = landmark_path(sequence_dir)
    if not path.is_file():
        raise FileNotFoundError("먼저 --stage landmarks를 실행하세요.")
    with np.load(path, allow_pickle=False) as archive:
        points = archive["face_landmark_2d"]
        ids = archive["timestep_id"] if "timestep_id" in archive else None
    expected = np.array([f"{index:06d}" for index in range(manifest["num_frames"])])
    if points.shape != (len(expected), 68, 3) or ids is None or not np.array_equal(ids, expected):
        raise ValueError("landmark의 frame 수/순서가 RGB와 다르다. 원본 대응을 확인하세요.")
    if not np.isfinite(points).all() or (points[..., 2] < 0).any():
        raise ValueError("landmark 값/confidence가 잘못됐다.")
    missing = (points[..., 2] == 0).any(axis=1)
    if missing[0]:
        raise ValueError("첫 frame에 검출점이 없어 초기화할 수 없다. 시작 구간을 다시 준비하세요.")
    if missing.any() and not allow_missing:
        raise ValueError("검출에 실패한 frame이 있다. landmarks_report.json을 확인하세요.")
    return points


def detect_landmarks(sequence_dir, manifest, allow_missing=False):
    import numpy as np
    from PIL import Image, ImageDraw
    from vhap.config.base import DataConfig
    from vhap.data.video_dataset import VideoDataset
    from vhap.util.landmark_detector_star import LandmarkDetectorSTAR

    destination = landmark_path(sequence_dir)
    if destination.exists():
        read_landmarks(sequence_dir, manifest, allow_missing)
        print("기존의 검증된 landmark를 사용:", destination)
        return
    detector = LandmarkDetectorSTAR()
    # Tracker가 보는 것과 같은 white-background RGB로 landmark를 검출한다.
    dataset = VideoDataset(DataConfig(root_folder=sequence_dir.parent, sequence=manifest["sequence"],
                                      use_landmark=False, use_alpha_map=True))
    landmarks, boxes, failures = [], [], []
    # 모든 frame에 한 row를 보관한다. 검출 실패 frame을 목록에서 빼면 이후 대응이 밀린다.
    for index, record in enumerate(manifest["frames"]):
        sample = dataset[index]
        if sample["timestep_id"] != Path(record["image"]).stem:
            raise ValueError("VHAP dataset과 manifest의 frame 대응이 다르다.")
        rgb = sample["rgb"]
        box, points = detector.detect_single_image(rgb)
        points = np.asarray(points, dtype=np.float32)
        if points.shape != (68, 3) or not np.isfinite(points).all() or (points[:, 2] <= 0).any():
            failures.append(index)
            points = np.full((68, 3), -1, dtype=np.float32)
            points[:, 2] = 0  # 추정 좌표를 만들지 않는다. 해당 frame의 landmark loss를 끈다.
        landmarks.append(points)
        boxes.append(np.asarray(box, dtype=np.float32).reshape(5))
        if index % 25 == 0 or index == manifest["num_frames"] - 1:
            print(f"Landmarks {index + 1}/{manifest['num_frames']} | failed={len(failures)}", flush=True)
    report_dir = LESSON_DIR / "outputs" / "02" / manifest["sequence"]
    save_json(report_dir / "landmarks_report.json", {
        "num_frames": len(landmarks), "failed_frame_ids": failures,
        "coordinates": "x/width, y/height, confidence; original frame order",
        "manifest_sha256": sha256(sequence_dir / "manifest.json"),
    })
    # 실패해도 전체 row가 든 진단 자료는 남긴다. 정상 입력으로 승인한 0.npz와 구분한다.
    if failures and (not allow_missing or 0 in failures):
        np.savez(report_dir / "failed_landmarks.npz", face_landmark_2d=np.stack(landmarks),
                 timestep_id=np.array([f"{i:06d}" for i in range(len(landmarks))]))
        raise ValueError(
            f"{len(failures)}개 frame의 얼굴 검출 실패. report의 frame을 확인하세요. "
            "첫 frame은 유효해야 한다. 이후 실패 frame의 photometric/temporal fitting을 허용하려면 "
            "--allow-missing-landmarks를 명시하세요."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.npz")
    np.savez(temporary, face_landmark_2d=np.stack(landmarks), bounding_box=np.stack(boxes),
             timestep_id=np.array([f"{i:06d}" for i in range(len(landmarks))]))
    temporary.replace(destination)
    width, height = manifest["width"], manifest["height"]
    indices = np.unique(np.linspace(0, len(landmarks) - 1, 8).astype(int))
    sheet = Image.new("RGB", (300 * 4, 460 * 2), "white")
    for slot, index in enumerate(indices):
        with Image.open(sequence_dir / manifest["frames"][index]["image"]) as image:
            tile = image.convert("RGB")
        draw = ImageDraw.Draw(tile)
        for x, y, confidence in landmarks[index]:
            if confidence <= 0:
                continue
            x, y = x * width, y * height
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="lime")
        tile.thumbnail((300, 440))
        sheet.paste(tile, ((slot % 4) * 300, (slot // 4) * 460 + 20))
        ImageDraw.Draw(sheet).text(((slot % 4) * 300, (slot // 4) * 460), f"frame {index:06d}", fill="black")
    sheet.save(report_dir / "landmarks_preview.jpg", quality=95)
    print("Landmarks:", destination)


def tracking_config(args, sequence_dir, run_dir):
    """각 단계의 기본 설정을 명시하고 photometric/static offset/teeth를 활성화한다."""
    import vhap.config.base as config

    pipeline = config.PipelineConfig(
        lmk_init_rigid=config.StageLmkInitRigidConfig(),
        lmk_init_all=config.StageLmkInitAllConfig(),
        lmk_sequential_tracking=config.StageLmkSequentialTrackingConfig(),
        lmk_global_tracking=config.StageLmkGlobalTrackingConfig(),
        rgb_init_texture=config.StageRgbInitTextureConfig(),
        rgb_init_all=config.StageRgbInitAllConfig(),
        rgb_init_offset=config.StageRgbInitOffsetConfig(),
        rgb_sequential_tracking=config.StageRgbSequentialTrackingConfig(),
        rgb_global_tracking=config.StageRgbGlobalTrackingConfig(num_epochs=args.epochs),
    )
    return config.BaseTrackingConfig(
        data=config.DataConfig(root_folder=sequence_dir.parent, sequence=args.sequence,
                               use_alpha_map=True, landmark_source="star", landmark_detector_njobs=1),
        model=config.ModelConfig(), render=config.RenderConfig(),
        log=config.LogConfig(), exp=config.ExperimentConfig(output_folder=run_dir / "tracking"),
        lr=config.LearningRateConfig(), w=config.LossWeightConfig(), pipeline=pipeline,
        batch_size=args.batch_size, async_func=False, device="cuda",
    )


def track(args, sequence_dir, manifest, run_dir):
    import torchvision
    from vhap.model.tracker import GlobalTracker

    read_landmarks(sequence_dir, manifest, args.allow_missing_landmarks)
    state_path = run_dir / "session.json"
    if state_path.exists() or (run_dir / "tracking").exists():
        raise FileExistsError("같은 run은 덮어쓰지 않는다. 새 --run 이름을 지정하세요.")
    configuration = tracking_config(args, sequence_dir, run_dir)

    class LessonTracker(GlobalTracker):
        # 계산/최적화는 원본을 상속한다. 기본 출력에서 반복 OBJ/texture dump를 생략한다.
        def log_media(self, verts, faces, lmks, albedos, output_dict, sample, timestep,
                      session, stage=None, frame_step=None, epoch=None):
            if args.save_debug_meshes:
                return super().log_media(verts, faces, lmks, albedos, output_dict, sample, timestep,
                                         session, stage, frame_step, epoch)
            grid = self.visualize_tracking(verts, lmks, albedos, output_dict, sample)
            path = self.prepare_output_path(session, timestep, "image_grid", "jpg",
                                            stage=stage, step=frame_step, epoch=epoch)
            torchvision.utils.save_image(grid, path)

    state = {
        "status": "running", "sequence": args.sequence, "run": args.run,
        "vhap_revision": VHAP_REVISION, "input_manifest_sha256": sha256(sequence_dir / "manifest.json"),
        "num_frames": manifest["num_frames"], "epochs": args.epochs, "batch_size": args.batch_size,
        "allow_missing_landmarks": args.allow_missing_landmarks,
        "configuration": json.loads(json.dumps(asdict(configuration), default=str)),
        "assets": {name: sha256(args.vhap / "asset" / "flame" / name)
                   for name in (*BUNDLED_ASSETS, "flame2023.pkl", "FLAME_masks.pkl")},
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "torchvision", "numpy", "pytorch3d", "nvdiffrast", "star")},
        "package_sources": {name: json.loads(importlib.metadata.distribution(name).read_text("direct_url.json") or "{}")
                            for name in ("pytorch3d", "nvdiffrast", "star")},
    }
    save_json(state_path, state)
    tracker = None
    try:
        tracker = LessonTracker(configuration)
        state["tracking_directory"] = str(tracker.out_dir)
        save_json(state_path, state)
        tracker.optimize()
        parameters_path = tracker.out_dir / f"tracked_flame_params_{args.epochs}.npz"
        # 원본의 평가 간격은 10 epoch다. 임의의 마지막 epoch도 저장/평가한다.
        if not parameters_path.is_file():
            tracker.evaluate(make_visualization=True, epoch=args.epochs)
        state.update(status="completed", parameters_path=str(parameters_path),
                     parameters_sha256=sha256(parameters_path))
        save_json(state_path, state)
        read_parameters(sequence_dir, manifest, run_dir)
    except Exception as error:
        state.update(status="failed", error=str(error))
        save_json(state_path, state)
        raise
    finally:
        if tracker is not None and tracker.tb_writer is not None:
            tracker.tb_writer.close()
    print("Tracking 완료:", parameters_path)
    print("다음: --stage inspect, 그다음 --stage preview")


def read_parameters(sequence_dir, manifest, run_dir):
    import numpy as np

    state = json.loads((run_dir / "session.json").read_text(encoding="utf-8"))
    if state.get("status") != "completed":
        raise ValueError("정상 완료한 tracking run이 필요하다. session.json을 확인하세요.")
    if state["input_manifest_sha256"] != sha256(sequence_dir / "manifest.json"):
        raise ValueError("tracking 이후 입력 manifest가 바뀌었다.")
    path = Path(state["parameters_path"])
    if sha256(path) != state["parameters_sha256"]:
        raise ValueError("저장한 tracking parameter 파일이 바뀌었다.")
    with np.load(path, allow_pickle=False) as archive:
        parameters = {key: archive[key] for key in archive.files}
    count = manifest["num_frames"]
    shapes = {"shape": (300,), "expr": (count, 100), "rotation": (count, 3),
              "translation": (count, 3), "neck_pose": (count, 3),
              "jaw_pose": (count, 3), "eyes_pose": (count, 6)}
    for key, shape in shapes.items():
        if key not in parameters or parameters[key].shape != shape or not np.isfinite(parameters[key]).all():
            raise ValueError(f"parameter shape/값 불일치: {key}, expected={shape}")
    expected = np.array([f"{i:06d}" for i in range(count)])
    if not np.array_equal(parameters["timestep_id"], expected):
        raise ValueError("tracking timestep과 원본 frame 순서가 다르다.")
    if tuple(parameters["image_size"]) != (manifest["height"], manifest["width"]):
        raise ValueError("tracking 이미지 크기가 원본과 다르다.")
    offset = parameters.get("static_offset")
    if offset is None or offset.ndim != 3 or offset.shape[0] != 1 or offset.shape[2] != 3:
        raise ValueError("개인 canonical static offset이 없다.")
    for key, value in parameters.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"NaN/Inf parameter: {key}")
    if parameters["focal_length"].size != 1 or parameters["focal_length"].item() <= 0:
        raise ValueError("유효한 monocular focal length가 필요하다.")
    return parameters, state


def inspect_parameters(sequence_dir, manifest, run_dir):
    os.environ["MPLCONFIGDIR"] = str(LESSON_DIR / ".cache" / "matplotlib")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parameters, state = read_parameters(sequence_dir, manifest, run_dir)
    height, width = manifest["height"], manifest["width"]
    focal_pixels = float(parameters["focal_length"].item()) * max(height, width)
    summary = {
        "parameters_path": state["parameters_path"], "num_frames": manifest["num_frames"],
        "intrinsics": [[focal_pixels, 0, width / 2], [0, focal_pixels, height / 2], [0, 0, 1]],
        "world_to_camera_opengl": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1], [0, 0, 0, 1]],
        "rotation_units": "axis-angle radians; global head rotation, neck, jaw, left/right eyes",
        "camera_source": "VHAP monocular estimated focal; NeRSemble calibration not used",
        "coordinate_note": "raw tracking space; export applies a separate mean-translation recentering",
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                   for key, value in parameters.items()},
    }
    offset_norm = np.linalg.norm(parameters["static_offset"][0], axis=-1)
    summary["static_offset_norm"] = {"mean": float(offset_norm.mean()), "max": float(offset_norm.max())}
    save_json(run_dir / "summary.json", summary)
    times = np.array([record["sequence_time_s"] for record in manifest["frames"]])
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for axis, key in zip(axes.flat[:5], ("expr", "rotation", "translation", "jaw_pose", "eyes_pose")):
        axis.plot(times, parameters[key][:, :6])
        axis.set(title=key, xlabel="sequence time (s)")
        axis.grid(alpha=.2)
    axes.flat[5].hist(offset_norm, bins=50)
    axes.flat[5].set(title="Canonical static offset norm", xlabel="FLAME model units")
    fig.tight_layout()
    fig.savefig(run_dir / "parameter_curves.png", dpi=150)
    plt.close(fig)
    for key, value in parameters.items():
        print(f"{key:20s}: {value.shape} {value.dtype}")
    print(f"Estimated fx=fy={focal_pixels:.3f}px | center=({width / 2}, {height / 2})")
    print("Summary:", run_dir / "summary.json")


def preview(args, sequence_dir, manifest, run_dir):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from vhap.model.flame import FlameHead
    from vhap.util.render_nvdiffrast import NVDiffRenderer

    parameters, state = read_parameters(sequence_dir, manifest, run_dir)
    detected = read_landmarks(sequence_dir, manifest, state.get("allow_missing_landmarks", False))
    destination = run_dir / "preview"
    model = FlameHead(300, 100, add_teeth=True).cuda().eval()
    renderer = NVDiffRenderer(use_opengl=False, lighting_type="front", lighting_space="camera")
    values = {key: torch.from_numpy(parameters[key]).float().cuda()
              for key in ("shape", "expr", "rotation", "translation", "neck_pose", "jaw_pose", "eyes_pose", "static_offset")}
    if values["static_offset"].shape[1] != model.v_template.shape[0]:
        raise ValueError("FLAME topology와 static offset vertex 수가 다르다.")
    height, width = manifest["height"], manifest["width"]
    focal = float(parameters["focal_length"].item()) * max(height, width)
    intrinsics = torch.tensor([[[focal, 0., width / 2], [0., focal, height / 2], [0., 0., 1.]]], device="cuda")
    # Tracking과 동일한 OpenGL w2c. Head의 frame별 rotation/translation은 FLAME에 적용한다.
    extrinsics = torch.eye(4, device="cuda")[None]
    extrinsics[:, 2, 3] = -1
    count = manifest["num_frames"]
    indices = np.arange(count) if args.preview_all else np.unique(np.linspace(0, count - 1, 8).astype(int))
    # 검증된 같은 checkpoint의 진단 이미지는 다시 그릴 수 있다. 학습 입력/parameter는 보존한다.
    destination.mkdir(parents=True, exist_ok=True)
    metrics = []
    thumbnails = []
    with torch.inference_mode():
        for slot, index in enumerate(indices):
            index = int(index)
            # Full FLAME: identity + expression + pose corrective + LBS + static offset.
            vertices, landmarks = model(
                shape=values["shape"][None], expr=values["expr"][[index]],
                rotation=values["rotation"][[index]], neck=values["neck_pose"][[index]],
                jaw=values["jaw_pose"][[index]], eyes=values["eyes_pose"][[index]],
                translation=values["translation"][[index]], static_offset=values["static_offset"],
            )
            rendered = renderer.render_rgba_vis(vertices, model.faces, extrinsics, intrinsics, (height, width))
            mesh_alpha = rendered["rgba"][0, ..., 3:4].cpu().numpy().clip(0, 1)
            shade = rendered["diffuse"][0].cpu().numpy().clip(0, 1)
            with Image.open(sequence_dir / manifest["frames"][index]["image"]) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
            blue_mesh = np.array([.2, .55, 1.], dtype=np.float32) * shade
            blended = rgb * (1 - .65 * mesh_alpha) + blue_mesh * .65 * mesh_alpha
            normal = (rendered["normal"][0].cpu().numpy() * .5 + .5).clip(0, 1)
            normal = normal * mesh_alpha + (1 - mesh_alpha)
            # NDC [-1,1] -> image pixel. y를 뒤집은 NDC라 위쪽이 row=0이다.
            projected = renderer.world_to_ndc(landmarks, extrinsics, intrinsics, (height, width), flip_y=True)
            predicted_pixels = (projected[0, :68, :2].cpu().numpy() + 1) * np.array([width, height]) / 2
            detected_pixels = detected[index, :, :2] * np.array([width, height])
            distance = np.linalg.norm(predicted_pixels - detected_pixels, axis=-1)
            valid = detected[index, :, 2] > 0
            inner_valid = valid[17:]
            metrics.append({"frame_id": index, "valid_detected_points": int(valid.sum()),
                            "detector_mean_error_px": float(distance[valid].mean()) if valid.any() else None,
                            "detector_inner_face_error_px": float(distance[17:][inner_valid].mean()) if inner_valid.any() else None})
            overlay = Image.fromarray(np.rint(blended.clip(0, 1) * 255).astype(np.uint8))
            draw = ImageDraw.Draw(overlay)
            for xy, color in ((detected_pixels[valid], "lime"), (predicted_pixels, "red")):
                for x, y in xy:
                    if np.isfinite([x, y]).all():
                        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
            row = Image.new("RGB", (width * 3, height + 24), "white")
            tiles = (Image.fromarray(np.rint(rgb * 255).astype(np.uint8)), overlay,
                     Image.fromarray(np.rint(normal * 255).astype(np.uint8)))
            for col, (tile, label) in enumerate(zip(tiles, ("RGB", "Mesh: green=detected, red=projected", "Camera normal"))):
                row.paste(tile, (col * width, 24))
                ImageDraw.Draw(row).text((col * width + 4, 4), f"{index:06d} | {label}", fill="black")
            row.save(destination / f"{index:06d}.jpg", quality=95, subsampling=0)
            if index in np.unique(np.linspace(0, count - 1, 8).astype(int)):
                tile = row.copy()
                tile.thumbnail((900, 460))
                thumbnails.append(tile)
            if slot % 25 == 0 or slot == len(indices) - 1:
                print(f"Preview {slot + 1}/{len(indices)}", flush=True)
    sheet = Image.new("RGB", (900, sum(tile.height for tile in thumbnails)), "white")
    y = 0
    for tile in thumbnails:
        sheet.paste(tile, (0, y))
        y += tile.height
    sheet.save(run_dir / "mesh_preview.jpg", quality=95)
    save_json(run_dir / "projection_report.json", {"frames": metrics,
              "note": "Agreement with detected 2D landmarks; not a 3D ground-truth error."})
    if args.preview_all:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("Preview 이미지는 저장했다. 영상 생성에는 FFmpeg가 필요하다.")
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-framerate", str(manifest["fps"]), "-start_number", "0", "-i", str(destination / "%06d.jpg"),
                   "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-crf", "18",
                   "-pix_fmt", "yuv420p", str(run_dir / "mesh_overlay.mp4")]
        subprocess.run(command, check=True)
    print("Mesh preview:", run_dir / "mesh_preview.jpg")


def export_dataset(sequence_dir, manifest, run_dir):
    _, state = read_parameters(sequence_dir, manifest, run_dir)
    from vhap.export_as_nerf_dataset import main as export_vhap

    destination = LESSON_DIR / "data" / "processed" / manifest["sequence"] / state["run"]
    if destination.exists():
        raise FileExistsError(f"기존 export를 덮어쓰지 않는다: {destination}")
    export_vhap(src_folder=Path(state["tracking_directory"]), tgt_folder=destination,
                flame_mode="param", create_mask_from_mesh=True, background_color="white")
    transforms = json.loads((destination / "transforms.json").read_text())
    if len(transforms["frames"]) != manifest["num_frames"]:
        raise ValueError("Export의 frame 수가 원본과 다르다.")
    for index, record in enumerate(transforms["frames"]):
        if record["timestep_id"] != f"{index:06d}" or record["timestep_index_original"] != index:
            raise ValueError("Export의 frame ID 대응이 원본과 다르다.")
        if (record["w"], record["h"]) != (manifest["width"], manifest["height"]):
            raise ValueError("Export의 이미지 크기가 원본과 다르다.")
        for key in ("file_path", "flame_param_path", "fg_mask_path"):
            if not (destination / record[key]).is_file():
                raise FileNotFoundError(f"Export 파일 누락: {record[key]}")
    save_json(run_dir / "export.json", {
        "directory": str(destination), "source_parameters_sha256": state["parameters_sha256"],
        "coordinate_note": "VHAP export recenters mean head translation and changes camera c2w consistently.",
        "canonical_jaw_note": "VHAP export canonical has jaw axis-angle [0.3,0,0]; neutral UV input is separate.",
        "alpha_note": "Upstream export uses JPEG alpha and applies below-line suppression; original lossless PNG remains in sequence.",
        "validation_note": "Official monocular export has no held-out camera validation; later lesson defines a temporal split.",
    })
    print("NeRF/3DGS 형식 export:", destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=("prepare", "check", "landmarks", "track", "inspect", "preview", "export"))
    parser.add_argument("--sequence", default="person")
    parser.add_argument("--run", default="full", help="서로 다른 fitting 설정의 결과를 구분하는 이름.")
    parser.add_argument("--vhap", type=Path, default=LESSON_DIR / "external" / "vhap")
    parser.add_argument("--flame-model", type=Path, default=REPO_DIR / "FLAME2023" / "flame2023.pkl")
    parser.add_argument("--flame-masks", type=Path, default=REPO_DIR / "FLAME2023" / "FLAME_masks.pkl")
    parser.add_argument("--epochs", type=int, default=30, help="Global photometric refinement epoch 수.")
    parser.add_argument("--batch-size", type=int, default=1, help="정합 확인을 우선해 원래 frame별 tracking 기본값 1.")
    parser.add_argument("--save-debug-meshes", action="store_true", help="원본 VHAP의 반복 OBJ/texture dump까지 저장.")
    parser.add_argument("--allow-missing-landmarks", action="store_true", help="첫 frame 이후 검출 실패는 confidence=0으로 photo/temporal fitting에 맡긴다.")
    parser.add_argument("--preview-all", action="store_true", help="모든 frame의 overlay와 mp4 생성. 기본은 8개 frame.")
    args = parser.parse_args()
    for name in (args.sequence, args.run):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
            parser.error("sequence/run 이름에는 영문/숫자/밑줄/하이픈을 사용하세요.")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs/batch-size는 1 이상이어야 한다.")
    args.vhap = args.vhap.expanduser().resolve()
    sequence_dir, manifest = read_sequence(args)
    run_dir = LESSON_DIR / "outputs" / "02" / args.sequence / args.run
    if args.stage == "prepare":
        prepare_assets(args)
    elif args.stage == "inspect":
        inspect_parameters(sequence_dir, manifest, run_dir)
    else:
        check_image_sizes(sequence_dir, manifest)
        with vhap_context(args):
            if args.stage == "check":
                check_runtime()
                prepare_star_weights()
                from vhap.util.landmark_detector_star import LandmarkDetectorSTAR
                print("STAR import: OK. --stage landmarks에서 실제 검출한다.")
            elif args.stage == "landmarks":
                check_runtime()
                provenance = prepare_star_weights()
                detect_landmarks(sequence_dir, manifest, args.allow_missing_landmarks)
                save_json(LESSON_DIR / "outputs" / "02" / args.sequence / "star_assets.json", provenance)
            elif args.stage == "track":
                check_runtime()
                track(args, sequence_dir, manifest, run_dir)
            elif args.stage == "preview":
                check_runtime()
                preview(args, sequence_dir, manifest, run_dir)
            elif args.stage == "export":
                check_runtime()
                export_dataset(sequence_dir, manifest, run_dir)


if __name__ == "__main__":
    main()
