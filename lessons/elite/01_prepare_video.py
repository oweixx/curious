"""01: 실제 영상에서 VHAP용 RGB frame과 foreground alpha를 만든다.

읽는 순서: probe_video -> ffmpeg_sync_options -> prepare_frames -> prepare_alpha -> main.
frames 단계는 CPU/FFmpeg, matting 단계는 사전학습된 RVM을 사용한다.
RVM은 전처리 모델이다. Gaussian Avatar network 학습은 이후 lesson에서 구현한다.

예시 (repo root, conda curious):
  python lessons/elite/01_prepare_video.py --video /path/person.mp4 --stage frames
  CUDA_VISIBLE_DEVICES=2 python lessons/elite/01_prepare_video.py --stage matting

출력: 이 파일 옆 data/sequences/person/{images,alpha_maps,manifest.json}.
실행 결과와 다운로드 cache는 lessons/elite/.gitignore로 제외된다.
"""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw


LESSON_DIR = Path(__file__).resolve().parent
RVM_REVISION = "53d74c6826735f01f4406b5ca9075eee27bec094"


def probe_video(video_path):
    """영상 stream을 골라 FPS와 해상도를 읽는다."""
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"{executable}가 필요하다. GUIDE의 FFmpeg 준비 항목을 읽으세요.")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    probe = json.loads(result.stdout)
    stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
    if stream is None:
        raise ValueError("영상 stream이 없다.")
    fps = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            value = Fraction(stream.get(key, "0/1"))
            if value > 0:
                fps = value
                break
        except (ValueError, ZeroDivisionError):
            continue
    if fps is None:
        raise ValueError("유효한 영상 FPS를 찾지 못했다.")
    if stream.get("color_transfer") in ("smpte2084", "arib-std-b67"):
        raise ValueError("HDR 영상이다. SDR로 촬영하거나 명시적으로 tone mapping한 영상을 준비하세요.")
    return {
        "coded_width": int(stream["width"]),
        "coded_height": int(stream["height"]),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "duration_s": probe.get("format", {}).get("duration"),
        "codec": stream.get("codec_name"),
        "color_transfer": stream.get("color_transfer"),
        "tags": stream.get("tags", {}),
        "side_data": stream.get("side_data_list", []),
    }, fps


def ffmpeg_sync_options():
    """현재 FFmpeg가 지원하는 frame passthrough 옵션을 선택한다."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-h", "full"],
        capture_output=True, text=True, check=True,
    )
    help_text = result.stdout + result.stderr
    if re.search(r"(?m)^\s*-fps_mode(?:\s|:)", help_text):
        return ["-fps_mode", "passthrough"]
    if re.search(r"(?m)^\s*-vsync\s", help_text):
        # 구버전의 vsync=0도 passthrough다. 최신 버전에서는 fps_mode를 사용한다.
        return ["-vsync", "0"]
    raise RuntimeError("FFmpeg가 -fps_mode와 -vsync를 모두 제공하지 않는다. 설치를 확인하세요.")


def prepare_frames(args, sequence_dir):
    """일정 FPS로 resample하고 frame 번호와 이미지 크기를 기록한다."""
    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"영상이 없다: {video_path}\n--video로 실제 영상 경로를 지정하세요.")
    if sequence_dir.exists():
        raise FileExistsError(
            f"이미 준비된 sequence 경로다: {sequence_dir}\n"
            "frame을 다시 만들려면 새 --sequence 이름을 쓰세요. alpha만 만들려면 --stage matting."
        )
    source_info, source_fps = probe_video(video_path)
    sync_options = ffmpeg_sync_options()
    # 30000/1001을 30000 FPS로 잘못 해석하지 않도록 Fraction을 사용한다.
    target_fps = Fraction(str(args.fps)) if args.fps is not None else min(source_fps, Fraction(25))
    if target_fps > source_fps:
        raise ValueError("--fps는 원본 평균 FPS 이하로 지정하세요. 불필요한 frame 복제를 줄이기 위함이다.")
    digest = hashlib.sha256()
    with video_path.open("rb") as video_file:
        for block in iter(lambda: video_file.read(1024 * 1024), b""):
            digest.update(block)

    sequence_dir.parent.mkdir(parents=True, exist_ok=True)
    # 부분 추출이 완료된 dataset처럼 보이지 않게 완성 후 directory를 이동한다.
    with tempfile.TemporaryDirectory(prefix=".frames_", dir=sequence_dir.parent) as temporary:
        staging = Path(temporary) / args.sequence
        image_dir = staging / "images"
        image_dir.mkdir(parents=True)
        filters = [f"fps=fps={target_fps.numerator}/{target_fps.denominator}"]
        if args.max_side:
            filters.append(
                f"scale=w='min(iw,{args.max_side})':h='min(ih,{args.max_side})':"
                "force_original_aspect_ratio=decrease:flags=lanczos"
            )
        # FFmpeg의 autorotate를 사용한다. crop/좌우 반전/배경 합성은 여기서 하지 않는다.
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
            "-i", str(video_path), "-map", "0:v:0", "-an", "-sn", "-dn",
            # FPS resampling은 fps filter가 한다. 출력 단계의 추가 복제/삭제를 막는다.
            "-vf", ",".join(filters), *sync_options,
            "-c:v", "mjpeg", "-q:v", "1", "-pix_fmt", "yuvj444p", "-start_number", "0",
        ]
        if args.max_frames is not None:
            command.extend(["-frames:v", str(args.max_frames)])
        command.append(str(image_dir / "%06d.jpg"))
        print("실행 명령:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        image_paths = sorted(image_dir.glob("*.jpg"))
        if len(image_paths) < 2:
            raise ValueError("2개 이상의 frame이 필요하다.")
        with Image.open(image_paths[0]) as image:
            width, height = image.size
        records = []
        for index, image_path in enumerate(image_paths):
            if image_path.name != f"{index:06d}.jpg":
                raise ValueError("frame 번호가 연속하지 않는다.")
            with Image.open(image_path) as image:
                if image.size != (width, height):
                    raise ValueError("sequence 안에서 이미지 크기가 바뀐다.")
            records.append({
                "frame_id": index, "image": f"images/{image_path.name}",
                # 준비된 CFR sequence의 시간이다. 원본 VFR의 정확한 source PTS는 아니다.
                "sequence_time_s": index / float(target_fps),
            })
        manifest = {
            "schema_version": 1, "sequence": args.sequence, "status": "frames_ready",
            "source_video": str(video_path), "source_sha256": digest.hexdigest(),
            "source_probe": source_info, "ffmpeg_command": command,
            "fps": float(target_fps), "fps_fraction": str(target_fps),
            "width": width, "height": height, "num_frames": len(records),
            "frames": records,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        staging.rename(sequence_dir)
    print(f"Frames: {len(records)} | {width}x{height} | FPS {float(target_fps):.6f}", flush=True)


def write_preview(sequence_dir, manifest):
    """몇 frame의 RGB, soft alpha, 흰 배경 합성을 한 그림에 모은다."""
    indices = np.unique(np.linspace(0, len(manifest["frames"]) - 1, 8).astype(int))
    tile_width, tile_height = 220, 220
    row_height = tile_height + 24
    sheet = Image.new("RGB", (tile_width * 3, row_height * len(indices)), "#dddddd")
    draw = ImageDraw.Draw(sheet)
    for row, index in enumerate(indices):
        record = manifest["frames"][index]
        with Image.open(sequence_dir / record["image"]) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        with Image.open(sequence_dir / record["alpha_png"]) as image:
            alpha = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        composed = rgb * alpha[..., None] + 255.0 * (1.0 - alpha[..., None])
        tiles = [Image.fromarray(rgb.astype(np.uint8)),
                 Image.fromarray(np.rint(alpha * 255).astype(np.uint8)).convert("RGB"),
                 Image.fromarray(np.rint(composed).clip(0, 255).astype(np.uint8))]
        for col, (tile, label) in enumerate(zip(tiles, ("RGB", "alpha", "white background"))):
            tile.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
            x = col * tile_width + (tile_width - tile.width) // 2
            y = row * row_height + 24 + (tile_height - tile.height) // 2
            sheet.paste(tile, (x, y))
            draw.text((col * tile_width + 4, row * row_height + 4),
                      f"{index:06d} | {label}", fill="black")
    preview_dir = LESSON_DIR / "outputs" / "01" / manifest["sequence"]
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / "preview.jpg"
    sheet.save(preview_path, quality=95, subsampling=0)
    print("Preview:", preview_path)


def check_cuda_runtime(torch, device):
    """GPU가 보이는 것과 실제 CUDA kernel이 실행되는 것은 다르다."""
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch가 필요하다. CPU matting은 --device cpu로 지정할 수 있다.")
    name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    try:
        # 메모리 할당만으로는 호환성을 확인할 수 없다. 실제 연산까지 실행한다.
        torch.ones(1, device=device).add_(1)
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        raise RuntimeError(
            f"CUDA 실행 확인 실패: {name} (sm_{major}{minor})\n"
            f"PyTorch={torch.__version__}, 빌드 CUDA={torch.version.cuda}, "
            f"지원 arch={torch.cuda.get_arch_list()}\n"
            f"원인: {error}\n"
            "'no kernel image'이면 GPU에 맞는 PyTorch/torchvision 빌드가 필요하다. "
            "GUIDE의 H200 항목을 확인하세요."
        ) from None


def prepare_alpha(args, sequence_dir):
    """RVM recurrent state를 시간 순서대로 전달해 실제 soft alpha를 예측한다."""
    import torch

    manifest_path = sequence_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("먼저 같은 --sequence에 대해 --stage frames를 실행하세요.")
    manifest = json.loads(manifest_path.read_text())
    if manifest["sequence"] != args.sequence or manifest.get("status") != "frames_ready":
        raise ValueError("새로 frame을 준비한 sequence가 필요하다. 완료된 alpha는 덮어쓰지 않는다.")
    alpha_dir = sequence_dir / "alpha_maps"
    if alpha_dir.exists():
        raise FileExistsError(f"alpha_maps가 이미 있다: {alpha_dir}")
    image_paths = sorted((sequence_dir / "images").glob("*.jpg"))
    expected = [sequence_dir / record["image"] for record in manifest["frames"]]
    if image_paths != expected:
        raise ValueError("manifest의 frame 목록과 images가 다르다.")
    device = torch.device(args.device)
    check_cuda_runtime(torch, device)
    # 모델 코드와 weight cache도 이 lesson 폴더 안에서만 관리한다.
    torch.hub.set_dir(str(LESSON_DIR / ".cache" / "torch" / "hub"))
    model = torch.hub.load(
        f"PeterL1n/RobustVideoMatting:{RVM_REVISION}", args.backbone,
        pretrained=True, trust_repo=True, skip_validation=True,
    ).to(device).eval().requires_grad_(False)
    ratio = min(args.matting_size / max(manifest["width"], manifest["height"]), 1.0)
    # rec를 frame마다 초기화하면 video matting이 아니라 독립적인 image matting이 된다.
    recurrent_state = [None] * 4
    print(f"RVM {args.backbone} | {device} | downsample_ratio={ratio:.4f}", flush=True)
    with tempfile.TemporaryDirectory(prefix=".alpha_", dir=sequence_dir) as temporary:
        staging = Path(temporary) / "alpha_maps"
        staging.mkdir()
        with torch.inference_mode():
            for index, record in enumerate(manifest["frames"]):
                with Image.open(sequence_dir / record["image"]) as image:
                    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
                if rgb.shape[:2] != (manifest["height"], manifest["width"]):
                    raise ValueError("RGB 해상도가 manifest와 다르다.")
                source = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device, dtype=torch.float32) / 255.0
                if index == 0:
                    for _ in range(args.warmup):
                        _, _, *recurrent_state = model(source, *recurrent_state, ratio)
                # RVM의 fgr는 사용하지 않는다. VHAP처럼 원래 RGB와 alpha를 함께 보관한다.
                _, alpha, *recurrent_state = model(source, *recurrent_state, ratio)
                alpha_float = alpha[0, 0].clamp(0, 1).cpu().numpy()
                if alpha_float.shape != rgb.shape[:2] or not np.isfinite(alpha_float).all():
                    raise ValueError("alpha의 크기/값이 잘못됐다.")
                alpha_image = Image.fromarray(np.rint(alpha_float * 255).astype(np.uint8))
                stem = Path(record["image"]).stem
                # PNG는 network supervision용 원본. 고정 VHAP fork는 alpha의 .jpg를 읽는다.
                alpha_image.save(staging / f"{stem}.png")
                alpha_image.save(staging / f"{stem}.jpg", quality=100, subsampling=0)
                record["alpha_png"] = f"alpha_maps/{stem}.png"
                record["alpha_vhap"] = f"alpha_maps/{stem}.jpg"
                # threshold는 진단에만 사용한다. 저장한 supervision은 soft alpha 그대로다.
                ys, xs = np.where(alpha_float >= 0.5)
                record["foreground_bbox_xyxy"] = (
                    [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
                )
                record["mean_alpha"] = float(alpha_float.mean())
                if index % 25 == 0 or index == len(manifest["frames"]) - 1:
                    print(f"Matting {index + 1}/{len(manifest['frames'])} | mean alpha {alpha_float.mean():.3f}", flush=True)
        staging.rename(alpha_dir)
    manifest["status"] = "matting_ready"
    manifest["matting"] = {
        "model": "RobustVideoMatting", "backbone": args.backbone, "revision": RVM_REVISION,
        "downsample_ratio": ratio, "warmup_frames": args.warmup,
        "supervision_format": "8-bit lossless PNG soft alpha; divide by 255",
        "vhap_compatibility_format": "JPEG quality=100",
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)
    write_preview(sequence_dir, manifest)
    print("다음 단계 입력:", sequence_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, default=LESSON_DIR / "data" / "input.mp4")
    parser.add_argument("--sequence", default="person", help="한 영상의 이름. 이후 lesson에서도 동일하게 사용.")
    parser.add_argument("--stage", choices=("frames", "matting", "all"), default="all")
    parser.add_argument("--fps", type=float, default=None, help="기본은 min(원본 평균 FPS, 25).")
    parser.add_argument("--max-side", type=int, default=1024, help="종횡비 유지. 0이면 원본 해상도.")
    parser.add_argument("--max-frames", type=int, default=None, help="처음부터 N개만 준비. 기본은 영상 전체.")
    parser.add_argument("--device", default="cuda:0", help="CUDA_VISIBLE_DEVICES=2 사용 시 cuda:0은 물리 GPU 2.")
    parser.add_argument("--backbone", choices=("resnet50", "mobilenetv3"), default="resnet50")
    parser.add_argument("--matting-size", type=int, default=512, help="RVM 내부 처리 최대 변 길이. 출력 alpha는 RGB 원래 크기.")
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.sequence):
        parser.error("--sequence에는 영문/숫자/밑줄/하이픈을 사용하세요.")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps는 양의 유한한 값이어야 한다.")
    if args.max_side < 0 or args.matting_size < 1 or args.warmup < 0:
        parser.error("해상도와 warmup 값을 확인하세요.")
    if args.max_frames is not None and args.max_frames < 2:
        parser.error("--max-frames는 2 이상이어야 한다.")
    sequence_dir = LESSON_DIR / "data" / "sequences" / args.sequence
    if args.stage in ("frames", "all"):
        prepare_frames(args, sequence_dir)
    if args.stage in ("matting", "all"):
        prepare_alpha(args, sequence_dir)


if __name__ == "__main__":
    main()
