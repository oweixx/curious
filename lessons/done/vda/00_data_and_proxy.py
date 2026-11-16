r"""00 — camera triplet과 pose-aware proxy conditioning.

실행 위치 / 환경 (아래 명령은 이 디렉터리 기준, 물리 GPU 4 사용):
    conda activate elite
    cd /media/vilab/mh/curious/lessons/done/vda

지금 시작할 명령: 공개 예제의 reference/proxy/skeleton을 이미지로 비교 (CPU).
학습 spec이나 target RGB 없이 실행되며 결과는 outputs/00_conditioning.jpg다.
    python 00_data_and_proxy.py inspect-conditioning \
        --cond .cache/upstream/examples/teaser_swimmer --out outputs/00_conditioning.jpg

다운로드한 실제 NeRSemble 검사 + camera.npz 준비 (CPU, FLAME 모델 불필요):
    python 00_data_and_proxy.py inspect-nersemble \
        --subject-dir data/042 --sequence FREE --out outputs/00_nersemble
출력: report.json, 원본 시각 0/중간/끝 카메라 비교 JPG, cameras/cam_*.npz.
이 단계는 RGB/calibration 검사이며 FLAME/MHR tracking이나 triplet 생성은 아니다.

공개 source 준비 확인 (현재 source/teacher weights/예제는 이미 준비되어 있음):
    python 00_data_and_proxy.py setup

새 환경에서 자산이 없을 때만 다운로드:
    python 00_data_and_proxy.py download -- weights
    python 00_data_and_proxy.py download -- base
    python 00_data_and_proxy.py download -- examples

원본 영상의 전처리는 별도 preprocess.py에서 수행한다 (그 파일 맨 위 실행 순서 참고).
preprocess.py export가 준비한 state와 clip_spec.json으로 여기서 proxy/triplet을 만든다.

이미 전처리한 state를 다른 출력 폴더에 다시 render할 경우:
--out은 아직 없는 새 폴더로 지정한다.
    CUDA_VISIBLE_DEVICES=4 python 00_data_and_proxy.py build-triplet \
        --spec data/preprocessed/042_FREE/clip_spec.json --out data/triplets/042_free_a
    python 00_data_and_proxy.py inspect \
        --manifest data/triplets/042_free_a/triplet.jsonl --out outputs/00

여러 triplet을 합친 manifest가 준비되어 있다면:
    python 00_data_and_proxy.py inspect \
        --manifest data/triplets.jsonl --out outputs/00

Raw X-NeMo code 추출을 따로 살펴볼 때 (현재 driver camera 222200042):
preprocess.py codes를 마쳤으면 data/preprocessed/042_FREE/codes_raw.npy에 이미
[77,512] code가 있다. build-triplet도 이 code를 사용하므로 아래 재추출은 선택이다.
원본 MP4는 73fps이므로 현재 학습 clip과 동기화된 24fps·77프레임 폴더를 사용한다.
    CUDA_VISIBLE_DEVICES=4 python 00_data_and_proxy.py extract-codes -- \
        --video /media/vilab/mh/curious/lessons/done/vda/data/preprocessed/042_FREE/frames/222200042 \
        --out /media/vilab/mh/curious/lessons/done/vda/data/codes/042_free_cam222200042.npy

현재 042는 preprocess.py export와 build-triplet까지 완료했으면 00 준비가 끝난다.
위 inspect로 확인한 뒤 02의 공개 disentangler/model audit를 진행한다.
reference/driving은 다른 단안 입력을 처리할 때 쓰는 별도 공개 CLI다. 현재 042의
후속 단계로 실행하지 않는다. cam_222200042 전체 영상의 단안 tracking은 Loss NaN으로
실패했으므로 해당 실패 cache를 학습에 사용하지 않는다 (GUIDE.md의 오류 설명 참고).
별도 CLI의 옵션을 살펴보려면:
    python 00_data_and_proxy.py reference -- --help
    python 00_data_and_proxy.py driving -- --help

reference/driving/conditioning/extract-codes는 -- 뒤 옵션을 공식 CLI에 전달한다.
공식 CLI의 작업 폴더는 .cache/upstream이므로 직접 지정하는 파일 경로는 절대
경로를 사용한다. driving 기본값은 canonical driving bank의 몸 움직임을 쓰며,
영상 자체의 몸 움직임은 별도 SAM/MHR 준비와 --motion driver 설정이 필요하다.
reference/driving 실행 로그: .cache/logs/00_<stage>_<timestamp>.log.
Loss NaN/Inf를 출력하면 자식 tracking 프로세스까지 중단한다.
출력: triplet/manifest, outputs/00/*.jpg, data/codes/*.npy.
도움말: python 00_data_and_proxy.py --help
        python 00_data_and_proxy.py conditioning -- --help

입력 계약을 먼저 읽는다: GUIDE.md의 데이터 준비 항목.
공개 전처리는 실제 함수를 사용한다. 학습용 camera는 반드시 tracking과 동일한
world 좌표계여야 한다. 추론용 canonical motion을 실제 target video와 섞지 않는다.
핵심 수식은 ray_colors / pelvis_objective에서 직접 따라갈 수 있다.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from common import ROOT, UPSTREAM, COMMIT, upstream, read_manifest, run_preprocessing


def download_base():
    """Use official WAN originals; the converted DiffSynth mirror may return 401.

    Pinned files are recognized by the original upstream state-dict converters.
    A text encoder/tokenizer is unnecessary with the released constant text cache.
    """
    from huggingface_hub import snapshot_download
    from vda.paths import weights_dir
    repo="Wan-AI/Wan2.2-Animate-14B"
    return snapshot_download(repo_id=repo,revision="cb93a225fbaf1ca100f54e79da8f994995b689b3",
        allow_patterns=["diffusion_pytorch_model*.safetensors","diffusion_pytorch_model.safetensors.index.json",
                        "Wan2.1_VAE.pth","models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
        local_dir=str(weights_dir()/"base"/repo))


def ray_colors(K, c2w, part_rotation, gray):
    """Eq.7. d_world -> R_part^T d_world. RGB는 UV/normal map이 아니다.

    공식 renderer와 같이 정수 pixel 좌표를 사용한다. row-vector이므로 d @ R.
    gray: Lambert [H,W], [0,1]. R: local -> world, c2w: OpenCV camera -> world.
    """
    h, w = gray.shape
    yy, xx = np.mgrid[:h, :w]
    ray_camera = np.stack((xx, yy, np.ones_like(xx)), -1) @ np.linalg.inv(K).T
    ray_world = ray_camera @ c2w[:3, :3].T
    local = ray_world @ part_rotation
    local /= np.linalg.norm(local, axis=-1, keepdims=True).clip(1e-12)
    return .5 * gray[..., None] + .25 * (local + 1)


def pelvis_objective(face, target_face, pelvis, shoulders, pose6d, initial_pose6d,
                     pelvis_anchor, shoulder_anchor):
    """Eq.6: 실제 MHR forward kinematics 출력에 적용할 differentiable 목적함수.

    face/pelvis/shoulders는 하나의 학습 가능한 global Sim(3) 적용 후 좌표.
    anchor는 최초 sequence median을 detach해서 고정. 단순 vertex smoothing으로
    pelvis-anchored joint-angle optimization을 대신하지 않는다.
    """
    import torch.nn.functional as F
    return (F.mse_loss(face, target_face) + F.mse_loss(pelvis, pelvis_anchor.expand_as(pelvis))
            + .3 * F.mse_loss(shoulders, shoulder_anchor.expand_as(shoulders))
            + .001 * F.mse_loss(pose6d, initial_pose6d)
            + .03 * F.mse_loss(pose6d[1:], pose6d[:-1]))


def inspect_conditioning(conditioning, output):
    """공개 추론 입력을 직접 비교한다. Target이나 가짜 학습 manifest는 만들지 않는다."""
    import cv2
    cond=Path(conditioning)
    codes=np.load(cond/"codes.npy",allow_pickle=False)
    if codes.ndim!=2 or codes.shape[1]!=512 or len(codes)<5 or not np.isfinite(codes).all():
        raise ValueError("유한한 raw X-NeMo codes [T,512], T>=5 필요")
    with Image.open(cond/"ref.png") as image:
        ref=image.convert("RGB")
    indices=[0,len(codes)//2,len(codes)-1]
    columns=[("reference", [ref]*3)]
    for name,title in (("pose_proxy.mp4","pose-aware proxy"),("pose_skel.mp4","skeleton")):
        cap=cv2.VideoCapture(str(cond/name))
        try:
            if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=len(codes):
                raise ValueError(f"{name}: codes와 같은 {len(codes)}프레임 필요")
            frames=[]
            for t in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES,t)
                ok,frame=cap.read()
                if not ok: raise ValueError(f"{name}: frame {t} 읽기 실패")
                frames.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
            columns.append((title,frames))
        finally:
            cap.release()
    canvas=Image.new("RGB",(540,28+3*344),"#222222"); draw=ImageDraw.Draw(canvas)
    draw.text((4,5),f"Conditioning only | raw codes {list(codes.shape)} | no target RGB",fill="white")
    for col,(title,frames) in enumerate(columns):
        for row,(t,im) in enumerate(zip(indices,frames)):
            if abs(im.width/im.height-9/16)>.005:
                raise ValueError(f"{title}: 9:16 crop 필요")
            x,y=180*col,28+344*row
            draw.text((x+3,y+4),f"{title} | frame {t}",fill="white")
            canvas.paste(im.resize((180,320)),(x,y+24))
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True); canvas.save(output)
    print(f"raw X-NeMo codes: {codes.shape}; frames: {indices}")
    print(f"reference / proxy / skeleton -> {output}")
    print("추론 조건 검사입니다. 실제 학습용 triplet/target은 생성하지 않았습니다.")


def inspect_nersemble(subject_dir, sequence, output):
    """원본 RGB 시간축/카메라 확인. W2C를 역변환하고 K를 native canvas로 축소한다."""
    import cv2
    from fractions import Fraction
    subject=Path(subject_dir).resolve(); output=Path(output)
    calibration_path=subject/"calibration/camera_params.json"
    calibration=json.loads(calibration_path.read_text())
    K=np.asarray(calibration["intrinsics"],dtype=np.float64)
    if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0:
        raise ValueError("유효한 3x3 camera intrinsics 필요")
    # Official raw intrinsics refer to 2200x3208. VDA renders at half this size.
    # RGB cropping happens later, with the matching camera crop transform.
    source_size=np.array([2200,3208]); native_size=np.array([1100,1604])
    scale=np.diag([.5,.5,1.]); native_K=scale@K
    serials=list(calibration["world_2_cam"]); rows=[]; previews={}
    color_path=subject/"calibration/color_calibration.json"
    colors=json.loads(color_path.read_text()) if color_path.is_file() else {}
    output.mkdir(parents=True,exist_ok=True); (output/"cameras").mkdir(exist_ok=True)
    for serial in serials:
        w2c=np.asarray(calibration["world_2_cam"][serial],dtype=np.float64)
        if (w2c.shape!=(4,4) or not np.isfinite(w2c).all()
                or not np.allclose(w2c[3],[0,0,0,1])
                or not np.allclose(w2c[:3,:3].T@w2c[:3,:3],np.eye(3),atol=1e-5)
                or not np.isclose(np.linalg.det(w2c[:3,:3]),1,atol=1e-5)):
            raise ValueError(f"{serial}: 유효한 OpenCV W2C pose 필요")
        c2w=np.linalg.inv(w2c)
        np.savez(output/"cameras"/f"cam_{serial}.npz",K=native_K,c2w=c2w,
                 image_size=native_size,K_source=K,source_image_size=source_size,
                 world_2_cam=w2c,source_calibration=str(calibration_path))
        video=subject/f"sequences/{sequence}/images/cam_{serial}.mp4"
        row={"camera":serial,"video":str(video),"exists":video.is_file(),
             "background_exists":(subject/f"sequences/BACKGROUND/image_{serial}.jpg").is_file(),
             "color_calibration_exists":serial in colors}
        rows.append(row)
        if not video.is_file(): continue
        probe=subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
            "-show_entries","stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,start_time",
            "-of","json",str(video)],capture_output=True,text=True)
        if probe.returncode:
            row["error"]="ffprobe failed"; continue
        stream=json.loads(probe.stdout)["streams"][0]
        n=int(stream["nb_frames"]); fps=Fraction(stream["avg_frame_rate"])
        row.update(bytes=video.stat().st_size,width=int(stream["width"]),height=int(stream["height"]),
                   frames=n,fps=str(fps),duration=float(stream["duration"]),
                   start_time=float(stream.get("start_time",0)))
        if [row["width"],row["height"]]!=source_size.tolist():
            row["error"]="raw calibration expects 2200x3208 RGB"; continue
        if n<5 or fps<=0:
            row["error"]="invalid frame count/fps"; continue
        cap=cv2.VideoCapture(str(video)); samples=[]
        row["sample_frames"]=[0,n//2,n-1]
        try:
            for t in row["sample_frames"]:
                cap.set(cv2.CAP_PROP_POS_FRAMES,t); ok,frame=cap.read()
                if not ok:
                    row["error"]=f"cannot decode frame {t}"; break
                rgb=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                rgb.thumbnail((220,321)); samples.append(rgb)
        finally:
            cap.release()
        if len(samples)==3: previews[serial]=samples
        print(f"{serial}: {n} frames, {fps} fps, samples={'ok' if serial in previews else 'failed'}",flush=True)
    present=[r for r in rows if r["exists"]]
    signatures={(r.get("frames"),r.get("fps"),r.get("start_time")) for r in present}
    metadata_match=len(signatures)==1 and bool(present) and not any("error" in r for r in present)
    missing=[r["camera"] for r in rows if not r["exists"]]
    report={"subject":str(subject),"sequence":sequence,"cameras":rows,
            "available_cameras":len(present),"missing_cameras":missing,
            "timeline_metadata_match":metadata_match,"raw_frames_sampled_only":True,
            "native_image_size":native_size.tolist(),"source_image_size":source_size.tolist(),
            "camera_convention":"OpenCV; c2w = inverse(world_2_cam); K_native = diag(.5,.5,1) @ K_source",
            "notes":["Matching metadata/sample frames alone does not prove physical synchronization.",
                     "73fps RGB must be resampled with tracking/codes to a common 24fps timeline.",
                     "FLAME/MHR fitting and dataset-world alignment still required; no training spec generated."]}
    for j,label in enumerate(("first","middle","last")):
        sheet=Image.new("RGB",(880,32+350*((len(serials)+3)//4)),"#222222")
        draw=ImageDraw.Draw(sheet); draw.text((4,5),f"{subject.name}/{sequence} | {label} raw frame (no crop)",fill="white")
        for i,serial in enumerate(serials):
            x,y=(i%4)*220,32+(i//4)*350
            status="MISSING" if serial in missing else "sample failed"
            if serial in previews:
                status=f"frame {next(r for r in rows if r['camera']==serial)['sample_frames'][j]}"
                sheet.paste(previews[serial][j],(x,y+24))
            draw.text((x+3,y+4),f"{serial} | {status}",fill="white")
        sheet.save(output/f"{label}.jpg")
    (output/"report.json").write_text(json.dumps(report,indent=2))
    print(f"{len(present)}/{len(serials)} cameras; missing={missing}; report -> {output}")
    if not metadata_match: raise ValueError("영상/시간축 검사가 실패했습니다. report.json을 확인하세요.")


def inspect_manifest(path, output):
    """실제 input 4종을 함께 저장해 crop / sync / RGB-BGR 문제를 먼저 본다."""
    import cv2
    rows = read_manifest(path)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        cond = Path(row["conditioning"])
        code = np.load(cond / "codes.npy", allow_pickle=False)
        n = row["frames"]
        if n < 5 or n % 4 != 1 or code.shape != (n, 512) or not np.isfinite(code).all():
            raise ValueError(f"{row['id']}: frames=4k+1, codes=[frames,512] 필요")
        ims = [Image.open(cond / "ref.png").convert("RGB")]
        for video in (cond / "pose_proxy.mp4", cond / "pose_skel.mp4", Path(row["target"])):
            cap = cv2.VideoCapture(str(video))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, frame = cap.read(); cap.release()
            if not ok or count != n:
                raise ValueError(f"{video}: 정확히 {n}개의 동기화 frame 필요, 현재 {count}")
            ims.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        canvas = Image.new("RGB", (4*180, 344), "#222222")
        for i,(im,title) in enumerate(zip(ims,("reference", "proxy: target camera", "skeleton", "target RGB"))):
            if abs(im.width/im.height - 9/16) > .005:
                raise ValueError(f"{row['id']}: 9:16 crop이 필요하다. resize로 찌그러뜨리지 않는다")
            canvas.paste(im.resize((180,320)), (180*i,24))
            ImageDraw.Draw(canvas).text((180*i+3,4),title,fill="white")
        canvas.save(output / f"{row['id']}.jpg")
    print(f"{len(rows)} triplets checked -> {output}")


def build_triplet(spec_path, destination):
    """Dataset-world fitted state + calibrated target camera -> actual paired training clip.

    This bridges already tracked/fitted geometry to the training manifest. It cannot infer
    dataset-world alignment from the public canonical-person driving bank.
    """
    spec_path=Path(spec_path).resolve()
    if not spec_path.is_file():
        raise FileNotFoundError(
            f"학습 데이터 spec 파일이 없습니다: {spec_path}\n"
            "먼저 python preprocess.py frames부터 전처리를 진행하세요.\n"
            "python preprocess.py export가 data/preprocessed/042_FREE/clip_spec.json을 생성합니다.\n"
            "전체 단계: python preprocess.py --help")
    spec=json.loads(spec_path.read_text())
    required=("id","identity","sequence","split","cameras","state","camera","reference","target_video")
    if not isinstance(spec,dict) or any(k not in spec for k in required):
        raise ValueError(f"spec에 필요한 항목: {', '.join(required)} (GUIDE.md 참고)")
    if spec.get("world_space")!="dataset" or spec.get("body_fitting")!="pelvis_anchored":
        raise ValueError("dataset world에 정렬하고 pelvis fitting한 state가 필요하다")
    def path(key): return (spec_path.parent/spec[key]).resolve()
    needed=[path("state"),path("camera"),path("target_video"),
            path("reference")/"flame_shape.npy",path("reference")/"ref_916.png"]
    missing=[str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError("spec이 가리키는 실제 입력 파일이 없습니다:\n"+"\n".join(missing))
    destination=Path(destination).resolve()
    if destination.exists(): raise FileExistsError("새 triplet 출력 폴더를 지정하세요")
    state_file=np.load(path("state"),allow_pickle=False)
    state={k:state_file[k] for k in state_file.files}
    if "codes" not in state: raise ValueError("state에 codes 필요")
    n=int(spec.get("frames",77)); start=int(spec.get("start",0)); count=len(state["codes"])
    if n<5 or n%4!=1 or start<0 or start+n>count: raise ValueError("state frame range")
    for key in ("codes","gtw_lm","verts","kp","z_expression","z_jaw","z_eyes","z_neck"):
        if key not in state: raise ValueError(f"state에 {key} 필요")
    for key,value in state.items():
        if value.ndim and len(value)==count and key!="rig": state[key]=value[start:start+n]
    camera=np.load(path("camera"),allow_pickle=False)
    if tuple(camera["image_size"])!=(1100,1604):
        raise ValueError("official renderer calibration canvas: image_size=[1100,1604]; K를 해당 canvas로 변환하세요")
    c2w=camera["c2w"]
    c2w=np.repeat(c2w[None],n,0) if c2w.shape==(4,4) else c2w[start:start+n]
    if c2w.shape!=(n,4,4): raise ValueError("camera timeline mismatch")
    ref=path("reference")
    import cv2
    upstream()
    from vda.preprocess.conditioning import render
    from vda.preprocess import geometry as G
    expected={"verts":(n,len(G.BIDX),3),"kp":(n,70,3),"gtw_lm":(n,68,3),
              "codes":(n,512),"z_expression":(n,100),"z_jaw":(n,3),"z_eyes":(n,6),"z_neck":(n,3)}
    for key,shape in expected.items():
        if state[key].shape!=shape or not np.isfinite(state[key]).all():
            raise ValueError(f"state {key}: expected finite {shape}, got {state[key].shape}; verts는 MHR[G.BIDX] 순서")
    if np.linalg.norm(state["codes"],axis=1).mean()<20:
        raise ValueError("RAW X-NeMo codes가 필요합니다 (normalized code 아님)")
    # Validate time synchronization BEFORE spending time rendering. Raw NeRSemble is
    # 73 fps; interpreting it as 24 fps silently pairs the wrong target expressions.
    target=path("target_video"); cap=cv2.VideoCapture(str(target))
    fps=cap.get(cv2.CAP_PROP_FPS); video_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    if abs(fps-24)>.01 or int(spec.get("fps",24))!=24 or video_count<start+n:
        raise ValueError(f"target은 state와 같은 24fps 시간축이어야 합니다: fps={fps}, frames={video_count}")
    destination.mkdir(parents=True)
    render(state,np.load(ref/"flame_shape.npy"),c2w,camera["K"],destination)
    shutil.copyfile(ref/"ref_916.png",destination/"ref.png")
    # Native RGB is cropped, never squeezed into a portrait aspect ratio.
    cap=cv2.VideoCapture(str(target)); cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    tmp=destination/"target.raw.mp4"
    writer=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*"mp4v"),24,(720,1280))
    if not writer.isOpened(): raise RuntimeError("target video writer unavailable")
    try:
        for _ in range(n):
            ok,frame=cap.read()
            if not ok: raise ValueError("target video와 state timeline 길이 불일치")
            h,w=frame.shape[:2]
            if abs(w/h-1100/1604)>.002: raise ValueError("target source aspect differs from calibrated canvas")
            # Match geometry.crop_cam EXACTLY, including its rounded resize width.
            # A floating half-pixel crop avoids an extra principal-point shift.
            scale=1280/h; left=(round(w*scale)-720)/2
            M=np.array([[scale,0,-left],[0,scale,0]],np.float64)
            cropped=cv2.warpAffine(frame,M,(720,1280),flags=cv2.INTER_LANCZOS4)
            writer.write(cropped)
    finally:
        cap.release(); writer.release()
    subprocess.run(["ffmpeg","-nostdin","-n","-loglevel","error","-i",str(tmp),"-c:v","libx264",
                    "-crf","15","-pix_fmt","yuv420p",str(destination/"target.mp4")],check=True)
    tmp.unlink()
    row={k:spec[k] for k in ("id","identity","sequence","split","cameras")}
    # start in manifests is a sequence clock for leakage checks, not a local video offset.
    row.update(start=int(spec.get("sequence_start",0))+start,frames=n,conditioning=".",target="target.mp4")
    (destination/"triplet.jsonl").write_text(json.dumps(row)+"\n")
    (destination/"provenance.json").write_text(json.dumps(spec,indent=2))
    inspect_manifest(destination/"triplet.jsonl",destination/"inspection")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    sub.add_parser("setup", help="공식 코드만 고정 revision으로 받는다. 모델/env 설치는 별도")
    preview=sub.add_parser("inspect-conditioning",help="공개 예제 입력 비교 (CPU, 학습 spec 불필요)")
    preview.add_argument("--cond",default=str(UPSTREAM/"examples/teaser_swimmer"))
    preview.add_argument("--out",default=str(ROOT/"outputs/00_conditioning.jpg"))
    raw=sub.add_parser("inspect-nersemble",help="NeRSemble RGB/calibration 검사 및 camera.npz 준비 (CPU)")
    raw.add_argument("--subject-dir",default=str(ROOT/"data/042"))
    raw.add_argument("--sequence",default="FREE")
    raw.add_argument("--out",default=str(ROOT/"outputs/00_nersemble"))
    audit = sub.add_parser("inspect"); audit.add_argument("--manifest", required=True)
    audit.add_argument("--out", default=str(ROOT / "outputs/00"))
    build=sub.add_parser("build-triplet"); build.add_argument("--spec",required=True); build.add_argument("--out",required=True)
    # 이 commands는 그대로 공식 CLI에 전달한다. 모든 옵션은 -- 뒤에 쓴다.
    for name in ("reference", "driving", "conditioning", "extract-codes", "download"):
        s = sub.add_parser(name); s.add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.stage == "setup":
        if not UPSTREAM.exists():
            UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git","clone","https://github.com/SimonGiebenhain/video-diffusion-avatars.git",str(UPSTREAM)],check=True)
            subprocess.run(["git","-C",str(UPSTREAM),"checkout",COMMIT],check=True)
        print(upstream()); return
    if a.stage == "inspect":
        inspect_manifest(a.manifest,a.out); return
    if a.stage == "inspect-conditioning":
        try: inspect_conditioning(a.cond,a.out)
        except (FileNotFoundError,ValueError) as exc: p.error(str(exc))
        return
    if a.stage == "inspect-nersemble":
        try: inspect_nersemble(a.subject_dir,a.sequence,a.out)
        except (FileNotFoundError,ValueError) as exc: p.error(str(exc))
        return
    if a.stage == "build-triplet":
        try: build_triplet(a.spec,a.out)
        except (FileNotFoundError,ValueError) as exc: p.error(str(exc))
        return
    upstream()
    module = {"reference":"vda.preprocess.reference", "driving":"vda.preprocess.driving",
              "conditioning":"vda.preprocess.conditioning", "extract-codes":"vda.preprocess.xnemo_codes",
              "download":"vda.download"}[a.stage]
    args = a.args[1:] if a.args[:1] == ["--"] else a.args
    if a.stage=="download" and args==["base"]:
        print(download_base()); return
    env=os.environ.copy()
    env["PYTHONPATH"]=os.pathsep.join([str(ROOT/".cache/runtime"),str(UPSTREAM),env.get("PYTHONPATH","")])
    command=[sys.executable,"-m",module,*args]
    if a.stage in ("reference","driving"):
        command=[sys.executable,str(ROOT/"preprocess.py"),"_official",a.stage,*args]
        from datetime import datetime, timezone
        stamp=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        log_path=ROOT/".cache/logs"/f"00_{a.stage}_{stamp}.log"
        print(f"전처리 로그: {log_path}",flush=True)
        try:
            run_preprocessing(command, UPSTREAM, env, log_path, echo=True)
        except RuntimeError as exc:
            p.error(str(exc))
    else:
        subprocess.run(command,cwd=UPSTREAM,env=env,check=True)


if __name__ == "__main__":
    main()
