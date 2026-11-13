"""00 — camera triplet과 pose-aware proxy conditioning.

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
from common import ROOT, UPSTREAM, COMMIT, upstream, read_manifest


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
    import cv2
    from PIL import ImageOps
    upstream()
    from vda.preprocess.conditioning import render
    spec_path=Path(spec_path).resolve(); spec=json.loads(spec_path.read_text())
    if spec.get("world_space")!="dataset" or spec.get("body_fitting")!="pelvis_anchored":
        raise ValueError("dataset world에 정렬하고 pelvis fitting한 state가 필요하다")
    def path(key): return (spec_path.parent/spec[key]).resolve()
    destination=Path(destination).resolve()
    if destination.exists(): raise FileExistsError("새 triplet 출력 폴더를 지정하세요")
    destination.mkdir(parents=True)
    state_file=np.load(path("state"),allow_pickle=False)
    state={k:state_file[k] for k in state_file.files}
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
    render(state,np.load(ref/"flame_shape.npy"),c2w,camera["K"],destination)
    shutil.copyfile(ref/"ref_916.png",destination/"ref.png")
    # Native RGB is cropped, never squeezed into a portrait aspect ratio.
    target=path("target_video"); cap=cv2.VideoCapture(str(target)); cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    tmp=destination/"target.raw.mp4"
    writer=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*"mp4v"),24,(720,1280))
    if not writer.isOpened(): raise RuntimeError("target video writer unavailable")
    try:
        for _ in range(n):
            ok,frame=cap.read()
            if not ok: raise ValueError("target video와 state timeline 길이 불일치")
            h,w=frame.shape[:2]
            if abs(w/h-1100/1604)>.002: raise ValueError("target source aspect differs from calibrated canvas")
            rgb=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            cropped=ImageOps.fit(rgb,(720,1280),method=Image.Resampling.LANCZOS)
            writer.write(cv2.cvtColor(np.asarray(cropped),cv2.COLOR_RGB2BGR))
    finally:
        cap.release(); writer.release()
    subprocess.run(["ffmpeg","-nostdin","-n","-loglevel","error","-i",str(tmp),"-c:v","libx264",
                    "-crf","15","-pix_fmt","yuv420p",str(destination/"target.mp4")],check=True)
    tmp.unlink()
    row={k:spec[k] for k in ("id","identity","sequence","split","cameras")}
    row.update(start=start,frames=n,conditioning=".",target="target.mp4")
    (destination/"triplet.jsonl").write_text(json.dumps(row)+"\n")
    (destination/"provenance.json").write_text(json.dumps(spec,indent=2))
    inspect_manifest(destination/"triplet.jsonl",destination/"inspection")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    sub.add_parser("setup", help="공식 코드만 고정 revision으로 받는다. 모델/env 설치는 별도")
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
    if a.stage == "build-triplet":
        build_triplet(a.spec,a.out); return
    upstream()
    module = {"reference":"vda.preprocess.reference", "driving":"vda.preprocess.driving",
              "conditioning":"vda.preprocess.conditioning", "extract-codes":"vda.preprocess.xnemo_codes",
              "download":"vda.download"}[a.stage]
    args = a.args[1:] if a.args[:1] == ["--"] else a.args
    if a.stage=="download" and args==["base"]:
        print(download_base()); return
    env=os.environ.copy()
    env["PYTHONPATH"]=os.pathsep.join([str(ROOT/".cache/runtime"),str(UPSTREAM),env.get("PYTHONPATH","")])
    subprocess.run([sys.executable,"-m",module,*args],cwd=UPSTREAM,env=env,check=True)


if __name__ == "__main__":
    main()
