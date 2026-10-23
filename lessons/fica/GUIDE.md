# FiCA — Figure 2의 다섯 이정표

목표는 **한 장의 사진 → partial UV → 생성된 textured mesh → 보정된 mesh →
움직이는 3D Gaussian avatar**의 각 경계를 직접 읽고 학습하는 것이다.
실험 자료는 NeRSemble이며, 원본 Meta 데이터/모델 성능의 동일 재현을 주장하지 않는다.

| Part | 파일 | 이번에 확인할 질문 | 먼저 읽을 함수 |
|---|---|---|---|
| 00 | `00_prepare_nersemble.py` | 이미지·camera·canonical mesh의 대응이 맞는가? | `prepare_dataset`, `rasterize_labels`, `MeshBridge` |
| 01 | `01_partial_uv.py` | RGB 한 장의 픽셀을 어디의 UV texel로 옮기는가? | `DensePredictor`, `UVValidityPredictor`, `train_uv_repair`, `resample_partial_uv` |
| 02 | `02_generate_uv.py` | 불완전한 관측에서 texture/geometry 전체를 어떻게 생성하는가? | `FrozenConditioners`, `UVFlowModel`, `flow_loss`, `sample_domains` |
| 03 | `03_refine_uv.py` | 이미지의 오차를 어떤 UV 수정으로 연결하는가? | `RefinementNet`, `constrain_geometry`, `refinement_loss`, `validate` |
| 04 | `04_gaussian_avatar.py` | Identity를 유지하면서 표정에 맞는 Gaussians를 만드는가? | `IdentityEncoder`, `UniversalPrior`, `decode`, `render` |

아래 명령의 기준 디렉터리는 `/media/vilab/mh/curious/lessons/fica`다. 기존 **conda elite**를
사용한다. venv나 새 학습 framework는 만들지 않는다. Smoke는 physical GPU **0**만 쓴다.

## 원본과 대응하는 부분, 구현에서 결정한 부분

| 구성 | 논문/공개 기반과 맞춘 내용 | 명시적인 구현 결정 |
|---|---|---|
| Sapiens | 1B, width1536, 40 blocks, 24 heads; UV/XYZ의 depth-style head; normal checkpoint; encoder/head를 함께 fine-tune | UV head 2ch, XYZ 단위 정규화, NeRSemble label 좌표계 |
| Partial UV | RGB3 + mask1 + normal3 + XYZ3 | Bilinear forward splat, alpha/UV 범위로 visibility 판정 |
| UV generation | Frozen SDXL VAE, CLIP image conditioning, 28 DiT+ControlMLP blocks, shared texture/geometry model, domain switcher, flow matching, 기본 50-step sampling | width1792/28 heads로 약 1.97B, CLIP ViT-L/14, linear conditional flow/Euler |
| Latent | SDXL VAE 8배 공간 압축과 논문의 32×32×16 latent 표기 | 64×64×4를 가역적인 pixel-unshuffle(2)로 packing; packing은 논문에 미기재 |
| UV refinement | 실제 Diffusers conditional U-Net, reference/render의 실제 Sapiens feature, cross-attention, photo/mask/keypoint와 texture/Laplacian/normal 규제 | 공개 기본 채널 폭, 저해상도 attention, residual 출력, feature type embedding, 중심 고정·foreground 평균·texture MSE·초기 geometry 변위 규제 |
| UPM | UV-conditioned identity hypernetwork의 multi-scale bias를 decoder 각 층에 더함; expression/view/gaze conditioning; 3축 scale의 실제 3DGS | 22.4M인 자체 채널 설정, FLAME100→256 expression adapter, learned guide, 고정 root pose adapter, 명시적인 geometry supervision |

**VAE와 CLIP, frozen feature encoder는 학습하지 않는다.** 01의 세 predictor는 encoder와
head를 함께 학습한다. 02는 DiT/ControlMLP, 03은 U-Net, 04는 identity encoder·expression
adapter·Gaussian decoders를 학습한다. UPM의 Gaussian decoder는 학습 Network이며,
그 다음 quaternion/scale/opacity를 해석하는 `decode`는 고정 수식이다.

DiT의 28 blocks는 기본 설정에서 유지한다. Smoke에서만 width/depth를 줄인다.
FiCA/Pippo의 image-only generative pretraining 및 private UPM checkpoint는 제공되지
않으므로 해당 weights가 있다고 가정하지 않는다. 구조를 만들고 scratch 학습할 수 있는
것과 대규모 사전학습의 지식/성능을 얻는 것은 별개다.

현재 실습의 범위는 **NeRSemble에서 각 모듈을 학습하고 연결하는 연구용 구현**이다.
03의 render alignment와 04의 animation은 준비된 camera/FLAME pose를 사용하는 명시적인
adapter다. 임의의 외부 사진을 넣기만 하는 tracking-free 제품형 inference까지 완성했다고
주장하지 않는다. 이 adapter와 비공개 Codec rig의 차이도 학습하면서 확인할 부분이다.

### 경계마다 실제로 전달하는 tensor

아래 크기는 기본 UV512 설정이며 batch 축은 생략했다.

| 경계 | Tensor | 다음 단계에서의 의미 |
|---|---|---|
| RGB → 01 | RGB `3×H×W` | 세 predictor가 같은 사진에서 UV2/normal3/XYZ3를 독립 예측 |
| 01 → 02 | Partial UV `10×512×512` | RGB3, visibility1, normal3, canonical XYZ3 |
| VAE → DiT | 각 domain `4×64×64` → pack `16×32×32` | texture/geometry는 동일 모델과 domain switcher를 공유 |
| DiT → 03 | Texture `3×512×512`, XYZ `3×512×512` | Gaussian parameter가 아니라 canonical textured mesh의 표현 |
| 03 → 04 | 보정된 Texture/XYZ, identity bias pyramid | 한 portrait로 만든 identity를 고정해서 사용 |
| 04 → renderer | position3, rotation4, scale3, opacity1, color3 | 유효 UV texel당 실제 3D Gaussian 하나 |

모든 단계를 한 번에 공동 학습하는 코드는 아니다. 01/02/03/04는 별도로 학습하며 사이의
결과를 cache한다. VAE/CLIP/Sapiens feature를 frozen으로 두는 이유와, `detach`/`no_grad`가
어디서 gradient를 끊는지도 각 파일에서 직접 확인할 수 있다.

## 환경과 외부 자산

이 작업에서 검증한 환경: Python 3.10, Torch 2.5.1+cu124, torchvision 0.20.1+cu124,
CUDA toolkit 12.4, H200. 추가 Python 버전은 `requirements.txt`에 기록했다.
기존 elite 환경에는 dependency가 준비되어 있다. 새 환경을 준비할 때만 설치한다.

```bash
conda activate elite
cd /media/vilab/mh/curious/lessons/fica
python -m pip install -r requirements.txt
python 00_prepare_nersemble.py --stage assets
```

`assets`는 공식 Sapiens-1B pretrain/normal, SDXL VAE, CLIP weight와 pinned Pippo source를
받는다. 이미 있는 SDXL VAE cache를 재사용한다. FiCA용 fine-tuned UV/XYZ weights는 없으므로
01에서 직접 학습한다. 원본 source와 LICENSE, revision/hash는 `.cache/reference/`에 있다.
Sapiens의 수 GB weight를 포함해 다운로드 자료는 `.cache/` 안에 두고 Git에서 제외한다.

실제 CUDA extension은 NVDiffrast와 **Graphdeco 3DGS**다. 후자는 이번에 기존 elite 환경에
빌드했다. 새 환경에서만 다음 명령이 필요하다. 기존 ELITE 2DGS backend를 대신 쓰지 않는다.

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=4
python -m pip install --no-deps --no-build-isolation 'git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git@59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d'
```

FLAME/VHAP/STAR/matting은 기존 ELITE 환경과 자산을 재사용한다. 데이터 준비 코드가 참조하는
`scene.pt` 안에 asset 경로가 있다. 새 asset을 가짜 mesh로 대체하거나 기존 파일을 복사하지 않는다.

## 00 — 자료가 맞는지 먼저 보기

처음에는 기존 ELITE 자료의 두 frame으로 좌표계/입출력을 확인할 수 있다.

```bash
CUDA_VISIBLE_DEVICES=0 python 00_prepare_nersemble.py --frames 0,32 --data data/inspection
```

`data/inspection/inspection.jpg`에서 reference, UV texture를 다시 올린 mesh, normal,
partial UV, visibility를 확인한다. 원본 tensor는 `frames/*.pt`, identity의 texture/geometry와
UV lookup은 `identities/*.pt`, split/단위/normalization은 `manifest.json`에 있다.

비교 패널은 종횡비를 유지해 가운데를 정사각형으로 crop한 뒤 표시한다. 직사각형 RGB를
정사각형으로 강제 변형하지 않는다. 이 crop은 시각화에만 적용하며 학습 RGB·alpha·label과
camera K는 00의 비율 유지 resize 좌표계를 사용한다.

두 frame의 대응이 맞는 것을 확인한 뒤, 학습 자료는 별도 기본 경로에 준비한다.
`--frames all`은 모든 frame을 사용한다. 옵션을 생략하면 빠른 확인을 위해 25 frame
간격으로 고른다. 이 작은 subset을 전체 데이터셋 학습으로 간주하지 않는다.
아래 이후의 명령은 기본 `data/prepared/`를 읽는다.

```bash
CUDA_VISIBLE_DEVICES=0 python 00_prepare_nersemble.py --frames all
```

- UV row0은 v=1. XYZ는 canonical head의 실제 world unit이며 normal과 다르다.
- Normal prediction label은 **OpenGL camera space**, 별도 `normal_world`는 world space다.
- 원본 mesh는 FLAME+teeth topology다. Hair/shoulder 전체의 정확한 geometry GT가 아니다.
- Texture는 관측 RGB다. Intrinsic albedo나 안 보인 영역의 정확한 GT가 아니다.
- Texture 누적에는 train camera/frame만 사용한다. Missing texture는 flow loss에서도 제외한다.
- `MeshBridge`는 G=A·V라는 barycentric 관계를 least squares로 역산한다. 미분 가능하며,
  UV seam의 여러 관측을 같은 vertex에 모은다. Rank가 부족한 낮은 해상도만 작은 ridge로 안정화한다.
- Canonical delta의 posing에는 고정된 tracking LBS rotation을 쓴다. 04의 Gaussian guide는
  Network가 별도로 예측하며 GT posed vertices를 Gaussian anchor로 직접 입력하지 않는다.

### Multi-view NeRSemble로 확장

NeRSemble raw calibration에 기존 monocular tracking을 그대로 붙이면 좌표계가 틀어진다.
같은 multi-view tracking에서 나온 **VHAP export의 images + cameras + FLAME parameters**를
함께 가져온다. 공식 절차는 [VHAP NeRSemble 문서](https://github.com/ShenhanQian/VHAP/blob/main/doc/nersemble.md)를 따른다.

기존 pinned VHAP의 v2 입력 예시(기존 assets/환경이 준비된 상태):

```bash
cd /media/vilab/mh/curious/lessons/elite/external/vhap
CUDA_VISIBLE_DEVICES=0 python vhap/preprocess_video.py --input /NeRSemble/042/sequences/FREE --target_fps 25 --matting_method robust_video_matting
CUDA_VISIBLE_DEVICES=0 python vhap/track_nersemble_v2.py --data.root_folder /NeRSemble --data.subject 042 --data.sequence FREE --exp.output_folder /tracking/042_FREE
CUDA_VISIBLE_DEVICES=0 python vhap/export_as_nerf_dataset.py --src_folder /tracking/042_FREE --tgt_folder /processed/042_FREE --flame_mode param
```

Full tracking은 smoke에서 새로 실행하지 않는다. 준비된 export를 다음처럼 등록한다.

```bash
cd /media/vilab/mh/curious/lessons/fica
python 00_prepare_nersemble.py --stage import-vhap --vhap-export /processed/042_FREE --data data/imported_042
```

`data/imported_042/scene.pt`와 `views.json`이 만들어진다. 원본 RGB/alpha를 참조하고 카메라의
split, intrinsic/extrinsic, recentering, per-frame parameters를 보존한다. Anatomical 68
landmark는 STAR 검출이 있으면 검출값을, export만 있으면 tracking projection을 target으로
사용한다. 후자는 detector/GT와 동일하다고 주장하지 않는다. VHAP dynamic offsets도 보존한다.

`fica_config.json`을 복사해 아래 항목을 수정한 뒤 모든 명령에 같은 `--config`를 전달한다.
경로는 `lessons/fica/` 기준이다. 여러 identity이면 두 목록에 같은 순서로 추가한다.

```json
{
  "sources": ["data/imported_042/scene.pt", "data/imported_043/scene.pt"],
  "views_manifests": ["data/imported_042/views.json", "data/imported_043/views.json"],
  "identity_splits": {"id_000": "train", "id_001": "validation"}
}
```

Held-out camera/identity를 texture 생성 전에 정한다. 00의 `--frames`는 **재색인된** frame
번호다. 같은 사람의 frame split은 unseen-identity 평가가 아니다. 현재 로컬 smoke 자료는
한 사람·한 카메라이며, multi-view/새 인물 일반화까지 검증했다고 해석하면 안 된다.
NeRSemble에는 완성된 FiCA용 고해상도 UV assets가 들어 있지 않으므로 tracking-derived
geometry의 한계와 UV coverage를 함께 평가한다.

## 01 — 사진에서 partial UV가 나오는지 보기

```bash
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage train --task uv
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage train --task xyz
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage train --task normal
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage finetune-uv --steps 3000
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage predict
```

세 task는 별도의 실제 Sapiens-1B encoder를 fine-tune한다. UV/XYZ는 L1, normal은 cosine
loss다. Encoder LR을 head보다 작게 사용한다. **학습 입력도 비율을 보존하는 crop**이다.
00의 RGB를 Sapiens 입력의 `H:W=4:3`에 맞게 자르고, 모델 내부에서 `1024×768`로
resize한다. 예를 들어 저장된 `512×351` 이미지는 validation에서 중앙 `468×351`로
crop되므로 얼굴 비율이 유지된다. 얼굴 검출 bbox crop은 아니다.
학습은 최대 crop의 80–100% 크기와 위치를 무작위로 정하고, validation/최종 예측은
최대 크기의 중앙 crop을 고정한다. RGB·alpha·UV·XYZ·normal·valid에 **동일한 crop**을
적용하며, dense label 값은 보간하거나 바꾸지 않는다. K/keypoints도 crop 좌표계로 이동한다.
RGB에는 photometric augmentation을 적용한다. Identity를 먼저 균등 sample하고 view angle
기반 importance weight로 frame/view를 고른다. Weight 식은 공개되지 않아 구현 선택으로 기록한다.

### 100 step마다 validation 보기

기본 실행은 `--run-name crop --val-every 100 --val-frames 8`이다. 학습 터미널은
**tqdm bar 하나에 train loss와 최근 validation loss**만 표시한다. Gradient 등의 상세
수치는 `training.jsonl`에 기록하고, step별 디버깅 문구는 출력하지 않는다.

`outputs/01/crop/uv/validation/0.jpg`, `100.jpg`, `200.jpg`, …로 저장한다.
XYZ/normal도 각 task 폴더 안에 같은 구조를 사용하며 step별 폴더를 만들지 않는다.
0은 학습 전 기준이고, 100은 **gradient accumulation을 마친 optimizer update 100회**다.
최종 step이 100의 배수가 아니면 별도 validation 이미지를 추가하지 않는다.
`--val-every`로 간격을 바꿀 수 있다. Checkpoint 저장 간격은 기존 1,000 step이다.

항상 같은 held-out frame을 미리보기로 사용한다. 평가는 manifest 순서상 처음 최대
8개 validation frame에서 수행하고 `--val-frames 0`이면 전체를 사용한다. 현재 로컬 자료는
validation 7개이므로 모두 평가한다. 학습 frame으로 validation을 대신하지 않는다.
각 패널은 실제 모델 입력, GT, 예측, 고정 색상 척도의 오차를 보여준다.
UV에는 GT/예측 대응으로 RGB를 UV에 splat한 비교도 추가한다. 이는 UV만의 진단이며
세 predictor를 합친 최종 10ch partial UV와는 구분한다. 화면 비율은 letterbox로 보존한다.

`validation/metrics.jsonl`에는 전체 평가 frame의 valid pixel 수로 가중 평균한 loss와
UV 거리(texel), XYZ 거리(world unit), normal 각도(degree)를 기록한다. 이미지 상단 수치는
표시된 한 frame의 값이다. 검증에는 `eval/no_grad`를 사용하고 optimizer는 갱신하지 않는다.
이 split은 같은 사람의 다른 frame이므로 새 사람에 대한 일반화 평가가 아니다.

**이전 학습 결과는 보존한다.** 새 checkpoint는 `checkpoints/01/crop/<task>.pt`에 저장한다.
이전에 학습한 `checkpoints/01/<task>.pt`는 수정하지 않는다. 전처리가 바뀐 이전 학습은
동일 실험의 `--resume`으로 취급하지 않는다. 기존 weight를 활용하려면 다음처럼 시작한다.
Optimizer와 step을 초기화하고, 새 `0.jpg`부터 crop 입력에 적응시키는 학습이다.

```bash
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --task uv --init-from checkpoints/01/uv.pt
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --task xyz --init-from checkpoints/01/xyz.pt
# normal.pt를 저장해 둔 경우에만 --init-from checkpoints/01/normal.pt를 추가한다.
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --task normal
```

새 run을 중단했다가 이어갈 때는 같은 `--run-name`과 `--resume`을 사용한다.
완전히 새 학습은 `--run-name crop_v2`처럼 이름을 바꾼다. `--steps`는 추가 횟수가 아니라
최종 목표 step이다. 기존 checkpoint를 학습 없이 crop 입력에서 평가할 수도 있다.

```bash
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage validate --task uv --checkpoint checkpoints/01/uv.pt --run-name legacy_validation
```

이는 기존 weight의 **새 전처리에서의 평가**다. 예전 resize 기반 성능과 동일한 측정이 아니다.
Weight만 읽는 평가/초기화에서도 task·dataset·전체 state key/shape를 확인하며,
이전 코드/설정은 provenance에 기록한다. Optimizer를 복원하는 resume의 엄격한
config/data/code 검사는 유지한다.

기존 `outputs/01/*.jpg`는 person mask와 point splat을 쓴 비교 결과다.
`--stage predict --legacy-splat`으로 같은 방식을 명시적으로 다시 확인할 수 있다.
학습한 영역보다 넓은 사람 mask를 사용해 머리카락/옷의 미감독 UV가 섞이는 문제가 있어,
기본 predict는 아래 추가 학습의 **예측 foreground + 면 기반 UV resampling**을 사용한다.
새 모델/예측이 없으면 중단하며 GT mask나 이전 person mask로 대신하지 않는다.
조합된 10ch partial UV는 `data/prepared/partial_pred/`에 저장된다.
최종 예측 tensor는 원본 사진의 `H×W`로 되돌려 저장한다. Crop 밖 UV는 `-1`로 채워
splat 대상에서 제외하고, XYZ/normal은 0으로 둔다. Crop/run/step은 같은 이름의 JSON에
남긴다. 02 이후의 RGB·camera·image-space prediction 대응을 유지하기 위한 처리다.
세 task 중 일부만 이전 resize 방식으로 남아 있으면 `predict`에서 중단한다.
UV/XYZ/normal 모두 같은 crop 좌표계의 예측을 준비한 뒤 조합한다.

### 기존 50K UV에서 추가 학습하기

이 과정은 **우리 구현의 보완**이다. Auxiliary validity head, BCE/Dice,
FP32 UV decoder, source-grid triangulation은 공개되지 않은 FiCA 저자의 원본 코드라고
주장하지 않는다. 01에서 신뢰할 수 있는 관측을 만들고, 남은 오차/미관측 부위는 02의
diffusion이 다룬다는 학습 목표는 유지한다.

이미 UV·XYZ·normal을 학습했다면 3개를 다시 학습할 필요가 없다.
아래 명령은 기본적으로 `checkpoints/01/crop/uv.pt`의 **weight만** 가져온다.
새 목표와 새 head 때문에 optimizer는 초기화하고, 추가 step을 0부터 센다.

```bash
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage finetune-uv --steps 3000
```

다른 UV checkpoint를 사용하려면 `--init-from checkpoints/01/다른_run/uv.pt`를 지정한다.
기존 50K checkpoint, XYZ/normal cache, 원래 UV 예측은 보존한다.

1. **처음 100 step:** backbone/기존 UV decoder를 고정하고 새 validity head만 학습한다.
2. **이후:** backbone, UV decoder, validity head를 함께 추가 학습한다.
3. **완료:** 새 UV2와 foreground probability1을 전체 frame에 export한다.

`DenseHead.forward_features`의 기존 384ch feature에서 두 분기로 나뉜다.
기존 `conv_seg`는 UV2를 출력하고, 새 `384→64→1` convolution head는 foreground logit을
출력한다. UV decoder/head/보간은 FP32이고 encoder는 기존 BF16 autocast를 사용한다.
이때 foreground 확률은 **이 픽셀이 UV 학습 대상인가**를 나타내며 UV 좌표의 정확도
자체를 보증하는 confidence는 아니다.

Loss는 기존 foreground UV L1 + `0.05 × (balanced BCE + 0.5 × Dice loss)`다.
UV L1은 valid 표면에서, mask loss는 배경까지 포함한 이미지 전체에서 계산한다.
Balanced BCE는 foreground/background pixel 평균에 같은 비중을 준다.
학습률은 encoder `1e-6`, 기존 UV decoder `1e-5`, 새 validity head `1e-3`다.
설정은 `fica_config.json`의 `uv_repair`에서 읽는다. `--warmup-steps`로 warmup을 조절한다.

추가 학습도 tqdm bar 하나를 사용하고, **0/100/200/... step**의 고정 validation을 저장한다.

- Checkpoint: `checkpoints/01/uv_repair/uv.pt` (기본 1,000 step 간격 + 마지막)
- Validation: `outputs/01/uv_repair/uv/validation/0.jpg`, `100.jpg`, …
- 세부 지표: 같은 폴더의 `metrics.jsonl`
- 새 cache: `data/prepared/predictions/uv_repair/{uv,validity}/`

새 mask head가 보수적으로 초기화되므로 초기 validation의 partial UV가 비어 있을 수 있다.
기존 UV가 지워진 것이 아니다. 학습하면서 **validity IoU/precision/recall, source precision,
UV 오차, 실제 RGB/normal partial UV**를 함께 확인한다. GT mask는 loss/지표/GT 비교
패널에만 사용한다. 실제 예측 패널은 예측 mask와 기존 XYZ/normal cache로 만든다.
추가 학습 이미지 상단의 수치는 validation frame 전체의 집계값이다.
따라서 UV loss만 낮고 background가 UV에 섞이는 문제도 확인할 수 있다.
Overlap RGB L1은 GT와 예측 둘 다 관측한 texel에서만 계산하므로 coverage와 함께 읽는다.

같은 추가 학습을 이어갈 때는 `--resume`을 사용한다. `--steps`는 최종 **추가 학습** step이다.
예를 들어 1,000에서 중단했다면 다음은 1,001–3,000을 진행한다.

```bash
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage finetune-uv --steps 3000 --resume
```

새 실험은 `--run-name uv_repair_v2`로 분리한다. 중간 checkpoint는 별도로 평가할 수 있다.

```bash
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage validate --run-name uv_repair
```

학습 완료 후 최종 partial UV를 만든다. 기본 threshold는 `.7`이다.

```bash
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage predict
```

이 단계는 예측 foreground와 person alpha의 교집합에서, UV 범위와 XYZ/normal의 유한성을
검사한다. 인접한 유효 pixel cell을 UV 삼각형으로 rasterize하고 원본 image 좌표를
보간한 뒤 `grid_sample`로 RGB/XYZ/normal을 함께 읽는다. Normal은 다시 단위 벡터로 만든다.
네 꼭짓점이 모두 유효한 cell만 쓰고, UV seam/국소 fold/퇴화 cell을 제외한다.
기본 `max_uv_edge=.03`은 정규화 UV 단위이며 길게 늘어나는 관측도 보수적으로 제외할 수 있다.
관측되지 않은 UV를 임의로 채우지 않는다. `weight`는 이 방식에서는 point splat 누적량이
아닌 raster coverage이며, metadata의 `weight_kind`에 구분해 기록한다.

최종 비교 이미지는 `outputs/01/uv_repair/predictions/`에 생긴다. `data/prepared/partial_pred/`
는 새 결과로 갱신하므로, 02를 시작하기 전에 **latent cache를 다시 만들어야 한다**.
02는 partial 파일의 SHA256을 함께 저장하고 오래된 latent의 학습/추론 재사용을 거부한다.

```bash
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage cache
```

학습은 끝났지만 export 도중 중단되었다면 학습을 다시 돌릴 필요 없이 다음을 사용한다.

```bash
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage export --run-name uv_repair
```

`CUDA_VISIBLE_DEVICES=0 python smoke.py --stage repair`는 실제 1B/기존 UV weight를 사용해
warmup→joint 학습, 저장/재로딩/optimizer resume, FP32 출력, UV seam, GT 독립성,
02 cache 검사를 별도 `.cache/smoke/repair_data`에서 수행한다. 입력은 smoke 해상도다.
새 head를 3 step 학습한 smoke 결과를 수렴한 결과로 해석하지 않는다.

## 02 — Noise에서 UV와 mesh가 만들어지는지 보기

이 part에서는 아래 **Pippo 원본을 직접 읽기**를 먼저 진행한다. 원본 source의
`ControlMLP`와 `DiTBlock`을 이해한 뒤 `UVFlowModel`이 이들을 어떻게 연결하는지 확인한다.

```bash
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage cache
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage train
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage sample
```

학습 화면은 **tqdm bar 하나에 train loss와 최근 validation flow loss**를 표시한다.
상세 gradient는 `outputs/02/training.jsonl`에만 기록한다. 시작 시점과 **100 optimizer
step마다**, 그리고 마지막 step에 고정 validation을 저장한다. 이미지 경로는
`outputs/02/validation/0.jpg`, `100.jpg`, `200.jpg`, … 이며 step별 폴더는 만들지 않는다.

기본 validation은 항상 첫 번째 held-out frame을 사용한다. Reference/partial UV와
noise seed를 고정하고, 기본 50-step Euler sampling을 수행한다. 한 장의 비교 이미지에
입력 사진·partial RGB·visibility, target/generated texture와 오차, target/generated XYZ와
오차, 두 mesh render, texture supervision mask를 배치한다. Target은 00에서 만든
pseudo target이며 미관측 texture를 정답으로 취급하지 않는다. 사진은 종횡비를 유지하고,
XYZ 표시와 error 색 범위도 모든 step에서 고정한다.

`validation/metrics.jsonl`에는 고정 noise와 `t=0.5`에서의 flow loss, 관측된 target
texel의 RGB L1, valid UV의 XYZ Euclidean RMSE(world 단위)를 기록한다. RGB/XYZ 지표는
sampling을 완료하고 VAE로 decode한 결과의 오차이므로 flow loss와 구분한다. RGB는
표시할 때만 [0,1]로 clip하고 지표는 원래 예측값으로 계산한다. Validation은 optimizer를
갱신하지 않으며 train/eval mode와 학습 RNG를 복원한다.

간격·평가 frame 수·sampling 비용은 다음처럼 지정할 수 있다. `--val-frames 0`은
전체 held-out frame의 지표를 평균내며, JPEG에는 같은 첫 frame을 계속 표시한다.

```bash
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage train --val-every 100 --val-frames 1 --val-sample-steps 50
```

`cache`는 실제 SDXL VAE/CLIP을 사용한다. Partial의 RGB/mask/normal/XYZ를 각각 VAE에 넣고
latent를 이어 ControlMLP에 전달한다. `--partial-source gt_correspondence`는 별도 oracle
ablation이며 기본값은 항상 `predicted`다. 어떤 경로인지 cache와 학습 log에 남는다.

학습식은 x_t=(1−t)·noise+t·target, v*=target−noise다. Texture/geometry는 같은 Network를
domain=0/1로 두 번 평가한다. CLIP+time+domain은 전역 modulation, partial UV는 위치별
ControlMLP로 전달한다. Sampling은 noise에서 data 방향으로 Euler 적분한다.

`outputs/02/predictions/*.jpg`는 입력 사진 / 생성 mesh render / 생성 texture / XYZ 표시다.
`generated/*.pt`에는 실제 float UV와 sampling 중간 latent가 있다. XYZ의 RGB 표시는
시각화일 뿐이며 표시 이미지를 mesh 좌표로 사용하지 않는다. VAE 재구성 자체의 geometry
오차도 별도로 검사해야 한다. 원본 모델처럼 unseen identity를 생성하려면 여러 identity의
학습이 필요하고, 한 사람 overfit만으로 이 능력을 주장할 수 없다.

### XYZ가 찢어질 때: 기존 weight 안정화와 geometry 추가 학습

50K 모델 진단에서 정답 geometry latent를 SDXL VAE로 decode하는 것만으로도
XYZ 오차가 생겼다. 또한 작은 UV triangle에 대응하는 vertex의 least-squares 해가
이 오차를 증폭했다. 현재 `MeshBridge`는 아래 식으로 template에 대한 변위를 안정화한다.

```text
min_V ||AV - XYZ||² + 1e-4 ||V - V_base||² + 10 ||L(V - V_base)||²
```

`L`은 vertex 이웃의 평균을 빼는 uniform Laplacian이다. `V_base`는 개인의 정답 mesh가
아닌 공개 FLAME base이며, inference에서 GT geometry로 바꿔치기하지 않는다.
이 변경은 02 render와 이후 03/04에도 같은 `PreparedData.bridge`를 통해 적용된다.
`MeshBridge(..., laplacian=0)`으로 기존 방식과 비교할 수 있다.

기존 50K weight/cache 그대로 안정화된 결과만 먼저 확인하려면:

```bash
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage stabilize
```

이미지는 `outputs/02/stabilized/predictions/`, tensor는 `data/prepared/generated_stabilized/`에
저장한다. 원래 checkpoint, latent cache, validation 이미지를 덮어쓰지 않는다.

추가 학습은 `geometry_repair` run에서 진행한다. DiT/ControlMLP 구조와 frozen SDXL VAE는
같고, 기존 50K weight를 초기값으로 사용한다. 새 optimizer/step으로 2,000 step을 학습한다.

- XYZ normalization: `(XYZ - mean) / (3 * std)`. Inverse에서도 같은 3을 곱한다.
  RGB/normal/mask의 전처리와 texture VAE는 그대로다. 입력을 hard clip하지 않는다.
- Geometry target latent는 atlas 밖의 알려진 padding까지 감독한다. 실제 surface의 미관측
  texture를 검정 정답으로 쓰는 것은 아니며, texture loss는 관측 mask를 계속 사용한다.
- 첫 100 step은 flow loss로 새 latent 분포에 적응한다. 이후 4 step마다 같은 DiT의
  `x_1_hat = x_t + (1-t) * v_hat`를 VAE로 decode하고, valid XYZ L1과 mesh vertex/edge
  오차를 추가한다. VAE parameter는 고정하지만 latent로의 gradient는 유지한다.
- 이 normalization, auxiliary loss, Laplacian 복원은 우리 재현의 보완이며 FiCA 저자의
  공개되지 않은 세부 구현과 동일하다고 주장하지 않는다.

```bash
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage codec-check
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage finetune-geometry
# 중단된 동일 run을 계속 학습하려면:
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage finetune-geometry --resume
```

`finetune-geometry`는 학습 전에 실제 XYZ→VAE→mesh 왕복 오차를 검사하고, 필요한 새
cache를 자동으로 준비한다. Codec의 평균 vertex 오차가 head 크기의 5%를 넘으면 본 학습을
시작하지 않는다. 이는 최소한의 이상 감지이며 품질을 보장하는 기준은 아니다.
새 cache는 `data/prepared/latents/geometry_repair/`, checkpoint는
`checkpoints/02/geometry_repair/latest.pt`, 비교 이미지는
`outputs/02/geometry_repair/validation/0.jpg`, `100.jpg`, … 에 저장한다.
`codec_check.json/jpg`에는 DiT와 무관한 codec 왕복 기준선이 있다.

Validation에는 UV 오차 외에 vertex RMSE/최대 오차, 정답 대비 10배 이상 늘어난 edge 수도
기록한다. 고정 frame/noise seed와 학습 RNG 복원은 유지한다. 새 학습 완료 후에는 모든
frame을 `data/prepared/generated/`로 export하여 03이 읽게 한다. World 좌표인 `xyz`와
진단용 mesh projection인 `xyz_surface`를 함께 저장하며 실제 texture/XYZ 예측에 GT는
사용하지 않는다. Render는 이전처럼 준비된 tracking pose adapter를 사용한다.
생성 결과를 바꾼 뒤에는 03 feature cache도 다시 만들어야 한다.

```bash
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage sample --run-name geometry_repair
CUDA_VISIBLE_DEVICES=0 python 03_refine_uv.py --stage cache
```

`--steps 4000`은 새 run의 총 목표 step을 뜻한다. 기본 2,000 step 이후 더 이어갈 때는
`--resume --steps 4000`을 사용한다. `--init-from /path/to/latest.pt`는 weight만 가져오는
새 run의 초기화이며 optimizer resume와 다르다. Cache의 codec/partial fingerprint가
맞지 않으면 학습·sampling을 중단해 이전 전처리와 섞이는 것을 막는다.

## 03 — 보정이 texture와 geometry를 어떻게 바꾸는지 보기

```bash
CUDA_VISIBLE_DEVICES=0 python 03_refine_uv.py --stage cache
CUDA_VISIBLE_DEVICES=0 python 03_refine_uv.py --stage train --run-name quality_repair --init-from checkpoints/03/latest.pt --steps 1000
CUDA_VISIBLE_DEVICES=0 python 03_refine_uv.py --stage predict
```

초기 생성 UV를 render하고 reference/render의 frozen Sapiens features를 cache한다.
U-Net의 UV queries가 두 이미지의 tokens를 cross-attend한다. 시간 입력은 0이며 별도의
diffusion sampling을 하지 않는다. 출력은 texture/XYZ residual이다.

입출력 UV는 모두 512×512다. 채널 폭은 `[320, 640, 1280, 1280]`을 유지하고,
512→256→128 구간은 convolution/skip connection, 64×64 구간과 bottleneck은
self-attention 및 Sapiens image feature와의 cross-attention을 사용한다. Up path는
64→128→256→512로 복원하며 고해상도 skip을 합친다. UV 전체를 64로 축소하거나,
작은 residual을 interpolation으로 확대하는 방식이 아니다.

이전 배치는 512×512에서도 global self-attention을 수행하여 head 하나당 약 687억
query/key 쌍을 처리했다. 64×64에서의 쌍 수는 그 1/4096이지만 전체 학습 속도가
4096배가 되는 것은 아니다. 고해상도 convolution, gradient, rendering은 계속 필요하다.
`refinement.attention_max_size=64`가 이 배치를 정한다. FiCA는 세부 attention 배치를
공개하지 않았으므로 이 변경은 우리의 구현 선택이다. 기존 attention 배치의 03 checkpoint와
구조가 달라 weight/optimizer를 그대로 재개할 수 없다. 01/02 출력과 Sapiens feature cache는
생성 UV가 동일하면 그대로 사용한다.

Image L1, alpha mask, pixel 단위 2D landmark L1과 초기 texture·cotangent Laplacian matrix·
vertex normal의 regularization을 사용한다. Loss reduction/정규화는 구현 선택이며
정확한 저자 implementation과 동일하다고 주장하지 않는다. Geometry의 Cholesky solve,
posing, CUDA rasterizer까지 gradient가 보존된다. 생성 UV가 바뀌면 기존 feature cache는
hash mismatch로 거부한다. 먼저 `cache`를 다시 실행한다.

Laplacian은 geometry-dependent cotangent matrix의 각 row를 L1 정규화한 값이다.
초기 생성 mesh의 가느다란 삼각형에서 cotangent가 커져 loss를 지배하는 현상을 줄인다.
행렬은 여전히 현재 geometry에 따라 변하고 gradient가 흐른다. 정확한 Laplacian 정의와
reduction은 FiCA에 공개되지 않았으므로 이 정규화를 저자 설정으로 주장하지 않는다.

학습은 tqdm 한 줄에 loss, 최근 validation photo L1, 진행률/속도/예상 시간을 표시한다.
개별 loss 항·gradient norm·validation/checkpoint를 제외한 optimizer step 소요시간은
`outputs/03/training.jsonl`에만 기록한다. 수백 개 gradient tensor마다 CPU로 동기화하던
debug 검사는 전체 gradient norm의 NaN/Inf 검사와 clipping으로 합쳤다.

기존 1K 실험에서는 texture 변화가 RGB 0~1 기준 평균 약 0.00026에 머물렀고, geometry
변위 에너지의 약 99.4%가 전체 translation이었다. `surface_refinement_v2`는 이 관찰을
보완하는 **우리의 loss/제약 변경**이며 FiCA 저자가 공개한 세부 설정이 아니다.

1. `constrain_geometry`: mesh의 중심을 초기 생성 mesh와 일치시킨다. UV texel 평균과
   vertex 평균은 다르므로, MeshBridge 선형부의 계수로 중심 변화량을 구한다.
   보정 UV에서 이 translation을 빼며 gradient는 Network까지 전달한다. GT mesh로
   위치를 맞추거나 geometry를 바꾸는 과정이 아니다. 3개 translation 자유도만 제거한다.
2. `photo`: 관측 FLAME 얼굴 mask 안에서 RGB L1을 평균한다. 쉬운 배경 pixel이 오차를
   희석하지 않도록 한다. 이전 전체 이미지 기준 지표는 `photo_full`로 따로 보존한다.
   이전 run의 `photo`와 새 run의 `photo_full`을 비교해야 같은 정의다.
3. `texture_reg`: pixel별 vector norm 대신 유효 UV의 RGB MSE를 사용하고 가중치를
   0.1로 둔다. 작은 보정에서는 규제 gradient도 작아져 사진에 맞춘 색 수정을 허용한다.
4. `geometry_reg`: 초기 생성 vertex로부터의 변위 제곱을 0.01 world 단위로 정규화하고
   가중치 0.25를 준다. 현재 FLAME 단위에서 0.01은 1 cm다. 전체 이동을 제거한 뒤에도
   얼굴 형태가 임의로 크게 변하지 않게 한다. 초기 생성 mesh가 기준이며 GT geometry는
   이 학습 항이나 best 선택에 사용하지 않는다.

Photo/mask/keypoint의 바깥 가중치는 2/0.5/0.01, Laplacian/normal은 각각 1을 유지한다.
Refiner Network/외부 Sapiens feature는 그대로이며, 새 run의 learning rate는 3e-5다.
새 objective와 기존 loss의 숫자는 직접 비교하지 않고, 같은 정의의 `photo_full`·mask·
keypoint·geometry 지표를 함께 본다. 검증에는 vertex 중심 이동, 이동을 뺀 형태 변화 RMS,
texture 평균 변화량도 기록한다. Coarse FLAME vertex RMSE는 계속 진단용이다.

3번의 기본 학습량은 `refinement.steps=1000`이다. 원 논문의 50K-step 대규모 학습과 구별되는
현재 단일 identity 실습 설정이다. Validation과 checkpoint의 기본 주기는 각각 100 step이다.
Validation은 학습 전과 마지막에도 실행하고, 학습 frame으로 대체하지 않는다. 기본
`val_frames=0`은 현재 held-out 7개 전체의 평균이며 표시는 같은 첫 frame으로 고정한다.

```bash
# 기존 weight만 가져와 별도 run을 시작한다. optimizer와 local step은 새로 시작한다.
CUDA_VISIBLE_DEVICES=0 python 03_refine_uv.py --stage train --init-from checkpoints/03/latest.pt --steps 1000
# 이미 quality_repair run이 있다면 총 목표 step까지 이어간다. init-from을 함께 쓰지 않는다.
CUDA_VISIBLE_DEVICES=0 python 03_refine_uv.py --stage train --resume --steps 1000
# 추론은 기본으로 best.pt를 사용한다. 마지막 모델 비교는 --weights latest.
CUDA_VISIBLE_DEVICES=0 python 03_refine_uv.py --stage predict
CUDA_VISIBLE_DEVICES=0 python 03_refine_uv.py --stage validate --weights best
```

CLI의 기본 `--run-name`은 `quality_repair`다. `outputs/03/quality_repair/validation/0.jpg`,
`100.jpg`, `200.jpg`처럼 step별 JPEG 한 장을 저장한다. 기존 `outputs/03/validation/`과
`checkpoints/03/latest.pt`는 비교용으로 그대로 보존한다.
항상 같은 첫 validation frame의 reference, 초기/보정 mesh render, 동일 scale의 photo error,
초기/보정 UV texture, XYZ 및 변화량을 비교한다. 사진 종횡비는 letterbox로 보존한다.
`--val-frames`를 늘려도 이미지는 첫 frame으로 고정하고 지표는 선택한 frames의 평균이다.
Validation의 RNG와 train/eval 모드는 복원하며 gradient/optimizer update를 수행하지 않는다.

`validation/metrics.jsonl`에는 loss의 각 항, 초기 결과의 동일 지표, coarse FLAME target에 대한
vertex RMSE를 남긴다. Vertex RMSE는 진단용이며 새 학습 loss로 추가하지 않는다.
Photometric loss 감소와 geometry 변화를 함께 확인한다. 이 held-out frame 평가는 새로운
identity에 대한 일반화 평가가 아니다.

`checkpoints/03/quality_repair/latest.pt`는 optimizer/RNG를 포함한 학습 재개용이다.
`best.pt`는 validation total loss가 가장 낮았던 weight만 보존하며 `best.json`에 step과
지표를 기록한다. Best는 optimizer를 재개하는 파일이 아니다. `--steps`는 총 목표 횟수이며
이미 해당 step에 도달한 checkpoint를 같은 steps로 재개하면 명시적으로 오류를 낸다.

`predict`는 선택한 checkpoint와 동일한 중심 제약을 적용해 모든 frame의 보정 UV를
`data/prepared/refined/`에 저장하고 04가 읽는다. 처음 quality run을 export할 때 기존
legacy tensor들은 `data/prepared/refined_legacy_1000/`로 보존한다. 각 tensor에 run/checkpoint
step과 objective를 기록하고, `outputs/03/quality_repair/predictions/`에는 실제 저장 UV의
초기/보정 render 및 texture 비교 JPEG, 전체 frame 지표 `metrics.json`을 함께 남긴다.
Predict는 추가 학습이 아니라 저장된 weight로 추론/export하는 단계다.

2026-10-20 quality 보완 검증: 기존 1K weight에서 `quality_repair`를 200 step 학습하고,
held-out 7개 validation total loss가 가장 낮은 150 step을 선택해 20개 frame을 export했다.
같은 정의의 전체 이미지 L1은 기존 1K의 0.03909→0.03852, coarse vertex RMSE는
0.01712→0.01014였다. Texture 평균 변화량은 약 0.00026→0.03194였고 vertex 중심 이동은
약 1.7e-9 world 단위였다. 색상 수정/translation 억제를 확인했지만 눈·입의 검은 영역과
coarse geometry는 남아 있다. 짧은 보완 검증을 최종 품질 수렴으로 해석하지 않는다.
`repair_report.json`과 `predictions/`에 실제 export 평가가 있다. 위치 제약의 gradient,
optimizer/RNG 재개, best weight 재로딩 및 04의 3DGS 연결 smoke도 통과했다.

2026-10-20, GPU 0의 실제 512×512 / 약 813M refiner와 기존 생성 UV·Sapiens cache로
5 optimizer step을 확인했다. Validation/checkpoint를 제외한 step 시간은 1.04~1.28초,
첫 step을 제외한 중앙값은 1.08초였다. GPU는 다른 학습과 공유 중이었다. Self-attention의
query 길이는 모든 layer에서 4096 이하였고, 512 입출력·condition 영향·checkpoint 재로딩을
검증했다. 이는 짧은 속도/동작 검사이며 수렴이나 최종 품질 검증이 아니다. 별도
`.cache/diagnostics/03_refinement_v2/`에 기록하고 본 학습 checkpoint를 만들지 않았다.

## 04 — Learned UPM이 실제 3DGS를 생성하는지 보기

```bash
CUDA_VISIBLE_DEVICES=0 python 04_gaussian_avatar.py --stage train
CUDA_VISIBLE_DEVICES=0 python 04_gaussian_avatar.py --stage train --val-every 100 --val-frames 1
CUDA_VISIBLE_DEVICES=0 python 04_gaussian_avatar.py --stage render --identity-source refined
```

UPM pretraining은 00의 identity target UV와 multi-view frames를 사용한다. 앞의 생성 모듈과
독립적으로 학습할 수 있다. 추론에서는 `refined` UV를 넣어 01→04 연결을 확인한다.
`--identity-source target`은 UPM 자체를 분리해서 보는 oracle 조건이다.

Identity encoder의 bias pyramid를 geometry/appearance decoder 각 layer에 더한다.
Geometry branch는 view-independent, appearance branch는 view 방향도 받는다. FLAME100의
표정은 학습 adapter를 통해 256D로 바꾸며 gaze는 eye axis-angle에서 direction으로 계산한다.
이 adapter는 비공개 Codec expression encoder와 같지 않다.

Decoder가 guide geometry와 position/color residual, quaternion, **3축** scale, opacity를
예측한다. Root pose만 tracking의 rigid transform을 적용한다. Mesh GT는 loss target이며
Gaussian의 움직임을 정답 mesh로 바꿔치기하지 않는다. Geometry는 XYZ std로 정규화해 감독하고,
guide/렌더링 normal과 Laplacian도 확인한다. RGB photometric·alpha·LPIPS gradient는 실제
Graphdeco CUDA rasterizer를 거친다. Relighting이나 별도 안구 시스템까지 복제하지는 않는다.

`outputs/04/refined/`에는 실제 Gaussian tensor와 RGB, reference/render/normal/alpha 비교가
남는다. Smoke의 두-step Network는 좋은 avatar를 만들기 위한 학습이 완료된 결과가 아니다.

추론에서는 identity마다 **한 portrait의 UV와 identity bias를 고정**하고 frame별 expression,
gaze, pose, camera만 바꾼다. 기본 portrait는 첫 frame이며 `--portrait-frame 0`으로 지정할 수
있다. `render.json`과 각 Gaussian tensor의 `reference_frame`/`driver_frame`으로 확인한다.
여러 frame의 UV를 계속 바꿔 넣는 것을 single-portrait animation으로 간주하지 않는다.

학습과 sequence rendering은 tqdm으로 진행률·속도·예상 시간을 표시한다. 학습 bar의
`loss`는 현재 optimizer step의 학습 loss, `val_target`/`val_refined`는 가장 최근의 고정
validation RGB L1이다. 각 loss 항·gradient norm·validation/checkpoint를 제외한 step
시간은 `outputs/04/training.jsonl`에 기록하고 매 step debug 문장을 출력하지 않는다.
Network 구조와 학습의 target UV 입력·loss 가중치는 유지한다.

Validation은 시작 전, 기본 100 step마다, 마지막 step에 실행한다. 기본은 첫 held-out
driving frame 1개이며 `--val-frames 0`이면 전체 held-out frame 평균을 평가한다.
`outputs/04/validation/0.jpg`, `100.jpg`, `200.jpg`처럼 한 step당 JPEG 한 장을 저장한다.
항상 같은 첫 frame을 표시하고, 사진은 letterbox로 종횡비를 보존한다.

기본 조건은 `--val-sources target refined`다. 두 조건 모두 **train split의 한 portrait를
고정**하고, held-out frame의 expression/neck/gaze/view와 camera만 바꾼다. Target은
UPM 자체를 검사하는 oracle 입력이고, refined는 03 결과를 실제로 이어 넣는 조건이다.
각 validation frame 자신의 보정 UV를 입력해 single-portrait 평가로 제시하지 않는다.
`--portrait-frame`으로 validation portrait를 지정할 때는 train split의 frame이어야 한다.
`refined` 파일이 없으면 이를 target으로 몰래 대체하지 않고, 필요한 경로를 알려준다.
04만 독립적으로 먼저 학습하려면 `--val-sources target`을 지정한다.

조건별 패널은 reference RGB, Gaussian RGB, 고정 scale의 RGB error, 고정 portrait의
입력 texture, reference/Gaussian alpha, world normal, depth다. Depth 표시 범위는 같은
driving frame의 tracked mesh에서 정하므로 예측마다 min/max를 다시 맞추지 않는다.
이는 표시용 범위이며 GT geometry를 Gaussian 위치로 바꿔 넣는 과정이 아니다.

`validation/metrics.jsonl`은 조건별 total/photo/mask/LPIPS/geometry/normal/Laplacian/offset,
전체 흰 배경 합성 이미지 기준 PSNR과 Gaussian 수를 기록한다. Geometry target은 기존
FLAME tracking의 local vertices이며 정확한 scan GT가 아니다. 하나의 identity에서 held-out
frame을 평가하는 범위이며 새 identity 일반화 검증으로 해석하지 않는다. Validation은
gradient/optimizer update 없이 실행하고 학습 RNG 및 train/eval 모드를 복원한다.

고정 UV tensor는 학습 시작 때 한 번 읽고 hash와 portrait 경로를 기록한다. 따라서 실행
중 03 export 파일이 바뀌어도 validation 입력은 바뀌지 않는다. Optimizer를 재개할 때는
같은 frame/source/portrait/UV 파일을 요구한다. `--steps`는 추가 횟수가 아닌 총 목표다.
새 validation 기능은 기존의 `latest.pt` checkpoint 저장 방식과 함께 동작한다.

```bash
# 03까지의 입력 없이 UPM 자체만 검증하는 독립 학습
CUDA_VISIBLE_DEVICES=0 python 04_gaussian_avatar.py --stage train --val-sources target
# 학습을 이어갈 때 validation 조건을 동일하게 지정한다.
CUDA_VISIBLE_DEVICES=0 python 04_gaussian_avatar.py --stage train --resume --steps 60000
# 저장된 latest weight로 validation만 다시 확인한다.
CUDA_VISIBLE_DEVICES=0 python 04_gaussian_avatar.py --stage validate --val-frames 0
```

2026-10-20, GPU 0에서 작은 연결 smoke와 실제 UV512/약 22.4M UPM으로 검증했다.
Full 설정은 frame당 235,498개 3D Gaussians를 사용하여 target/refined validation,
held-out 7개 평가, optimizer의 2→3 step 재개 및 checkpoint 재로딩을 통과했다.
기존 실행과 새 실행의 첫 학습 loss도 부동소수점 오차 범위에서 일치했다. 이는 표시와
학습 연결을 확인한 짧은 검사이며 avatar 품질 수렴 검증은 아니다. 본 학습 자료와
결과는 보존했고 검사 결과는 `.cache/diagnostics/04_validation_full/`에 격리했다.

## 학습 기록, checkpoint, smoke

각 학습은 loss 항·gradient를 JSONL에 남긴다. Checkpoint는 model/optimizer/RNG/config,
dataset fingerprint, 코드 hash를 보존한다. `--steps`는 총 optimizer step이며 재개는
`--resume`을 명시한다. 본 학습 checkpoint는 fresh 실행으로 덮어쓰지 않는다. Microbatch는
camera별 가변 해상도를 위해 1이고 `training.grad_accumulation`으로 effective batch를 늘린다.
학습 이론은 각 part의 실제 loss와 gradient를 읽을 때 함께 살펴본다.

```bash
CUDA_VISIBLE_DEVICES=0 python smoke.py
```

Smoke는 `.cache/smoke/` 안에 격리한다. 실제 NeRSemble 출신 ELITE frame 0/32, 실제 FLAME,
실제 Sapiens-1B/SDXL VAE/CLIP/LPIPS, 실제 mesh/3DGS CUDA backend를 사용한다.
Sapiens 세 task는 각각 2 step, flow는 4 step, refiner/UPM은 각각 2→3 step 재개를 확인한다.
DiT/refiner/UPM의 smoke 크기와 이미지 해상도만 줄인다. 본 설정 생성 모델은 약 1.97B,
refiner는 저해상도 attention 배치의 약 813M, UPM은 약 22.4M이다. 다음 명령은 축소하지 않은 모델의 실제
forward/backward를 추가로 검사한다. Sapiens 입력은 1024×768, UV는 512다.

```bash
CUDA_VISIBLE_DEVICES=0 python smoke.py --stage full
```

이 검사는 큰 모델의 optimizer/checkpoint를 만들지 않는다. Full refinement/UPM에는
입출력과 gradient를 분리 검증하기 위해 target UV를 넣는다. 앞의 작은 end-to-end
smoke는 실제 predictor→생성→보정 UV를 전달한다. Full 검사의 조건 partial은 작은
predictor smoke 결과를 512로 resize한 것이며, 전체 크기 모델의 수렴 검증은 아니다.

`report.json`은 단계별 성공/실패·소요시간·GPU peak memory와 검사 범위를 기록한다.
검사에는 UV→mesh gradient, pretrained strict load, 실제 prediction→generation 연결,
ControlMLP의 출력 영향, checkpoint reload, refinement 변화, expression별 guide 변화,
quaternion/3D scale/alpha와 실제 렌더링이 포함된다. 이 결과를 원본 FiCA 품질이나 새로운
인물에 대한 일반화 성능으로 해석하지 않는다.

2026-10-17, H200 physical GPU 0 / conda elite에서 00~04 smoke와 `--stage full`의
forward/backward가 모두 통과했다. Full UPM은 유효 UV texel의 235,498개 3D Gaussians를
실제로 render하고 역전파했다. 02에서 optimizer/RNG checkpoint의 4→5 step 재개와
2 microbatch→1 optimizer step도 별도로 검증했다. 실제 VHAP export의 single-camera
재가져오기/68 landmark/white composite 검증 결과는 `.cache/smoke/vhap_import.json`,
재개 검증은 `training_controls.json`, 단계별 결과는 `report.json`에 있다.

## Pippo 원본을 직접 읽기

FiCA의 UV 생성기를 이해할 때 Pippo는 핵심 학습 대상이다. 현재 02도 원본의
`DiTBlock`, `ControlMLP`, `TimestepEmbedder`를 그대로 import하지만, 필요한 파일만
`.cache/reference/pippo/`에서 읽는다. 전체 모델의 흐름을 공부하기 위해 공식 저장소를
**수정하지 않은 전체 checkout**으로 `external/pippo/`에 준비했다.

원본: [facebookresearch/pippo](https://github.com/facebookresearch/pippo/tree/6f80b22f32f58f7ae08f7b20a4901a3c66809368).
Commit은 `6f80b22f32f58f7ae08f7b20a4901a3c66809368`로 본편과 같다. 외부 checkout은
Git에서 제외하고, 이 가이드와 `pippo_walkthrough.py`를 관리한다. 다른 PC에서 준비하려면:

```bash
cd /media/vilab/mh/curious/lessons/fica
git clone https://github.com/facebookresearch/pippo.git external/pippo
git -C external/pippo checkout --detach 6f80b22f32f58f7ae08f7b20a4901a3c66809368
```

Pippo는 사진에서 여러 시점의 **이미지**를 생성하는 모델이다. FiCA는 그 기반에서
canonical **UV texture/geometry**를 생성한다. 따라서 원본의 전체 `Pippo` class와
현재 `UVFlowModel`은 같은 모델이 아니다. 원본 block을 가져오는 부분과 FiCA 문제에 맞게
구성한 부분을 아래 순서로 구별해서 읽는다. 원본 전용 pretrained weight는 배포되지 않았다.

### 1. 파일보다 tensor의 이동 순서부터 보기

`external/pippo/` 기준으로 읽을 순서는 다음과 같다.

| 순서 | 원본 파일·정의 | 확인할 내용 |
|---|---|---|
| 1 | `blocks/patch.py`의 `PatchifyPS`, `UnpatchifyPS` | 공간 배열이 token sequence로 바뀌고 돌아오는 과정 |
| 2 | `blocks/timestep.py`의 `TimestepEmbedder` | scalar noise time을 feature vector로 바꾸는 과정 |
| 3 | `models/control_mlp.py`의 `ControlEncoderLatent` → `ZeroControlBlock` → `ControlMLP` | 공간 조건을 block별 control token으로 만드는 과정 |
| 4 | `models/dit.py`의 `DiTBlock.forward` → `DiT.forward` | control이 attention/MLP 중 어디에 들어가고 residual과 어떻게 합쳐지는가 |
| 5 | `models/pippo.py`의 `forward` → `get_conds` → `predict` | RGB/VAE, reference, camera, spatial anchor, Plücker 조건이 모이는 전체 과정 |
| 6 | `config/full/512_4v.yml`, `train.py`, `losses.py`, `schedulers/diffusion.py` | 실제 configuration, noise target과 backward의 연결 |

위 파일들은 모두 `latent_diffusion/` 아래에 있다. `train.py`와 `config/`는 저장소 root다.
다음 명령은 원본 정의의 **실제 파일/줄 번호**를 출력한다.

```bash
python pippo_walkthrough.py --part files
```

처음부터 1,000줄이 넘는 `Pippo` class를 전부 읽기보다, 작은 단위에서 이해한 뒤 5번에서
전체 연결을 확인한다. 공식 source를 별도 교육용 Network로 다시 작성하지 않는다.

### 2. 작은 tensor로 원본 연산을 직접 따라가기

`pippo_walkthrough.py`는 원본 class를 호출하고 tensor shape를 출력하는 읽기 보조 파일이다.
`patch/control/block`의 입력과 target은 **연산 확인용 작은 합성 tensor**이며 avatar 학습
결과로 사용하지 않는다. 원본 file은 변경하지 않고 hook과 debugger로 내부를 살펴본다.

```bash
CUDA_VISIBLE_DEVICES=0 python pippo_walkthrough.py --part patch
CUDA_VISIBLE_DEVICES=0 python pippo_walkthrough.py --part control
CUDA_VISIBLE_DEVICES=0 python pippo_walkthrough.py --part block
```

**Patch:** 작은 packed latent `1×16×8×8`이 `1×16×64` token으로 바뀐다.
두 번째 16은 4×4개의 공간 주소, 마지막 64는 학습된 feature 차원이다.
원본 `PatchifyPS`는 PixelUnshuffle → 1×1 conv → LayerNorm을 사용한다.
현재 FiCA 본편은 packing → `unfold` → Linear로 연결하며 원본 `PatchifyPS` 전체를
그대로 사용한 것은 아니다. 이 차이도 비교할 대상이다.

**Control:** 본 설정에서는 partial의 네 VAE latent가 합쳐져 `16×64×64`, packing 후
`64×32×32`가 된다. `ControlEncoderLatent`가 공간 크기를 절반으로 줄여 `D×16×16`을
만들고 `256×D` token으로 펼친다. Target도 같은 16×16 patch 위치를 사용한다.
공유 encoder가 만든 h를 block마다 별도의 `ZeroControlBlock`에 전달한다.
이 block들은 depth 방향으로 직렬 연결된 MLP가 아니다.

ZeroControlBlock은 timestep에서 shift/scale을 만들고 다음을 계산한다.

```text
control_i = zero_linear_i((1 + scale_i(time)) * h + shift_i(time))
```

처음에는 zero_linear가 0이므로 control도 0이다. 예제의 첫 backward에서는
zero_linear의 gradient는 존재하지만 앞쪽 encoder의 gradient는 0임을 직접 확인한다.
이는 zero initialization의 결과다. 한 번 update하면 control이 0에서 벗어난다.

**DiTBlock:** 이 원본은 익숙한 attention→MLP 직렬 block과 연산 배치가 다르다.
같은 `Linear(LN(x))` 결과를 MLP feature와 Q/K/V로 나눈다. 두 branch를 계산하고,
Control은 **MLP branch에 더한 뒤**, global condition의 shift/scale/gate를 적용한다.
Cross-attention을 끈 현재 FiCA 경로를 간추리면:

```text
m, q, k, v = split(Linear(LN(x)))
a = attention_output(SDPA(normalize(q), normalize(k), v))
m = mlp_output(GELU(m)) + control_i
x_next = x + gate_a * modulate(a) + gate_m * modulate(m)
```

`cond_adaln`은 attention과 MLP 각각에 쓸 shift/scale/gate, 총 6D를 만든다.
원본 `DiT.initialize_weights`는 gate를 0으로 초기화한다. 따라서 첫 block 출력이
입력 x와 같은 현상과, gate update 뒤 control이 출력에 영향을 주는 현상을 확인한다.
Block class 자체의 생성자만 호출하는 것과 `DiT`의 초기화까지 거치는 것은 구별해야 한다.

원본 첫 줄에서 멈춰 직접 진행하려면:

```bash
CUDA_VISIBLE_DEVICES=0 python pippo_walkthrough.py --part block --break
```

`p x.shape`, `p control.shape`로 값을 확인하고 `n`으로 다음 줄, `s`로 함수 내부,
`c`로 계속 진행한다. 여기서 실제 `external/pippo/latent_diffusion/models/dit.py`를 읽는다.
Hook은 `in_proj`, Q normalization, 6D modulation의 shape를 출력하며 연산을 바꾸지 않는다.

### 3. 원본 전체 흐름을 FiCA 02에 연결하기

`Pippo.forward`에서는 이미지를 VAE로 encoding하고 noise를 섞는다. `get_conds`가
reference/camera/spatial 조건과 controls를 구성하고, `predict`가 generator를 호출한다.
원본의 multi-view reference token 결합과 attention biasing은 이 전체 문맥에서 읽는다.

| 항목 | 원본 Pippo 공개 설정/코드 | 현재 FiCA lesson 02 |
|---|---|---|
| 생성 대상 | 여러 시점의 RGB image latent | canonical texture/geometry latent |
| Reference | reference image tokens 및 camera conditioning | CLIP embedding + texture/geometry domain embedding |
| 공간 조건 | spatial anchor, camera/Plücker 관련 경로 | partial RGB/mask/normal/XYZ의 VAE latent |
| 공통으로 직접 사용하는 부분 | DiTBlock, ControlMLP, timestep embedding | 동일한 원본 class |
| Token/모델 연결 | 원본 `DiT`, `Pippo`가 수행 | lesson의 `UVFlowModel`이 구성 |
| 학습 target | 공개 config의 eps + cosine noise schedule | FiCA에 맞춘 linear flow velocity `target-noise` |
| 세부 설정 | full/512 config의 width1536, depth28, heads24 | width1792/depth28/heads28; FiCA 미공개 width의 구현 선택 |

Pippo의 scheduler에서 말하는 eps/v와 이 lesson의 flow velocity를 같은 정의로 취급하지
않는다. Architecture를 가져오는 것과 noise schedule/loss까지 그대로 가져오는 것은 다르다.
특히 FiCA 전용 CLIP/domain 연결과 latent packing은 우리가 구성한 adapter다.

아래는 이전 `smoke.py`가 만든 **실제 partial UV latent와 학습된 02 smoke checkpoint**로
본편의 호출을 따라간다. 작은 합성 진단에서 실제 lesson으로 돌아오는 단계다.

```bash
CUDA_VISIBLE_DEVICES=0 python pippo_walkthrough.py --part fica
```

이 호출에서는 전체 checkout의 class를 우선 import하고, 본편 cache와 source hash가
일치하는지 확인한다. 기존 checkpoint의 parameter 이름/값과 원본 class가 호환되는지,
partial control이 velocity 출력에 영향을 주는지 검사한다. VAE/CLIP은 다시 다운로드하거나
학습하지 않는다. Cache/checkpoint가 없다면 앞서 `smoke.py`를 실행해야 한다.

2026-10-18, physical GPU 0에서 네 part를 모두 실행해 통과했다. 결과는
`.cache/pippo_walkthrough/all.json`, 출력은 `.cache/pippo_walkthrough.log`에 있다.
이 검증은 원본 전체 Pippo의 대규모 학습이나 FiCA 품질 검증을 의미하지 않는다.

영상·가중치·캐시·출력은 `.gitignore`로 제외한다. 루트 README와 기존 ELITE lesson은 수정하지 않는다.
논문/기반 코드 링크와 구현 귀속은 `NOTICE.md`에 있다.
