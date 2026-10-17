# FiCA — Figure 2의 다섯 이정표

목표는 **한 장의 사진 → partial UV → 생성된 textured mesh → 보정된 mesh →
움직이는 3D Gaussian avatar**의 각 경계를 직접 읽고 학습하는 것이다.
실험 자료는 NeRSemble이며, 원본 Meta 데이터/모델 성능의 동일 재현을 주장하지 않는다.

| Part | 파일 | 이번에 확인할 질문 | 먼저 읽을 함수 |
|---|---|---|---|
| 00 | `00_prepare_nersemble.py` | 이미지·camera·canonical mesh의 대응이 맞는가? | `prepare_dataset`, `rasterize_labels`, `MeshBridge` |
| 01 | `01_partial_uv.py` | RGB 한 장의 픽셀을 어디의 UV texel로 옮기는가? | `DensePredictor`, `prediction_loss`, `compose_predictions` |
| 02 | `02_generate_uv.py` | 불완전한 관측에서 texture/geometry 전체를 어떻게 생성하는가? | `FrozenConditioners`, `UVFlowModel`, `flow_loss`, `sample_domains` |
| 03 | `03_refine_uv.py` | 이미지의 오차를 어떤 UV 수정으로 연결하는가? | `RefinementNet`, `feature_pair`, `refinement_loss` |
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
| UV refinement | 실제 Diffusers conditional U-Net, reference/render의 실제 Sapiens feature, cross-attention, Eq.2의 loss 종류와 2/.5/.01/1 가중치 | U-Net 채널은 공개 기본값, residual 출력, reference/render type embedding, loss reduction |
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
CUDA_VISIBLE_DEVICES=0 python 01_partial_uv.py --stage predict
```

세 task는 별도의 실제 Sapiens-1B encoder를 fine-tune한다. UV/XYZ는 L1, normal은 cosine
loss다. Encoder LR을 head보다 작게 사용하고, dense labels에 같은 crop/scale을 적용한다.
RGB에는 photometric augmentation을 적용한다. Identity를 먼저 균등 sample하고 view angle
기반 importance weight로 frame/view를 고른다. Weight 식은 공개되지 않아 구현 선택으로 기록한다.

`outputs/01/*.jpg`에서 GT-correspondence로 만든 UV와 **Network가 예측한 대응**으로 만든
UV를 비교한다. 전자는 디버깅 기준이다. 예측이 틀려도 다음 단계에 GT를 몰래 넣지 않는다.
이미지→UV는 forward mapping이므로 `scatter_add`로 bilinear splat한다. 겹치는 관측의
가중치와 빈 texel을 함께 저장한다. 학습 완료 시의 전체 자료 예측은 `data/prepared/predictions/`,
조합된 10ch partial UV는 `partial_pred/`에 저장된다.

## 02 — Noise에서 UV와 mesh가 만들어지는지 보기

```bash
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage cache
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage train
CUDA_VISIBLE_DEVICES=0 python 02_generate_uv.py --stage sample
```

`cache`는 실제 SDXL VAE/CLIP을 사용한다. Partial의 RGB/mask/normal/XYZ를 각각 VAE에 넣고
latent를 이어 ControlMLP에 전달한다. `--partial-source gt_correspondence`는 별도 oracle
ablation이며 기본값은 항상 `predicted`다. 어떤 경로인지 cache와 학습 log에 남는다.

학습식은 x_t=(1−t)·noise+t·target, v*=target−noise다. Texture/geometry는 같은 Network를
domain=0/1로 두 번 평가한다. CLIP+time+domain은 전역 modulation, partial UV는 위치별
ControlMLP로 전달한다. Sampling은 noise에서 data 방향으로 Euler 적분한다.

`outputs/02/*.jpg`는 입력 사진 / 생성 mesh render / 생성 texture / XYZ 표시다.
`generated/*.pt`에는 실제 float UV와 sampling 중간 latent가 있다. XYZ의 RGB 표시는
시각화일 뿐이며 표시 이미지를 mesh 좌표로 사용하지 않는다. VAE 재구성 자체의 geometry
오차도 별도로 검사해야 한다. 원본 모델처럼 unseen identity를 생성하려면 여러 identity의
학습이 필요하고, 한 사람 overfit만으로 이 능력을 주장할 수 없다.

## 03 — 보정이 texture와 geometry를 어떻게 바꾸는지 보기

```bash
CUDA_VISIBLE_DEVICES=0 python 03_refine_uv.py --stage cache
CUDA_VISIBLE_DEVICES=0 python 03_refine_uv.py --stage train
CUDA_VISIBLE_DEVICES=0 python 03_refine_uv.py --stage predict
```

초기 생성 UV를 render하고 reference/render의 frozen Sapiens features를 cache한다.
U-Net의 UV queries가 두 이미지의 tokens를 cross-attend한다. 시간 입력은 0이며 별도의
diffusion sampling을 하지 않는다. 출력은 texture/XYZ residual이다.

Image L1, alpha mask, pixel 단위 2D landmark L1과 초기 texture·cotangent Laplacian matrix·
vertex normal의 L2 regularization을 사용한다. Loss reduction/정규화는 구현 선택이며
정확한 저자 implementation과 동일하다고 주장하지 않는다. Geometry의 Cholesky solve,
posing, CUDA rasterizer까지 gradient가 보존된다. 생성 UV가 바뀌면 기존 feature cache는
hash mismatch로 거부한다. 먼저 `cache`를 다시 실행한다.

Laplacian은 geometry-dependent cotangent matrix의 각 row를 L1 정규화한 값이다.
초기 생성 mesh의 가느다란 삼각형에서 cotangent가 커져 loss를 지배하는 현상을 줄인다.
행렬은 여전히 현재 geometry에 따라 변하고 gradient가 흐른다. 정확한 Laplacian 정의와
reduction은 FiCA에 공개되지 않았으므로 이 정규화를 저자 설정으로 주장하지 않는다.

`outputs/03/preview/100.jpg` 등의 비교와 각 loss 항을 함께 읽는다. 최종 UV는 `refined/`다.
Photometric loss 감소만으로 geometry 개선을 판단하지 않는다.

## 04 — Learned UPM이 실제 3DGS를 생성하는지 보기

```bash
CUDA_VISIBLE_DEVICES=0 python 04_gaussian_avatar.py --stage train
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
Sapiens 세 task는 각각 2 step, flow는 4 step, refiner/UPM은 각각 2 step 학습한다.
DiT/refiner/UPM의 smoke 크기와 이미지 해상도만 줄인다. 본 설정 생성 모델은 약 1.97B,
refiner는 약 879M, UPM은 약 22.4M이다. 다음 명령은 축소하지 않은 모델의 실제
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

영상·가중치·캐시·출력은 `.gitignore`로 제외한다. 루트 README와 기존 ELITE lesson은 수정하지 않는다.
논문/기반 코드 링크와 구현 귀속은 `NOTICE.md`에 있다.
