# PointDiT — Flow Matching을 RGB-conditioned geometry estimation으로 옮기기

목표는 **RGB → image feature → point-map tokens → clean XYZ prediction → velocity loss → Euler reconstruction**을
하나의 계산으로 이해하는 것이다. 앞의 [DiT](../done/dit/GUIDE.md), [Flow Matching](../flow_matching/GUIDE.md)과 연결된다.

기준 논문은 [PointDiT: Pixel-Space Diffusion for Monocular Geometry Estimation](https://arxiv.org/abs/2607.02515),
기준 구현은 [Google Research 공식 저장소](https://github.com/google-research/pointdit)다.
**학습 코드·추론 코드·pretrained checkpoint가 모두 공개되어 있다.** DINOv3 가중치는 Meta의 별도 접근 승인이 필요하다.
출처·수정 사항·논문과 코드의 차이는 [NOTICE.md](NOTICE.md)에 기록했다.

## 다섯 단계

| 단계 | 집중할 질문 | 눈으로 확인할 결과 |
|---|---|---|
| [00 Point maps](00_pointmaps.py) | RGB pixel에 XYZ를 담는다는 뜻은? 정규화하면 무엇을 잃는가? | RGB/XYZ/Z, 3D cloud, PLY, scale/shift 확인 |
| [01 Tokens and model](01_tokens_and_model.py) | PointDiT가 원 DiT에 무엇을 연결했나? | patch grid, feature PCA, token shape와 fusion |
| [02 Training](02_train_pointdit.py) | x-prediction인데 왜 v-loss를 쓰나? | time/loss 곡선, 실제 학습, 고정 validation PNG |
| [03 Sampling and ablation](03_sampling_and_ablation.py) | 1-step이 어떻게 가능한가? 더 적분하면 무엇이 바뀌나? | 1/2/4/8-step, zero/random 시작, GIF, 3D 결과 |
| [04 Official pretrained](04_pretrained_and_geometry.py) | 공식 모델로 실제 RGB에서 geometry를 얻으려면? | 공식 B 모델의 XYZ·PLY·step 비교·생성 GIF |

`common.py`는 입출력·그림 helper, `assets.py`는 자산 준비, `smoke.py`는 검증 도구다.
새 lesson 단계가 아니며, 핵심 수식과 학습 loop는 번호 파일에 있다.

## 무엇이 contribution이고 무엇이 기존 아이디어인가?

이 연구를 새로운 Transformer block 하나를 발명한 논문으로 읽으면 핵심을 놓친다.
원본 `upstream/model.py`에는 JiT/SiT 계열 구조와 RMSNorm, RoPE, SwiGLU, adaLN 계열 modulation이 보인다.
Flow Matching, DINOv3, image patchification, x-prediction도 각각 선행 연구에서 온 구성 요소다.

가져갈 부분은 **이미지 생성에서 유효했던 설계를 geometric signal에 적용하는 선택과 검증**이다.

| 선택 | 해결하려는 문제 | 코드에서 볼 위치 |
|---|---|---|
| Raw point map 공간에서 학습 | geometry VAE의 압축·복원 오차를 생성 모델의 상한으로 만들지 않음 | `00.backproject`, `01.build_model` |
| Clean point map을 직접 예측 | 큰 raw patch에서 noise/velocity를 직접 예측하는 방식과 다른 inductive bias | `upstream/model.py: FinalLayer`, `02.pointdit_loss` |
| Frozen DINOv3의 4개 intermediate layer | RGB에서 얻은 표현을 공간적으로 정렬된 geometry 조건으로 제공 | `01.condition_tokens`, `extract_y_embedding` |
| Channel concatenation + linear fusion | 복잡한 multi-scale convolution head 없이 image/point 정보를 결합 | `concat_proj`, `PointDiT.forward` |
| Geometry normalization·sky·t=0 학습 | scene scale 차이와 infinite sky, 시작 시간의 train/test 차이를 다룸 | `00.normalize_points`, `02.sample_time` |
| 동일한 모델의 1/few-step 추론 | 강한 image conditioning에서 효율과 refinement의 관계를 검증 | `03.sample` |

논문이 제시하는 sharp geometry·transparent region 개선은 대규모 학습과 benchmark의 결과다.
작은 연습 dataset에서 loss가 줄었다는 사실만으로 그 주장을 재현했다고 말할 수는 없다.
반대로, 어느 선택이 어떤 tensor와 loss를 바꾸는지는 작은 실험에서도 정확히 확인할 수 있다.

## 시작하기

기존 `elite` Conda 환경을 사용한다. venv를 만들거나 환경을 자동으로 변경하지 않는다.
기본 의존성은 PyTorch, NumPy, matplotlib, Pillow, tqdm, einops다.
다운로드에는 huggingface_hub/safetensors를 사용한다. 최신 transformers 설치는 필요하지 않다.
FlashAttention extension 없이 PyTorch SDPA로 실행한다.

아래 명령은 repo root `/media/vilab/mh/curious` 기준이다.
예시의 GPU 1은 검증 당시 비어 있던 GPU다. 실행할 때 비어 있는 번호로 바꿀 수 있다.

```bash
# 00: 라이선스 자산 없이 시작. 작은 연습 RGB-D dataset도 생성한다.
python lessons/pointdit/00_pointmaps.py

# 02의 수식만 먼저 보고 싶다면: 네트워크/가중치 없이 실행 가능
python lessons/pointdit/02_train_pointdit.py --inspect-objective

# DINOv3 공식 source와 승인된 공식 HF weights 준비
python lessons/pointdit/assets.py source
python lessons/pointdit/assets.py dinov3

# 01~03: frozen DINOv3 조건 + 작은 PointDiT를 실제 학습
CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/01_tokens_and_model.py --device cuda
CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/02_train_pointdit.py --device cuda
CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/03_sampling_and_ablation.py --device cuda

# 04: 공식 pretrained PointDiT-B로 별도 추론
python lessons/pointdit/assets.py pointdit --resolution 512
CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/04_pretrained_and_geometry.py --device cuda
```

Lesson 폴더에서는 `python 00_pointmaps.py`처럼 파일명만 사용한다.
기본 경로는 파일 위치 기준이며, 직접 지정한 `--out`, `--data`, `--checkpoint`는 현재 작업 폴더 기준이다.
`data/`, `outputs/`, `weights/`, `.cache/`는 Git에서 ignore된다.
Root README와 다른 lesson 파일은 이 작업에서 수정하지 않는다.

**DINOv3 접근 준비:** [Meta의 ViT-B/16 HF 모델](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)에
접근 승인을 받은 계정으로 서버에 HF login이 되어 있어야 한다. 동의 제출과 작성자의 승인은 별개일 수 있다.
`assets.py dinov3`는 기존 token을 사용한다. Token을 코드나 명령행에 붙여 넣지 않는다.

이미 Meta 원본 `.pth`를 가지고 있다면 01/02/04에 다음처럼 지정할 수 있다.

```bash
CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/01_tokens_and_model.py --device cuda \
    --dino-weights /absolute/path/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

HF 파일은 transformers 형식이므로 `assets.py`가 이름·QKV packing을 Meta 형식으로 변환하고
**모든 tensor shape와 state-dict key를 검사**한다. 원본 DINOv3 code로 실행하며 가중치를 재학습하지 않는다.
승인된 weights가 없을 때 random DINO로 조용히 대체하는 경로는 없다.

가중치 없이도 계산과 학습을 살펴볼 때는 **별도의 linear RGB-conditioning ablation**을 명시한다.

```bash
CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/01_tokens_and_model.py \
    --encoder rgb --device cuda --out lessons/pointdit/outputs/01_rgb
CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/02_train_pointdit.py \
    --encoder rgb --device cuda --out lessons/pointdit/outputs/02_rgb
CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/03_sampling_and_ablation.py \
    --checkpoint lessons/pointdit/outputs/02_rgb/best.pt --device cuda \
    --out lessons/pointdit/outputs/03_rgb
```

이 모드의 RGB patch projection은 **학습된다**. DINO의 semantic feature를 사용하는 실험과 구분한다.
공식 코드에는 현재 DINO branch만 공개되어 있어, 논문 Table 3의 linear-conditioning 비교를 위한
작은 branch를 lesson에 추가했다. 이 branch를 논문 checkpoint에 연결하지 않는다.

## 00 — XYZ map은 canonical UV map과 무엇이 다른가?

| 표현 | 2D 인덱스의 의미 | 저장한 값 |
|---|---|---|
| RGB image | 현재 camera의 pixel | RGB |
| PointDiT point map | 같은 camera의 같은 pixel | 해당 pixel에서 보이는 surface의 XYZ |
| Canonical UV position map | template surface의 UV texel | template와 대응하는 surface XYZ |

둘 다 3채널 tensor일 수 있지만 대응 관계가 다르다. PointDiT는 FLAME topology나 고정 semantic UV를 사용하지 않는다.
다른 시점의 image에서 같은 pixel index가 같은 얼굴 vertex라는 보장도 없다.
보이는 표면의 point map이므로 image 한 장에서 뒤통수까지 완성된 avatar mesh를 출력하는 모델도 아니다.

GT를 만들 때는 RGB-D와 camera intrinsics를 사용한다. Depth가 camera Z라면:

```python
ray = inverse(K) @ [u, v, 1]
XYZ = depth_z * ray / ray.z
```

`K`는 pixel 단위이며 `u,v`는 정수 pixel-center 좌표다.
Depth sensor의 ray-distance와 camera Z-depth는 다르다. 저장된 depth의 정의를 확인해야 한다.
Crop/resize를 한다면 RGB와 depth를 함께 변환하고 focal/principal point도 갱신한다.
00의 기본 연습 dataset은 처음부터 같은 camera ray에서 RGB와 Z-depth를 생성하므로 이 대응이 정확하다.

정규화는 valid non-sky points에 대해 다음과 같다.

```python
mu = points.mean(axis=points_axis)
s = mean(norm(points - mu))
normalized = (points - mu) / s
```

색 범위를 [0,1]로 맞추는 min-max normalization이 아니다.
Noise의 scale과 scene geometry의 scale을 비슷하게 만들어 Flow Matching 경로를 학습하기 쉽게 한다.
결과는 **positive uniform scale과 translation에 대해 불변**이다. 일반적인 모든 affine transform이나 rotation을 제거하는 식은 아니다.

Normalized Z는 음수일 수도 있다. Centroid를 빼었기 때문이다.
Inference에서 GT의 `mu,s`를 가져와 metric scale을 복구하면 정답 정보를 사용한 것이 된다.
04에서는 그 값을 읽지 않고, 상대 좌표의 point cloud를 그대로 저장한다.

Sky는 centroid/scale 계산에서 제외한 뒤 원 camera ray 방향으로 radius 3의 proxy sphere에 놓는다.
Stage-2 모델에서는 norm > 2.9인 예측을 sky 후보로 제거한다. 이 threshold를 모든 모델의 진짜 물리적 거리로 해석하면 안 된다.
Invalid depth는 sky와 별개다. Loss에서 invalid를 제외하고, noisy input의 빈 부분에는 noise를 채운다.

생각해볼 것: 한 사람의 point map 두 장을 각각 정규화했다고 해서 두 얼굴이 동일한 canonical 좌표계로 정렬된 것일까?

## 01 — image token과 point token이 어디에서 만나는가?

공식 B 모델의 256×256 입력을 기준으로 계산을 따라가자.

```text
RGB [B,3,256,256]
  -> frozen DINOv3-B blocks 3,6,9,12 (1-based)
  -> 각 [B,256,768] -> channel concat -> [B,256,3072]

noisy XYZ [B,3,256,256]
  -> 16×16 patches, 3*16*16 = 768 values per patch
  -> BottleneckPatchEmbed: 768 -> 128 -> 768
  -> [B,256,768]

image tokens + point tokens, 같은 pixel patch끼리 channel concat
  -> [B,256,3840] -> learned Linear -> [B,256,768]
  -> 12 Transformer blocks, time modulation
  -> learned Linear: 768 -> 3*16*16
  -> fixed unpatchify -> clean XYZ [B,3,256,256]
```

Concat해도 token 수 N은 그대로다. 두 sequence를 이어 붙여 N을 두 배로 만들거나
별도 cross-attention layer를 넣는 연산이 아니다.
`BottleneckPatchEmbed`의 `pca_dim`이라는 upstream 변수명도 실제 사전 PCA를 수행한다는 뜻이 아니다.
두 convolution으로 구현한 학습 가능한 linear projection이며, kernel/stride가 patchification을 수행한다.
이 projection에도 정보 손실은 가능하다. VAE를 없앴다는 말은 모든 압축을 없앴다는 뜻이 아니다.
별도로 학습한 geometry tokenizer/decoder의 복원 오차를 거치지 않고 raw XYZ에 직접 loss를 건다는 차이다.

Transformer block 안에서는 q/k RMS normalization, 2D RoPE, self-attention, SwiGLU가 쓰인다.
시간 조건은 adaLN 계열 scale/shift/gate로 들어간다. 원 DiT lesson처럼 초기 modulation/output은 zero initialization이다.
이 공개 구현은 `t∈[0,1]`을 timestep embedding에 그대로 넣는다. 앞 FM lesson의 `1000*t` scaling을 가져와 덧붙이지 않았다.
01의 초기 XYZ 출력이 0이라고 모델이 작동하지 않는 것은 아니다. 아직 geometry 학습 전이다.
`feature_pca.png`의 색 역시 feature를 표시한 것으로, XYZ 예측이나 segmentation 정답이 아니다.

**DINO freeze와 cache:** DINO parameter를 freeze하고 eval mode를 유지한다.
RGB가 Euler step마다 변하지 않으므로 DINO feature와 image position embedding을 한 번 계산해 재사용한다.
PointDiT에는 매 step의 noisy point map과 t가 바뀌어 들어간다.

| profile | PointDiT width/depth/heads | image encoder | 용도 |
|---|---|---|---|
| `B` | 768 / 12 / 12, bottleneck 128 | DINOv3-B | 공식 B backbone, resolution은 입력에 맞게 지정 |
| `mini` | 192 / 4 / 6, bottleneck 128 | DINOv3-B 유지 | 작은 학습 실험; 논문의 B라고 부르지 않음 |
| `smoke` | 64 / 2 / 4, bottleneck 32 | 선택한 encoder | 빠른 실행 검사 |

모든 profile의 patch size는 원본과 같은 16이다.
Mini에서는 DINO feature width는 유지되어 concat dimension이 192+3072다. 원본의 `5D`와 숫자가 다르다.
`--encoder rgb`를 명시하면 위 image encoder가 trainable linear patch encoder로 바뀐다.

## 02 — x-prediction, v-loss, relative point loss

앞 Flow Matching lesson은 network가 velocity를 직접 출력했다. PointDiT는:

```python
z_t = t * clean_xyz + (1-t) * noise
clean_prediction = model(z_t, t, rgb_condition)
v_prediction = (clean_prediction - z_t) / max(1-t, 0.05)
```

`prediction`의 의미와 loss를 계산하는 공간을 분리하자.
Clean XYZ를 예측해도 그 출력으로 velocity를 만들고 velocity끼리 비교할 수 있다.
Released 코드의 GT에도 동일한 denominator를 쓰므로 다음이 성립한다.

```text
v_pred - v_target = (x_pred - x_target) / max(1-t,0.05)
v-MSE = x-MSE / max(1-t,0.05)^2
```

단순한 unweighted XYZ MSE와는 timestep weighting이 다르다.
`--inspect-objective`의 `objective.png`에서 이를 확인한다.
Network의 parameterization까지 x-output으로 정하는 것과, 이미 학습된 v-output의 이름만 바꾸는 것은 다르다.

시간은 uniform 대신 `sigmoid(N(-0.8,0.8²))`로 뽑고, 10%를 정확히 t=0으로 바꾼다.
연속 sigmoid sampling만으로는 정확한 t=0이 나오지 않지만 inference는 거기서 시작한다는 관찰이다.
Noise amplitude는 Stage 1에서 1, 512px Stage 2에서는 2다. Lesson 기본 연습 학습은 1을 사용한다.

Relative point loss는 clean map에서 직접 계산한다.

```text
L_rel(pixel) = ||predicted_XYZ - target_XYZ||_1 / (||target_XYZ||_2 + 1e-6)
L_total = L_velocity + 0.1 * L_relative
```

원점에 가까운 point의 상대 오차를 강조한다. Metric world-space의 camera 거리라고 해석하지 말고
현재 loss에 들어간 normalized coordinates의 원점이라는 점을 확인하자.

**Paper와 released code를 같다고 뭉뚱그리지 않는다.**

| 항목 | `--recipe released` (기본) | `--recipe paper` |
|---|---|---|
| GT velocity | `(x-z)/clamp(1-t)` | `x-noise` |
| XYZ channel reduction | mean | squared L2 norm의 sum |
| Sky velocity reduction | non-sky mean + 0.01×sky mean | valid pixel 수로 나눈 weighted sum |
| Relative loss의 sky | valid에 포함, 별도 sky downweight 없음 | 같은 sky weight 0.01 적용 |

따라서 t>0.95, sky 비율이 달라지는 batch 등에서는 값이 달라진다.
`paper` 모드의 M은 sample별 valid pixel 수로 해석하고, sample 평균 후 batch 평균을 계산한다.
기본 loss를 공개 `denoiser.py`와 수치 비교하는 검사가 `smoke.py`에 있다.
`upstream/loss.py`의 relative loss 함수는 그대로 사용한다.
좌표 정규화는 논문에 따라 valid non-sky 통계로 고정했다. 공개 loader의 다양한 선택 flag 전체를 흉내 내지는 않는다.

02는 실제 optimizer update를 수행한다. `mini` 기본값은 2,000 steps / batch4 / lr 2e-4 / warmup100 / EMA0.99다.
작은 연습용 budget이며 원 논문의 16~64 GPU, 수백만 RGB-D 데이터 학습 schedule과 다르다.
공식 학습은 AdamW betas=(0.9,0.95), EMA1=0.9999 등으로 동작하며 원 launch script는 NOTICE에서 연결한다.

기본 100 step마다 validation의 첫 4개 고정 scenes를 **GT 없이 zero-start inference**하고 `0.png,100.png,...`에 저장한다.
GT는 loss/비교 그림에만 사용한다. `best.pt`는 EMA의 heldout normalized XYZ MSE 기준이다.
이 값은 논문의 affine-aligned benchmark metric이 아니다.
`latest.pt`에는 optimizer/EMA/RNG를 함께 저장한다. `--steps`는 재개 시에도 누적 목표 step이다.

별도 비교 실험:

```bash
# 같은 data/profile로 출력 parameterization을 바꾼 독립 학습
CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/02_train_pointdit.py \
    --encoder rgb --device cuda --prediction velocity --out lessons/pointdit/outputs/02_velocity

# 논문 수식과 released reduction의 차이를 비교하는 독립 학습
CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/02_train_pointdit.py \
    --encoder rgb --device cuda --recipe paper --out lessons/pointdit/outputs/02_paper_loss
```

## 03 — 1-step은 왜 clean prediction과 같은가?

기본 inference는 공개 코드처럼 zero tensor에서 시작한다. `initial_std=1`로 noise 시작도 비교한다.
Image condition을 dropout하는 CFG 학습은 여기 없다. 앞 FM lesson의 class CFG를 임의로 붙이지 않는다.

```python
x_hat = model(z, t, image_features)
velocity = (x_hat-z)/(1-t)
z = z + dt*velocity
```

1-step이면 t=0, dt=1이므로:

```text
z_1 = z_0 + (F(z_0,0,c)-z_0) = F(z_0,0,c)
```

추가 distillation network나 특별한 one-step solver가 숨어 있지 않다.
하지만 이 항등식만으로 좋은 geometry가 보장되지는 않는다. T=0에서 잘 작동하도록 학습한 조건부 모델이 필요하다.
Zero initialization은 Gaussian source에서 무작위로 sampling한 것과도 다르며, 논문에서 관찰·평가한 deterministic 추론 선택이다.

Released sampler의 denominator epsilon은 0이다. Euler는 t=1에서 field를 평가하지 않으므로 0으로 나누지 않는다.
Training의 0.05 clamp를 inference에 무조건 복사하지 않은 이유다.
More steps는 현재 state에 맞춰 clean map을 다시 예측한다. 학습 pair의 직선과 실제 marginal trajectory가 같다고 가정하지 않는다.

`comparison.png`는 동일 RGB/seed에 대해 step 수와 초기 noise만 바꾼다.
`pointcloud.png`와 `target_pointcloud.png`는 같은 validation 장면의 예측/GT를 3D에서 비교하는 그림이다.
`conditioning.png`의 blank RGB는 image condition에 민감한지 보는 **OOD 진단**이다.
이를 DINO 제거 ablation이나 정식 unconditional 생성 성능으로 읽으면 안 된다.

## 04 — 공식 weights로 무엇까지 복원할 수 있나?

04는 원본 B/16 architecture를 만들고 공식 checkpoint의 `model_ema1`을 기본으로 사용한다.
Checkpoint에서 제거된 DINO tensor만 missing으로 허용하며, 다른 missing/unexpected key는 오류로 처리한다.
`--raw-model`로 raw weights를 따로 비교할 수 있다.

RGB 전체를 aspect ratio를 유지해 resize하고 /16까지 최대 15px padding한다.
출력에서 padding을 제거한 후 원 image 크기로 돌린다. 공개 demo의 가까운 /16 크기로 직접 resize하는 방식과
표시 전처리가 조금 다르며, 이 단계 결과를 공식 benchmark 숫자라고 주장하지 않는다.

출력 `points.ply`는 normalized XYZ와 원 RGB 색이다. 완성된 watertight mesh나 rigged avatar가 아니다.
`prediction.npz`의 points도 metric meter 좌표가 아니다. 중심/scale이 사라진 상대적인 camera-oriented geometry다.
추가 intrinsic/depth fitting, cross-view alignment, topology 구성은 별도의 연구 문제다.

처음에는 제공된 연습 RGB로 경로를 확인하고, 이후 `--image /실제/사진.jpg`로 일반 장면을 시도하자.
실제 portrait에서도 출력의 의미는 **현재 image에서 보이는 surface의 상대 point map**이다.

## 실제 RGB-D 데이터로 바꾸려면

`--data` 폴더에 `train/*.npz`, `validation/*.npz`를 만든다.

| key | shape / 값 |
|---|---|
| `rgb` | `[H,W,3]`, uint8 또는 float [0,1] |
| `depth` | `[H,W]`, camera Z-depth |
| `K` | `[3,3]`, 해당 image 해상도의 pixel-unit intrinsics |
| `valid` | optional `[H,W]` bool, depth validity |
| `sky` | optional `[H,W]` bool; ray 방향 계산용 finite positive depth 필요 |

같은 batch는 같은 해상도이며 training crop은 정사각형, /16 크기다.
NeRSemble RGB 영상만으로 정답 XYZ가 생기지는 않는다. Calibrated multiview reconstruction이나 신뢰할 GT depth가 별도로 필요하다.
같은 sequence의 거의 동일한 frame을 train/validation에 섞으면 일반화를 과대평가할 수 있다.

## 마친 뒤 설명할 수 있어야 할 것

1. Image-aligned XYZ와 canonical UV의 대응 관계 차이.
2. Patchification이 VAE geometry compression과 다른 이유, bottleneck projection 자체는 학습된다는 점.
3. DINO 4개 layer를 channel로 합쳤을 때 token 수가 그대로인 이유.
4. Clean-output parameterization과 velocity-space loss의 차이.
5. 1-step identity가 성립하는 이유와, one-step 품질이 별도의 학습 결과인 이유.
6. Prediction에서 metric scale/translation을 알 수 없는 이유.

## 검증

```bash
CUDA_VISIBLE_DEVICES=1 python lessons/pointdit/smoke.py --device cuda
```

동일 결과 폴더가 있다면 새 `--out`을 지정하거나 `--reuse`로 기존 결과의 수식 검사를 실행한다.
DINO 접근 승인 후에는 `--with-dino --out .cache/smoke_dino`로 실제 frozen encoder 경로까지 검사할 수 있다.
기본 smoke는 RGB ablation으로 학습/추론하고, 이를 pretrained DINO 검증으로 기록하지 않는다.

검사 범위: camera projection, normalization 불변성, sky radius, rectangular patch roundtrip,
one-step 항등식, released loss 일치, invalid gradient mask, 원본 block의 gradient,
동일 seed sampling, inference 중 weight 불변성, 연속 8 steps와 4+4 resume 일치.

### 이 작업에서 실제 확인한 결과 (2026-11-15)

- 기존 `elite` 환경, PyTorch 2.5.1+cu124, 비어 있던 H200 GPU 1에서 검증했다.
- RGB ablation과 **실제 pretrained DINOv3** 경로 모두 학습/추론/재개 smoke를 통과했다.
- 공개 denoiser의 velocity loss와 local released loss의 차이는 0.
  실제 DINO 조건에서 공개 Euler sampler와 local sampler의 출력 최대 차이도 0.
- 8 steps 연속 학습과 4+4 resume의 model/EMA 최대 차이는 모두 0.
- 공식 B/256 EMA1 checkpoint의 non-DINO tensor 184개를 로드하고, 같은 cached features에서
  수정 전 upstream forward와 local forward의 최대 차이가 0임을 확인했다.
- 연습 dataset의 RGB mini 실험 400 steps에서 고정 validation EMA XYZ MSE가 0.3644 → 0.0560으로 감소했다.
  이는 작은 ablation의 동작 확인이며 공식 모델의 성능 수치가 아니다.
- **공식 B/512 EMA1 + 실제 DINOv3**로 upstream 건물 demo image의 XYZ/PLY/GIF/1·3·8-step 비교를 생성하고 그림을 확인했다.
  결과는 현재 작업 환경의 `.cache/official_demo/`에 있다. 공식 benchmark 재현은 수행하지 않았다.
