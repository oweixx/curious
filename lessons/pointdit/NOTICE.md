# 출처와 구현 범위

논문: [PointDiT, ICML 2026](https://arxiv.org/abs/2607.02515).
공식 repository: [google-research/pointdit](https://github.com/google-research/pointdit).
확인한 revision은 `11f53a30aeb4d34d49d2beda655ee6f19b2ad7f8`이다.
원본 file SHA256는 [upstream_sources.json](upstream_sources.json)에 기록했다.

## 직접 가져온 코드

| 로컬 파일 | 역할 / 수정 |
|---|---|
| `upstream/model.py` | 원본 PointDiT 전체 backbone. 아래의 명시적 수정 적용 |
| `upstream/denoiser.py` | 원본 학습 loss/추론. Import만 상대경로로 변경. 수식 parity test 기준 |
| `upstream/loss.py` | Relative point loss, 원본 그대로 |
| `upstream/util/model_util.py` | RMSNorm, position embedding, rotate-half, 원본 그대로 |
| `upstream/util/paths.py` | 원본 path helper, 그대로 |
| `upstream/LICENSE`, `upstream/THIRD_PARTY_NOTICES` | Apache-2.0 및 원본의 제3자 notice 보존 |

`model.py` 수정 내용:

1. `util.*` import를 local package의 상대 import로 변경.
2. `image_encoder` constructor argument 추가. 공식 DINOv3 model이나 명시적인 RGB ablation encoder를 주입한다.
3. Pretrained DINO의 Linear weights가 PointDiT 초기화에 의해 덮이지 않도록 encoder를 초기화 대상에서 제외.
   원본 `main.py`도 같은 문제 때문에 PointDiT 생성 **후** DINO weights를 다시 로드한다.
   Local 방식은 처음부터 보존하며, 난수 소비 순서가 원본 초기화와 bitwise 같다는 뜻은 아니다.
4. `rgb_linear` conditioning branch를 추가. Paper Table 3의 별도 ablation을 위한 경로이며 기본 DINO 경로를 대체하지 않는다.

Transformer attention/MLP, RoPE, adaLN modulation, bottleneck patch embedding, clean head,
point/image channel fusion, cached-feature forward의 수식은 원본을 유지한다.
Local subclass는 frozen DINO가 `train()` 호출로 training mode가 되지 않도록 `eval()`을 유지한다.

원본 전체 research checkout은 `.cache/upstream`에서 확인할 수 있다(현재 작업 환경의 ignored reference copy).
새 clone에서도 위의 필요한 model 파일은 Git에 포함되어 있으며, 전체 cache가 없어도 lesson이 동작한다.

## 논문과 코드의 차이

다음은 오류라고 임의로 수정하거나 숨기지 않고 둘을 구분했다.

- Paper Eq.(5)의 GT velocity는 `x-epsilon`. Released `denoiser.py`는 `(x-z)/clamp(1-t,0.05)`.
- Paper의 squared L2는 XYZ 합, code의 tensor MSE는 XYZ 평균. Paper 비교 모드의 M은 sample별 valid pixel 수로 해석한다.
- Paper는 pixel별 sky weight를 둔 weighted reduction을 서술. Code는 non-sky/sky 각각의 mean을 계산해 더함.
- Paper Eq.(6)에는 sky weight가 있음. Released `engine.py`의 relative loss는 valid 전체 mean.
- Paper는 noise 시작을 설명하고 zero 시작도 분석함. Released demo의 기본 `generate_noise_scale`은 0.
- 공식 pretrained evaluation은 존재하는 경우 `model_ema1` 사용. Local 04도 EMA1을 기본으로 선택.

02는 `--recipe released`를 기본으로 하고 `--recipe paper`를 비교용으로 제공한다.
지정된 loss reduction과 output parameterization은 checkpoint에 저장한다.
전처리 normalization은 논문처럼 valid non-sky의 mean distance를 사용한다.
12개 dataset별 depth parser, 대규모 mixed sampling, color augmentation, distributed epoch schedule,
공식 7개 benchmark metric 전체는 작은 lesson에서 다시 복제하지 않았다.

## 공식 데이터·학습으로 확장할 때

- [Training / main.py](https://github.com/google-research/pointdit/blob/11f53a30aeb4d34d49d2beda655ee6f19b2ad7f8/main.py)
- [Optimizer loop / engine.py](https://github.com/google-research/pointdit/blob/11f53a30aeb4d34d49d2beda655ee6f19b2ad7f8/engine.py)
- [Stage 1 B launch](https://github.com/google-research/pointdit/blob/11f53a30aeb4d34d49d2beda655ee6f19b2ad7f8/scripts/train_stage1_256_b.sh)
- [Stage 2 B launch](https://github.com/google-research/pointdit/blob/11f53a30aeb4d34d49d2beda655ee6f19b2ad7f8/scripts/train_stage2_512_b.sh)
- [Data preparation](https://github.com/google-research/pointdit/blob/11f53a30aeb4d34d49d2beda655ee6f19b2ad7f8/DATASETS.md)
- [11-dataset mixture](https://github.com/google-research/pointdit/blob/11f53a30aeb4d34d49d2beda655ee6f19b2ad7f8/dataloader/configs/res512mix.yaml)

Small practice data는 analytical sphere/plane ray tracing으로 만들며 SceneNet-RGBD라고 부르지 않는다.
`mini/smoke`는 학습 실행 시간을 줄인 명시적인 축소 구성이다. 원본 B checkpoint는 `B` profile에서만 로드한다.
00~03의 MSE는 normalized frame의 직접 오차이고 공식 affine alignment / BF1 평가 결과가 아니다.
04는 모든 모델 tensor key를 검사하지만 image resize/padding 방식은 GUIDE에 설명한 local 변경이 있다.

## DINOv3

[공식 source](https://github.com/facebookresearch/dinov3)의 revision
`6876159a11b4df116f30f667f8c9888617df0751`을 ignored cache로 준비한다.
DINOv3 code/weights는 해당 DINOv3 License Agreement를 따르며 이 repository의 Git에 재배포하지 않는다.

`assets.py`는 승인된
[Meta HF ViT-B/16](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)의 safetensors를
원본 Meta model의 state-dict 형식으로 변환한다. 변환 규칙은
[Hugging Face의 공식 converter](https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py)의
역방향으로 구현했다. 학습 tensor 전체와 shape를 검사하고, 고정 K-bias mask/frequency는 원본 constructor의 값을 유지한다.
실제 HF revision·가중치 SHA256는 다운로드 시 local JSON에 기록한다.

원본 Meta `.pth`를 직접 지정하는 경우에는 변환 없이 strict load한다.
승인 없이 다른 mirror에서 weights를 가져오거나 random encoder를 pretrained로 취급하는 동작은 없다.
