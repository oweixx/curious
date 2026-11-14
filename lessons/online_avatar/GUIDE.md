# Online Canonical Avatar

관측이 늘어날 때 **같은 사람의 canonical Avatar를 갱신하는 과정**을 직접 읽고 실험한다.
FLAME을 추적하고 실제 surface Gaussian을 렌더링하며, 그 rendering loss로 parameter를 갱신한다.
다섯 단계는 각각 데이터 계약, 상태 분리, 학습 대상, 온라인 갱신, 평가에 집중한다.

이 lesson은 특정 논문의 재현이 아니라 README Ideation의 연구 질문을 시험하기 위한 첫 구현이다.
작은 모델·한 사람으로 갱신 원리를 확인하고, 어떤 부분을 더 연구해야 하는지 드러내는 것이 목표다.
코드에서 구현한 것과 이후 확장할 것을 아래에서 구분한다.

## 5단계와 결과물

| 단계 | 읽을 핵심 | 실행 후 직접 볼 것 |
|---|---|---|
| [00_observation_stream.py](00_observation_stream.py) | 두 camera의 DLT, 입력/평가 분리, crop 후 K 변환 | `data/042_FREE/inspection.jpg`, `stream.json` |
| [01_canonical_and_dynamic.py](01_canonical_and_dynamic.py) | 첫 관측의 shape와 각 시각의 expression/pose를 다른 변수로 최적화 | `tracking/0.jpg ...`, `tracking/dynamics.png` |
| [02_avatar_and_updates.py](02_avatar_and_updates.py) | coordinate field, UV→surface Gaussian decode, representation와 adapter의 차이 | `initialization/0.jpg,100.jpg,...`, `canonical.jpg` |
| [03_online_adaptation.py](03_online_adaptation.py) | 현재 조명 fitting → canonical update → 과거 replay, 시간 제한 | 방법별 `2.jpg,3.jpg,...`, 학습된 `checkpoint.pt` |
| [04_evaluate_retention.py](04_evaluate_retention.py) | 평가 camera, 과거 시각을 고정한 retention, renderer/update 속도 구분 | `comparison.jpg`, `curves.png`, `canonical_drift.png`, 방법별 `progress.gif` |

`common.py`는 입출력·경로·그림만 공유한다. 학습 모델과 loss는 번호 파일에 있다.
`smoke.py`는 실제 데이터로 전체 경로와 gradient를 검사한다.
실행으로 생성되는 데이터·그림·checkpoint·cache는 `.gitignore`로 제외한다.

## 시작하는 방법

기존 **elite Conda 환경**을 사용한다. venv를 만들거나 package를 자동 설치하지 않는다.
필요한 기존 의존성은 PyTorch, NumPy, Pillow, matplotlib, tqdm, PyTorch3D,
nvdiffrast, diff_surfel_rasterization과 VDA에서 준비했던 pixel3DMM FLAME 의존성이다.
GPU 번호는 환경변수에서 선택하고, Python 안의 `cuda:0`은 그 중 첫 번째 GPU다.

repo root `/media/vilab/mh/curious`에서:

```bash
python lessons/online_avatar/00_observation_stream.py
CUDA_VISIBLE_DEVICES=7 python lessons/online_avatar/01_canonical_and_dynamic.py
CUDA_VISIBLE_DEVICES=7 python lessons/online_avatar/02_avatar_and_updates.py --steps 400
CUDA_VISIBLE_DEVICES=7 python lessons/online_avatar/03_online_adaptation.py --seconds 2
CUDA_VISIBLE_DEVICES=7 python lessons/online_avatar/04_evaluate_retention.py
```

각 파일 맨 위에도 해당 실행 명령과 읽는 순서가 있다. 기본 경로는 파일 위치 기준이므로
lesson 폴더에서 `python 00_observation_stream.py`처럼 실행해도 같다.
직접 지정하는 `--work`, `--out`, `--runs`는 현재 터미널 위치를 기준으로 해석한다.

각 단계를 읽으면서 하나씩 실행한다. 먼저 00에서 얼굴 crop을 보고, 01에서 녹색 검출점과
주황색 FLAME 점이 맞는지 확인한다. Tracking이 나쁘면 이후의 texture 최적화로 덮지 않는다.
기존 결과를 덮으려면 `--overwrite`를 명시하거나 다른 `--work`를 사용한다.
00/01/02를 바꿨다면 그 아래 단계도 새로 생성해야 한다. 파일 hash가 다른 실행의 결과를 섞지 못하게 한다.

기본 설정은 24개 시각을 사용한다. 시각마다 camera 두 개가 함께 들어오므로 RGB 48장이 입력이다.
`--stride 3`은 전처리된 24fps stream에서 3칸 간격이다. 원본 영상 73fps의 frame 번호와 다르다.
앞의 2시각은 warm-start prefix이고, 나머지 22시각을 하나씩 공개한다.
400 steps는 시작 설정이며 수렴이나 고품질을 보증하는 숫자가 아니다.

## 00 — 평가 정보를 어디까지 막아야 할까?

현재 repo의 실제 자료를 기본값으로 쓴다:

```text
lessons/vda/data/preprocessed/042_FREE/
  run.json
  frames/{222200037,222200042,222200036}/*.jpg
  cameras/{222200037,222200042,222200036}.npz
  landmarks/{222200037,222200042}.npz
```

`222200037`, `222200042`가 입력, `222200036`이 평가 camera다.
기존 NeRSemble 동기화 프레임과 camera별 검출 landmark를 재사용한다.
3개 camera를 모두 사용해 최적화했던 `face_world.npz`, 영상 전체의 shape를 공유한
`mono_flame.npz`, ELITE의 전체 영상 texture/학습된 checkpoint는 초기화에 쓰지 않는다.

평가 camera의 calibration을 아는 것은 novel-view render를 위해 허용한다.
평가 camera의 RGB나 2D landmark로 tracking·crop·조명·texture를 맞추는 것은 허용하지 않는다.
평가용 crop도 첫 입력 두 camera에서 삼각측량한 3D landmark를 그 camera에 투영해서 정한다.
입력의 첫 crop을 고정하고, **crop 후의 principal point와 resize scale을 K에도 반영**한다.

전체 RGB를 미리 디스크에 준비하는 것과 미래 관측으로 모델을 갱신하는 것은 다르다.
00은 시간표를 만들고, 01은 `0→1→2...`로 fitting하며, 02는 prefix만 읽고, 03은 현재·과거만 읽는다.
`Stream.rgb(t, heldout)`은 학습 중 예외를 낸다. 04에서만 `evaluation=True`를 명시한다.

## 01 — canonical은 '첫 프레임'이라는 뜻이 아니다

이 단계가 최적화하는 값은 다음과 같다.

```text
고정 함수 FLAME(shape, expression_t, jaw_t) -> local mesh_t
world mesh_t = scale * R_t @ local mesh_t + translation_t
```

`shape`와 `scale`은 첫 입력 시각의 두 camera만으로 fit한 뒤 고정한다.
각 시각에서 `expression`, `jaw`, `R`, `translation`은 현재 두 camera의 landmark로 fit한다.
과거 값은 초기값과 약한 temporal prior로만 사용한다. 미래 방향 smoothing은 없다.
`neutral = FLAME(shape, expression=0, jaw=0)`를 따로 계산한다.
처음 들어온 얼굴이 웃고 있다면, 그 posed mesh를 그대로 canonical이라고 저장하는 방식과 다르다.

여기서는 **network weight가 아니라 FLAME에 넣을 입력 parameter를 최적화**한다.
pixel3DMM이 공개한 FLAME2020 forward/LBS를 그대로 호출하며, dense prediction network는 돌리지 않는다.
눈·목의 관절은 이 작은 tracker에서 neutral로 둔다. 얼굴 landmark만으로 세밀한 주름·시선까지
복원하는 tracker가 아니며, pose/표정 오차가 03에서 geometry로 잘못 흡수될 수 있다.
`tracking/*.jpg`와 `median_error_px`, `confidence`를 확인하는 이유다.

Canonical 형태에도 모호함은 남는다. 한 시각의 landmark만으로 identity와 expression이 완벽히
분리된다고 주장할 수 없다. 이후에 첫 prefix 여러 시각으로 identity를 추정하거나, 이전에 학습된
identity prior를 연결하는 실험이 필요하다. 현재 코드는 미래 정보가 섞이지 않는 작은 출발점이다.

## 02 — 값을 갱신하는 것과 함수를 갱신하는 것

먼저 canonical mesh를 UV에 대응시킨다. 한 유효 texel은 다음 정보를 가진다.

```text
face index + barycentric coordinates -> 현재 posed surface 위치와 TBN
canonical position -> positional features -> coordinate MLP -> raw 6 channels
raw[:3] -> 5 mm * tanh -> local XYZ displacement
raw[3:] + prefix RGB logit -> sigmoid -> canonical appearance
```

RGB 초기값은 prefix의 입력 camera만 z-buffer/표면 방향으로 확인해 합성한다.
관측되지 않은 texel은 회색으로 두고 coverage를 표시한다. 일반적인 사람 얼굴을 생성하는 prior는 없다.
Gaussian의 위치는 FLAME의 해당 표면에 붙어 표정과 자세를 따라간다.
회전은 현재 TBN에서, scale은 mesh/UV 면적 비율에서 얻고 opacity는 0.95로 고정한다.
이번 비교에서 학습할 자유도는 **local XYZ 3채널 + appearance 3채널**이다.
ELITE의 16채널 전체 MGPM과 동일한 모델은 아니다.

실제 gradient 경로를 따라 읽는다:

```text
trainable value/weight -> raw Gaussian attributes -> fixed decode -> CUDA 2DGS render
       ^                                                      |
       +--------------- rendering loss gradient -------------+
```

`CoordinateField`는 27차원 position feature를 64차원 hidden feature `h`로 바꾼 뒤
마지막 Linear `Wh+b`로 6채널을 만든다. 처음 두 시각으로 이 작은 network를 warm-start한다.
다인물 pretrained encoder는 사용하지 않는다. 따라서 여기서의 `frozen`은 **개인 prefix에
학습된 field를 고정한 baseline**이며, 일반적인 feed-forward Avatar model의 성능을 뜻하지 않는다.

| 방법 | 관측이 새로 들어오면 영구적으로 바뀌는 것 | 의미 |
|---|---|---|
| `frozen` | 없음 | 같은 canonical field, 현재 pose/표정/조명만 반영 |
| `representation` | texel별 `delta_i ∈ R⁶` | test-time representation fitting |
| `adapter` | 작은 Linear A/B의 weight | prefix로 학습한 field의 마지막 Linear를 LoRA 형태로 갱신 |
| `adapter_replay` | 같은 A/B, 신뢰도 gate와 작은 과거 buffer 사용 | 정보 누적·보존 가설을 시험하는 정책 |

Adapter는 `Wh+b+B(Ah)`다. A를 작은 random 값, B를 0으로 초기화하면
첫 출력은 기존 `Wh+b`와 정확히 같다. A와 B를 둘 다 0으로 두면 첫 gradient가 둘 다 0이 된다.
`smoke.py`에서 네 방법의 시작 raw가 같은지, 실제 geometry/color와 adapter까지 gradient가 오는지 검사한다.

2DGS renderer와 UV 수식은 기존 ELITE lesson에서 재사용한다. `Surface.decode()`와
`Surface.gaussians()`의 고정 계산을 갱신하는 것이 아니라 **그 함수에 넣는 raw를 만드는 값/weight**를 갱신한다.

## 03 — 어떤 차이를 영구적으로 저장할까?

하나의 관측 시각은 아래 순서로 처리한다.

1. 01이 추정한 해당 시각의 pose/expression으로 Avatar를 움직인다.
2. Canonical raw를 `detach()`하고 현재 조명 계수를 짧게 fit한다.
3. 조명 계수를 고정하고 canonical representation 또는 adapter를 갱신한다.
4. `adapter_replay`는 tracking confidence가 너무 낮으면 영구 갱신을 건너뛴다.
5. 과거 입력 하나를 같은 canonical state로 다시 렌더링해 현재 loss에 더한다.
6. 업데이트 후 현재 관측을 작은 memory에 넣는다. 첫 anchor를 보존하고 나머지는 최근 관측을 유지한다.

조명은 `exp(.35*tanh([1,nx,ny,nz]L_t))`라는 양수의 저차 근사다.
두 입력 camera에 같은 `L_t`를 쓰며, 평가 camera에 맞춰 다시 최적화하지 않는다.
첫 prefix 시각의 조명을 1로 고정하고 계수 크기를 정규화해 기준을 정한다.
이 제한된 모형으로 cast shadow, 정반사, 모든 카메라 response를 분리할 수는 없다.
출력 색은 **canonical appearance / albedo proxy**이며 실제 intrinsic albedo의 정답은 아니다.

`adapter_replay`의 pixel gate는 현재 설명되지 않는 색 차이를 downweight하는 고정 heuristic이다.
confidence가 낮은 신규 detail을 영구적으로 배우지 못할 수도 있다. 그러므로 replay와 gate를
합친 방법만 보고 효과를 단정하지 않고 각각 제거하는 실험을 한다.
현재의 regularization은 변화량을 제한하지만 보이지 않는 texel을 완전히 잠그지는 않는다.
공유 adapter weight가 미관측 영역까지 바꿀 수 있으므로, 표면별 visibility memory와
미관측 출력 보존 제약을 추가하는 것이 다음 연구 항목이다.

```bash
# repo root: gate만 제거
CUDA_VISIBLE_DEVICES=7 python lessons/online_avatar/03_online_adaptation.py \
    --methods adapter_replay --no-gate --out lessons/online_avatar/outputs/no_gate

# replay만 제거
CUDA_VISIBLE_DEVICES=7 python lessons/online_avatar/03_online_adaptation.py \
    --methods adapter_replay --no-replay --out lessons/online_avatar/outputs/no_replay

CUDA_VISIBLE_DEVICES=7 python lessons/online_avatar/04_evaluate_retention.py \
    --runs lessons/online_avatar/outputs/no_gate --out lessons/online_avatar/outputs/no_gate_eval
```

`--seconds 2`는 새 관측 묶음당 갱신 시간 상한이다. RGB 읽기, transient light fit,
render/backward/optimizer/replay가 포함된다. 마지막 한 iteration은 중간에 끊지 않아 조금 초과할 수 있다.
`run.json`에 실제 시간, 초과량, step 수, `time_cap/step_cap/gate/frozen` 종료 이유를 남긴다.
CUDA warmup, checkpoint/그림 저장, 04 평가는 별도로 제외한다. Tracking 시간도 따로 기록한다.
엄밀한 시간 비교는 다른 작업이 없는 GPU에서 반복 측정하고 실제 총 시간에 대한 품질 곡선을 비교해야 한다.
각 방법의 학습률·parameter 수가 다르므로 이것만으로 최적의 방법이 결정되는 것도 아니다.

이 구현은 pretrained initialization의 **online adaptation**을 작게 연습하는 것이다.
ZipMap처럼 outer training으로 학습한 key/value/query와 inner fast-weight loss,
TTT-E2E처럼 update를 통과하는 meta-gradient는 구현하지 않았다. 나중에 붙일 연구 모듈이다.

## 04 — '좋아졌다'를 어떻게 확인할까?

`curves.png`는 세 가지를 구분한다.

- **Current heldout PSNR:** 같은 시각의 관측하지 않은 camera로 잘 보이는가? 곡선의 점마다 표정이 달라지므로 상승만으로 누적 효과를 단정하지 않는다.
- **Anchor retention:** 항상 처음 시각을 다시 렌더링한다. 새 관측을 배운 뒤 과거에 잘 맞던 부분이 무너지는지 본다.
- **Forgetting:** 각 과거 시각에서 지금까지 가장 높았던 PSNR과 현재 PSNR의 차이를 평균한다. 낮을수록 좋다.

`retention_matrix`의 행은 '몇 번째 관측까지 학습했는가', 열은 '어느 과거 시각을 렌더링했는가'다.
아직 관측하지 않은 미래 열은 `null`이다. 열을 고정해 읽으면 같은 표정·같은 camera에 대한 변화가 된다.
평가 이미지를 학습에 쓰거나 가장 좋아 보이는 checkpoint를 자동 선택하지 않는다.

평가 mask는 입력만으로 추적한 FLAME mesh를 rasterize한 뒤 조금 침식해 고정한다.
학습한 Gaussian alpha가 작아졌다고 평가 픽셀도 줄어들지는 않는다.
현재 PSNR은 sRGB의 **mesh 표면 내부 지표**다. 외곽 머리카락·배경·full-image 성능을 대표하지 않는다.
실측 3D canonical geometry 정답이 없으므로 `canonical_drift.png`의 변화량은 진단값이다.
Identity preservation의 정확도나 geometry accuracy라는 이름으로 보고하면 안 된다.

`metrics.json`에는 실제 update 시간, 별도 tracking 시간, trainable parameter 수, PyTorch peak GPU 할당량,
이미 pose가 준비된 상태의 decode+render FPS를 남긴다. 마지막 수치에는 landmark 검출과
FLAME tracking이 포함되지 않는다. **높은 렌더링 FPS가 실시간 reconstruction을 뜻하지 않는다.**
`efficiency.png`는 공통 prefix 초기화 이후의 누적 갱신 시간에 대한 고정 anchor 품질이다.

## 확인된 범위와 후속 연구

현재 구현은 실제 두 시점 관측, calibrated projection, FLAME/LBS, surface Gaussians,
photometric gradient, parameter/weight update, bounded replay, 별도 평가 경로를 모두 연결한다.
일반적인 pretrained Avatar network, 학습된 confidence policy, expression residual network,
hair Gaussians의 증식, 장기 sequence의 identity drift 보정은 포함하지 않았다.

연구로 확장할 때의 주요 실험은 다음과 같다.

1. Prefix initialization을 여러 identity로 사전학습한 prior로 교체하고 평가 identity를 완전히 분리한다.
2. 현재의 frame/pixel heuristic 대신 표면별 관측 신뢰도·coverage·uncertainty로 canonical update를 결정한다.
3. 같은 얼굴 영역에서 표정/조명으로 생기는 차이와 persistent shape/appearance 차이를 구분하는 목적함수를 만든다.
4. 고정 anchor+최근 관측 대신 pose/표면 coverage를 고려한 memory selection을 비교한다.
5. 여러 표정 sequence, 길어진 stream, 관측 가림, 알려진 밝기 변화로 forgetting과 잘못 저장된 색을 확인한다.
6. Inner update를 포함하는 outer training으로 학습된 초기화/learning rate/fast-weight memory를 비교한다.

새로운 adapter를 하나 붙였다는 사실만으로 novelty가 되지는 않는다. 이 실험에서 확인하려는
기여 후보는 **동적 얼굴의 정보를 안전하게 분리·누적하는 방법과 제한된 갱신 비용에서의 효과**다.
Flow Matching lesson은 이 작업에 포함하지 않는다.

## Smoke test

```bash
CUDA_VISIBLE_DEVICES=7 python lessons/online_avatar/smoke.py
```

기본 full run과 분리된 `.cache/smoke`에 5시각·128px·64UV의 작은 실행을 만든다.
각 stage의 명령도 별도 프로세스로 실제 실행한다. 기본 smoke 폴더가 이미 있으면
`--work lessons/online_avatar/.cache/smoke_new`로 새 실행을 만들거나 `--reuse`로 기존 결과를 검사한다.
`--reuse`는 단순 출력 확인만 하는 게 아니라 실제 renderer backward 검사도 수행한다.

검사: heldout RGB read 차단, 같은 초기값, frozen canonical 불변, geometry/appearance/adapter의
유한하고 0이 아닌 gradient, frozen trunk의 gradient 부재, 과거만 포함한 replay,
네 방법의 실제 갱신·유한한 평가 수치. 최종 그림도 눈으로 확인한다.
작은 smoke의 품질 차이를 연구 성능·수렴·실시간 성능의 근거로 사용하지 않는다.

2026-11-14에 기존 `elite` 환경과 GPU 7에서 전체 smoke를 실행했다.
5시각, 입력 2 camera/평가 1 camera, 128px 이미지, 64×64 UV의 2,861개 Gaussian을 사용했다.
네 방법의 실행과 위 검사를 통과했고, 낮은 confidence의 관측을 거절하는 분기도 확인했다.
입력 landmark fitting의 시각별 median error를 다시 중앙값 낸 값은 약 0.716px였다.
이는 128px crop에서의 tracking 진단값이며 3D geometry 정확도는 아니다.
로컬 결과: [비교 이미지](.cache/verified/evaluation/comparison.jpg),
[평가 곡선](.cache/verified/evaluation/curves.png), [검사 기록](.cache/verified/smoke_report.json).
이 결과는 Git에 올리지 않으며, 다른 checkout에서는 위 명령으로 직접 생성한다.

## 논문·원본 코드와 연결

- [TTT, ICML 2020](https://proceedings.mlr.press/v119/sun20b.html): test observation의 self-supervision으로 model을 갱신하는 개념.
- [ELITE §3.2](https://arxiv.org/html/2601.10200v1#S3.SS2): observed video rendering loss로 pretrained MGPM을 개인화한다. 우리의 streaming/replay 정책은 별도 설계다.
- [TTT3R](https://arxiv.org/abs/2509.26645): sequential 3D memory update와 online learning의 연결. FLAME Avatar용 모듈을 제시하는 논문은 아니다.
- [ZipMap](https://arxiv.org/abs/2603.04385), [공개 코드](https://github.com/Haian-Jin/ZipMap): fast weights에 scene 정보를 저장한다. 그 inner key–value loss와 여기의 rendering loss를 혼동하지 않는다.
- [NeRSemble](https://tobias-kirschstein.github.io/nersemble/): 동기화 multi-view human head capture.
- [pixel3DMM](https://github.com/SimonGiebenhain/pixel3dmm): 기존 VDA checkout에 들어 있는 FLAME2020 forward를 호출한다. 라이선스 자산을 복사·재배포하지 않는다.
- [기존 ELITE lesson](../elite/GUIDE.md): UV barycentric/TBN과 실제 2DGS CUDA renderer만 import해 재사용한다.
