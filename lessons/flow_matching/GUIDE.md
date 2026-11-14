# Flow Matching — 경로를 정하고, 속도를 배우고, 적분해서 생성하기

**목표:** “loss로 무엇을 맞추는가?”, “생성할 때 무엇을 입력하는가?”, “왜 ODE가 필요한가?”를
수식과 코드, 그림으로 연결한다. 앞에서 공부한 DiT를 **velocity predictor**로 직접 학습하는 것까지 이어진다.

기준은 [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)과
[Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264)다.
03에서는 별도 연구인 Rectified Flow와 minibatch OT-CFM으로 범위를 넓힌다.
공식 구현과의 대응·변경 사항은 [NOTICE.md](NOTICE.md)에 적었다.

## 여섯 단계

| 단계 | 집중해서 이해할 것 | 직접 확인할 결과 |
|---|---|---|
| [00 Probability paths](00_probability_paths.py) | noise→data 경로와 그 미분 | 분포 변화, pair별 경로, coefficient 곡선, GIF |
| [01 Conditional Flow Matching](01_conditional_flow_matching.py) | conditional target으로 marginal velocity를 학습하는 이유 | 정답/학습 vector field 비교, loss 곡선 |
| [02 ODE sampling](02_ode_sampling.py) | 학습한 속도장을 적분해서 생성 | 실제 생성 궤적·GIF, Euler/Heun/RK4의 같은 NFE 비교 |
| [03 Coupling & reflow](03_coupling_and_reflow.py) | noise와 data를 어떻게 짝지을지 | independent/OT/teacher pair, 재학습 결과, 적은 NFE의 차이 |
| [04 Train image flow](04_train_image_flow.py) | DiT backbone + velocity loss + optimizer/EMA | 실제 MNIST 학습, `0.png,100.png,200.png,...` |
| [05 Image sampling & CFG](05_image_sampling_and_cfg.py) | conditional generation, CFG, step 수와 계산량 | 숫자 생성·GIF, 조건·solver 비교, endpoint 외삽 |

00~03은 2D 점 분포에서 계산이 눈에 보이도록 만들었다. 04~05는 **실제 이미지 학습과 생성**이다.
각 `.py` 맨 위에 실행 명령·출력·읽는 순서가 있다. 모델 수식이 큰 공통 framework에 숨지 않도록
번호 파일에 구현했다. `common.py`에는 입출력·시각화·seed 설정과 finite gradient를 확인하는
짧은 optimizer step을 모았다.

## 실행 순서

기존 `elite` Conda 환경을 사용한다. venv 생성이나 package 자동 설치는 없다.
필요한 package는 PyTorch, torchvision, timm, NumPy, SciPy, matplotlib, Pillow, tqdm이다.
00~03은 CPU에서도 실행 가능하고, 아래에서는 GPU 7로 시간을 줄인다.

repo root `/media/vilab/mh/curious`에서:

```bash
python lessons/flow_matching/00_probability_paths.py
CUDA_VISIBLE_DEVICES=7 python lessons/flow_matching/01_conditional_flow_matching.py --device cuda
CUDA_VISIBLE_DEVICES=7 python lessons/flow_matching/02_ode_sampling.py --device cuda
CUDA_VISIBLE_DEVICES=7 python lessons/flow_matching/03_coupling_and_reflow.py --device cuda
CUDA_VISIBLE_DEVICES=7 python lessons/flow_matching/04_train_image_flow.py --download
CUDA_VISIBLE_DEVICES=7 python lessons/flow_matching/05_image_sampling_and_cfg.py
```

Lesson 폴더로 이동했다면 `python 00_probability_paths.py`처럼 파일명만 써도 된다.
기본 출력 경로는 파일 위치 기준이다. 직접 지정한 `--out`, `--checkpoint`는 현재 터미널 위치 기준이다.
01이 끝나야 02/03을 실행할 수 있다. 04의 이미지 학습은 2D checkpoint와 독립적이고, 05는 04 결과가 필요하다.

MNIST cache가 있으면 재사용한다. 없으면 `--download`가 ignored `data/`에 내려받는다.
실행 결과·가중치·cache도 모두 ignore되며 코드와 설명만 Git으로 관리한다.
기본 설정은 01 3,000 steps, 04 6,000 steps다. 단계 수와 optimizer step 수, ODE step 수는 다른 숫자다.

## 먼저 구분할 네 가지

| 기호 | 의미 | 값을 알 수 있는 곳 |
|---|---|---|
| `x0` | source sample. 여기서는 Gaussian noise | 학습·생성 모두 직접 뽑음 |
| `x1` | target data sample | 학습 시 dataset에서 읽음 |
| `x_t` | 정한 경로 위의 중간 상태 | 학습은 식으로 만들고, 생성은 ODE로 갱신 |
| `v_theta(x_t,t,c)` | 현재 위치와 시간에서의 velocity 예측 | 학습되는 network의 출력 |

이 lesson은 **t=0이 noise, t=1이 data**다. 기존 DDPM/DiT lesson의 clean→noise 시간 관례와 다르다.
또한 t는 실제 영상의 frame 번호나 online Avatar의 관측 시각이 아니라, 생성 경로의 가상 시간이다.

```text
학습: x0 + 실제 x1 + random t -> 식으로 x_t와 정답 velocity -> network -> MSE -> weight update
생성: 새로운 x0 + 조건       -> v_theta(x,t)를 여러 번 평가 -> ODE state update -> 생성 x1
```

학습 때는 x1을 알고 있어 경로 위의 한 점과 정답을 바로 만들 수 있다.
생성 때는 x1을 모르므로 network가 알려주는 현재 속도로 이동한다.
생성 중 model weight를 다시 학습하는 것은 아니다. 변하는 것은 image/point state다.

## 무엇이 논문의 핵심 기여인가?

Continuous Normalizing Flow는 시간에 따른 velocity field로 분포를 운반한다.
ODE `dx/dt=v_t(x)`와 density의 continuity equation `∂_t p_t + div(p_t v_t)=0`가 연결된다.
FM은 정해진 probability path를 따르는 vector field를 **회귀로 학습**하는 틀을 제공한다.
중요한 지점은 직접 계산하기 어려운 marginal field 대신, 쉽게 계산하는 conditional field로
동일한 model-gradient를 얻을 수 있다는 것이다.

학습 step마다 긴 ODE를 풀고 그 과정 전체를 역전파할 필요가 없다는 의미에서 simulation-free다.
생성할 때 ODE solver가 사라진다는 뜻은 아니다. 또한 FM은 특정 U-Net/Transformer 구조를 요구하지 않는다.
여기서는 작은 MLP와 기존 DiT에 같은 원리를 적용한다.

Rectified Flow의 reflow, minibatch OT coupling, DiT backbone은 각기 별도의 아이디어다.
아래 실험에서 함께 사용하더라도 이를 모두 원 FM 논문 하나의 신규 기여로 묶지 않는다.
대규모 최신 모델의 성능을 재현하는 대신, 그 모델들에 재사용할 계산을 분해해서 배우는 구성이다.

## 00 — path와 velocity를 직접 만들기

일반적인 affine path를 다음처럼 쓴다.

```python
x_t = alpha(t) * x1 + sigma(t) * x0
u_t = d_alpha(t) * x1 + d_sigma(t) * x0
```

가장 단순한 linear path라면:

```python
x_t = (1 - t) * x0 + t * x1
u_t = x1 - x0
```

`u_t`는 이 pair의 경로를 t에 대해 미분한 값이다. 정답 endpoint를 아는 학습 중에는 계산할 수 있다.
`t`를 batch마다 무작위로 한 번 뽑고 그 점만 만든다. 0부터 t까지 작은 step으로 시뮬레이션하지 않는다.
`conditional_path()`의 `[B,1,...]` reshape는 image tensor의 모든 pixel/channel에 같은 t를 적용한다.

00에서 세 가지를 비교한다.

- `linear`: 정확히 noise에서 data까지 이어진다.
- `residual`: `sigma(t)=1-(1-sigma_min)t`. 끝이 `x1+sigma_min*x0`여서 작은 noise가 남는다. 원 FM의 finite endpoint noise를 확인하는 예다.
- `cosine`: `alpha=sin(πt/2)`, `sigma=cos(πt/2)`. FM이 직선 하나에 제한되지 않음을 보여준다. 계수의 제곱 합이 1이라는 뜻이지, 이 mixture dataset의 실제 분산까지 일정하다는 뜻은 아니다.

`path.gif`는 **정답 pair를 아는 학습 경로**다. 아직 network가 이미지를 생성한 결과가 아니다.

생각해볼 것: sigma_min을 0.2로 바꾸면 endpoint가 깨끗한 데이터와 다른 이유를 그림과 식으로 설명할 수 있을까?

## 01 — 같은 위치에 서로 다른 정답 속도가 생겨도 괜찮을까?

서로 다른 `(x0,x1)` pair의 선이 같은 공간 근처를 지나면 방향이 서로 다를 수 있다.
Network에는 `x_t,t`만 주므로 원래의 pair를 완벽하게 알아맞힐 수 없다.
MSE 회귀가 배우는 것은 다음 conditional expectation이다.

```text
v*(x,t) = E[u_t | x_t=x]
```

이는 pair별 conditional field를 해당 위치에서 평균한 **marginal field**다.
경로와 coupling이 model parameter와 독립적으로 고정되어 있으면:

```text
E ||v_theta - u||²
  = E ||v_theta - E[u|x_t,t]||² + E ||u - E[u|x_t,t]||²
```

뒤 항은 model과 무관한 conditional variance다. 그래서 conditional target을 사용하는 loss와
marginal target을 사용하는 loss는 값은 달라도 model에 대한 기대 gradient가 같다.
공간마다 pair velocity가 충돌한다고 회귀 자체가 잘못된 것이 아니고, CFM loss가 0이 되어야 하는 것도 아니다.

01의 target은 8개의 작은 Gaussian이 원형으로 놓인 분포다.
이 특수한 경우에는 `oracle_velocity()`로 정답 marginal field를 해석적으로 계산할 수 있다.
**이 oracle은 진단·그림에만 사용하고, 학습 정답은 계속 `x1-x0`를 사용한다.**

`0.png,100.png,...`의 위쪽은 oracle, 아래쪽은 학습한 MLP다.
`loss.png`에서 CFM validation과 marginal MSE를 같이 보자. 두 값이 다른 이유가 위 분해식이다.
CFM의 conditional은 여기서 endpoint/pair를 조건으로 삼는다는 뜻이다. class label을 넣는 conditional generation은 04에서 추가한다.

생각해볼 것: Network에 x1도 넣으면 loss는 쉽게 낮아질 수 있다. 그런데 실제 생성 때 그 입력은 어디서 얻을까?

## 02 — ODE를 풀어야 하는 이유

생성은 새 noise에서 다음 초기값 문제를 푼다.

```text
x(0) ~ N(0,I)
dx(t)/dt = v_theta(x(t),t)
```

Euler는 `x_next = x + dt*v(x,t)`다. Heun은 Euler로 한 번 예측한 위치에서 속도를 다시 구해
두 속도의 평균을 쓴다. RK4는 한 구간에서 네 번 평가한다.
`integrate()` 안에 이 계산을 전부 적었으므로 solver 라이브러리 내부를 찾아갈 필요가 없다.

학습 pair의 경로가 직선이고 `x1-x0`가 상수라고 해도 **marginal field의 생성 궤적은 휘어질 수 있다**.
Network는 생성 시 알 수 없는 특정 pair의 상수를 외우는 함수가 아니다.
따라서 처음의 velocity를 한 번 더한 `x0+v(x0,0)`가 좋은 data가 된다는 보장은 없다.
02의 `trajectories.png`와 00의 pair 경로를 비교해보자.

`nfe.png`는 두 오차를 구분한다.

| 측정값 | 비교 대상 | 의미 |
|---|---|---|
| endpoint RMSE | 같은 noise를 같은 model의 촘촘한 RK4로 적분한 결과 | 수치 적분 오차의 근사 |
| empirical sliced W2 | 별도로 뽑은 실제 target 분포 sample | 생성 분포와 데이터의 차이 |

Solver를 정밀하게 바꿔도 model의 학습 오차가 없어지지는 않는다.
Euler 32 steps와 Heun 32 steps를 같은 비용이라고 부르면 안 된다.
여기서는 Euler32 / Heun16 / RK4 8이 모두 **NFE=32**인 비교다.
NFE는 batch 크기와 별개의 field 평가 횟수다. 높은 차수 solver가 모든 큰 step에서 무조건 우월한 것도 아니다.

## 03 — conditional OT path, minibatch OT, reflow는 다르다

| 선택 | 무엇을 바꾸나? | 여기의 구현 |
|---|---|---|
| Linear conditional path | 정해진 pair 사이를 어떻게 이동하나 | `(1-t)x0+t*x1` |
| Independent coupling | 어떤 noise와 data를 짝짓나 | 두 분포에서 독립적으로 draw |
| Minibatch OT coupling | batch 안의 pairing을 바꿈 | squared distance 비용의 Hungarian assignment |
| Reflow | teacher가 만든 source→endpoint pairing으로 다시 학습 | `x1_teacher = ODESolve(teacher,x0)` |

공식 코드의 이름 `CondOTScheduler`만 보고 전체 분포의 global OT를 해결했다고 이해하면 안 된다.
Independent pairing에서도 각 pair 사이의 conditional path는 직선일 수 있다.
그 pair 전체의 결합분포와, marginalize한 생성 flow가 전역적으로 최적인가는 다른 질문이다.

Minibatch OT는 같은 source/target batch의 marginal을 유지하며 대응 순서만 바꾼다.
개별 batch의 transport cost를 줄이지만 전체 dataset의 전역 optimal map이라는 보장은 없다.

Reflow는 기존 field로 noise를 운반한 결과를 endpoint로 저장한다. 이후 새 학생이 그 pair의
직선을 따라가도록 다시 CFM을 학습한다. 더 적은 NFE로 생성할 가능성을 살펴보는 실험이다.
Teacher 자체가 데이터와 다르면 그 오차도 전달될 수 있다. 자동으로 정확한 one-step generator가 되는 절차는 아니다.

03은 independent teacher, 새 OT model, 새 reflow model을 비교한다.
기본 새 모델 step 수는 teacher의 학습 step과 같지만, reflow에는 **teacher 학습 + pair 생성 + 학생 학습** 비용이 있다.
`metrics.json`에 이를 구분해 저장한다. 세 방법의 training MSE 자체는 coupling이 달라 직접적인 품질 순위가 아니다.
생성 sample, 데이터에 대한 sliced W2, trajectory bending을 함께 읽는다.

생각해볼 것: 경로가 덜 휘었는데 데이터 분포와의 거리는 오히려 커질 수 있을까? Teacher error와 학습 부족을 분리해서 설명해보자.

## 04 — DiT를 FM에 연결하기

FM과 DiT는 서로 다른 축이다. **FM은 무엇을 어떻게 학습할지, DiT는 그것을 예측하는 network 구조**다.
원 DiT의 patch embedding·Transformer·adaLN-Zero를 그대로 사용한다.
아래 부분이 기존 DiT lesson과 달라진다.

| 항목 | 기존 DiT lesson | 이번 image flow |
|---|---|---|
| 공간 | frozen VAE latent | 32×32 grayscale pixels |
| 입력 | noisy latent, diffusion step, class | interpolated image, continuous t, class |
| 출력 | epsilon + learned variance | velocity 1채널 |
| loss | epsilon MSE + variance 관련 항 | `MSE(v_theta, x1-x0)` |
| 생성 | diffusion sampler | Euler/Heun/RK4 ODE |

`learn_sigma=False`는 별도 variance head를 만들지 않게 한다. 이 설정만 바꿔서 FM이 되는 것은 아니고,
실제 path·target·loss·sampler를 모두 바꾼다. Diffusion 구현에서 말하는 `v_prediction`이라는
이름 역시 자동으로 이 `dx/dt`와 같은 정의는 아니다. 수식과 시간 관례를 확인해야 한다.

`FlowDiT.forward()`에서 t를 embedding 직전에 `1000*t`로 표현한다.
이는 sinusoidal embedding이 읽을 시간 feature의 scale이고 1,000번 적분한다는 뜻이 아니다.
ODE의 시간 단위는 계속 [0,1]이므로 target velocity를 1,000배 하지 않는다.

MNIST train 60,000장을 고정 seed로 섞어 55,000/5,000 train/validation으로 나눈다.
원본 28px은 검은 padding으로 32px로 만든다. Class 의미를 바꾸는 flip augmentation은 없다.
Mini 설정은 4 blocks / width192 / 6 heads / patch4다. 논문에 쓰인 대규모 모델의 이름이나 성능을 뜻하지 않는다.

Training은 tqdm bar로 진행하고, 기본 100 step마다:

- 같은 validation 이미지·noise·t의 velocity MSE를 계산한다.
- 같은 initial noise와 class 0~9로 current model/EMA 결과를 `100.png,200.png,...`에 저장한다.
- `latest.pt`, EMA validation loss 기준 `best.pt`, `history.json`, `loss.png`를 저장한다.

EMA는 optimizer로 따로 학습하는 두 번째 model이 아니라 weight의 지수이동평균이다.
초기 출력은 아직 거의 noise일 수 있다. loss 감소와 생성 이미지의 의미가 다르므로 둘 다 확인한다.
Validation 생성은 기본 Heun 24 steps다. 이것은 관찰용 설정이며 최적 sampler/FID 실험이 아니다.

기본 설정으로 시작했다면 repo root에서 이렇게 이어서 학습한다:

```bash
CUDA_VISIBLE_DEVICES=7 python lessons/flow_matching/04_train_image_flow.py \
    --resume lessons/flow_matching/outputs/04/latest.pt --steps 12000
```

`--steps`는 추가량이 아니라 누적 목표다. Custom profile/batch/lr/precision으로 시작했다면 같은 값을 재개 명령에 넣는다.
Optimizer·EMA·RNG도 복원하고, data/source/config가 달라졌으면 조용히 섞지 않고 오류를 낸다.
Validation 빈도가 다음 training random sample을 바꾸지 않게 RNG도 격리한다.
`--val-every 100`은 변경할 수 있고, 생성 preview가 오래 걸리면 `--preview-steps 12`로 줄일 수 있다.
BF16 지원 GPU가 아니면 `--precision fp32`를 사용한다.

## 05 — 생성, CFG, endpoint 외삽

학습 중 10%의 label을 null label로 바꿔 conditional/unconditional velocity를 같은 network에 학습시켰다.
이것을 다음처럼 결합한다.

```python
v_cfg = v_null + w * (v_cond - v_null)
```

우리 관례는 w=0이면 unconditional, w=1이면 일반적인 conditional이다.
`cfg.png`의 모든 칸은 **같은 noise**를 사용한다. w=0의 열들이 같고, 조건에 따라 나머지 행의 숫자가
달라지는지 확인할 수 있다. 큰 guidance가 항상 품질이나 다양성에 유리한 것은 아니다.

CFG의 두 branch를 하나의 batch로 합쳤다고 계산이 공짜가 되지는 않는다.
`metrics.json`은 field NFE, Python network call 수, branch 수, single-branch batch 환산 비용을 구분한다.
`solvers.png`의 마지막 세 줄은 Euler32 / Heun16 / RK4 8로 동일한 field NFE를 사용한다.

Linear path에서는 정답 pair에 대해:

```text
x1 = x_t + (1-t)*u_t
epsilon = x_t - t*u_t
```

Network의 marginal prediction을 넣은 `x_t+(1-t)v_theta`는 endpoint의 추정이지
그 위치에서 출발한 ODE의 미래 endpoint를 정확히 계산한 것이 아니다.
w=1인 잘 학습된 모델에서는 conditional mean 관점으로 읽을 수 있고, 강한 CFG에서는 그것도 바뀐다.
`extrapolation.png`에서 t가 다른 위치들의 한 번 외삽 결과를 실제 최종 생성 결과와 비교한다.

Noise 예측과 velocity 예측은 특정 path에서 변환할 수 있지만 loss weighting까지 같지는 않다.
예를 들어 여기서는 `epsilon_theta=x_t-t*v_theta`이므로 noise MSE는 같은 velocity 오차에 t²가 곱해진다.
그저 출력 변수 이름만 바꾸어 동일한 학습이라고 생각하면 안 된다. t=0에서 역변환으로 나누는 것도 불가능하다.

ODE state는 적분 중 [-1,1]로 clamp하지 않는다. PNG를 만들 때만 clipping한다.
중간에 state를 clamp하면 실제로 푸는 동역학이 달라진다.

## 연구 주제로 가져갈 때

- Avatar의 UV latent·Gaussian parameter를 state로 두고, 이미지·text·pose를 condition으로 주는 field를 생각할 수 있다.
- Coarse reconstruction을 source, 정제된 reconstruction을 target으로 두는 conditional transport도 가능하다. 이 경우 coupling의 의미와 사용할 수 있는 paired supervision부터 정해야 한다.
- Pixel 생성과 다르게 geometry에는 topology·identity·surface consistency 제약이 있다. Flow loss만으로 이러한 성질이 자동 보장되지는 않는다.
- 같은 backbone을 쓰면서 path, coupling, conditioning, prediction target, inference budget을 각각 바꾸면 어떤 가정이 효과를 냈는지 비교하기 좋다.
- One-step 생성이 다음 관심사라면 reflow 이후 distillation·consistency·mean-flow 계열과 무엇이 다른지 살펴볼 수 있다. 이번 코드에서 그 방법들까지 재현한 것은 아니다.

이 lesson을 마쳤을 때 직접 설명할 질문:

1. 학습 시 target endpoint를 쓰면서도 생성 시에는 주지 않는 이유는?
2. CFM loss가 0이 되지 않더라도 marginal field가 맞을 수 있는 이유는?
3. Straight conditional paths가 curved sampling trajectories를 만들 수 있는 이유는?
4. Coupling을 바꾸는 것과 ODE solver를 바꾸는 것은 무엇이 다른가?
5. DiT를 flow model로 바꾸려면 architecture 외에 무엇을 바꿔야 하는가?
6. Reflow·CFG·많은 solver step 각각의 추가 비용은 무엇인가?

## Smoke test와 검증 범위

```bash
CUDA_VISIBLE_DEVICES=7 python lessons/flow_matching/smoke.py
```

`.cache/smoke`에 여섯 CLI의 작은 실행을 만들며, 이미 같은 결과가 있으면 새 `--out`을 사용한다.
`--reuse`는 기존 결과의 수식/gradient/재개 계약을 다시 검사한다.

검사 항목은 path derivative의 수치 미분, analytic field의 endpoint 성질, solver NFE와 수렴,
minibatch OT의 비용/주변분포 보존, 2D 실제 학습, DiT block까지의 gradient, CFG 항등식,
동일 seed의 deterministic inference와 weight 불변성, 8 steps와 4+4 resume의 model/EMA 일치다.
Default `mini`의 BF16 실행은 축소된 smoke profile과 별도로 확인한다.
짧은 이미지 smoke는 noise처럼 보여도 이상하지 않다. 생성 품질·ImageNet FID 검증과 구분한다.

2026-11-14, 기존 `elite` 환경 / H200 / GPU 7에서 여섯 단계의 smoke test를 통과했다.
경로 미분의 최대 오차는 3.6×10⁻¹⁰ 미만이고, 중단 없는 8 steps와 4+4 resume의
model/EMA weight 차이는 모두 0이었다. 별도로 2D MLP를 1,000 steps 학습했을 때 고정
validation의 analytic marginal MSE는 1.826에서 0.108로 감소했다.
이 수치는 해당 seed·작은 실험의 동작 확인이며 논문의 대규모 생성 성능 지표는 아니다.

이미지 `mini` 설정도 BF16 / 실제 MNIST train 55,000장으로 5,700 steps까지 별도 확인했다.
고정 validation의 EMA velocity MSE는 1.945에서 0.147로 줄었고, Heun NFE=64 / CFG=2로
noise에서 숫자 이미지와 생성 GIF를 만들었다. 일부 숫자 형태는 아직 거칠어 추가 학습 여지가 있다.
이 부가 실행은 SIGTERM으로 종료되어 6,000 steps 완료로 기록하지 않는다.
저장된 5,700-step checkpoint의 05 생성은 정상 완료했으며, 위의 전체 smoke test도 모두 통과했다.
