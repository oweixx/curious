# DiT를 코드와 실험으로 이해하기

논문: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748),
William Peebles · Saining Xie, ICCV 2023.
원본 모델/학습/샘플링 코드를 기준으로 구성했다. 번호가 붙은 파일의 맨 위에 실행 명령,
결과 파일, 관찰할 내용이 있다. root README는 변경하지 않는다.

**목표는 입력부터 loss, gradient, 생성까지 설명할 수 있게 되는 것**이다.
원본의 모든 ImageNet 실험/FID를 재현했다고 주장하지 않는다.
논문 ImageNet256 모델과 같은 블록을 사용해 작게 직접 학습하고, 마지막에는 실제 원본
DiT-XL/2 가중치로 ImageNet 생성을 확인한다.

## 환경과 실행 순서

기존 conda elite에서 실행한다. 별도의 venv를 만들거나 환경을 자동 변경하지 않는다.
기본 경로는 파일 위치 기준이고, 다음 위치에서 주석의 명령을 그대로 실행하면 된다.

    conda activate elite
    cd /media/vilab/mh/curious/lessons/done/dit

GPU 작업 예시는 물리 GPU4를 지정한다. 그 안의 cuda:0이 GPU4다.
00/06/07/08/10은 GPU, 나머지는 CPU로 충분하다.
검증 환경: Python3.10, PyTorch2.5.1+cu124, torchvision0.20.1, timm1.0.16,
diffusers0.25.1, numpy, matplotlib, Pillow, tqdm. 신규 환경이라면 이 의존성이 필요하다.

| 단계 | 직접 이해할 계산 | 시각적 결과 |
|---|---|---|
| [00](00_pixels_and_latents.py) | frozen VAE, posterior mean/sample, latent scaling | RGB/복원 비교, latent 채널 |
| [01](01_forward_diffusion.py) | forward noising, beta/alpha_bar/SNR | 시간별 noise, schedule 곡선 |
| [02](02_patch_tokens.py) | patch embedding, positional encoding, unpatchify | patch 경계와 token feature heatmap |
| [03](03_attention_and_conditions.py) | Q/K/V, attention, timestep/class/null embedding | head별 attention, 조건 embedding |
| [04](04_adaln_zero.py) | adaLN-Zero, shift/scale/gate, gradient 경로 | 초기 identity, backward별 gradient, 조건별 modulation |
| [05](05_objective.py) | epsilon MSE, learned variance, VB, detach | loss별 값과 head별 gradient |
| [06](06_train.py) | 실제 optimizer loop, CFG dropout, EMA, validation/resume | 고정 seed 생성 이미지, 학습 곡선 |
| [07](07_reverse_sampling.py) | reverse diffusion, timestep respacing, DDPM/DDIM | 생성 GIF, noise/variance → reverse Gaussian 비교 |
| [08](08_classifier_free_guidance.py) | conditional/unconditional 결합, 채널 관례 | class×CFG 비교, 3/4채널 비교 |
| [09](09_scaling.py) | depth/width/patch와 계산량, model vs sampling compute | parameter/MAC 비교 |
| [10](10_original_pretrained.py) | 원본 XL/2의 실제 end-to-end inference | ImageNet 생성, 원본 attention |

    CUDA_VISIBLE_DEVICES=4 python 00_pixels_and_latents.py
    python 01_forward_diffusion.py
    python 02_patch_tokens.py
    python 03_attention_and_conditions.py
    python 04_adaln_zero.py
    python 05_objective.py
    CUDA_VISIBLE_DEVICES=4 python 06_train.py
    CUDA_VISIBLE_DEVICES=4 python 07_reverse_sampling.py --checkpoint checkpoints/mini/best.pt
    CUDA_VISIBLE_DEVICES=4 python 08_classifier_free_guidance.py --checkpoint checkpoints/mini/best.pt
    python 09_scaling.py
    CUDA_VISIBLE_DEVICES=4 python 10_original_pretrained.py

10은 독립적인 공개 checkpoint 실험이므로06 학습 완료 전에도 실행할 수 있다.
07/08은06 결과가 필요하다. 작은 smoke checkpoint는 최종 생성 품질을 보장하지 않는다.
06에서 `--out checkpoints/small`처럼 경로를 바꾸면03/04/07/08의 `--checkpoint`도
`checkpoints/small/best.pt`로 맞춘다. model profile은 checkpoint에서 읽는다.
데이터/가중치/cache/출력은 gitignore에 들어가며, 코드/설명/원본 출처만 Git으로 관리한다.

## 논문에서 새롭게 배울 것

Attention, Transformer, VAE, diffusion, CFG, learned variance 각각은 이전 연구의 구성 요소다.
DiT의 기여는 **latent diffusion의 backbone을 Transformer로 구성하고, 조건 주입 방식과
모델/토큰 크기를 체계적으로 비교해 확장성을 보인 것**으로 읽는다.

1. U-Net의 down/up 경로 대신 같은 token resolution의 Transformer blocks를 쌓는다.
2. diffusion timestep/class를 어떻게 넣을지 비교한다. 논문의 비교 대상은
   in-context tokens, cross-attention, adaLN, adaLN-Zero다.
3. 좋은 결과를 보인 adaLN-Zero 구조와 초기화를 관찰한다.
4. parameter 수만 보지 않고 depth, width, patch 크기로 변하는 계산량을 함께 분석한다.

공개 models.py는 대표 adaLN-Zero 모델을 제공한다. 이 lesson도 그 모델을 직접 쓴다.
다른 세 conditioning 변형의 ImageNet 재학습까지 포함했다고 해석하지 않는다.
원본을 읽는 길: models.py의 DiT.forward → DiTBlock.forward → initialize_weights,
diffusion/gaussian_diffusion.py의 training_losses → p_mean_variance → p_sample,
diffusion/respace.py의 SpacedDiffusion과 _WrappedModel.

논문의 conditioning 비교를 코드 관점에서 구분하면 다음과 같다.

| 방식 | 조건이 들어가는 위치 | 영상 patch token과의 관계 |
|---|---|---|
| in-context | timestep/class를 별도 token으로 붙임 | self-attention에서 이미지/조건 token이 함께 상호작용 |
| cross-attention | 이미지 query가 조건 token의 key/value를 조회 | self-attention 외에 조건을 읽는 attention 경로 추가 |
| adaLN | 조건으로 LayerNorm 이후의 shift/scale을 생성 | 모든 공간 token에 같은 feature별 조절값을 적용 |
| adaLN-Zero | adaLN + 조건부 residual gate + identity 초기화 | 초기에는 residual branch를 닫고 학습하며 기여를 조절 |

이 중 공개 원본이 제공하는 **adaLN-Zero를 실제 실행·학습**한다.
위 표의 다른 세 구조는 비교 개념이며, 별도 구현/성능 측정을 했다는 뜻은 아니다.

이전 avatar lesson과 연결해 읽으면, **model이 예측하는 표현**과 **그 표현을 사용하는
고정 계산**을 구별하는 습관이 그대로 이어진다. 이곳의 DiT는 latent 이미지 자체를
한 번에 완성하는 함수가 아니다. 현재 noisy latent에서 noise/variance를 예측하고,
sampler가 그 예측으로 다음 latent를 계산한다. VAE decoder는 마지막 latent를 RGB로 바꾼다.

## 00 — 이미지와 latent는 서로 다른 공간이다

64×64 숫자를 RGB로 만든 뒤 frozen SD VAE를 통과하면4×8×8 posterior가 나온다.
숫자 데이터를 택한 이유는 대규모 비디오 전처리 없이 class 조건의 효과를 볼 수 있기 때문이다.
VAE는 숫자용으로 새로 학습하지 않고 원본 DiT에서 사용한 SD VAE-EMA를 쓴다.

- encode가 만드는 것은 mean과 logvar를 가진 분포다.
- 학습에서는 z = (mean + std * noise) * 0.18215를 사용한다.
- decode에서는 scale을 다시 나눈다. DiT token/latent를 RGB처럼 해석하지 않는다.
- posterior mean을 decode한 복원도 원본 픽셀과 정확히 같지는 않다.

00은 mean/logvar를 cache해서 매 training visit에 posterior를 다시 sample한다.
이미지10,000개를 고정 latent10,000개로 바꾸는 것과 구분한다.
MNIST train에서10,000개, test에서1,000개를 뽑아 두 split을 분리한다.
학습 중test split을 고정 validation으로 사용하므로 별도의 최종 test 성능은 보고하지 않는다.

질문: scale을 encode 쪽에서만 빼먹으면01의 SNR과07의 decode에 각각 어떤 문제가 생길까?

## 01 — 학습할 때1000번 순차적으로 noise를 넣지 않는다

alpha_t = 1 - beta_t, alpha_bar_t = product(alpha_0 ... alpha_t).

    z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1-alpha_bar_t) * epsilon
    epsilon ~ N(0,I)

하나의 training batch에서 t를 무작위로 뽑고 원하는 z_t를 바로 만든다.
1000은 noising schedule의 길이이며 optimizer step 수나 network block 수가 아니다.
코드의 t는0..999 index다. t=0에서도 beta_0만큼의 작은 noise가 이미 섞인다.
수식의 clean latent `z_0` 표기와 Python의 `t == 0`을 같은 상태라고 혼동하지 않는다.
schedule.png에서 beta, 누적 signal, SNR을 서로 구분한다.
noising.png의 같은 epsilon으로 만든 열들은 서로 다른 t의 주변분포 예시다.
그 열들을 이어 붙였다고 실제 Markov chain sample이 되는 것은 아니다.

질문: 정답 epsilon을 알면 z0를 역산할 수 있다. 그런데 생성 시에는 왜 이것만으로 끝나지 않을까?
생성 때는 정답 epsilon과 z0를 모르므로 network와 reverse sampler가 필요하다.

## 02 — patch embedding과 unpatchify

논문256 RGB 기준:

    [B,3,256,256] --VAE--> [B,4,32,32]
    patch2 --Conv2d--> [B,D,16,16] --flatten--> [B,256,D]
    DiT blocks --> [B,256,D]
    output Linear --> [B,256,2*2*8]
    unpatchify --> [B,8,32,32]

8채널은05에서 설명하는 noise4 + variance parameter4다.
Conv2d의 patch projection은 모든 patch에 동일한 Linear를 적용하는 것과 같다.
출력 unpatchify는 학습 없는 재배열이다. Transformer feature를 알아서 영상으로
변환하는 network가 별도로 숨어 있는 것이 아니다.

02의 roundtrip은 **raw patch 재배열**에 대한 검증이다.
learned embedding의 inverse를 계산했다고 해석하지 않는다.
질문: latent 해상도32를 유지하고 patch2를 patch4로 바꾸면 token 수/출력 Linear 크기는?

## 03 — token 사이 계산과 조건 주입을 나누어 읽는다

Self-attention:

    Q = X Wq, K = X Wk, V = X Wv
    A = softmax(Q K^T / sqrt(head_dim))
    output = concat_heads(A V) Wo

03은 실제 timm Attention의 weight로 이 식을 직접 계산하고 출력이 일치하는지 검사한다.
첫 block/head의 query token0이 어느 공간 token을 보는지 표시한다.
attention map 하나만으로 모델의 결정 원인을 증명하지는 못한다.
미학습 그림은 연산 구조 관찰이고, 학습 후 --checkpoint로 다시 실행하면 실제 학습된 weight를 본다.

원본 DiT의 조건은 c = timestep_embedding(t) + class_embedding(y).
조건은 기본 모델에서 attention K/V에 text처럼 들어가는 것이 아니라 다음04의 modulation으로 들어간다.
num_classes=10이면 class10은 null token이다. 모델이 train일 때 일부 label을 null로 교체해
같은 network가 conditional/unconditional 예측을 모두 배우게 한다.

## 04 — adaLN-Zero의 의미

각 block에서 condition을 MLP로 바꿔6개의 D차원 벡터를 만든다.
attention branch에 shift/scale/gate, MLP branch에도 shift/scale/gate가 하나씩 있다.
모두 token dimension에 broadcast한다.

    h = (1 + scale_attn) * LayerNorm(x) + shift_attn
    x = x + gate_attn * Attention(h)
    h = (1 + scale_mlp) * LayerNorm(x) + shift_mlp
    x = x + gate_mlp * MLP(h)

shift/scale은 feature를 조절하고 gate는 residual branch의 기여를 조절한다.
LayerNorm 자체는 `elementwise_affine=False`다. 고정된 학습 gamma/beta를 쓰는 대신
condition에 따라 shift/scale을 새로 만든다. `[B,D]`를 `[B,1,D]`로 바꾸어
`[B,T,D]` 전체 공간 token에 broadcast한다. 이 값은 pixel mask가 아니다.
초기 gate=0이면 block output=input이다. Attention/MLP weight 전체가0인 것은 아니다.
최종 output projection도0으로 초기화하므로 첫 model output은0이다.
gradient_flow.png에서 처음에는 output projection, 이후 modulation/gate, 이후 attention으로
학습 신호가 전달되는 것을 확인한다. 이 단계의4회 update는 효과를 분리한 수치 실험이다.

`modulation.png`는 t만 변경/class만 변경/null로 변경했을 때6개 벡터를 보여준다.
기본 실행은4회 진단 update 이후의 작은 값이고, 의미 있는 학습 조건을 관찰하려면06 이후:

    python 04_adaln_zero.py --checkpoint checkpoints/mini/best.pt --out outputs/04_trained

이 파일의 `inspect_modulation`은 LayerNorm → modulation → attention → residual
→ LayerNorm → modulation → MLP → residual을 직접 풀어 쓰고 원본 block과 비교한다.

질문: block identity와 network output0은 어떻게 동시에 성립할까?

## 05 — epsilon만 예측한다고 말하면 빠지는 부분

원본 설정 learn_sigma=True에서 출력은 epsilon_hat4채널과 v4채널이다.
v는 std 자체가 아니라 아래 log-variance 식에 들어간다.

    fraction = (v + 1) / 2
    log_variance = fraction * log(beta_t)
                 + (1 - fraction) * log(posterior_variance_t)

원본은 v를 hard clamp하지 않으므로 fraction이 반드시0..1인 것은 아니다.
기본 loss는 noise MSE + VB다. VB는 t>0에서 Gaussian posterior KL,
t=0에서 decoder likelihood 항을 쓴다.
VB를 계산할 때 epsilon_hat을 detach해 variance 학습이 mean prediction head에 직접
간섭하지 않도록 한다. variance head로 연결된 공유 trunk의 gradient까지 끊는 것은 아니다.
gradient_routing.png에서 출력8채널에 도달하는 MSE/VB gradient를 비교한다.

여기서 learned variance, hybrid objective 자체를 DiT만의 새로운 기법으로 소개하지 않는다.
원본은 이전 diffusion 구현의 이 요소를 Transformer backbone과 함께 사용한다.
`training_losses` 내부를 읽을 때 `model_output`이 epsilon 부분을,
`model_var_values`가 나머지4채널을 가리키도록 split된 시점을 확인한다.
VB의 t=0 likelihood는 원본 diffusion objective의 계산이며 VAE를 다시 학습시키는 항이 아니다.

## 06 — 진짜 학습과 재개

숫자 이미지 → frozen VAE posterior → random t/noise → DiT → MSE+VB → backward
→ AdamW → EMA 순서다. optimize 대상은 DiT이며 VAE는 업데이트하지 않는다.

mini는4 blocks/width192/6heads인 lesson 크기다. S/B/L/XL은 원본 profile과 같다.
학습률1e-4, weight_decay0는 원본 설정을 사용한다.
원본 EMA는0.9999이고 S/B/L/XL에서는 그대로 쓴다. mini 기본은 짧은 실습을 위해0.999로
시간척도를 줄인다. 원본 EMA를 비교하려면 --ema-decay 0.9999와 별도 --out을 지정한다.
단일 GPU, mini/MNIST, batch128, 샘플링 방식과 검증 절차는 이 lesson 선택이다.
훈련 sample은 replacement로 뽑고, 의미가 바뀌는 숫자 flip은 하지 않는다.

100step마다 같은 seed/label/t/noise로 검증하고 samples/100.png처럼 이미지를 저장한다.
loss 검증에는 validation cache의 고정128개를 사용한다. 전체1,000개 평균이나 FID가 아니다.
이미지 윗줄은 현재 model, 아랫줄은 EMA이며 같은 seed로 생성한다.
curves.png는 학습 MSE/VB와 고정 validation의 model/EMA loss를 보여준다.
model은 현재 weight, EMA는 과거 weight의 지수이동평균이다.
EMA decay0.9999는 대략10,000 update의 시간척도라 짧은 실습에서 현재 model보다
생성 품질이 늦게 개선될 수 있다. 반드시07의 --weights model과도 비교한다.

    CUDA_VISIBLE_DEVICES=4 python 06_train.py --resume checkpoints/mini/latest.pt --steps 40000

latest는 optimizer와 RNG를 포함하고 best는 EMA validation loss 기준이다.
loss 최소가 시각적 품질 최고를 보장하지 않으므로 이미지도 함께 본다.
설정/데이터/원본 source가 다르면 resume을 거부한다. --steps는 누적 목표다.
EMA 설정 변경은 새 학습 실험으로 분리한다.
기본20,000 step은 실습의 시작 설정이다. 완성된 생성 품질이나 class 정확도를 보장하는
수치가 아니다. 작은 모델에서도 low-noise 복원, class 구분, EMA 수렴이 서로 다른 속도로
진행하므로07/08 이미지를 보고 학습을 이어갈지 판단한다.

## 07 — sampler와 network를 구분한다

network는 현재 z_t,t,y를 받아 epsilon/variance parameter를 반환한다.
sampler가 이를 사용해 z_(t-1)을 계산한다. sampling step마다 전체 network를 호출한다.
DiT block28개를 썼다고 sampling28step을 실행하는 것은 아니다.

DDPM은 predicted posterior mean/variance로 다음 sample을 뽑는다.
DDIM eta0는 같은 시작 noise에서 결정론적으로 이동하며 learned variance를 쓰지 않는다.
step을 줄일 때는1000-step schedule의 alpha_bar를 보존한 새 beta들을 만들고,
원래 timestep을 network에 전달한다. respace.py의 mapping을 꼭 읽는다.

trajectory.png/gif는 각 단계의 predicted clean z0를 decode한다.
초기의 그럴듯하지 않은 예측과 마지막 생성 결과가 어떻게 변하는지 관찰한다.
이것은 noisy z_t 자체를 decode한 그림과 다르다.

`explain_reverse_step`은 한 시점의 원본 계산을 직접 풀어 쓴다.

    z0_hat = (z_t - sqrt(1-alpha_bar_t)*epsilon_hat) / sqrt(alpha_bar_t)
    mean = posterior_coef1(t)*z0_hat + posterior_coef2(t)*z_t
    std = exp(0.5*log_variance)
    DDPM: z_previous = mean + std*fresh_noise

마지막 t=0에서는 fresh_noise 항을0으로 만든다. `reverse_step.png`의 mean/std는
이 reverse Gaussian의 파라미터다. RGB값이 아니며 각 그림의 color scale도 독립적이다.
`reverse_step_check.json`에서 수동 계산이 `p_mean_variance`와 일치하는지 확인한다.
실제 respaced sampler에서는 같은 원리를 **새 beta schedule**에 적용한다.

DDIM eta=0의 핵심은 다음과 같다. s는 다음으로 선택한 더 작은 원래 timestep이다.

    z_s = sqrt(alpha_bar_s)*z0_hat + sqrt(1-alpha_bar_s)*epsilon_hat

따라서 DDPM의 learned std 대신 같은 noise 방향을 이용하는 결정론적 업데이트를 한다.
네트워크 구조/가중치를 바꾸지 않고 sampler를 바꿀 수 있는 이유다.
sampling에서는 원본처럼 `clip_denoised=False`다. VAE latent는 RGB처럼[-1,1]로
제한된 값이 아니므로, 추정한 clean latent를 그 범위로 잘라 버리면 의미가 달라진다.

## 08 — classifier-free guidance

    epsilon_guided = epsilon_uncond
                  + scale * (epsilon_cond - epsilon_uncond)

같은 network에 label과 null label을 넣어 두 예측을 얻는다.
scale1은 conditional, scale0은 unconditional, 큰 값은 차이 방향을 외삽한다.
cfg_sweep.png는 같은 initial noise에서 class/scale만 바꾼다.
같은 seed여도 batch에서 서로 다른 noise를 주면 class만 바꾸는 실험이 아니므로,
이 파일은 각 class에 동일한 시작 noise를 명시적으로 재사용한다.

원본 forward_with_cfg는 재현 관례상 앞3개 epsilon 채널에만 CFG를 적용한다.
latent는4채널이다. RGB3채널이라서 그렇게 해야 한다는 일반 원리가 아니다.
all 모드는 epsilon4채널 모두에 적용하며 별도 실험으로 표시한다.
variance head는 guide하지 않는다. 원본 pretrained를 확인할 때는 원본 관례를 사용한다.
이번 mini의 고정 noise 비교에서도 큰 CFG가 획을 끊거나 형태를 과장하는 모습이 보인다.
조건을 강하게 주는 것과 좋은 이미지를 만드는 것을 같은 의미로 취급하지 않는다.
07에서 직접 비교하려면 `--cfg 2 --cfg-mode all --out outputs/07_cfg2`를 붙인다.

## 09 — scaling의 계산적 근거

T=token 수, D=width, L=depth일 때 한 block의 주요 MAC은 다음과 같다.

    QKV/output projections: 4 T D²
    MLP (ratio4):           8 T D²
    attention QK/AV:        2 T² D
    condition modulation:  6 D²

patch를 줄이면 parameter 수는 거의 유지되지만 token과 계산량은 크게 증가한다.
XL/2의 약118.6G MAC, XL/8의 약7.39G 차이를 실제 원본 parameter 수와 함께 확인한다.
FMA를1 unit으로 세는 관례다. FLOPs를2배로 세는 도구와 직접 비교하지 않는다.
depth_vs_steps.png는 모델을 키우는 것과 sampler를 오래 돌리는 것이 서로 다른 축임을 보여준다.
이 그래프에는 FID를 임의로 붙이지 않는다. 논문의 scaling 품질 실험은 논문 Fig.2/5/8과 비교한다.

## 10 — 실제 원본 모델로 연결

00~09에서 사용하는 mini는 직접 학습을 빠르게 관찰하기 위한 작은 모델이다.
10은 배포된 XL/2의 약675M parameter, latent32×32, patch2, token256,
block28, width1152, head16을 그대로 사용한다. ImageNet의1000 class 조건이다.
50-step 결과는 빠른 검사이고 원본 기본 sampling은250-step DDPM이다.
VAE-EMA는00과 공유하며 원본 sample.py의 기본 VAE-MSE와 차이를 run.json에 남긴다.
이미지 품질을 확인했다고 논문 FID50K를 재현했다고 말하지 않는다.

## 스스로 설명할 수 있어야 할 질문

1. RGB, latent, patch token, epsilon prediction 각각의 shape는?
2. timestep embedding과 positional embedding은 서로 무엇을 알려주는가?
3. shift/scale/gate가 각각0이면 block은 어떤 함수를 계산하는가?
4. VB에서 epsilon을 detach해도 공유 backbone이 학습되는 이유는?
5. null label은 어디서 배우며 inference에서 dropout을 꺼야 하는 이유는?
6. output head의8채널과 VAE의4채널은 어떻게 연결되는가?
7. 50 sampling steps와4 DiT blocks를 쓰면 DiT 전체를 몇 번 호출하는가?
8. EMA decay를 그대로 둔 채 학습을 크게 줄이면 어떤 일이 생기는가?
9. patch 크기 변화가 parameter 수와 attention 메모리에 미치는 영향은?
10. Wan의 시간 축/RoPE/cross-attention은 원본 DiT의 어느 부분을 확장하는가?

학습 이후03을 trained checkpoint로 다시 실행하고07의 model/EMA,08의 CFG를
같은 seed로 비교하면 각 계산의 효과를 실제 결과와 연결할 수 있다.
원본 출처/수정 범위는 [NOTICE.md](NOTICE.md), hash는 source_manifest.json에 있다.

여기서 직접 학습하는 것은 원본의 class-conditioned epsilon/variance prediction이다.
비디오 시간축, text cross-attention, RoPE, flow-matching objective를 이 코드에
포함했다고 해석하지 않는다. 그런 확장을 읽을 때도 먼저 입력 token, 조건 주입,
예측 target, sampler 중 어디가 달라졌는지 구분하면 된다.

## 준비하면서 실제 확인한 범위

검증은 기존 elite 환경, GPU4에서 수행했다. CPU로 충분한 연산 진단은 CPU에서 실행했다.

- 00: MNIST train10,000/validation1,000 posterior cache와 VAE 복원 이미지를 생성했다.
- 01~05: noising, patch projection/unpatchify, attention, 초기화/gradient,
  variance 분기의 detach를 원본 계산과 비교했다.
- 06: mini를 실제20,000 optimizer steps 학습했다. EMA0.999, batch128, lr1e-4.
  고정128개 validation의 EMA noise MSE는1.00068 → 0.07935,
  MSE+VB는1.02720 → 0.09456이었다. 생성 숫자는 구분되지만 일부 가는 획/형태 오류가 남는다.
- 저장/재개: 실제 CLI에서3step 연속 학습과2step + resume1step을 비교했다.
  model/EMA weight 최대 차이는 모두0이었다.
- sampling: DDIM50과DDPM250, CFG0/1의 식, 원본3채널 관례,
  원래1000-step timestep mapping을 확인했다.
- 09: 원본12개 profile의 parameter 수와 analytic 계산량을 확인했다.
- 10: 원본 DiT-XL/2 checkpoint로50-step DDPM 실제256×256 이미지4장을 생성했다.
  이 실행에서는250-step 생성/FID50K까지 평가하지 않았다.

실행 검증용 학습은 `.cache/verification/mini_ema/`에 두었다.
기본 `checkpoints/mini/`는 본인이00~06을 따라 학습할 때 만들어진다.
검증 실행의 저장 간격은500step, resume 이후2,000step이었고, 코드 기본은100step이다.
원본 EMA0.9999를10,000step 적용한 초기 비교 실행도 `.cache/verification/train/`에 있다.
그 설정의 짧은 학습에서는 EMA가 현재 model보다 크게 뒤처졌으므로, 작은 실습의
기본 EMA 시간을 줄인 이유를 비교할 수 있다.

바로 확인할 파일:

- [직접 학습한 mini의20,000step 이미지](.cache/verification/mini_ema/samples/20000.png)
- [mini 학습/validation 곡선](.cache/verification/mini_ema/curves.png)
- [생성 과정 GIF](outputs/07/trajectory.gif)
- [같은 noise에서 class/CFG를 바꾼 결과](outputs/08/cfg_sweep.png)
- [원본 DiT-XL/2의 실제 생성](outputs/10_fast/samples.png)
- [수식/gradient/재개 검사 결과](.cache/smoke/report.json)
- [전체 실행 검증 기록](outputs/verification.json)

위 데이터/결과는 로컬 검증 산출물이므로 Git에 포함하지 않는다. 새로 clone한 환경에서는
각 lesson을 실행해 생성한다. 직접 검증만 다시 하려면:

    python smoke.py
    # 00의 기본 cache를 준비한 뒤 실제 CLI 학습/재개까지 확인:
    CUDA_VISIBLE_DEVICES=4 python smoke.py --device cuda --training

생성 품질을 판단할 때는 loss와 이 이미지들을 같이 본다. 위 검증은 원본 ImageNet 학습이나
FID 재현을 대신하지 않는다.
