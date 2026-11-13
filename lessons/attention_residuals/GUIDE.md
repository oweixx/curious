# Attention에서 Attention Residuals까지 — 5개 lesson

목표는 **attention을 직접 계산하고, 같은 Transformer의 residual 연결을 바꾸어
학습·추론하며, 논문에서 새롭게 제안한 부분을 설명할 수 있게 되는 것**이다.

논문: [Attention Residuals, 2026](https://arxiv.org/abs/2603.15031).
[공식 저장소](https://github.com/MoonshotAI/Attention-Residuals)는 논문과 핵심 의사코드를 제공한다.
여기의 실행 가능한 모델과 학습 루프는 수식에 맞춰 작성한 독립적인 교육용 구현이다.
Kimi Linear의 대규모 LLM 사전학습을 재현한 코드는 아니다. 코드 출처와 범위는 [NOTICE.md](NOTICE.md).

## 무엇을 만들고 무엇을 보나?

| 단계 | 중심 질문 | 직접 실행할 결과 | 학습 여부 |
|---|---|---|---|
| [00_attention.py](00_attention.py) | Q/K/V, softmax, multi-head는 어떻게 읽고 섞는가? | RGB lookup, scaling/entropy, causal attention heatmap | 고정 숫자·무작위 초기값 |
| [01_residual_stream.py](01_residual_stream.py) | residual에 과거 정보가 어떻게 누적되는가? | 누적 RMS, 상대적 branch 크기, 정규화 감도 | 통제된 수치 실험 |
| [02_full_attnres.py](02_full_attnres.py) | 이전 **층**의 출력을 어떻게 선택하는가? | 학습 전후 depth selection, query, key 정규화 효과 | selector만 실제 학습 |
| [03_block_attnres.py](03_block_attnres.py) | 묶어서 저장하면 무엇을 잃고 무엇을 줄이는가? | block별 계수, source 저장 수, two-phase 동치 검사 | 수치·gradient 검사 |
| [04_train_and_compare.py](04_train_and_compare.py) | 모델 전체를 학습해도 같은 원리가 작동하는가? | 고정 예측 패널, loss, token/depth attention, gradient, checkpoint 추론 | 전체 모델 실제 학습 |

각 `.py` 맨 위에 실행 명령·결과 파일·관찰 질문이 있다.
네트워크를 읽을 때는 [models.py](models.py)의 `TokenAttention`, `DepthAttention`,
`RecallTransformer.forward` 순서로 보면 된다. `common.py`는 그림과 데이터 생성만 담당한다.

## 실행

기존 `conda elite`를 사용한다. 별도 venv, 외부 모델, 데이터셋 다운로드가 없다.
의존성은 PyTorch, NumPy, matplotlib, tqdm이다. 00~03은 CPU로 충분하다.
GPU 예시는 사용자 지정 **물리 GPU 7번**이며 해당 프로세스 안에서는 `cuda:0`이다.

```bash
conda activate elite
cd /media/vilab/mh/curious/lessons/attention_residuals

python 00_attention.py
python 01_residual_stream.py
python 02_full_attnres.py
python 03_block_attnres.py

CUDA_VISIBLE_DEVICES=7 python 04_train_and_compare.py train --device cuda
python 04_train_and_compare.py infer --checkpoint outputs/04/full/best.pt
python 04_train_and_compare.py infer --checkpoint outputs/04/block/best.pt --two-phase
```

기본 경로는 파일 위치 기준이고, 직접 지정한 상대 경로는 실행 디렉터리 기준이다.
기본 학습은 baseline/full/block을 차례로1000 step씩 수행한다.
`outputs/04/<variant>/0.png,100.png,200.png,...`가 고정 validation 이미지다.
step별 디렉터리를 만들지 않는다. `comparison.png`는 세 모델의 비교 그림이다.

추가 학습과 별도 실험:

```bash
# steps는 추가 step이 아니라 최종 도달할 총 step.
CUDA_VISIBLE_DEVICES=7 python 04_train_and_compare.py train --device cuda --resume --steps 2000
# CPU smoke. 결과는 정식 학습과 별도 경로.
python 04_train_and_compare.py train --steps 100 --out outputs/smoke
# 핵심 요소 ablation을 원하면 같은 구조/데이터 설정으로 실행.
CUDA_VISIBLE_DEVICES=7 python 04_train_and_compare.py train --device cuda \
    --variants static no_key_norm --out outputs/ablation
```

`last.pt`가 있으면 자동 덮어쓰지 않는다. 재시작 실험은 새 `--out`, 이어 학습은 `--resume`.
checkpoint는 model, optimizer, 데이터 generator 상태와 이력을 저장한다.
resume에서는 모델 크기·seed·batch·learning rate를 이전과 동일하게 지정한다.
LR은 50 step warm-up 후 일정하다. 긴 학습으로 재개해도 과거 LR schedule이 바뀌지 않는다.
`best.pt`는 고정 validation CE가 가장 작았던 상태이며 마지막 step과 다를 수 있다.
출력·가중치는 이 폴더의 `.gitignore`에 포함된다.

## 이 논문에서 새로운 부분

Q/K/V attention, residual connection, MLP, RMSNorm은 배경 지식이다.
이전 layer를 연결하는 dense connection이나 가중합 역시 그 자체로 최초 아이디어가 아니다.
이 lesson은 다음 설계와 그 이유에 집중한다.

1. **Depth-wise selection:** token 위치마다 이전 layer의 출력들을 softmax로 선택한다.
2. **Pseudo-query + key RMSNorm:** layer별 학습 벡터로 조회하고 입력 feature를 key로 쓴다.
   query는 현재 hidden으로부터 만들지 않지만, key가 바뀌므로 가중치는 입력 의존적이다.
3. **Block AttnRes:** 여러 branch를 하나의 합으로 저장하여 유지할 source 수를 줄인다.
4. **시스템 기여:** 완성된 block 조회를 모아 계산하고 현재 block 결과와 병합한다.
   분산 pipeline에서는 이미 보낸 block을 cache해 중복 전송을 줄인다.

논문 §3/Eq.1–6은02/03과 models.py, §4/Algorithm1은03의 two-phase 실험과 연결된다.
zero-query 초기화는 §5, content-independent mixing/key RMSNorm ablation은 §5.3에 나온다.
성능 수치의 대규모 재현과 pipeline/tensor parallel runtime, fused kernel은 구현 범위 밖이다.

## 00: 주소와 내용, 그리고 계산의 축

입력 `X[B,T,D]`에 learned projection을 적용해 Q/K/V를 얻는다.

```text
Q = X Wq, K = X Wk, V = X Wv
A[b,h,t,s] = softmax_s(Q[b,h,t] dot K[b,h,s] / sqrt(D_head))
Y[b,h,t] = sum_s A[b,h,t,s] V[b,h,s]
```

Q는 찾는 조건, K는 비교할 주소, V는 가져올 내용이다. 이 비유가 의미를 자동 보장하지는 않는다.
projection의 실제 역할은 loss를 통해 학습한다. 첫 RGB 예제는 이 구조를 숫자로 확인하는 것.
head마다 다른 projection을 쓰고 결과를 합쳐 output projection을 통과시킨다.
causal mask는 미래 token을 가린다. 00에서는 PyTorch SDPA와의 출력 일치 및 미래 정보 차단을 검사한다.

`scaling.png`에서 entropy가 작으면 몇 개의 항목에 가중치가 집중된 것이다.
항상 entropy가 낮은 것이 좋은 건 아니다. 이 그림은 random Q/K의 scale 현상만 보여준다.

## 01: residual은 왜 연구 대상이 되는가?

PreNorm sublayer는 보통 다음처럼 동작한다.

```text
h_next = h + F(RMSNorm(h))
h_final = embedding + branch_0 + branch_1 + ...
```

`branch_i`는 실제 layer 함수의 출력이며 이미 그 이전 입력에 의존한다.
대수적으로 펼친 식에서 명시적인 합산 계수는 모두1이다. implicit한 정보 변환은 F 안에서 일어난다.
정규화는 F의 입력에 적용되므로 바깥의 누적 stream까지 정규화한 것은 아니다.

01에서는 F 출력을 독립 random increment로 바꿔서 누적 효과만 관찰한다.
이 조건에서는 누적 RMS가 대략 sqrt(depth)에 비례하고 새 branch의 상대적 크기는 줄어든다.
RMSNorm 주변의 감도도 관찰하지만, 이 값은 residual identity 경로의 전체 gradient가 아니다.
이 실험만으로 깊은 Transformer가 반드시 실패한다고 결론 내리지 않는다.

## 02: Full AttnRes의 실제 연산

source 목록에는 embedding과 **각 branch의 출력**을 넣는다. 누적 hidden을 반복해서 넣지 않는다.

```text
V[S,B,T,D] = stack(embedding, branch_0, ..., branch_previous)
K = RMSNorm(V)                         # feature D축 정규화
q_l[D] = 학습 가능한 layer별 벡터
score[S,B,T] = dot_D(q_l, K)
alpha = softmax_S(score)               # source/depth축!
h[B,T,D] = sum_S(alpha * V)
branch_l = F_l(RMSNorm(h))
```

일반 token attention의 `Q=XWq`와 달리 q는 Parameter다.
source key 내용은 batch/token마다 다르므로 같은 q로도 서로 다른 alpha가 나온다.
여기서는 head를 나누거나 `sqrt(D)`로 나누지 않는다. 마지막 출력 head 앞에도 mixer가 있다.
ATTN 앞 mixer와 MLP 앞 mixer는 별도의 query와 key RMSNorm을 가진다.

q는0으로 초기화한다. 첫 alpha는 source별 균등 가중치이고 output은 평균이다.
**초기 AttnRes의 함수가 baseline residual sum과 똑같다는 뜻은 아니다.**
key RMSNorm은 큰 source가 크기만으로 score를 지배하는 것을 줄인다.
V는 정규화하지 않으므로 output 크기까지 source scale에 불변인 것은 아니다.

02는 marker가 있는 source를 선택하도록 query와 key norm을 학습한다.
따라서 learning.png는 실제 optimizer update 결과지만, 전체 Transformer 학습은04에서 한다.
key_normalization.png는 별도의 고정 query/source로 크기만 바꾸는 통제 실험이다.
정규화 유무로 학습 성능이 얼마나 달라지는지는04의 no_key_norm ablation에서 관찰한다.

## 03: Block AttnRes와 추론 계산

`depth=6`이면 ATTN/MLP sublayer는 L=12개다.
`block_size=4`이면 branch를4개씩 합쳐3개의 block representation을 만든다.
embedding은 이들과 별도 source다.

```text
completed = [embedding, complete_block_1, ...]
partial = 현재 block에서 이미 계산한 branch들의 합
sources = completed + [partial]       # partial이 존재할 때만 추가
h = depth_attention(sources)
partial = partial + F(h)
```

완성된 block을 다음 block의 partial에 다시 더하지 않는다.
block 내부의 source별 구별은 합 안에 압축되며 이후 mixer는 block 전체에 하나의 계수를 준다.
`grouping.png`가 원래 branch들에 적용되는 동일한 block 계수를 보여준다.
block별 alpha의 합은1이지만, block 합을 개별 branch로 펼친 계수들의 합은1일 필요가 없다.
`block_size=1`은 Full과 일치한다. `block_size=4`는 다른 모델이다.
`block_size`가 전체 layer 수를 넘는다고 baseline과 수치적으로 같아지는 것은 아니다.
embedding이 분리되어 있고 학습되는 softmax 혼합이 남기 때문이다.

추론의 two-phase는 **같은 Block 모델을 계산하는 순서**를 바꾸는 방법이다.

1. 현재 block에 속한 여러 layer의 q로 이미 완성된 block들을 한꺼번에 조회한다.
2. 아직 계산되지 않은 partial sum은 layer를 진행하면서 순서대로 조회한다.
3. 두 결과를 softmax 분모까지 고려해 병합한다. output끼리 단순 합산하지 않는다.

03은 nonzero query, 서로 다른 RMSNorm scale, 마지막 불완전 block까지 검사한다.
Full↔size1 Block, naive↔two-phase의 **출력과 모든 parameter gradient**를 비교한다.
실제04의 `--two-phase` 추론도 같은 함수를 사용한다.

`source_cache.png`는 표현 저장 수다. B/T/D와 dtype bytes를 곱하면 그 표현의 용량을 얻는다.
이 수치를 전체 학습 VRAM 감소로 해석하면 안 된다. autograd 저장과 token KV cache는 별도이며,
원논문도 작은 일반 학습과 pipeline parallel 상황을 구분한다.

## 04: 실제로 무엇을 학습하는가?

과제는 매번 새로운 key-color 표에서 query의 색을 찾는 것이다.

```text
입력: (C, red), (A, blue), (B, green), query=A
정답: blue

key/color embedding + position
    -> [token attention, MLP] 반복 (residual 방식만 교체)
    -> 마지막 query token -> 8색 classifier -> cross entropy
    -> backward -> AdamW.step
```

query token의 색은 항상 unknown ID=8이다. 정답 색을 query에 주지 않는다.
key-color 관계와 key 순서는 매번 바뀐다. 따라서 고정된 A→blue 관계만 암기해서 풀 수 없다.
이 과제에서는 embedding, Q/K/V/output projection, MLP, norm, classifier,
AttnRes query/key norm이 함께 학습된다. static ablation에서는 source별 고정 학습 logits를 쓴다.

모델 세 개는 backbone 초기값, batch 순서, loss, optimizer, LR를 공유한다.
추가 mixer 때문에 parameter 수는 약간 다르며 comparison.json에 실제 수를 저장한다.
train은 매번 생성한 표, validation은 독립 seed의 고정512개 표를 사용한다.
추론은 별도 seed의2048개 표로 평가한다. validation을 보고 선택한 best 모델을 쓴다는 점도 기록한다.

`0.png`와 `100.png`를 같은 위치끼리 비교한다.

- 위 왼쪽: 고정 memory 표 / query / 정답 색 / 예측 색. x는 오답.
- 위 오른쪽: 실제 minibatch CE와 고정 validation CE.
- 아래 왼쪽: query token이 context의 어느 token을 읽는지. head 평균이라 개별 head 차이는 숨겨진다.
- 아래 가운데: 같은 query token이 어떤 이전 layer/block을 읽는지. 마지막 행은 출력 head mixer.
- 아래 오른쪽: sublayer 입력과 branch 출력의 RMS. 빨간 선은 고정16개 probe의 입력 gradient RMS.

validation gradient는 진단에만 사용하고 optimizer를 업데이트하지 않는다.
학습 후 CE가 거의0이면 gradient도 작아지는 것이 자연스럽다. 절대 크기만 보고
vanishing gradient라고 판단하지 말고, 같은 probe/step의 loss와 층별 분포를 함께 본다.
`infer`는 eval + no_grad로만 실행하여 inference 그림에는 gradient 선을 그리지 않는다.
추론에서 AttnRes query도 업데이트하지 않는다. 학습된 query와 새 입력의 key로 alpha만 새로 계산한다.
depth cache는 이 forward 안의 이전 layer 표현이며 이전 영상/이전 호출의 장기 기억이 아니다.

작은 lookup 과제는 세 모델 모두 빠르게 높은 정확도에 도달할 수 있다.
그 결과는 코드와 학습 경로가 작동한다는 증거이며 대규모 논문 우위의 증명은 아니다.
관찰된 우열을 미리 정해 두지 않는다. step 시간은 이 구현에서 측정한 값이며 fused kernel 성능이 아니다.
후속으로 DiT에 옮길 수 있지만, 이5단계에는 diffusion 학습까지 섞지 않는다.

## 구현 검증 기록

2026-11-13, 기존 conda elite / PyTorch2.5.1+cu124에서 확인했다.
00~03은 CPU, 아래 전체 모델 학습·추론·재개 검증은 물리 GPU7에서 수행했다.

- Q/K/V 직접 계산과 PyTorch SDPA 일치, causal mask의 미래 정보 차단.
- 02의 selector가 고정 marker validation에서100% 선택 정확도에 도달.
- Full과 size1 Block의 출력·gradient 일치. 여러 block 크기의 two-phase 최대 오차는
  float64 출력 약7e-16, gradient 약5e-15 이내였다.
- baseline/full/block 각각1000 step 학습, validation512개 및 별도 추론2048개에서 모두100%.
  쉬운 인공 과제에서 학습 경로를 확인한 결과이며, 논문 성능 우위를 검증한 것은 아니다.
- 중단 후 재개와 연속 학습의 parameter·loss·데이터 RNG 일치.
- static/no_key_norm ablation의 forward/backward, checkpoint 저장 확인.

미리 실행한 결과는 `outputs/verification/`에 있고 정식 실행 기본값은 `outputs/04/`다.
저장된 검증 checkpoint를 먼저 관찰하려면:

```bash
python 04_train_and_compare.py infer --checkpoint outputs/verification/full/best.pt
python 04_train_and_compare.py infer --checkpoint outputs/verification/block/best.pt --two-phase
```

## 마친 뒤 설명해볼 질문

1. token attention과 depth attention의 softmax 축 및 query 생성 방식은 어떻게 다른가?
2. source 목록에 누적 hidden을 넣으면 왜 논문의 계산과 달라지는가?
3. query가0인 초기 상태와 학습 후의 alpha는 어떻게 달라지는가?
4. Block 압축으로 사라지는 구분 능력과 줄어드는 저장 항목은 무엇인가?
5. 모델의 Full→Block 변경과 계산 순서의 naive→two-phase 변경은 왜 다른가?
6. 실제 학습 결과에서 어느 것이 관찰이고 어느 것이 일반화할 수 없는 가정인가?
