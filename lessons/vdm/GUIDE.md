# Video Diffusion Avatars — 5개의 이정표

이 lesson의 질문은 **한 장의 reference를 가진 video diffusion model에 표정·목 자세·카메라를 독립적으로 전달하려면 무엇을 학습해야 하는가?** 이다.
최종 결과물은 조건에 맞춰 생성한 영상이다. ELITE/FiCA처럼 Gaussian UV Map을 decode해 얻는 명시적 3D asset은 없다.

논문: [Video Diffusion Avatars, arXiv:2610.23772v1](https://arxiv.org/html/2610.23772v1)
공개 구현: [SimonGiebenhain/video-diffusion-avatars](https://github.com/SimonGiebenhain/video-diffusion-avatars)
확인한 revision: `8ba8b8f3e43c98df2e99fe46906cf011b10d34df` (2026-11-12 조회).

## 진행 순서

| 단계 | 직접 읽을 함수 | 확인할 핵심 | 결과물과 완료 기준 |
|---|---|---|---|
| [00 데이터·조건](00_data_and_proxy.py) | `ray_colors`, `pelvis_objective`, `inspect_manifest` | camera triplet, FLAME+MHR proxy, skeleton, local ray coloring | reference/proxy/skeleton/target 검사 이미지. 실제 crop·카메라·시간축 일치 |
| [01 표정 분리](01_expression_disentanglement.py) | `supervised_contrastive`, `cycle_loss`, `objective` | X-NeMo raw code의 viewpoint bias와 ex/cam factorization | `best.pt`, held-out reconstruction과 retrieval. encoder/entangler를 직접 학습 |
| [02 WAN 연결](02_connect_wan.py) | `build_pipeline`, `parameter_groups`, `encode_triplet` | translator, temporal face encoder, face fuser, zero-init proxy branch | 모듈별 학습 parameter 수, frozen VAE/CLIP cache, tensor 시간축 검증 |
| [03 Avatar 학습](03_train_avatar.py) | `flow_pair`, `conditions`, `predict_velocity`, `train` | camera-triplet rectified flow, frozen backbone의 gradient 전달 | stage1→stage2 adapter, 고정 validation JPG, `best/latest.pt` |
| [04 생성·개입](04_generate_and_intervene.py) | `generate`; 03의 `sample_latents` | expression/identity/camera 제어, Euler integration, 조건 ablation | 같은 seed의 원본/no-expression/freeze-expression/no-proxy 영상 |

각 단계에서 먼저 위 함수의 입력/출력 shape를 직접 적고, 그 다음 실행한다. 결과만 얻는 CLI는 마지막 확인 도구다.

## 무엇이 novelty인가

- **DISTRACT와 disentanglement (§3.1, App.H):** expression encoder를 새로 발명한 것이 아니라, X-NeMo/WAN code의 시점 누출을 드러내고 multi-view supervision으로 분리한다. 같은 `(identity, sequence, frame)`의 다른 camera는 expression positive다. 다른 frame의 같은 camera는 camera positive다.
- **Latent translation (§3.2):** X-NeMo expression을 WAN의 pretrained motion conditioning에 연결한다. `translator.f`는 `512→1024→1024→1024→512`, GELU. 논문의 T와 WAN entangler는 공개 구현에서 하나의 MLP로 합쳐져 있다. 별도 WAN disentangler를 main pipeline 필수 단계로 만들어 중복 학습하지 않는다.
- **Pose-aware proxy (§3.3, App.F):** FLAME head + SAM-3D-Body/MHR torso를 target camera에서 render한다. head와 torso 각각의 local frame에서 ray direction을 색으로 표현하므로 목 회전과 camera 이동을 구분할 수 있다. RGB proxy는 UV map도 normal map도 아니다. Plücker의 6채널 전체가 아니라 **direction 부분**을 쓴다.
- **Camera triplets (§3.4):** reference/driver/target camera를 다르게 뽑으면 driver code에 남은 시점 정보는 target을 설명하는 데 방해가 된다. 따라서 표정은 driver에서, 시점은 proxy에서 읽게 된다. 데이터 구성이 학습 신호의 일부다.

WAN-Animate의 VAE/DiT/temporal face encoder/face fuser와 rectified flow 자체는 기반 기술이다. Frozen DiT라는 이유로 `no_grad()`를 씌우면 앞쪽 adapter로 gradient가 전달되지 않는다. 02의 checkpoint wrapper와 03의 `model_fn` 호출을 함께 읽는다.

## 공개 코드와 논문의 차이 — 숨기지 않는 재구현 선택

| 항목 | 확인한 근거 | 이 lesson |
|---|---|---|
| Main training loop | 공개 repo에 없음 | Eq.5/Table7로 재구현. 저자 training code라고 부르지 않음 |
| X-NeMo expression head | 논문 Table7은 fine-tune; 공개 `TranslatorHead.pre`는 frozen | 기본 `--mode paper`는 train, `release`는 freeze |
| Reference slot | Eq.5는 reference concat/loss 제외; 공개 full sampler는 noisy slot0를 매 step 고정하지 않음 | 자체 학습/추론은 clean reference 고정. 공개 `.safetensors` 추론은 원본 sampler 사용. 서로 bitwise 동일하다고 주장하지 않음 |
| Translator 초기값 | MLP 구조 공개, main-training 초기값 recipe 미공개 | stock face modules 유지, 새 translator는 PyTorch 초기값. `--adapter`로 명시적 warm start 가능 |
| Sigma sampling / bell weight | bell-shaped w(sigma)만 명시 | uniform sigma, Gaussian bell width .25. config의 `implementation_choices` |
| CFG dropout | 공개 코드의 null/blur/proxy-absent 처리 확인, 확률 미공개 | 각 .1, blur8. 확률은 재구현 선택 |
| Optimizer betas / epoch 길이 | 구체값 미공개 | AdamW (.9,.999); disentangler epoch는 ceil(N/64) steps |
| XN cycle weight | Eq.19의 cycle 계수 .1; 실제 checkpoint의 별도 `cycle_contrastive_*` 값도 .1 | L1 cycle에는 .1을 한 번만 적용. 공개되지 않은 별도 cycle-contrastive 학습 절차까지 동일하다고 주장하지 않음 |
| Disentangler identity/residual | App.H는 일반 구조도 설명, deployed XN은 없음 | XN deployed 512/512/32, 3-layer ReLU width512 |
| Disentangler val | 논문은 identity 10% hold-out | 사용자 manifest에서 identity 분리 강제. 한 사람으로 identity 일반화 주장 불가 |
| DISTRACT | 수동 cross-identity pairing과 전용 distractor 구성이 필요 | retrieval proxy만 구현. 공식 DISTRACT score로 부르지 않음 |
| Pelvis fitting | 논문 400-step MHR FK 최적화; 공개 driver prep는 smoothing/canonical transfer | Eq.6 loss 제공. 실제 데이터용 MHR FK fitting/export 연결은 추가 준비 필요 |
| Streaming distillation | App.K는 prototype, fast weights/추론 공개 | 5단계 core는 full teacher. 4-step student 학습이나 3DGS probe까지 재현했다고 주장하지 않음 |

## 학습 대상과 시간축

Table7: face fuser 839M (lr5e-6), face encoder17.8M (1e-5), translator3.1M (1e-5), XN expression head.8M (1e-5), skeleton/proxy patch encoder 각.3M (5e-5).
DiT/VAE/CLIP/text encoder 및 raw X-NeMo image encoder는 고정한다. Adapter optimizer의 weights/moments는 FP32, forward는 BF16 autocast다. Frozen DiT와 trainable fuser는 gradient checkpointing한다. 추론용 CPU block offload hook을 학습에 재사용하지 않는다.

```text
reference image ─ VAE ─ 1 clean latent slot ─────────────┐
target video 77 frames ─ VAE ─ 20 latent slots ─ noise ─┼─ WAN DiT ─ velocity
skeleton + proxy 77 frames ─ VAE ─ patch embeddings ───┤
raw XN codes [1,77,512] ─ ex head ─ translator ─ face encoder
                           └ [1,20,5,5120] + zero ref slot
                                  └ face fuser every 5 DiT blocks
```

API의 `num_frames=81`은 **77 supervised frames + reference용 4-frame bookkeeping**이다. 81개의 실제 target frame을 넣는 뜻이 아니다. 20 video latent slot과 1 reference slot의 loss mask를 `flow_pair`에서 확인한다.

Stage1: 288×512,5000 steps. Stage2: 576×1024,250 steps. 둘 다 77 supervised frames/step, AdamW weight decay1e-3, gradient clip1.0. 논문의 34시간은 저자 데이터/장비/구현의 수치이며 이 lesson runtime 보장이 아니다.

## 환경과 source 준비

기존 conda `elite` 환경을 자동 변경하지 않는다. `.cache/upstream`에는 위 revision의 원본을 받아 두었다. 원본 source와 license를 유지하고 lesson에서 직접 import한다. source/weights/data/output은 모두 이 폴더의 `.gitignore`로 제외한다.

```bash
cd /media/vilab/mh/curious/lessons/vdm
python 00_data_and_proxy.py setup
# 원본 환경 스크립트를 먼저 읽는다. 사용자 선택 시 conda vda를 설치하는 공식 절차:
# (cd .cache/upstream && bash envs/setup_env.sh)
# conda activate vda
python 00_data_and_proxy.py download -- weights
python 00_data_and_proxy.py download -- base
python 00_data_and_proxy.py download -- examples
```

공식 환경은 Python3.12 / torch2.10+cu128이며 generation/preprocessing dependency 목록은 `.cache/upstream/envs/requirements_generate.txt`, `requirements_preprocess.txt`에 있다. CUDA compiler와 nvdiffrast/pytorch3d, FLAME2020, pixel3DMM, X-NeMo checkpoint가 필요하다. Driver body motion을 추정하려면 SAM-3D-Body/MHR도 별도 준비해야 한다. FLAME2023을 FLAME2020처럼 조용히 대체하지 않는다. 공식 installer의 SM architecture 설정은 H200 환경에 맞춰 확인한다.

`00 setup`은 source만 설치한다. 이번 사전 검증에서는 공개 VDA/X-NeMo 및 WAN base 자산 17개(약39GiB)를 `weights/`에 받았고 HF LFS SHA256과 대조했다. Converted DiffSynth mirror가401을 반환해서 `00 download -- base`는 공식 `Wan-AI/Wan2.2-Animate-14B`의 원본 `.pth` VAE/CLIP과 DiT shards를 받도록 했다. 02는 두 weight 형식을 모두 읽는다.

GPU4/H200에서 기존 `elite`(Python3.10,torch2.5.1+cu124)로 실제 전체 모델 실행을 확인했다. 부족한 import는 conda를 변경하지 않고 `.cache/runtime`에 추가했고 `common.upstream()`이 이 경로를 연결한다. 설치한 주요 항목은 `modelscope==1.38.0`, `modelscope-hub==0.7.1`, `ftfy==6.3.1`, `face-alignment==1.3.3`, `numba==0.60.0`, `llvmlite==0.43.0`이다. Face alignment는 공개 preprocessing requirements의 버전을 따른다. 이 경로는 Git에서 제외되므로 새 머신에서는 공식 환경을 준비해야 한다. Torch 버전이 다르므로 공식 환경과 bitwise 동일한 출력을 보장하지 않는다.

## 데이터 준비: 추론과 학습을 구분한다

**추론 입력**은 공개 전처리 그대로 만들 수 있다.

```bash
CUDA_VISIBLE_DEVICES=4 python 00_data_and_proxy.py reference -- --image /path/portrait.jpg --name person
CUDA_VISIBLE_DEVICES=4 python 00_data_and_proxy.py driving -- --video /path/driver.mp4 --name performance
CUDA_VISIBLE_DEVICES=4 python 00_data_and_proxy.py conditioning -- --ref person --driving performance \
  --camera over_the_top_zoom --max_frames 77 --out_dir data/cond/demo
```

**학습 입력**에는 같은 identity/clip의 동기화된 다른 camera 3개와 calibration이 필요하다. 한 카메라 영상을 복사해서 3개라고 표시해서는 안 된다. 현재 확인한 `lessons/elite/data/042`에는 이전 단일-view 준비 자료가 있고, 이번 lesson을 위한 calibrated multi-view triplet은 아직 준비되지 않았다. 공개 모델 weights와 예제 conditioning은 준비되어 있다.

실제 NeRSemble 전처리 순서는 다음과 같다.

1. identity/sequence별 train/validation 구간을 분리한다. 논문 일반화 검증은 identity hold-out, 한 사람 학습은 비중첩 sequence/time hold-out으로 제한해서 해석한다.
2. driver의 raw X-NeMo codes와 pixel3DMM FLAME tracking을 얻는다. Reference camera 이미지에서 FLAME shape를 얻는다.
3. dataset world의 head landmarks에 head를 정렬한다. SAM-3D-Body 결과는 MHR forward kinematics + Eq.6으로 pelvis anchoring한다. `pelvis_objective`는 이 최적화의 loss이며 공개 repo에는 원본 training-data builder가 없다. **이 fitting/export adapter는 현재 외부 준비가 필요한 경계**다. `00 build-triplet`은 그 이후의 calibrated rendering, target crop, manifest 생성을 연결한다.
4. 같은 dataset world의 target camera `c2w,K`로 official conditioning renderer를 실행한다. 공식 `conditioning.render`의 state에는 `codes`, `z_expression/jaw/eyes/neck/eyelids`, `gtw_lm`, `verts`, `kp` 등이 필요하다. inference용 canonical-person state를 그대로 dataset target RGB에 붙이면 좌표계가 맞지 않는다.
5. Target RGB와 reference를 동일한 9:16 crop으로 저장한다. K에는 crop/resize transform을 같이 적용한다. proxy/skeleton/target 모두 같은 77프레임,같은 fps여야 한다. 공식 renderer의 native calibration canvas는 1100×1604, conditioning은720×1280이다.

정렬된 state가 준비되면 `00 build-triplet --spec data/clip_spec.json --out data/triplets/042_free_a`로 실제 학습 clip을 만들 수 있다. Spec은 아래 형식이며 경로는 spec 파일 기준이다. `camera.npz`는 `K[3,3]`, OpenCV `c2w[4,4]` 또는 timeline별 `[T,4,4]`, `image_size=[1100,1604]`를 담는다. Target 원본 해상도가 배수이면 K를 이 calibration canvas에 맞춘다.

```json
{"id":"042_free_a","identity":"042","sequence":"FREE","split":"train","start":0,"frames":77,"cameras":{"reference":"cam_A","driver":"cam_B","target":"cam_C"},"world_space":"dataset","body_fitting":"pelvis_anchored","state":"aligned_state.npz","camera":"camera.npz","reference":"reference_cache","target_video":"cam_C.mp4"}
```

`reference_cache`에는 공식 reference preprocessing의 `ref_916.png`와 `flame_shape.npy`가 있어야 한다. State와 target video는 같은 원본 timeline을 사용한다. Builder의 결과 `triplet.jsonl`들을 하나의 training manifest로 합칠 때는 conditioning/target 상대 경로를 새 manifest 위치에 맞춘다.

학습용 JSONL manifest의 한 줄 (`data/triplets.jsonl` 기준 상대 경로):

```json
{"id":"042_free_a","identity":"042","sequence":"FREE","start":0,"frames":77,"split":"train","cameras":{"reference":"cam_A","driver":"cam_B","target":"cam_C"},"conditioning":"cond/042_free_a","target":"targets/042_free_a.mp4"}
```

`conditioning`에는 `ref.png, pose_proxy.mp4, pose_skel.mp4, codes.npy`가 필요하다. 별도 validation row도 있어야 한다. `start`는 **동기화된 원본 timeline의 frame index**이며 파일 자체는 이미 잘라둔 77-frame clip이다. 서로 다른 camera를 다른 split에 넣어 같은 시간대를 누출시키는 manifest는 거부한다.

```bash
python 00_data_and_proxy.py inspect --manifest data/triplets.jsonl
```

00의 이미지 검사는 길이/비율/선언된 split을 검증한다. Calibration, 실제 synchronization, face landmark 정렬의 정확성은 시각 검사와 dataset metadata로 확인해야 한다.

## 01 실행

각 camera video에서 frozen X-NeMo raw codes를 추출한다.

```bash
CUDA_VISIBLE_DEVICES=4 python 00_data_and_proxy.py extract-codes -- --video /path/camera_frames \
  --out data/codes/042_free_camA.npy
```

`data/codes.json`은 아래 row들의 JSON list다. validation identity도 포함한다. 서로 다른 identity에서 physical camera label이 실제로 동일한 rig 위치를 뜻하는지 확인한다.

```json
[{"codes":"codes/042_free_camA.npy","identity":"042","sequence":"FREE","camera":"cam_A","start":0,"split":"train"}]
```

```bash
python 01_expression_disentanglement.py pack --manifest data/codes.json --out data/codes.npz
CUDA_VISIBLE_DEVICES=4 python 01_expression_disentanglement.py train --codes data/codes.npz
```

실제 pack에는 expression batch32×4, camera batch8×16을 만들 수 있는 multi-view 자료가 필요하다. 부족하면 임의의 positive를 만들어 학습하지 않고 중단한다. 검증된 cross-identity expression pair가 있다면 `pack --verified-pairs pairs.json`로 class를 합친다. 수동 pairing이 없으면 identity-invariant expression 학습을 완전히 재현했다고 말할 수 없다.

## 02~04 실행

01을 직접 학습한 뒤 `checkpoints/01/best.pt` 사용. 공식 pretrained disentangler를 먼저 분석하려면 `weights/xnemo_disentangler.pt`를 명시적으로 선택한다.

```bash
CUDA_VISIBLE_DEVICES=4 python 02_connect_wan.py audit --disentangler checkpoints/01/best.pt
CUDA_VISIBLE_DEVICES=4 python 02_connect_wan.py cache --manifest data/triplets.jsonl \
  --size 288 512 --out data/cache_stage1
CUDA_VISIBLE_DEVICES=4 python 03_train_avatar.py --cache data/cache_stage1 \
  --disentangler checkpoints/01/best.pt --out checkpoints/03_stage1
# 재개: 동일 명령에 --resume checkpoints/03_stage1/latest.pt

CUDA_VISIBLE_DEVICES=4 python 02_connect_wan.py cache --manifest data/triplets.jsonl \
  --size 576 1024 --out data/cache_stage2
CUDA_VISIBLE_DEVICES=4 python 03_train_avatar.py --stage 2 --cache data/cache_stage2 \
  --disentangler checkpoints/01/best.pt --adapter checkpoints/03_stage1/best.pt \
  --out checkpoints/03_stage2

CUDA_VISIBLE_DEVICES=4 python 04_generate_and_intervene.py --cond data/cond/demo \
  --disentangler checkpoints/01/best.pt --adapter checkpoints/03_stage2/best.pt \
  --size 576 1024 --ablation all
```

Validation은 같은 held-out clip/seed/sigma(.25,.5,.75)로100step마다 수행한다. 기본으로20-step 생성 JPG도 저장한다 (`validation/0.jpg,100.jpg,…`). 비용이 크면 `--no-val-render`로 고정 flow validation만 남길 수 있다. Best는 held-out flow MSE 기준이며 실제 영상 품질과 반드시 함께 봐야 한다. Train loss의 감소나 몇 step smoke만으로 학습 성공을 판단하지 않는다.

`latest.pt`는 optimizer/RNG까지 포함, `best.pt`는 최저 validation 시점이다. `--adapter`는 가중치만 이어받는 명시적 warm start이고 `--resume`과 다르다. Data/config hash mismatch는 재개를 거부한다. Stage2는 stage1 best를 시작점으로 새 optimizer를 만든다.

공개 결과를 먼저 확인할 때:

```bash
CUDA_VISIBLE_DEVICES=4 python 04_generate_and_intervene.py \
  --cond .cache/upstream/examples/teaser_swimmer --adapter weights/vda_full.safetensors \
  --disentangler weights/xnemo_disentangler.pt --protocol release --size 576 1024
```

카메라 제어 검증은 reference/driver/seed를 고정한 채 00에서 camera만 바꿔 다시 conditioning을 만든다. 단순 no-proxy ablation은 완전한 camera invariance 평가가 아니다. Multi-view consistency를 공식 수치로 평가하려면 논문의 고정-expression camera sweep과 별도3DGS optimization/evaluation protocol을 추가해야 한다.

## 검증 범위

`smoke.py`는 exact upstream MLP checkpoint round-trip,contrastive/cycle gradient,synthetic optimization,ray-frame equivariance,reference loss exclusion을 확인한다. CUDA에서는 실제5120-wide face encoder와 face fuser 1개(전체8개 중)를 실행하고 raw code까지 backward를 확인한다. 전체 모듈 수는 meta tensors로 계산한다. 실제5120-wide frozen DiT block 1개도 gradient checkpointing으로 통과시켜, DiT 가중치에는 gradient가 없고 앞의 expression 입력에는 gradient가 있는지 확인한다.

```bash
CUDA_VISIBLE_DEVICES=4 python smoke.py --device cuda
```

`smoke.py` 자체는 전체14B 실행이나 학습 품질 검증을 대신하지 않는다. 이 모듈 검사 결과는 `.cache/smoke/report.json`에 기록한다.

### 2026-11-12 추가 사전 검증 — GPU4 / H200

모듈 검사에 더해 실제 pretrained backbone과 공개 conditioning으로 다음을 실행했다. **학습 테스트의 target은 reference를 반복한 합성 fixture**다. 실제 camera triplet, held-out 성능, 학습 완료 Avatar로 해석하지 않는다. Fixture는 `.cache/preflight/`에 격리했다.

| 구간 | 실제 확인한 것 |
|---|---|
| 00 | RGB 영상 입출력·crop 계약·manifest 검사, 시간 구간 누출/중복 camera 거부, 실제 CUDA rasterizer의 좌표 방향 |
| X-NeMo | 공식 pretrained image encoder + face-alignment1.3.3으로 실제 얼굴에서 유한한512차원 code 추출. 얼굴 검출 fallback0회 |
| 01 | contrastive 수식 대조, cycle 포함 backward, 작은 합성 데이터 학습, uninterrupted/resume 가중치 최대 차이0 |
| 02 | 실제 VAE/CLIP으로77프레임 cache를288×512와576×1024 모두 생성 |
| 03 전체 모델 | 40 DiT blocks + 8 face fusers. 학습 대상6개 그룹 모두 nonzero gradient와 실제 parameter update, frozen DiT weight에는 gradient 없음 |
| 03 저장/재개 | adapter/optimizer/RNG 저장·복원, 재로드 prediction 최대 차이0. 실제 CLI stage1에서1step→resume2step 완료 |
| 03 해상도 | 288×512·77프레임 backward 최대48.3GiB. Stage1 checkpoint로 시작한576×1024·77프레임 stage2 학습/validation/저장 최대71.4GiB. 별도 테스트 프로세스의 PyTorch peak allocated 기준 |
| 03 validation | 고정 seed/sigma flow 평가와20-step 생성 JPG 저장, validation 전후 RNG·train/eval 상태 보존 |
| 04 | 공개 adapter와 release sampler로77프레임 영상 생성·시각 확인. Stage2 테스트 checkpoint를 실제 CLI에서 재로드해 자체 sampler로도77프레임 생성. Seed 재현성, target 미참조, expression/proxy 개입에 대한 출력 변화 확인 |

실행 중 converted mirror 접근 오류, 공개 adapter의 미사용 `gates.*` 키 로딩 오류, cycle L1 계수 중복 적용을 수정했다. 공개 원본 source는 수정하지 않았다.

- 공개 모델 결과: [77프레임 영상](outputs/preflight/public_teacher/release.mp4), [첫·중간·마지막 프레임](outputs/preflight/public_teacher/preview.jpg). **우리 재학습 결과가 아니라 공개 pretrained 결과**다.
- 상세 기록: [.cache/preflight/report.json](.cache/preflight/report.json), 실제 전체 모델 검사: [.cache/preflight/full/report.json](.cache/preflight/full/report.json).
- 아직 확인하지 못한 범위: 실제 calibrated NeRSemble triplet, pixel3DMM/FLAME2020 + SAM/MHR raw preprocessing 및 pelvis fitting/export,5000+250step 수렴, 논문의 정량 평가와 distillation.

추론 예제와 네트워크·학습 경로를 공부할 준비는 되어 있다. 본격적인 사용자 데이터 학습의 다음 선행 작업은 동기화된 다중 카메라 RGB/calibration 확보와00의 dataset-world 정렬 경계를 연결하는 것이다. 합성 테스트의 대용량 optimizer checkpoint는 검사 후 제거하고 로그·결과·공개 weights만 남긴다.
