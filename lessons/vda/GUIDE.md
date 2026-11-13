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
| Disentangler val | 논문은 identity 10% hold-out | `--protocol paper`는 identity 분리 강제. 현재042 실습은 명시적 `single-subject`: 같은 사람의 비중첩 시간 검증, 3-camera batch. identity 일반화 주장 불가 |
| DISTRACT | 수동 cross-identity pairing과 전용 distractor 구성이 필요 | retrieval proxy만 구현. 공식 DISTRACT score로 부르지 않음 |
| Pelvis fitting | 논문 400-step MHR FK 최적화; 공개 driver prep는 smoothing/canonical transfer | `preprocess.fit_body`에서 실제 MHR FK, 공유 Sim(3), continuous rotation prior와 400-step Adam 구현. face 대응점/초기화는 lesson 선택이며 미공개 저자 solver와 동일하다고 주장하지 않음 |
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

**지금 00에서 시작할 명령**은 공개 예제의 입력 비교다. GPU나 학습 spec 없이 실행된다.

```bash
python 00_data_and_proxy.py inspect-conditioning \
  --cond .cache/upstream/examples/teaser_swimmer --out outputs/00_conditioning.jpg
```

첫·중간·마지막 프레임의 reference/proxy/skeleton을 비교하고 raw code shape를 출력한다.
이 예제에는 실제 학습 target이 없으며, 명령은 학습용 triplet을 생성하지 않는다.
실제 학습 데이터는 별도 **`preprocess.py`**로 준비한다. `python preprocess.py export`가
검사된 실제 state의 경로로 `clip_spec.json`을 만든다. 그 다음 00의 `build-triplet`을 실행한다.

**현재 042의 다음 단계:** `preprocess.py export`와 `build-triplet`을 마쳤으면
`reference`/`driving`을 다시 실행하지 않는다. 이미 만들어진77프레임 clip을 확인한다.

```bash
python 00_data_and_proxy.py inspect \
  --manifest data/triplets/042_free_a/triplet.jsonl --out outputs/00
```

공개 `reference`/`driving` CLI는 다른 단안 입력을 처리할 때 쓰는 별도 경로다.
옵션은 `python 00_data_and_proxy.py driving -- --help` 등으로 확인한다.
`driving`은 원본 영상 전체를24fps로 바꿔 tracking하므로, 현재77프레임 학습용
전처리의 후속 명령으로 사용하지 않는다. `conditioning --max_frames 77`도 앞 단계
tracking의 길이를 제한하지 않는다.

**`042_free_driver`의 Loss NaN 오류:** 원본 `cam_222200042.mp4`의460프레임을
별도로 단안 tracking하다가 loss가 NaN이 된 뒤, 최종 시각화에서 NaN 좌표를 정수로
변환하며 실패했다. `cv2.circle`에서 좌표만 건너뛰어 해결할 문제가 아니다.
해당 실패 경로는 `.cache/preprocess/driving/042_free_driver/`이며 유효한 최종
checkpoint가 없다. `data/preprocessed/042_FREE/`의77프레임 calibrated state와
`data/triplets/042_free_a/`는 별도 결과다. 이77프레임 state의 유한값과 정렬 report를 확인했다.
첫 발산 연산은 기존 실행 로그만으로 확정할 수 없으므로 단안 tracking 전체가
수정·검증됐다고 간주하지 않는다.
GPU4에서 첫3프레임을 별도 출력 경로로 재실행했다. 일반 실행과 `torch.compile`
실행 모두 online200회/프레임 + global10회 및 유한한 checkpoint 저장을 통과했지만,
원래460프레임 실행의 NaN을 이 짧은 구간에서는 재현하지 못했다.
진단 기록: [.cache/nan_diagnosis/report.json](.cache/nan_diagnosis/report.json).

`00 reference/driving`은 이제 `.cache/logs/00_<stage>_<timestamp>.log`에 전체 로그를
남기고, `Loss NaN/Inf` 출력 시 해당 실행의 하위 tracking 프로세스도 함께 중단한다.
`preprocess.py`의 외부 단계에도 같은 중단 검사를 적용하며, FLAME parameter의
NaN/Inf를 확인해 잘못된 완료 결과를 저장하거나 재사용하지 않는다.

**validation tracking의 원인과 수정:** `042_FREE_val`의26번 프레임에서 다시 발생한
NaN을 별도 재현했다. 재현 실행의27번 프레임/101번째 iteration에서 loss와 parameter는
유한했지만, 카메라 평면에 걸린 일부 vertex의 투영이 NaN/Inf였다. 이 vertex가 UV
correspondence에서 제외되어 UV loss 값은0이어도, 전체 투영의 backward에서 `0 × NaN`이
전파되어 파라미터 gradient가 오염됐다.

`preprocess.py:stable_tracker_projection`은 공개 투영식의 분모에 부호를 유지하는
`1e-4 m` 하한을 적용한다. 렌더러의 near plane(`0.1 m`)보다 훨씬 안쪽에 있는
카메라 평면 부근만 바꾸며, 원래 카메라 깊이는 visibility 판단에 그대로 전달한다.
공개 loss·학습률·iteration 수는 유지하고, optimizer update 직전 gradient도 검사한다.
외부 checkout을 수정하지 않고 `_track` 실행 진입점에서 이 함수를 연결한다.
`00 reference/driving`도 같은 진입점을 사용한다.

실패 직전 동일 parameter로 재생했을 때 total loss `8816.1376953125`는 유지됐고,
기존 UV loss의 NaN gradient가 유한한0으로 바뀌었으며 모든 loss의 gradient가 유한했다.
카메라 평면에서 떨어진 점에서는 원본과 forward/gradient가 동일함도 확인했다.
이 검사는 수치 오류 수정의 근거이며 원본460프레임 단안 tracking 전체의 품질 검증과는 다르다.
진단 파일은 `.cache/val_nan_diagnosis/`에 보관한다.
합성 vertex로 같은 문제를 다시 검사하려면
`CUDA_VISIBLE_DEVICES=4 python smoke.py --device cuda --tracking-projection`을 실행한다.

수정 후 GPU4에서 `042_FREE_val`의 **77프레임 online tracking + global 5,000회**를
끝까지 실행하고, calibrated alignment/export와 lesson00 triplet 생성까지 확인했다.
최종 state의 모든 수치가 유한하며 얼굴 landmark 재투영 오차는 중앙값5.18px,
95백분위22.19px, 몸통-얼굴 대응점 오차 중앙값은7.70mm다. 정렬 이미지는
`data/preprocessed/042_FREE_val/alignment.jpg`, 학습 입력 비교는
`outputs/00/042_free_000240_222200036.jpg`에 있다. 이는 전처리의 정렬 검사 결과이며
최종 avatar 생성 품질을 뜻하지 않는다. 상세 기록: `.cache/val_nan_diagnosis/report.json`.

**학습 입력**에는 같은 identity/clip의 동기화된 다른 camera 3개와 calibration이 필요하다. 한 카메라 영상을 복사해서 3개라고 표시해서는 안 된다. 실제 raw 데이터는 `data/042`에 있으며, 기존 ELITE의 `222200037` 영상도 상대 symlink로 연결돼 있다. 현재 기본 reference/driver/target은 각각 `222200037`, `222200042`, `222200036`이다.

```bash
python 00_data_and_proxy.py inspect-nersemble \
  --subject-dir data/042 --sequence FREE --out outputs/00_nersemble
```

이 명령은 원본의 해상도/FPS/프레임 수와 샘플 프레임을 검사하고, `first.jpg`,
`middle.jpg`, `last.jpg`, `report.json`을 만든다. `cameras/cam_*.npz`에는 실제
calibration을 `c2w=inverse(world_2_cam)`, `K_native=diag(.5,.5,1) @ K_source`로
변환해 저장한다. 원본2200×3208에서 renderer의1100×1604 canvas로 바꾸는 것이며,
9:16 crop은 이후 renderer와 RGB에 동일하게 적용해야 한다. 원본은73FPS이므로
tracking/codes/target을 공통24FPS 시간축으로 맞추는 전처리도 필요하다.

FLAME2020 준비는 2026-11-13 완료했다. 사용자 제공 `data/assets/FLAME2020.zip`의
전체 CRC를 검사하고, 필요한 `generic_model.pkl`과 `Readme.pdf`를
`data/assets/FLAME2020/`에 추출했다. 공개 installer가 지정한 MICA revision
`af22e7a5810d474bc28a1433db533723d6bd2b07`의 landmark/mask 자산을 사용하며,
pixel3DMM이 읽는 `preprocessing/MICA/data/FLAME2020/generic_model.pkl`은 이 모델의
상대 symlink다. 원본 ZIP과 공개 source는 보존한다.

기존 conda를 변경하지 않고 `.cache/runtime`에 `environs`와 필요한 의존성을
추가하고 vendored `pixel3dmm` package를 연결했다. GPU4에서 실제 VDA의
`geometry.flame20/neutral20/fwd20` 경로로5023개 vertex·9976개 triangle·68개
landmark 생성, jaw 변형 및 expression gradient를 확인했다.
기록은 [outputs/00_flame/report.json](outputs/00_flame/report.json)에 있다.
이 기록은 FLAME 모델 연결에 한정한다. 아래 새 전처리의 검증 범위는 별도로 구분한다.

### 실제 전처리 실행

구현과 실행 진입점은 [preprocess.py](preprocess.py)다. 원본 영상 전처리는 lesson 밖에서 진행한다.
기존 00은 준비된 geometry로 camera triplet, proxy coloring, skeleton conditioning을 배우는 단계로 유지한다.

```bash
cd /media/vilab/mh/curious/lessons/vdm
python preprocess.py check
# 공개 pixel3DMM/PIPNet/facer source와 가중치 준비. 현재 준비 완료된 파일은 재사용.
python preprocess.py assets

python preprocess.py frames
CUDA_VISIBLE_DEVICES=4 python preprocess.py landmarks
CUDA_VISIBLE_DEVICES=4 python preprocess.py face
CUDA_VISIBLE_DEVICES=4 python preprocess.py codes
CUDA_VISIBLE_DEVICES=4 python preprocess.py body
CUDA_VISIBLE_DEVICES=4 python preprocess.py align
CUDA_VISIBLE_DEVICES=4 python preprocess.py export

# 전처리 완료 후 기존 lesson 00: proxy/skeleton/target RGB를 묶어 학습 clip 구성
CUDA_VISIBLE_DEVICES=4 python 00_data_and_proxy.py build-triplet \
  --spec data/preprocessed/042_FREE/clip_spec.json --out data/triplets/042_free_a
```

기본 작업 폴더는 `data/preprocessed/042_FREE`다. **SAM 가중치 접근 승인 전에는
`body`부터 실행할 수 없다.** [SAM 모델 페이지](https://huggingface.co/facebook/sam-3d-body-dinov3)에서
승인 및 HF 로그인을 마친 뒤 `python preprocess.py assets --download-sam`을
실행한다. 기대 경로는 `weights/third_party/sam-3d-body-dinov3/{model.ckpt,model_config.yaml,assets/mhr_model.pt}`다.
이미 다른 곳에 있으면 `body --sam-weights /실제/폴더`로 지정할 수 있다.
SAM source는 `.cache/sam-3d-body`에 고정 revision으로 받는다.
이 서버에서는 2026-11-13 접근 승인과 다운로드를 확인했다. 기본 HF 로그인 파일을
참조하면서 모델 캐시는 lesson 안에 유지하도록 설정했다. 인증 파일은 repo로 복사하지 않는다.

| 명령 | 실제 계산 | 확인할 파일 (work 내부) |
|---|---|---|
| `frames` | 공통 24fps 시각의 가장 가까운 원본 frame을 선택. 2200×3208→1100×1604 등비 축소와 K 변환 | `run.json`, `frames.jpg`, `frames/<camera>/*.jpg`, `cameras/*.npz` |
| `landmarks` | 각 camera의 68점 검출. 시점에 따라 달라지는 턱 윤곽을 제외한 inner51의 robust triangulation | `landmarks_world.npz`, `triangulation.json` |
| `face` | 정면 camera에서 **공식 pixel3DMM crop→facer→UV/normal prediction→FLAME2020 dense fitting** | `mono_flame.npz`, `pixel3dmm/`, `logs/pixel3dmm_*.log` |
| `codes` | driver camera의 **공식 frozen X-NeMo**로 raw 512D code 추출. face 검출 실패 시 중단 | `codes_raw.npy` |
| `body` | 실제 K를 전달해 **공식 SAM3D/MHR**로 해당 사람의 body/shape/scale 추정 | `body_sam3d.npz`, `logs/sam3d.log` |
| `align` | 실제 camera 관측에 FLAME identity/expression/jaw와 world transform 재최적화. MHR FK의 관절각을 최적화하면서 pelvis/shoulder anchor 고정 | `face_alignment.jpg`, `alignment.jpg`, `*_alignment.json`, `aligned_state.npz` |
| `export` | 품질 검사 통과 state로 reference cache, 동기화 target 영상, spec 생성 | `reference/`, `target_24fps.mp4`, `clip_spec.json` |

여기까지가 `preprocess.py`의 역할이다. 이후 **00의 `build-triplet`**에서 공식 proxy/skeleton
renderer, target의 동일 crop, 학습 manifest 생성을 진행한다.

SAM 준비 전 얼굴 정렬만 살펴보려면 `python preprocess.py align-face`를 실행한다.
`face_alignment.jpg`는 실제 RGB에 fitted mesh를 겹친 첫/중간/끝 프레임이다.
모든 자산을 준비한 뒤에는 `CUDA_VISIBLE_DEVICES=4 python preprocess.py all`도 가능하다.
완료된 frames/추정 파일은 재사용하며 align은 재최적화한다. `all`도 전처리까지만 수행한다.

**시간축:** `--start`는 원본 sequence의 **24fps clock**이다. 처음77프레임은 원본73fps의
0..231번 프레임에서 샘플링한다. 원본77프레임을 단순히24fps라고 표시하지 않는다.
`run.json`에 source index/시각/카메라/원본 파일 정보를 남긴다. Export spec의 `start=0`은
잘린 state/video 내부 offset, `sequence_start`는 전체 sequence에서의 위치다.
학습 manifest의 `start`는 후자로 환산해서 train/validation 시간 누출을 검사한다.

별도 validation 구간의 예 (이후 모든 단계에도 같은 `--work` 사용):

```bash
python preprocess.py frames \
  --work data/preprocessed/042_FREE_val --start 240 --frames 77 --split validation
CUDA_VISIBLE_DEVICES=4 python preprocess.py landmarks --work data/preprocessed/042_FREE_val
# 이어서 face, codes, body, align, export에도 같은 --work를 붙인다.
```

현재 서버에서는 이 validation 구간도 전처리와 `build-triplet`까지 완료했다.
`data/triplets.jsonl`에는 train의24fps frame0..76 (`triplets/042_free_a`)와
validation의240..316 (`triplets/042_free_val`)이 연결돼 있다. 두 구간의 시간 비중첩,
각77프레임의 영상/code 길이와 상대 경로를 검사했다. 그대로 다음 단계로 진행한다:
**아래의 01 실행부터** 직접 disentangler를 학습한다. 이어서 02에서 cache를 만들고
03에서 Avatar를 학습한다. 00 완료가 01 학습까지 끝났다는 뜻은 아니다.

Triangulation에 시점을 더 쓰려면 **처음 frames 실행할 때** `--extra-cameras 222200038 222200044`처럼 추가한다.
기존 work의 구간/카메라/원본이 바뀌면 새 work를 요구한다. 별도 identity가 없는 현재042 데이터에서는
한 사람에 대한 비중첩 시간 검증만 가능하며, 논문의 unseen-identity 일반화 실험과 구분한다.

**재구현 경계:** pixel3DMM, SAM3D, X-NeMo, FLAME/MHR, renderer는 공개 구현을 쓴다.
공개되지 않은 training-data fitting은 inner51 다중 시점 정렬과 MHR joint-DOF optimization으로
구현했다. SAM의 scale PCA 계수를 MHR에 잘못 넣지 않도록 최종 `mhr_model_params`를 저장한다.
얼굴은 pixel3DMM을 초기값으로 사용하고 calibrated multi-view reprojection으로 shared shape,
expression, jaw, eyelids와 world transform을 재최적화한다. 1rad를 넘는 jaw 회전은 발산한
초기값으로 보고 해당 frame의 expression/pose를 중립에서 다시 fit한다. 원본
`mono_flame.npz`는 보존하며 `face_alignment.json`에 재초기화 frame을 기록한다.
jaw 범위/regularizer와 이 multi-view 보정은 lesson 구현 선택이다. 몸은 sequence의
shape/scale을 고정한 뒤 공유 Sim(3)와 frame별 회전을 함께 학습한다. 공식 SAM의 3-DOF 관절6D,
hinge 관절sin/cos 변환을 그대로 사용하고 Eq.6 계수(1, .3, .001, .03),400-step Adam을 적용한다.
공개되지 않은 dense face 대응표 대신 확실히 식별 가능한 코끝/양쪽 눈 중심 3점을 사용한다.
이 sparse 대응은 저자의 정밀 face fitting과 다르므로 overlay와 잔차 검사로 한계를 확인해야 한다.
공개 canonical driving-bank의 다른 사람 몸을 042의 tracking 결과로 대체하지 않는다.
보이지 않는 하체는 관측값이 아닌 SAM prior이므로 portrait 바깥의 정확도는 주장하지 않는다.

원본 출력/영상/가중치와 smoke 결과는 모두 Git ignore 경로에 둔다. 기존 conda를 변경하지 않고
필요한 Python package는 `.cache/runtime`에서 읽는다. 새 머신에서는 공식 환경 요구사항도 준비해야 한다.

수식 검사: `python smoke.py --preprocess-only`. 실제 데이터 검사는
`.cache/preprocess_smoke/`에 분리한다. 아래에 명시한 단계만 검증했으며 전체 pipeline 통과를 주장하지 않는다.

2026-11-13 GPU4 검증: 실제042/FREE의3개 camera×5프레임에서 추출/동기화,
landmark triangulation, 공식 pixel3DMM crop·segmentation·UV/normal prediction·5000-step tracking,
X-NeMo `[5,512]` 추출, calibrated FLAME 보정과 mesh overlay까지 실행했다.
단안 tracking의 jaw/eye 발산을 발견해 원본 parameter를 보존하고 multi-view 단계에서
복구했다. 최종 재투영 오차는 중앙값5.12px,95백분위21.37px(native1100×1604)다.
이는 landmark 정렬 검사이며 전체 머리 silhouette/최종 생성 품질을 보장하지 않는다.
접근 승인 후 SAM/MHR 가중치를 다운로드하고 공식 SHA256과 일치함을 확인했다.
GPU4에서 같은 5프레임의 SAM 추론이 완료됐고 모든 출력이 유한값이다.
출력 parameter를 실제 MHR에 다시 넣어 복원한 관절 좌표는 SAM 출력과 최대
`2.38e-7 m` 차이였으며 parameter까지의 gradient도 유한하고 0이 아님을 확인했다.
이번 접근 검사에서는 calibrated MHR fitting·최종 state export·triplet 생성을 실행하지 않았다.
상세 기록: [.cache/preprocess_smoke/report.json](.cache/preprocess_smoke/report.json).

Builder 결과 `triplet.jsonl`들을 하나의 training manifest로 합칠 때는 conditioning/target 상대 경로를
새 manifest 위치에 맞춘다. 각 clip에는 `ref.png`, `pose_proxy.mp4`, `pose_skel.mp4`, `codes.npy`, `target.mp4`가 있다.

학습용 JSONL manifest의 한 줄 (`data/triplets.jsonl` 기준 상대 경로):

```json
{"id":"042_free_a","identity":"042","sequence":"FREE","start":0,"frames":77,"split":"train","cameras":{"reference":"cam_A","driver":"cam_B","target":"cam_C"},"conditioning":"cond/042_free_a","target":"targets/042_free_a.mp4"}
```

`conditioning`에는 `ref.png, pose_proxy.mp4, pose_skel.mp4, codes.npy`가 필요하다. 별도 validation row도 있어야 한다. `start`는 **공통24fps sequence clock의 frame index**이며 파일 자체는 이미 잘라둔 77-frame clip이다. 서로 다른 camera를 다른 split에 넣어 같은 시간대를 누출시키는 manifest는 거부한다.

```bash
python 00_data_and_proxy.py inspect --manifest data/triplets.jsonl
```

00의 이미지 검사는 길이/비율/선언된 split을 검증한다. Calibration, 실제 synchronization, face landmark 정렬의 정확성은 시각 검사와 dataset metadata로 확인해야 한다.

## 01 실행

현재는 **042 한 사람을 직접 학습하는 순서**로 진행한다. 00의 Avatar용 triplet에는
driver camera code만 있지만, 01은 같은 순간의 **서로 다른 camera code**가 필요하다.
`prepare`가 이미 전처리된 train/validation 구간의 세 camera에서 공식 frozen X-NeMo로
각각 `[77,512]` raw code를 추출한다. FLAME/MHR는 재실행하지 않는다.
입력/가중치 hash가 같은 추출 결과는 재사용하고, face 검출 실패는 중단한다.

```bash
cd /media/vilab/mh/curious/lessons/vdm
CUDA_VISIBLE_DEVICES=4 python 01_expression_disentanglement.py prepare \
  --train-work data/preprocessed/042_FREE \
  --validation-work data/preprocessed/042_FREE_val --out data/01
python 01_expression_disentanglement.py pack \
  --manifest data/01/codes.json --out data/01/codes.npz --protocol single-subject
python 01_expression_disentanglement.py inspect \
  --codes data/01/codes.npz --protocol single-subject
CUDA_VISIBLE_DEVICES=4 python 01_expression_disentanglement.py train \
  --codes data/01/codes.npz --protocol single-subject --out checkpoints/01
# 재개: 마지막 train 명령에 --resume checkpoints/01/latest.pt 추가
```

`codes.npz`가 이미 있으면 pack을 반복하지 않고 inspect/train으로 진행한다.
실제 데이터는 train231개(77×3), validation231개이며, 시간 구간은 각각0..76과240..316이다.
현재 서버의 `data/01/codes.npz`는 준비와 검사를 마쳤다. GPU4에서2epochs와1epoch
재개, cycle 포함4개 loss의 backward, 공식 translator의 checkpoint 호환을 검사했다.
테스트 checkpoint는 `.cache/01_workflow_smoke/`에 있으며 본 학습 결과가 아니다.
본 학습은 위 train 명령으로 `checkpoints/01`에 처음부터 진행한다.
공식 MLP 구조와 reconstruction/expression/camera/cycle loss는 유지한다.
`single-subject`는 expression batch를32×3, camera batch를3×16으로 맞추고,
동일 시간은 camera가 달라도 train/validation 양쪽에 들어가지 못하게 검사한다.
새로운 사람에 대한 일반화나 논문 규모 학습의 재현을 의미하지 않는다.
학습은 기본1000epochs이고, cycle loss는 기존 설정대로 epoch index35부터 사용한다.
`checkpoints/01/best.pt`가 **02 audit → 03 → 04의 공통 disentangler 입력**이다.

학습 결과는 아래 명령으로 재학습 없이 시각화한다. 모델 계산은 CPU에서 수행한다.

```bash
python 01_expression_disentanglement.py visualize \
  --codes data/01/codes.npz --checkpoint checkpoints/01/best.pt \
  --manifest data/01/codes.json --work data/preprocessed/042_FREE_val --out outputs/01
```

- `learning_curves.png`: validation의512D code 복원 L1과 전체 gallery의 동일 frame 검색률.
  빨간 점선은 선택한 checkpoint의 epoch다. best는 복원 L1 기준이므로 검색률 최고와 다를 수 있다.
- `retrieval.jpg`: query / 다른 시점의 동일 시간 정답 / raw code 검색 / 학습한 expression code 검색.
  실제 RGB를 검색해 나란히 표시하며 새 RGB를 생성하지 않는다. 초록 테두리는 정확히 같은 frame,
  주황 테두리는 다른 frame이다. 조회 구간에서 균등하게 선택한5개 시점이라 실패 사례도 포함한다.
- `frame_matching.png`: driver→reference camera의 모든 시간 쌍의 cosine similarity.
  흰 대각선이 동일 시간, 빨간 점이 각 query의 top-1이다. 각 패널의 color scale은 별도다.
- `summary.json`: 선택한 checkpoint의 복원 오차와 검색 지표. 기존 로그의 검색은 자기 자신만 제외하고,
  RGB 비교의 검색은 같은 camera 전체를 제외하므로 두 검색률은 서로 다르다.

정확히 동일 frame을 정답으로 세므로 유사한 표정의 인접 frame도 오답이다.
01은512D 표정/32D 카메라 latent를 배우는 단계이며, 생성 영상은03 학습 후04에서 확인한다.

여러 사람 데이터를 확보했을 때는 별도 manifest로 `pack/train --protocol paper`를 사용한다.
이 모드는 identity를 분리하고 원래 expression batch32×4, camera batch8×16을 요구한다.
Manifest는 아래 row들의 JSON list이며 `codes`는 manifest 기준 상대 경로다.
각 camera의 code는 같은24fps 시간축이어야 한다. 원본73fps MP4를 그대로 넣어
추출하면 현재77프레임 구간과 맞지 않는다.

```json
[{"codes":"codes/042_free_000000_cam222200042.npy","identity":"042","sequence":"FREE","camera":"222200042","start":0,"split":"train"}]
```

검증된 cross-identity 표정 pair는 `pack --verified-pairs pairs.json`로 class를 합칠 수 있다.
수동 pairing이 없으면 identity-invariant expression 학습을 완전히 재현했다고 말할 수 없다.

## 02~04 실행

01을 직접 학습한 뒤 `checkpoints/01/best.pt` 사용. 공식 pretrained disentangler를 먼저 분석하려면 `weights/xnemo_disentangler.pt`를 명시적으로 선택한다.

```bash
CUDA_VISIBLE_DEVICES=4 python 02_connect_wan.py audit --disentangler checkpoints/01/best.pt
CUDA_VISIBLE_DEVICES=4 python 02_connect_wan.py cache --manifest data/triplets.jsonl \
  --size 288 512 --out data/cache_stage1
CUDA_VISIBLE_DEVICES=4 python 03_train_avatar.py --cache data/cache_stage1 \
  --disentangler checkpoints/01/best.pt --out checkpoints/03_stage1
# 재개: 동일 명령에 --resume checkpoints/03_stage1/latest.pt

# Stage1 결과는 Stage2 전에도 확인 가능:
CUDA_VISIBLE_DEVICES=4 python 04_generate_and_intervene.py \
  --cond data/triplets/042_free_val --adapter checkpoints/03_stage1/best.pt \
  --disentangler checkpoints/01/best.pt --protocol paper --size 288 512 --out outputs/04_stage1

CUDA_VISIBLE_DEVICES=4 python 02_connect_wan.py cache --manifest data/triplets.jsonl \
  --size 576 1024 --out data/cache_stage2
CUDA_VISIBLE_DEVICES=4 python 03_train_avatar.py --stage 2 --cache data/cache_stage2 \
  --disentangler checkpoints/01/best.pt --adapter checkpoints/03_stage1/best.pt \
  --out checkpoints/03_stage2

CUDA_VISIBLE_DEVICES=4 python 04_generate_and_intervene.py --cond data/triplets/042_free_val \
  --disentangler checkpoints/01/best.pt --adapter checkpoints/03_stage2/best.pt \
  --protocol paper --size 576 1024 --ablation all --out outputs/04_ablation
```

01의 `--protocol single-subject`는 데이터 분할, 03의 `--mode paper`는 학습할 parameter,
04의 `--protocol paper`는 sampler를 선택하는 서로 다른 설정이다.

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
- 아직 확인하지 못한 범위: MHR pelvis fitting/export, 완성된 calibrated NeRSemble triplet,5000+250step Avatar 학습 수렴, 논문의 정량 평가와 distillation. 얼굴 전처리와 SAM/MHR 추론 검증은 위 전처리 항목에 별도로 기록했다.

추론 예제와 네트워크·학습 경로를 공부할 준비는 되어 있다. 본격적인 사용자 데이터 학습의 다음 선행 작업은 동기화된 다중 카메라 RGB/calibration 확보와00의 dataset-world 정렬 경계를 연결하는 것이다. 합성 테스트의 대용량 optimizer checkpoint는 검사 후 제거하고 로그·결과·공개 weights만 남긴다.
