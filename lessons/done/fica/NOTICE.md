# 구현 출처와 재현 범위

- FiCA v2: https://arxiv.org/abs/2606.24232v2
- FiCA project: https://youwangk.github.io/FiCA
- Sapiens: https://github.com/facebookresearch/sapiens/tree/2cb07227a740cf09896309ea3a3b8fa44429865c
- Pippo: https://github.com/facebookresearch/pippo/tree/6f80b22f32f58f7ae08f7b20a4901a3c66809368
- URAvatar: https://arxiv.org/abs/2410.24223
- Diffusers: https://github.com/huggingface/diffusers/tree/v0.25.1
- Graphdeco 3DGS rasterizer: https://github.com/graphdeco-inria/diff-gaussian-rasterization/tree/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d

01의 Sapiens encoder/head는 공개 architecture/weight naming을 따르는 PyTorch port다.
원본 Sapiens의 전체 framework를 복사하지 않는다. 공개 checkpoint를 strict하게 읽으며,
pretrain의 cls token을 dense task에서 제거하는 변환을 명시적으로 수행한다.
01의 `finetune-uv`는 기존 UV 모델에 별도 foreground head를 추가하는 우리 구현의 보완이다.
Balanced BCE/Dice, FP32 decoder, UV 삼각형 rasterization과 image grid sampling의
세부 설정은 FiCA 저자의 공개 코드가 아니다. GT foreground는 학습/평가에만 사용한다.
02의 DiTBlock/ControlMLP는 pinned Pippo source를 그대로 import한다. 03은 실제
Diffusers UNet2DConditionModel을 사용한다. 캐시의 원본 source에는 각 프로젝트 LICENSE를
함께 보존한다. 해당 코드/가중치/데이터의 이용 조건은 원래 배포처를 따른다.

02의 geometry repair는 우리 데이터에서 확인한 XYZ VAE 복원 오차와 UV→mesh의
ill-conditioning을 보완하는 구현이다. XYZ를 dataset std의 3배로 나누는 전처리,
atlas padding까지 포함한 geometry latent loss, 고정 VAE를 통과한 endpoint XYZ/vertex/edge
supervision, template-relative Laplacian MeshBridge는 저자의 공개 설정이 아니다.
DiT/ControlMLP 구조와 SDXL VAE/CLIP weight는 유지하며, 원본 50K checkpoint와 cache는
새 geometry_repair run의 weight 초기화/비교용으로 보존한다.

03은 512×512 입출력과 고해상도 convolution/skip connection을 유지하되, self-attention과
Sapiens feature cross-attention을 64×64 이하의 U-Net stage에 배치한다. 채널 폭은
Diffusers 기본값을 사용한다. 이 attention 배치와 단일 identity의 기본 1K-step 학습량은
우리의 계산량 조정이며 FiCA 저자의 공개 architecture/training 설정이 아니다.
초기 구성은 논문의 refinement loss 종류와 바깥 가중치를 따르며, held-out frame의 전후 비교 및
coarse FLAME vertex RMSE는 validation 진단으로 사용한다. 이후 quality 보완은 아래와 같다.

03의 `surface_refinement_v2`는 관찰된 품질 문제를 보완하는 별도 구현 선택이다.
UV→mesh 선형부로 계산한 중심 이동 제거, 얼굴 mask 내 평균 photo L1, 가중치 0.1의
texture MSE, 초기 생성 mesh로부터의 변위 규제(0.01 world scale/가중치 0.25)를 사용한다.
Photo/mask/keypoint의 바깥 가중치 2/0.5/0.01과 Laplacian/normal 규제는 유지하지만,
위 reduction/추가 제약을 FiCA의 원본 loss 구현이라고 주장하지 않는다. GT vertex는
진단 지표에만 사용하며 중심 고정/변위 규제/best 선택의 target으로 사용하지 않는다.

Pippo 전체 원본은 동일한 pinned commit의 `external/pippo/` checkout에서 읽을 수 있다.
`pippo_walkthrough.py`는 이 원본 class를 직접 호출하며 원본 source를 변경하지 않는다.
본편의 cached module과 동일한 파일 hash를 확인하고, 기존 02 checkpoint도 읽어 검증한다.
전체 checkout의 LICENSE와 저작권 표기를 보존한다. 자세한 읽기 순서는 GUIDE에 있다.

FiCA는 코드·전용 checkpoint·전체 architecture 설정·내부 데이터를 공개하지 않았다.
따라서 이 lesson을 저자 구현과의 bitwise 일치나 원본 성능 재현으로 부르지 않는다.
`fica_config.json/implementation_choices`와 GUIDE의 대응표가 구현 결정의 기록이다.

UPM의 identity bias pyramid, driving decoder, learned guide와 Gaussian parameter
출력은 논문 및 선행 URAvatar의 구조를 따른다. 정확한 FiCA UPM의 채널 수, 내부 Codec
expression encoder와 shoulder/eye topology는 복제할 수 없어 공개 FLAME 기반 adapter를
명시적으로 사용한다. ELITE의 Mesh2Gaussian network나 2DGS backend를 UPM/3DGS로
대체하여 같은 모델이라고 주장하지 않는다.
