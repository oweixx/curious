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
