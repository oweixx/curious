# Sources and implementation boundaries

이 lesson은 논문의 수식과 공식 codebase를 참고한 교육용 구현이다.
FM 라이브러리의 `sample()`만 호출하는 wrapper 대신, conditional path·regression·ODE update를
번호 파일 안에서 직접 계산한다. 공식 대규모 ImageNet 실험의 재현은 아니다.

| 출처 | 이 lesson에서 대응하는 부분 |
|---|---|
| [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), Lipman et al., ICLR 2023 | conditional probability path, conditional/marginal vector field, simulation-free regression |
| [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264), Lipman et al., 2024 | endpoint coupling 관점의 affine path, 구현 구성 |
| [Flow Straight and Fast / Rectified Flow](https://arxiv.org/abs/2209.03003), Liu et al., ICLR 2023 | independent straight interpolation, teacher endpoint로 coupling을 다시 만드는 reflow |
| [Improving and generalizing flow-based generative models with minibatch optimal transport](https://arxiv.org/abs/2302.00482), Tong et al., TMLR 2024 | minibatch OT coupling 실험 |
| [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748), Peebles & Xie, ICCV 2023 | 기존 DiT lesson의 backbone, adaLN-Zero, class dropout을 실제 import |

확인한 Meta `flow_matching` 코드 revision:
`11568d37f8d5a080e12aa7b5305d9c35ae07d136`.
이 codebase는 2024 Guide와 함께 공개된 자료이며 2022 원 논문의 전체 실험 코드를 그대로 복제했다는 뜻은 아니다.

- [AffineProbPath.sample](https://github.com/facebookresearch/flow_matching/blob/11568d37f8d5a080e12aa7b5305d9c35ae07d136/flow_matching/path/affine.py): `00.conditional_path`의 affine sample/velocity 수식과 대응.
- [CondOTScheduler / CosineScheduler](https://github.com/facebookresearch/flow_matching/blob/11568d37f8d5a080e12aa7b5305d9c35ae07d136/flow_matching/path/scheduler/scheduler.py): `00.coefficients`와 대응. residual sigma는 원 FM의 finite endpoint noise를 설명하기 위해 추가했다.
- [ODESolver](https://github.com/facebookresearch/flow_matching/blob/11568d37f8d5a080e12aa7b5305d9c35ae07d136/flow_matching/solver/ode_solver.py): 공식 코드는 `torchdiffeq`를 호출한다. `02.integrate`는 Euler/Heun/RK4 계산 자체를 읽을 수 있게 별도로 작성했다.

공식 FM 파일을 복사하거나 `flow_matching` package를 설치하지 않는다.
02의 solver는 adaptive solver/likelihood/divergence를 구현하지 않는다.
03의 Hungarian assignment는 균일 가중치 minibatch의 exact assignment이며 entropic Sinkhorn이나 전역 분포 OT가 아니다.
Reflow는 작은 2D teacher/student 실험이고 image reflow/distillation은 포함하지 않는다.

04/05는 [../dit/models.py](../dit/models.py)를 직접 재사용한다.
그 파일의 저작권 헤더, [원본 출처 기록](../dit/NOTICE.md), [CC BY-NC 4.0 라이선스](../dit/LICENSE.txt)를 유지한다.
새 `FlowDiT` wrapper는 1-channel 32px pixels, `learn_sigma=False`, continuous time embedding,
velocity supervision을 사용한다. pretrained DiT noise weights를 FM checkpoint라고 바꿔 부르지 않는다.
원본 코드와 현재 로컬 파일의 hash를 image checkpoint에 기록한다.

MNIST는 torchvision으로 읽는다. 기존 cache가 있으면 재사용하고, 없으면 `--download`로 lesson의
ignored data 디렉터리에 받는다. torchvision은 train/test raw 파일을 함께 받을 수 있으나,
이 lesson의 학습·validation은 공식 train split만 나누어 사용한다. 공식 test 성능이나 FID는 보고하지 않는다.
