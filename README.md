<h1 align="center">Curious</h1>

<details>
<summary>Canonical UV Map (26/10/02)</summary>

- 3D Head, Face Avatar에서 Canonical UV Map을 이용하여 Geometry를 표현하는 논문들을 많이 볼 수 있다. 대부분 3D -> 2D로 표현하면서 가져가는 Network의 이점들을 가져가는듯 하기도 하다.
- [RealDenseFace (CVPRW 2026)](https://arxiv.org/abs/2606.24144)
- [FiCA (Preprint)](https://arxiv.org/pdf/2606.24232)

</details>

<details>
<summary>ELITE (26/10/03 ~ 10/11)</summary>

- I want to implement from scratch. First, extract FLAME meshe from a monocular video in NeRSemble dataset. Second, We make Canonical geometry/texture uv map to aligned with previous extracted FLAME mesh. And then, we predict/bind on Gaussian Parameter map. So we can render using naive 2DGS.

- [x] Video → Tracking (2, 3)
- [x] Tracking Result → Canonical UV (4)
- [x] UV represent → Gaussian Map (5)
- [x] Gaussian Map → Avatar (6, 7, 8)
- [x] Animation & Eval (9, 10)

- [ELITE (CVPR 2026)](https://arxiv.org/abs/2601.10200)

</details>

<details>
<summary>FiCA (26/10/17 ~ 10/21)</summary>

- 기존 FLAME을 사용한다면 Partial uv map 예측은 필요 없어진다. 아마 FiCA의 dataset자체가 내부 multi-view iphone 촬영 데이터셋이기도 하고 그중에서도 single image로만 이용하기 때문에 기존에 Sapiens가 Foundation model로서 학습하던 것을 이용하여 partial uv map을 예측하게 fine-tune하게 된다.
- 1에서 진행하는 것은 결국 FLAME을 쓰지 않고 대규모 데이터로 fine-tune된 Sapiens를 이용해서 Feed-Forward uv map 예측을 해보자 라는 것으로 해석된다. 그렇다 보니 이번에 진행한 한 명이라는 dataset으로 overfitting시킨다고 해도 novel pose에 대해서 깨지거나 점들이 번지는 현상이 보이고 있다. 결국 Meta에서 수행한 연구이기도 하고 내부에서 수집한 대규모 데이터셋을 통해서 학습 하기 때문에 이러한 Feed-Forward 작업이 가능한 것은 사실인 것 같다.
- 2자체는 Partial UV Map을 이용하여 Canonical UV Map을 생성하는 부분이다. 이렇게 되면 안보이는 부분에 대한 texture를 일관성있게 생성해 낼 수 있기 때문에 이러한 부분이 2D UV Map에서의 장점이 되는 것 같다.

- [x] Prepare dataset (NeRSemble)
- [x] Extract partial uv map
- [x] UV Texture / Geometry Diffusion
- [x] UV Refine Network
- [x] Gaussian Avatar Training

- [FiCA (arXiv 2026)](https://arxiv.org/pdf/2606.24232)

</details>

<details>
<summary>Video Diffusion Avatars(VDA) (26/11/13 ~ 11/13)</summary>

- 해당 논문은 기본적으로 지금까지 관심있게 봐왔던 Gaussian Avatar 방법론들과 다르게 Diffusion based Video 형식으로 출력하는 연구이다. explicit한 representation(point-cloud, gs, mesh...)으로 내놓지 않기 때문에 조금은 다른 결이라고 볼 수도 있을 것 같다.
- 논문을 읽으면서 헷갈렸던 부분으로는 분명히 3D Mesh model(proxy mesh)을 사용하는 것으로 보아 explicit한 무엇인가가 있는 줄 알았는데 논문에도 잘 명시되어 있듯이 **"the VDM is the avatar","no 3D asset is ever built"**라고 설명한다.
- 해당 논문의 Contribution & Novelty는 VDM에서 강력한 multi-view consistency를 확보하면서도 입력으로 들어오는 Single Reference에 대해서 일관성을 유지하며 Video Model안에서도 expression, camera view를 분리하여 Avatar를 표현한다는 것이다.
- 하지만 실제로 [WAN-Animate-2](https://humanaigc.github.io/wan-animate-2/)를 보자면 똑같이 Single Reference에 대해 입력으로 들어오는 drive video에 대응하는 결과 video를 그대로 출력하는 것을 볼 수 있다. 심지어 Novel-view에 대한 multi-view consistency도 지켜주는 것을 볼 수 있고 더불어 배경또한 일관성있게 잘 처리하는 것을 볼 수 있다. 만약 실제 conference review라면 이렇게 지적할 수 있을까? "WAN-Animate-2에서도 이미 잘되고 있으면서 같은 기능을 똑같이 잘 수행하는 연구가 있는데 이 연구의 독창적인 차별점이 보이지 않는다." 그냥 단순하게 생각하면 이렇게 치부할 수 있을 것이라 생각되기도 한다.

- [x] Data and proxy model preparation
- [x] Expression disentanglement
- [x] Connect wan
- [x] Train full-pipeline avatar
- [x] Generate

- [Video Diffusion Avatars (arXiv 2026)](https://simongiebenhain.github.io/video-diffusion-avatars/)

</details>

<details>
<summary>Diffusion Transformer(DiT) (26/11/13 ~ 11/14)</summary>

- DiT는 간단하게 noise가 섞인 latent를 입력으로 같이 주어진 t와 조건을 통해 해당 noise를 예측하는 형식으로 학습을 진행한다. 여기서 noise를 맞추는 것이 어떻게 이미지 생성으로 이어지는 것인가? 단순 DDPM과 같이 실제 inference의 경우에는 입력으로 들어오는 random noise와 condition에 따라 noise를 예측하여 이미지를 생성하는 형식으로 이루어진다. DDPM과 흐름은 완전히 똑같지만, DDPM에서는 U-Net을 통해 noise를 예측하였다면 DiT는 해당 부분 대신에 Transformer를 이용해 노이즈를 예측하는 것으로 이해할 수 있다. (0, 1)
- 03_attention_and_condition.py에서 알게된 것은 Condition을 어떤식으로 처리할 것인지에 대한 부분이다. time, y를 embedding을 통해서 하나의 condition으로 표현하여 이를 patchify된 token에 scale, shift 연산을 하게 된다. 여기서 말하는 scale, shift는 단순히 해당 token vector에 값을 곱하고 더하는 형식으로 나타난다. 그렇게 값이 조정된 token은 attention을 통해 값이 계산되게 된다. 여기서 신기한 부분은 condition이 noise 예측 transformer의 영향을 주는 방식이 scale, shift 형식으로 나타난다는 것이다. 이렇게 표현될 수 있기 위해 condition을 embedding하는 MLP와 embedding값들을 이용해 shift와 scale로 변환해주는 Linear의 weight와 bias가 learnable parameter로서 이를 표현하게 해준다. 자연스럽게 이러한 표현을 이용해서 attention에서는 token 사이의 관계에서의 정보를 가져오고, MLP에서 해당 조건들을 반영하여 노이즈를 예측하는 형식으로 진행이 되게 된다. (3,4)
- 간단하게 정리한 아래 수식을 설명해보도록 하자. Condition c는 입력으로 들어오는 t와 target condition인 y를 각각 embedding을 하고 더해서 들어오는 condition 입력이다. 이를 학습시키는 parameter로는 embedding mlp와 이를 합쳐서 shift, scale, scalar값으로 변환하는 linear parameter가 있다. 왜 굳이 attention과 MLP에 별도의 shift, scale, gate가 있는것인가? 엄연히 두 모듈의 기능이 다르기 때문이다. attention은 동작시 token들의 정보를 주고 받는 형태로 동작하게 되고, MLP는 각 Token 내부의 vector feature들을 변환하는 역할을 가지기 때문이다. 더 쉽게 말하면 어떤 정보를 가지고 있는지 계산하고, 해당 정보를 어떻게 가공할 것인지 변환할 것인지를 하므로 각각 맡은 일이 다르다. 그리고 입력 값을 새로 생성하는 것이 아닌 residual 하게 x + delta(x) 형식으로 더해주는 것은 기존 표현을 유지하면서도 필요한 변화량을 학습하기 위함이다. 더 안정적으로 네트워크를 학습하기 위한 설계라고도 볼 수 있지 않을까 싶다.
  ```python
  sa, sca, ga, sm, scm, gm = modulation(c)

  u = norm1(x) * (1 + sca) + sa
  x = x + ga * attention(u)

  u = norm2(x) * (1 + scm) + sm
  x = x + gm * mlp(u)
  ```        


- [x] study frozen VAE, pixels and latents
- [x] Forward diffusion
- [x] Patch embedding and token
- [x] Attention and conditions
- [x] AdaLN-Zero
- [x] Optimize objective loss
- [x] Optimize train
- [x] Reverse diffusion
- [ ] CFG
- [ ] Scaling
- [ ] Original pretrained XL/2 inference

- [Scalable Diffusion Models with Transformers (ICCV 2023 Oral)](https://www.wpeebles.com/DiT)

</details>

<details>
<summary>Attention Residuals (26/11/14 ~ )</summary>
- 해당 논문은 그냥 추천 받아서 하는거긴한데 논문을 직접 보지 않고 Agent가 만들어준 lesson으로 직접 진행해보면서 배우는 식으로 진행한다.
- 

- [x] Basic attention
- [x] Residaul stream
- [x] Full AttnRes
- [ ] Block AttnRes
- [ ] Train and compare

- [Attention Residuals (Arxiv Mar)](https://arxiv.org/abs/2603.15031)

</details>

---
### TODO

- 3D Reconstruction using 2D Prior
  - References
    - [Splatter Image (CVPR 2024)](https://github.com/szymanowiczs/splatter-image)
    - [SynShot (3DV 2025)](https://wojciechzielonka.com/synshot/)
    - [EDM]
- Dense UV Map?
  - [WarpHE4D (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Yun_WarpHE4D_Dense_4D_Head_Map_toward_Full_Head_Reconstruction_ICCV_2025_paper.html)
- Learnable Token
- Gaussian Map to Gaussian Splatting

---
### Ideation

<details>
<summary>Deformation about FLAME (Geometry Transfer in 3D Head Avatar)</summary>
<div markdown="1">


  - Idea
    - Appearance와 Geometry를 분리하여 reference text, image에 맞는 geometry로 transfer하는 방향성.
    - identity는 유지한채로 reference에 맞게 변화해야함.
  - References
    - [SOAP (SIGGRAPH 2025)](https://tingtingliao.github.io/soap/)
    - [HeadEvolver (3DV 2025)](https://www.duotun-wang.co.uk/HeadEvolver/)
    - [GeoStyle (CVPR 2026)](https://changwoonchoi.github.io/GeoStyle/)
    - [InvSculpt (SIGGRAPH 2026)](https://cislab.hkust-gz.edu.cn/media/documents/SIGGRAPH2026_InvSculpt_Cameraready.pdf)
    - [RegHead (ECCV 2026)](https://arxiv.org/abs/2607.12206)
    - [Geometry in Style (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Dinh_Geometry_in_Style_3D_Stylization_via_Surface_Normal_Deformation_CVPR_2025_paper.html)
    - [ShapeUP (SIGGRAPH 2026)](https://inbar-2344.github.io/ShapeUp-page/)
    - [Prox-E (SIGGRAPH 2026)](https://etaisella.github.io/Prox-E/)
  

</div>
</details>


<details>
<summary>Online Canonical Avatar — 관측이 쌓일수록 개선되는 개인 Avatar (26/11/14)</summary>

**연구 질문:** 처음 만든 Avatar를 고정하는 대신, 동기화된 카메라 관측이 들어올 때마다 개인의 canonical geometry와 appearance를 개선할 수 있을까? 새로운 표정·자세·조명을 설명하면서도, 이전에 복원한 개인 정보는 보존해야 한다.

| 상태 | 포함할 정보 | 갱신 방식 |
|---|---|---|
| Persistent canonical state | Identity geometry, 피부 appearance, 머리 형태 | 여러 관측에서 일관된 정보만 누적하고 천천히 갱신 |
| Transient state | Expression, pose, 현재 조명 | 현재 관측을 설명하도록 빠르게 추정하고 canonical과 분리 |
| Observation memory | 과거의 신뢰할 수 있는 입력과 해당 시각의 상태 | 작은 replay buffer로 이전 관측에 대한 성능 유지 |

**핵심 가설과 갱신 흐름**

1. 현재 canonical state를 고정하고, 새 입력 두 시점으로 expression·pose·조명을 먼저 추정한다.
2. 가림, tracking 오차, 조명으로 설명되는 차이를 구분한 뒤 canonical에 남길 정보를 결정한다.
3. 현재 관측의 rendering loss와 과거 관측의 replay loss로 canonical 표현 또는 작은 adapter를 갱신한다.
4. 관측하지 못한 부분은 현재 추정치를 보존한다. 웃는 입 모양이나 잠깐의 그림자가 identity geometry·texture에 영구적으로 남는 것을 줄이는 것이 목표다.

**첫 실험: NeRSemble 한 사람, 입력 camera 2개 + 보류 camera 1개**

- 시간순으로 입력을 공개한다. 초기화는 앞부분의 짧은 prefix만 사용하고, 미래 RGB·전체 영상으로 만든 UV texture·평가 camera로 맞춘 tracking은 사용하지 않는다.
- 같은 초기 Avatar에서 `frozen`, `representation optimization`, `adapter update`, `adapter + confidence gate + replay`를 비교한다. 관측당 시간 예산은 같게 두고 실제 소요 시간·step 수·메모리를 함께 기록한다.
- 현재 시각의 보류 camera 품질, 동일한 과거 시각의 품질 유지, canonical 변화량을 각각 측정한다. 변화량이 작다는 사실만으로 복원 정확도가 높다고 판단하지 않는다.
- Rendering FPS와 tracking·transient fitting·canonical update 시간을 따로 측정한다. NeRSemble의 비교적 일정한 조명만으로 실제 환경의 조명 분리가 검증되지는 않으므로, 조명 변화·가림 실험도 별도로 필요하다.

**기여를 구체화할 지점**

- 단순한 test-time fine-tuning 자체보다, **동적 얼굴에서 무엇을 개인 정보로 누적할지 결정하는 방법**에 초점을 둔다.
- 후보는 표정·조명 오차의 canonical 유입을 억제하는 갱신 규칙, 관측 신뢰도·표면 coverage에 따른 공간별 갱신, 작은 메모리에서의 forgetting 억제, 제한된 시간 안의 adapter/fast-weight 갱신이다.
- 각각 `transient 분리 없음`, `gate 없음`, `replay 없음`, `직접 표현 최적화`와 비교해야 한다. 한 사람의 개선을 확인한 다음 여러 identity·다른 표정·가림·조명으로 확장한다. 아직 검증된 novelty나 성능 결과는 아니다.

**관련 연구와 구분**

- [Test-Time Training (ICML 2020)](https://proceedings.mlr.press/v119/sun20b.html): 관측으로 만든 self-supervised objective를 사용해 추론 시 모델을 갱신한다는 출발점.
- [ELITE (CVPR 2026)](https://arxiv.org/abs/2601.10200): 입력 영상으로 pretrained MGPM을 개인화한다. 여기서는 지속적으로 들어오는 관측의 상태 분리·누적·보존을 더 구체적인 질문으로 삼는다.
- [TTT3R](https://arxiv.org/abs/2509.26645): streaming 3D reconstruction의 memory update를 온라인 학습 관점으로 해석한다.
- [ZipMap (CVPR 2026)](https://arxiv.org/abs/2603.04385): fast-weight TTT layer로 scene 정보를 압축한다. 이 inner key–value learning과 Avatar의 photometric adaptation은 서로 다른 학습 목적이다.

**실행 가능한 첫 검증:** [Online Avatar lesson — 5단계](lessons/online_avatar/GUIDE.md)

- 실제 NeRSemble 관측 → 입력 두 camera만의 causal FLAME fitting → canonical UV surface Gaussians → 순차 갱신 → 보류 camera·forgetting 평가.
- 첫 구현은 prefix로 학습한 작은 coordinate MLP를 공통 초기값으로 쓴다. 다인물 사전학습 모델의 일반화, 학습된 fast-weight update rule, 머리카락 전체 복원은 후속 연구 범위다.

</details>


<details>
<summary>Not specified.</summary>
<div markdown="1">


  - Idea
  - References
    - [ELITE (CVPR 2026)](https://arxiv.org/abs/2601.10200)
    - [URHead (ECCV 2026)](https://arxiv.org/abs/2607.22673)
    - [GTAvatar (CGF/Eurographics 2026)](https://kelianb.github.io/GTAvatar/)
    - [MATCH (CVPR 2026)](https://malteprinzler.github.io/projects/match/)
    - [MeGA (CVPR 2025)](https://conallwang.github.io/MeGA_Pages/)
    - [AnimPortrait3D (SIGGRAPH 2025)](https://onethousandwu.com/animportrait3d.github.io/)
    - [Video Diffusion Avatars(Preprint 2026)](https://simongiebenhain.github.io/video-diffusion-avatars/)

</div>
</details>