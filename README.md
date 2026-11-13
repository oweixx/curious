# Curious
궁금한 것들 Codex와 티키타카 하는중

<details>
<summary>Canonical UV Map (26/10/02)</summary>
<div markdown="1">

- 3D Head, Face Avatar에서 Canonical UV Map을 이용하여 Geometry를 표현하는 논문들을 많이 볼 수 있다. 대부분 3D -> 2D로 표현하면서 가져가는 Network의 이점들을 가져가는듯 하기도 하다.
- [RealDenseFace (CVPRW 2026)](https://arxiv.org/abs/2606.24144)
- [FiCA (Preprint)](https://arxiv.org/pdf/2606.24232)

</div>
</details>

<details>
<summary>ELITE (26/10/03 ~ 10/11)</summary>
<div markdown="1">

- I want to implement from scratch. First, extract FLAME meshe from a monocular video in NeRSemble dataset. Second, We make Canonical geometry/texture uv map to aligned with previous extracted FLAME mesh. And then, we predict/bind on Gaussian Parameter map. So we can render using naive 2DGS.
  - [X] Video → Tracking (2, 3)
  - [X] Tracking Result → Canonical UV (4)
  - [X] UV represent → Gaussian Map (5)
  - [X] Gaussian Map → Avatar (6, 7, 8)
  - [X] Animation & Eval (9, 10)
- [ELITE (CVPR 2026)](https://arxiv.org/abs/2601.10200)

</div>
</details>

<details>
<summary>FiCA (26/10/17 ~ 10/21)</summary>
<div markdown="1">

- 기존 FLAME을 사용한다면 Partial uv map 예측은 필요 없어진다. 아마 FiCA의 dataset자체가 내부 multi-view iphone 촬영 데이터셋이기도 하고 그중에서도 single image로만 이용하기 때문에 기존에 Sapiens가 Foundation model로서 학습하던 것을 이용하여 partial uv map을 예측하게 fine-tune하게 된다.
- 1에서 진행하는 것은 결국 FLAME을 쓰지 않고 대규모 데이터로 fine-tune된 Sapiens를 이용해서 Feed-Forward uv map 예측을 해보자 라는 것으로 해석된다. 그렇다 보니 이번에 진행한 한 명이라는 dataset으로 overfitting시킨다고 해도 novel pose에 대해서 깨지거나 점들이 번지는 현상이 보이고 있다. 결국 Meta에서 수행한 연구이기도 하고 내부에서 수집한 대규모 데이터셋을 통해서 학습 하기 때문에 이러한 Feed-Forward 작업이 가능한 것은 사실인 것 같다.
- 2자체는 Partial UV Map을 이용하여 Canonical UV Map을 생성하는 부분이다. 이렇게 되면 안보이는 부분에 대한 texture를 일관성있게 생성해 낼 수 있기 때문에 이러한 부분이 2D UV Map에서의 장점이 되는 것 같다.
  - [X] Prepare dataset (NeRSemble)
  - [X] Extract partial uv map
  - [X] UV Texture / Geometry Diffusion
  - [X] UV Refine Network
  - [X] Gaussian Avatar Training
- [FiCA (arXiv 2026)](https://arxiv.org/pdf/2606.24232)

</div>
</details>

<details>
<summary>Video Diffusion Avatars(VDA) (26/11/13 ~ 11/13)</summary>
<div markdown="1">

  - 해당 논문은 기본적으로 지금까지 관심있게 봐왔던 Gaussian Avatar 방법론들과 다르게 Diffusion based Video 형식으로 출력하는 연구이다. explicit한 representation(point-cloud, gs, mesh...)으로 내놓지 않기 때문에 조금은 다른 결이라고 볼 수도 있을 것 같다.
  - 논문을 읽으면서 헷갈렸던 부분으로는 분명히 3D Mesh model(proxy mesh)을 사용하는 것으로 보아 explicit한 무엇인가가 있는 줄 알았는데 논문에도 잘 명시되어 있듯이 **"the VDM is the avatar","no 3D asset is ever built"**라고 설명한다. 
  - 해당 논문의 Contribution & Novelty는 VDM에서 강력한 multi-view consistency를 확보하면서도 입력으로 들어오는 Single Reference에 대해서 일관성을 유지하며 Video Model안에서도 expression, camera view를 분리하여 Avatar를 표현한다는 것이다.
  - 하지만 실제로 [WAN-Animate-2](https://humanaigc.github.io/wan-animate-2/)를 보자면 똑같이 Single Reference에 대해 입력으로 들어오는 drive video에 대응하는 결과 video를 그대로 출력하는 것을 볼 수 있다. 심지어 Novel-view에 대한 multi-view consistency도 지켜주는 것을 볼 수 있고 더불어 배경또한 일관성있게 잘 처리하는 것을 볼 수 있다. 만약 실제 conference review라면 이렇게 지적할 수 있을까? "WAN-Animate-2에서도 이미 잘되고 있으면서 같은 기능을 똑같이 잘 수행하는 연구가 있는데 이 연구의 독창적인 차별점이 보이지 않는다." 그냥 단순하게 생각하면 이렇게 치부할 수 있을 것이라 생각되기도 한다. 

  - [X] Data and proxy model preparation
  - [X] Expression disentanglement
  - [X] Connect wan
  - [X] Train full-pipeline avatar
  - [X] Generate 
- [Video Diffusion Avatars (arXiv 2026)](https://simongiebenhain.github.io/video-diffusion-avatars/)

</div>
</details>

<details>
<summary>Diffusion Transformer(DiT) (26/11/13 ~ )</summary>
<div markdown="1">
  - 

  - [] study frozen VAE, pixels and latents
  - [] Forward diffusion
  - [] Patch embedding and token
  - [] Attention and conditions
  - [] AdaLN-Zero
  - [] Optimize objective loss
  - [] Optimize train
  - [] Reverse diffusion
  - [] CFG
  - [] Scaling
  - [] Original pretrained XL/2 inference

- [Scalable Diffusion Models with Transformers (ICCV 2023 Oral)](https://www.wpeebles.com/DiT)

</div>
</details>


<details>
<summary>Attention Residuals (later)</summary>
<div markdown="1">
  - 

  - [] Basic attention
  - [] Residaul stream
  - [] Full AttnRes
  - [] Block AttnRes
  - [] Train and compare

- [Attention Residuals (Arxiv Mar)](https://arxiv.org/abs/2603.15031)

</div>
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