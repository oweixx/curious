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
<summary>FiCA (26/10/17 ~ )</summary>
<div markdown="1">

- 기존 FLAME을 사용한다면 Partial uv map 예측은 필요 없어진다. 아마 FiCA의 dataset자체가 내부 multi-view iphone 촬영 데이터셋이기도 하고 그중에서도 single image로만 이용하기 때문에 기존에 Sapiens가 Foundation model로서 학습하던 것을 이용하여 partial uv map을 예측하게 fine-tune하게 된다.
- 1에서 진행하는 것은 결국 FLAME을 쓰지 않고 대규모 데이터로 fine-tune된 Sapiens를 이용해서 Feed-Forward uv map 예측을 해보자 라는 것으로 해석된다. 그렇다 보니 이번에 진행한 한 명이라는 dataset으로 overfitting시킨다고 해도 novel pose에 대해서 깨지거나 점들이 번지는 현상이 보이고 있다. 결국 Meta에서 수행한 연구이기도 하고 내부에서 수집한 대규모 데이터셋을 통해서 학습 하기 때문에 이러한 Feed-Forward 작업이 가능한 것은 사실인 것 같다.
  - [X] Prepare dataset (NeRSemble)
  - [X] Extract partial uv map
  - [] UV Texture / Geometry Diffusion
  - [] UV Refine Network
  - [] Gaussian Avatar Training
- [FiCA (arXiv 2026)](https://arxiv.org/pdf/2606.24232)

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


</div>
</details>