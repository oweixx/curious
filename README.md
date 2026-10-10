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
  - [ ] Animation & Eval (9, 10)
- [ELITE (CVPR 2026)](https://arxiv.org/abs/2601.10200)

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