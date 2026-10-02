# Curious

호기심이 생긴 키워드에서 시작해 개념, naive 구현, 시각화와 실험으로 이어가는 학습 repo.
연구에 필요한 모듈의 가정과 동작을 직접 확인하며 기술 선택의 근거를 쌓는다.

첫 실험은 **FLAME의 canonical mesh를 하나의 UV atlas에 배치하고, 다섯 종류의 정보를
같은 주소 체계에서 표현하기**다. 설정과 학습용 최소 구현을 포함하며 학습/최적화는 하지 않는다.

```text
FLAME v_template (+ identity shape)
  → UV rasterization: face index + barycentric weights + valid mask
  → canonical base position / vertex displacement / displaced position / normals
  → texture + position + displacement + normal + feature
  → raw arrays + PNG + mesh/point cloud + interactive HTML
```

Python 3.11.13, PyTorch 2.6.0 CUDA 12.4 환경을 repo의 `.venv`에 격리한다.
전역 Conda 환경을 바꾸지 않는다. GPU rasterizer나 OpenGL 빌드 없이 barycentric 계산은
NumPy CPU에서 수행하고, 속성 보간과 Feature CNN은 PyTorch의 지정 device에서 수행한다.

현재 `FLAME2023/flame2023.pkl`을 확인하고 canonical 배열을 `assets/flame/`에 변환해
연결했다. 원본은 그대로 보존한다. Chumpy 변환은 별도의 `.venv-legacy`에서 수행한다.
공식 DECA의 공개 UV template을 고정 commit에서 받아 모델과 topology도 확인했다.
기본 경로로 바로 실행할 수 있다. 원본/변환 모델과 출력은 Git에서 제외한다.

Git에는 소스, 테스트, 문서, 실행/변환 스크립트, `pyproject.toml`, `.python-version`,
`uv.lock`을 저장한다. 가상환경, Python 캐시, FLAME 원본/다운로드/변환 파일,
`outputs/`, 로컬 `.env`는 `.gitignore`에서 제외한다. 새 checkout에서는 공식 모델을
`FLAME2023/`에 준비하고 `bash scripts/setup.sh`로 환경과 canonical asset을 재생성한다.
학습용 notebook과 문서에 넣을 그림은 추적할 수 있고, 실행 출력은 `outputs/`에 저장한다.

```bash
# 환경 설치 (uv 필요)
bash scripts/setup.sh

# 입력 파일/변환 설명
cat assets/flame/README.md

# 모델 변환과 UV template를 다시 준비하려면 (CPU)
bash scripts/prepare_flame.sh FLAME2023/flame2023.pkl

# 입력 파일과 topology만 검증 (CPU)
uv run --locked flame-uv --check-assets

# 물리 GPU 2에서 실행: 프로세스 내부에서는 cuda:0
bash scripts/run_gpu2.sh --resolution 256 --output outputs/flame_uv

# 실제 UV texture 입력 (없으면 기본 체크무늬)
bash scripts/run_gpu2.sh --texture assets/flame/texture.png
```

`CUDA_VISIBLE_DEVICES=2`를 실행 스크립트에서만 설정한다. CLI를 직접 실행하면 기본은 CPU다.
GPU 실행은 사용자가 위 스크립트를 실행할 때 시작된다.

초기 확인 결과: CPU 테스트 10개 통과, 실제 FLAME의 64×64 map 생성 성공
(`outputs/flame_cpu_check/`), 유효 texel 3,629개, `P=P_base+D` 최대 오차 약 3.2e-8.
GPU 실행/학습은 수행하지 않았다. 작은 해상도 결과는 설치와 출력 확인용이다.

각 map의 정의:

| Map | Raw shape | 저장하는 정보 | 기본 실험 |
|---|---|---|---|
| Texture | H × W × 3 | RGB, [0,1] | UV 체크무늬; 실제 피부 albedo가 아님 |
| Position | H × W × 3 | 변위 적용 후 canonical XYZ | P = P_base + D |
| Displacement | H × W × 3 | canonical 기준 표면에서의 XYZ 변위 | Geometry 공간의 작은 normal 방향 bump |
| Normal | H × W × 3 | 변위 적용 mesh의 object-space 단위 normal | 원래 mesh 연결관계에서 계산 후 보간·정규화 |
| Feature | H × W × C | 학습 전 CNN 출력 | 기본 C=16; 11 → 32 → C pointwise CNN |

모든 map의 픽셀 주소는 같다. 별도의 UV unwrap을 다섯 번 하는 구조가 아니다.
`Position`은 카메라 좌표나 depth map이 아니며, `Normal`도 tangent-space normal map이
아니다. Feature의 입력은 정규화한 position(3), displacement(3), normal(3), UV(2)다.
Seed로 초기화한 1×1 CNN은 학습하지 않았으므로 semantic feature라고 해석하면 안 된다.
1×1 kernel은 이 첫 실험에서 서로 다른 UV island 사이로 convolution이 새지 않게 한다.

Displacement는 vertex에서 정의한 뒤 보간한다. UV seam 양쪽의 같은 geometry 지점은
같은 변위를 받는다. 기본 amplitude는 **모델 단위로 0.002**이며, 입력 모델이 m 단위일 때
2 mm다. 임의 UV 위치에 주름을 생성하거나 세분화하는 모델은 아직 아니다. FLAME
identity 계수는 `--shape shape.npy`, 직접 만든 [V,3] 변위는 `--offsets offsets.npy`로 넣는다.

```bash
# 원리를 쉽게 보기 위한 실험: 변위 0 vs 확대
bash scripts/run_gpu2.sh --amplitude 0 --output outputs/no_displacement
bash scripts/run_gpu2.sh --amplitude 0.005 --output outputs/larger_displacement

# 환경 검증용 curved patch (FLAME 머리가 아닌 작은 synthetic fixture)
CUDA_VISIBLE_DEVICES='' uv run --locked flame-uv --synthetic --device cpu \
  --resolution 64 --output outputs/setup_check

# 수학/대응관계 검증; 실제 FLAME asset 없이 CPU에서 수행
CUDA_VISIBLE_DEVICES='' uv run --locked pytest -q
```

출력에서 `overview.png`는 다섯 map과 mask를 한 화면에 보여주고, `uv_layout.png`는
UV 삼각형 배치를 보여준다. `viewer.html`은 브라우저에서 열어 기준 mesh, 변위 mesh,
UV position 픽셀에서 복원한 point cloud를 회전하며 비교한다. Dropdown으로 point
cloud에 칠하는 속성을 바꿀 수 있다. Plotly를 HTML에 포함하므로 인터넷 연결이 필요 없다.
HTML은 최대 16,000점을 표시하고 raw 파일은 모든 유효 픽셀을 보존한다.

`maps.npz`와 각 `.npy`는 실제 float32 값(HWC)이며, `valid_mask`는 bool이다.
`base_position`도 저장해 `position ≈ base_position + displacement`를 확인할 수 있다.
PNG는 표시용이다: position은 채널별 min/max, displacement는 크기를 viridis 색으로,
normal은 (N+1)/2, feature는 처음 최대 3채널의 min/max를 시각화한다. PNG에서 raw
geometry 값을 복원하지 않는다. `metadata.json`에 입력 hash, convention, seed, device,
shape, reconstruction 오차를 기록하고, `feature_encoder.pt`에 초기화된 가중치를 저장한다.

`canonical_base.obj`와 `canonical_displaced.obj`는 원래 topology와 UV corner indices를
유지한다. `uv_reconstructed_points.obj`는 position map에서 얻은 **point cloud**다.
2D 격자의 인접 픽셀을 무조건 삼각형으로 연결하면 seam과 island를 잘못 연결하므로
복원 점에는 임의의 faces를 붙이지 않는다.

읽을 코드 순서는 [assets.py](src/flame_uv/assets.py) → [rasterize.py](src/flame_uv/rasterize.py)
→ [maps.py](src/flame_uv/maps.py) → [export.py](src/flame_uv/export.py)다.
핵심 관계는 `P(u,v) = b1*v1 + b2*v2 + b3*v3`이며,
pixel center convention은 `u=(col+0.5)/W`, `v=1-(row+0.5)/H`다.
빈 UV 영역은 mask=false/값 0으로 처리한다. 겹치는 UV island는 하나의 픽셀에 두
표면 지점을 저장할 수 없어서 오류로 알려준다. UV uniform sampling은 surface-area
uniform sampling과 다르고, 해상도를 올려도 원래 mesh에 없는 geometry detail이 생기지 않는다.

참고: [FLAME](https://flame.is.tue.mpg.de/),
[FLAME PyTorch](https://github.com/soubhiksanyal/FLAME_PyTorch),
[PRNet](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Yao_Feng_Joint_3D_Face_ECCV_2018_paper.php),
[PyTorch CUDA wheel 설치](https://pytorch.org/get-started/previous-versions/).
