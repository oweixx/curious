# Canonical UV Map: 12개 lesson

목표는 **UV 좌표가 표면의 주소라는 사실을 이해하고, 그 주소에 geometry와 appearance를
직접 저장하고 읽을 수 있게 되는 것**이다. 먼저 숫자가 작은 삼각형으로 원리를 확인하고,
그 계산을 실제 FLAME으로 확장한다. 각 `.py`를 위에서부터 읽고, 출력값을 예상하고,
입력값을 바꿔서 결과를 설명한다.

## 순서와 결과물

| 번호 | 파일 | 이번 단계에서 답할 질문 | 결과물 |
|---|---|---|---|
| 01 | [01_uv_triangle.py](01_uv_triangle.py) | UV 주소 하나를 어떻게 3D 위치로 바꾸는가? | Barycentric 가중치와 XYZ 출력 |
| 02 | [02_uv_position_map.py](02_uv_position_map.py) | 같은 계산을 모든 픽셀에 반복하면 무엇이 되는가? | 8×8 Position Map, mask, 복원한 3D 점 |
| 03 | [03_two_triangles.py](03_two_triangles.py) | 삼각형이 여럿일 때 어떤 face를 골라야 하는가? | Face index, 가중치, 접힌 표면의 Position Map |
| 04 | [04_read_flame.py](04_read_flame.py) | FLAME vertex와 OBJ의 v/vt/f는 어떻게 연결되는가? | 실제 FLAME 배열, 대응된 UV faces, atlas 그림 |
| 05 | [05_flame_position_map.py](05_flame_position_map.py) | 실제 머리 전체를 어떻게 UV 격자로 표현하는가? | Canonical Position, mask, face index, barycentric lookup |
| 06 | [06_texture_map.py](06_texture_map.py) | 같은 주소에 위치 대신 색을 담으면 어떻게 쓰이는가? | 체크무늬 Texture Map과 색칠된 3D 점 |
| 07 | [07_normal_map.py](07_normal_map.py) | 표면 방향은 어디에서 계산하고 어떻게 UV로 옮기는가? | Object-space Normal Map과 vertex normals |
| 08 | [08_displacement_map.py](08_displacement_map.py) | 기준 표면을 조금 움직이면 어떤 배열이 변하는가? | Displacement, 변위 적용 Position과 Normal, P=P_base+D 확인 |
| 09 | [09_canonical_and_expression.py](09_canonical_and_expression.py) | Canonical 기준과 표정/회전 상태는 UV에서 어떻게 연결되는가? | 같은 주소의 canonical/expression/rotation Position 비교 |
| 10 | [10_feature_map.py](10_feature_map.py) | 픽셀에 C개 특징을 담는다는 것은 무엇인가? | 8채널 Feature Map, network weights, 선택적 복원 학습 |
| 11 | [11_collect_five_maps.py](11_collect_five_maps.py) | 다섯 종류의 정보를 같은 표면에 어떻게 연결하는가? | Five-map NPZ, 비교 그림, OBJ point cloud, 회전 가능한 HTML |
| 12 | [12_seams_and_sampling.py](12_seams_and_sampling.py) | UV 표현은 어떤 이웃관계와 면적 정보를 왜곡하는가? | Seam의 여러 주소, 면적별 밀도, 64/128 sampling 비교 |

01–03은 작은 숫자로 원리를 배우고, 04–05는 실제 데이터로 옮긴다.
06–10에서는 각 표현의 의미를 분리해서 관찰한다. 11에서 모은 뒤, 12에서 연구에 쓸 때의
제약을 이해한다. 아직 배우지 않은 단계의 코드는 미리 준비했으며, 읽는 순서를 생략할
필요는 없다. 앞 파일을 import하거나 앞 단계를 자동으로 실행하지 않는다.

## 최종 목표와 범위

완료하면 다음을 직접 설명하고 수정할 수 있어야 한다.

- 3D vertex 좌표와 UV 주소의 차이, face/UV-face index가 필요한 이유.
- Barycentric 보간, 픽셀 중심의 좌표 convention, 유효 영역 mask.
- Position / Texture / Normal / Displacement / Feature 채널의 서로 다른 의미.
- 고정된 UV 주소에 다른 표정/pose의 XYZ를 저장할 수 있는 이유.
- Seam과 UV 면적 왜곡이 CNN, sampling, loss 설계에 미치는 영향.

최종 결과는 `outputs/canonical_uv_map/11/five_maps.npz`다.

| 배열 | Shape | 정의 |
|---|---|---|
| texture | H×W×3 | RGB [0,1]; 기본은 synthetic checker |
| position | H×W×3 | 변위 적용 canonical XYZ |
| displacement | H×W×3 | 기준 canonical 표면에서의 XYZ 변위 |
| normal | H×W×3 | 변위 적용 표면의 object-space 단위 normal |
| feature | H×W×8 | 1×1 CNN의 출력; 기본은 학습 전 특징 |
| base_position | H×W×3 | 변위 전 canonical XYZ |
| valid_mask | H×W | 유효 표면 여부 |
| face_index | H×W | 픽셀에 대응하는 triangle 번호; 무효는 -1 |
| barycentric | H×W×3 | 해당 triangle의 세 꼭짓점에 대한 가중치 |

`five_maps.png`는 전체 비교, `points.obj`와 `points.html`은 Position에서 복원한 점들이다.
12에서는 표현의 한계를 정리하는 별도 그림과 수치도 얻는다.

이 과정은 FLAME의 평균 identity, expression=0, pose=0을 기준으로 시작한다.
09의 expression basis와 global 회전은 대응관계 관찰을 위한 최소 변형이다.
전체 FLAME LBS/턱 articulation, inverse deformation, 얼굴 사진에서의 fitting,
실제 피부 texture 복원, hair/teeth 모델링, 완성된 avatar renderer는 다음 학습 주제다.
10의 선택적 복원 학습은 semantic feature나 identity feature를 자동으로 만들지는 않는다.

## 진행 방법

Repo root에서 각 파일을 직접 실행한다. 예:

```bash
python lessons/canonical_uv_map/03_two_triangles.py
python lessons/canonical_uv_map/04_read_flame.py
python lessons/canonical_uv_map/05_flame_position_map.py
```

01–03은 독립이다. 04는 로컬 FLAME asset을 읽고, 05는 04의 출력을 읽는다.
06/07/09는 04–05 준비 후 실행할 수 있다. 08은 07도 필요하다.
10은 06/08을, 11은 05/06/08/10을 읽는다. 12는 04만 있으면 된다.
번호순으로 한 파일씩 실행하면 모든 선행 자료가 준비된다.

각 파일의 `height/width`, 좌표, amplitude, expression coefficient, feature 채널 수 등을
직접 바꾼다. 05의 해상도를 바꾸면 06–11에서 사용하는 lookup의 크기가 달라지므로
후속 단계를 다시 실행한다. Lesson 번호별 출력은 덮어써지고 `outputs/` 전체는 Git에서
제외된다. `.py` 파일과 이 안내 문서는 Git에 포함한다. README는 사용자의 짧은 학습
로그로 유지한다.

01–09, 11–12는 CPU 계산으로 충분하다. 10도 기본은 CPU이고 `TRAIN_STEPS=0`이다.
10의 `device`를 `'cuda:0'`으로 바꾼 경우 물리 GPU 2는 다음처럼 지정한다.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python lessons/canonical_uv_map/10_feature_map.py
```

## 환경과 입력 자료

기존에 사용하는 Python/Conda 환경에서 필요한 라이브러리를 설치한다.
가상환경 생성이나 repo 패키지 설치 과정은 없다.

```bash
python -m pip install -r requirements.txt
python lessons/canonical_uv_map/02_uv_position_map.py
```

`requirements.txt`에는 전체 lesson에서 사용하는 NumPy, Matplotlib, Pillow, PyTorch,
Plotly만 적었다. 01–05, 07–09, 12에는 NumPy/Matplotlib이면 충분하고, Pillow는 06,
PyTorch는 10, Plotly는 11에서 사용한다. PyTorch의 CUDA 지원은 사용하는 환경에 맞춘다.
현재 셸의 기본 Python에는 이 라이브러리들이 설치되어 있지 않다.

FLAME 원본은 `FLAME2023/`에 보존했다. 실제 lesson 입력은 이미 준비한 아래 두 파일이다.

```text
assets/flame/flame2023_canonical.npz
assets/flame/head_template.obj
```

Canonical NPZ는 원본의 v_template/f/shapedirs를 NumPy로 변환한 자료다. UV OBJ는
[공식 DECA repository의 고정 commit](https://github.com/yfeng95/DECA/blob/a11554ae2a2b0f3998cf1fa94dd4db03babb34a2/data/head_template.obj)에서
준비했다. 04에서 geometry/UV topology를 직접 확인한다. 모델/UV asset은 Git에 포함되지
않으므로 새 checkout에서는 따로 준비해야 한다. 원본 Chumpy pickle의 변환은 별도 환경
호환 작업이고, 현재 lesson을 실행할 때는 필요하지 않다.

이 폴더의 예제는 이전 `scripts`, `src`, `tests`와 독립적이다.
