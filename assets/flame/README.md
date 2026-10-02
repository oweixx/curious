# FLAME 입력 파일

이 폴더에 모델과 UV atlas를 넣으면 `scripts/run_gpu2.sh`가 기본 경로로 읽는다.
모델 파일은 Git에 포함하지 않는다.

```text
assets/flame/
├── flame2023_canonical.npz  # v_template, f, shapedirs: legacy pickle에서 변환
├── head_template.obj   # 모델과 같은 vertex indexing/topology, v/vt/f 포함
└── texture.png         # 선택: 위 OBJ의 UV 배치를 사용하는 texture
```

현재 입력은 사용자가 제공한 `FLAME2023/flame2023.pkl`이다. 5,023 vertices와 9,976 faces,
400개 shape/expression basis가 있다. `flame2023_no_jaw.pkl`도 있지만 이 실험에서는
`flame2023.pkl`을 사용한다. 원본은 수정하지 않았다.

원본의 Chumpy 객체를 분리된 `.venv-legacy`(Python 3.10, NumPy 1.23.5, Chumpy 0.70)에서
NumPy로 평가하고, 이 실험에 필요한 세 배열만 canonical NPZ에 저장한다. 현재 모델은
메인 `.venv`에서 Chumpy 없이 읽을 수 있다. 재생성:

```bash
bash scripts/prepare_flame.sh FLAME2023/flame2023.pkl
```

UV template은 [공식 DECA repository의 head_template.obj](https://github.com/yfeng95/DECA/blob/a11554ae2a2b0f3998cf1fa94dd4db03babb34a2/data/head_template.obj)를
고정 commit에서 내려받았다. 5,118 UV vertices를 사용하며, 제공된 모델과 geometry
vertex indexing/triangle topology가 일치함을 확인했다. `scripts/fetch_uv_template.py`에
URL과 SHA-256이 고정되어 있고, `.source.json`에도 출처를 저장한다.

다른 모델을 쓸 경우 [공식 FLAME 다운로드](https://flame.is.tue.mpg.de/download.php)에서
받은 호환 파일과 해당 UV atlas를 지정한다.
`--model /path/to/model.pkl --uv-template /path/to/template.obj` 또는 변환한 NPZ를 지원한다.

UV는 임의의 얼굴 OBJ가 아니라 **해당 FLAME 모델과 대응되는 triangulated template**을
사용해야 한다. 파일에 `vt`와 `f v/vt/...`가 있어야 한다. 코드가 모델과 template의
삼각형 집합을 비교하고, face/corner 순서 차이를 정렬한다. Vertex 번호 자체가 다르면
자동으로 대응을 추정하지 않고 오류를 낸다. UV seam의 v/vt 중복은 유지한다.

NPZ도 `--uv-template`으로 지원한다. 키는 `f`(geometry triangles), `vt`(UV coordinates),
`ft`(UV triangle indices)이며, 모두 zero-based indexing이다. `vt/ft`만 있는 texture
archive는 geometry correspondence를 검증할 수 없어 직접 입력으로 받지 않는다.
그 경우 함께 제공되는 matching OBJ를 사용한다.

이 구현은 기본적으로 `v_template`을 사용한다. Shape 계수가 있으면 identity
shapedirs의 처음 K개를 더한다(K <= 300). Expression/pose/LBS는 구현 범위에 포함하지
않으며, expression과 pose가 0인 canonical 상태를 다룬다. 표정이 있는 fitted mesh를
neutral 상태로 되돌리는 기능은 아니다. Shape 계수의 모델 버전을 맞춰야 한다.

일부 과거 pickle에는 Chumpy 객체가 들어 있다. 메인 환경은 Chumpy를 설치하지 않는다.
위 변환 스크립트를 사용하면 된다. 공식/신뢰할 수 있는 로컬 모델 pickle만 사용한다.
Canonical NPZ는 `allow_pickle=False`로 읽는다.

파일을 넣은 후 GPU 사용 없이 확인:

```bash
uv run --locked flame-uv --check-assets
```

Texture는 이미 이 UV atlas로 펼쳐진 이미지여야 한다. 얼굴 사진을 `--texture`에
넣으면 올바른 대응이 만들어지지 않는다. 사진에서 UV texture를 추출하려면 카메라,
mesh fitting, visibility, image sampling 단계가 별도로 필요하다.
