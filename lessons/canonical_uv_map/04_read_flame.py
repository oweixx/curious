"""04: 실제 FLAME의 vertex/faces와 OBJ의 v/vt/f를 직접 읽는다.

필요: assets/flame/flame2023_canonical.npz, assets/flame/head_template.obj.
원본 pickle의 Chumpy 변환은 환경 준비 단계에서 이미 완료했다.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

root = Path(__file__).resolve().parents[2]
model_path = root / 'assets/flame/flame2023_canonical.npz'
obj_path = root / 'assets/flame/head_template.obj'
if not model_path.exists() or not obj_path.exists():
    raise FileNotFoundError('변환한 canonical NPZ와 matching UV OBJ를 assets/flame/에 준비하세요.')
with np.load(model_path, allow_pickle=False) as model:
    vertices = model['v_template'].astype(np.float64)
    faces = model['f'].astype(int)
    shapedirs = model['shapedirs'].astype(np.float64)

# OBJ의 v는 geometry, vt는 UV. f의 각 corner는 'v번호/vt번호/vn번호'다.
# OBJ는 1부터 세고 NumPy는 0부터 센다. 이 template은 양수 index/삼각형을 쓴다.
obj_vertices, uv, obj_faces, obj_uv_faces = [], [], [], []
for line in obj_path.read_text().splitlines():
    fields = line.split('#', 1)[0].split()
    if not fields:
        continue
    if fields[0] == 'v':
        obj_vertices.append([float(x) for x in fields[1:4]])
    elif fields[0] == 'vt':
        uv.append([float(x) for x in fields[1:3]])
    elif fields[0] == 'f':
        if len(fields) != 4:
            raise ValueError('이 예제는 triangulated OBJ만 읽는다.')
        geometry_indices, texture_indices = [], []
        for corner in fields[1:]:
            indices = corner.split('/')
            vi, ti = int(indices[0]), int(indices[1])
            if vi <= 0 or ti <= 0:
                raise ValueError('이 예제의 OBJ reader는 양수 index를 사용한다.')
            geometry_indices.append(vi - 1)
            texture_indices.append(ti - 1)
        obj_faces.append(geometry_indices)
        obj_uv_faces.append(texture_indices)
uv = np.array(uv)
obj_faces = np.array(obj_faces)
obj_uv_faces = np.array(obj_uv_faces)
if len(obj_vertices) != len(vertices) or obj_faces.shape != faces.shape:
    raise ValueError('FLAME과 UV template의 vertex/face 수가 다르다.')

# 두 파일의 face 순서가 같다는 가정을 하지 않는다.
# 같은 geometry triangle을 찾고 corner에 맞는 UV index도 정렬한다.
triangle_lookup = {}
for geo, tex in zip(obj_faces, obj_uv_faces):
    key = tuple(sorted(geo))
    if key in triangle_lookup or len(set(geo)) != 3:
        raise ValueError('Template에 중복/퇴화 geometry triangle이 있다.')
    triangle_lookup[key] = dict(zip(geo, tex))
uv_faces = []
seen = set()
for face in faces:
    key = tuple(sorted(face))
    if key not in triangle_lookup or key in seen:
        raise ValueError('모델과 template의 vertex indexing/topology가 다르다.')
    seen.add(key)
    uv_faces.append([triangle_lookup[key][vertex] for vertex in face])
uv_faces = np.array(uv_faces)
if uv.min() < -1e-6 or uv.max() > 1 + 1e-6:
    raise ValueError('UV atlas는 [0,1]^2 안에 있어야 한다.')
print('Vertices / faces / UV / UV faces:', vertices.shape, faces.shape, uv.shape, uv_faces.shape)
print('첫 face의 geometry index:', faces[0], 'UV index:', uv_faces[0])
print('그 face의 XYZ:\n', vertices[faces[0]])
print('그 face의 UV:\n', uv[uv_faces[0]])
print('같은 vertex가 여러 vt를 가질 수 있다. 그 이유는 12에서 seam으로 확인한다.')

fig = plt.figure(figsize=(11, 5))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(*vertices.T, s=1)
ax.set(xlabel='X', ylabel='Y', zlabel='Z', title='Canonical FLAME vertices')
ax.set_box_aspect(np.maximum(np.ptp(vertices, axis=0), 1e-6))
ax = fig.add_subplot(122)
triangles = uv[uv_faces]
edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
ax.add_collection(LineCollection(edges, linewidths=.2, colors='royalblue'))
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='u', ylabel='v', title='Matching UV atlas')
ax.set_aspect('equal')
out = root / 'outputs/canonical_uv_map/04'
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / 'flame_data.npz', vertices=vertices, faces=faces,
                    uv=uv, uv_faces=uv_faces, shapedirs=shapedirs)
fig.tight_layout()
fig.savefig(out / 'flame_data.png', dpi=150)
plt.close(fig)
# 실험: faces[0]과 uv_faces[0]의 index를 서로 바꾸면 왜 올바른 삼각형이 안 나올까?
