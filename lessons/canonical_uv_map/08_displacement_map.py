"""08: 기준 vertex를 조금 움직이고 D와 P_base+D를 비교한다. 필요: 04,05,07.

UV 주소와 face 연결관계는 유지된다. 변위는 geometry vertex에서 정의한다.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
data_path = root / 'outputs/canonical_uv_map/04/flame_data.npz'
lookup_path = root / 'outputs/canonical_uv_map/05/lookup.npz'
normal_path = root / 'outputs/canonical_uv_map/07/vertex_normals.npy'
if not all(path.exists() for path in (data_path, lookup_path, normal_path)):
    raise FileNotFoundError('먼저 04, 05, 07을 실행하세요.')
with np.load(data_path) as data:
    vertices, faces = data['vertices'], data['faces']
with np.load(lookup_path) as data:
    base_position, mask, face_index, barycentric = [data[k] for k in
                       ('position', 'valid_mask', 'face_index', 'barycentric')]
vertex_normals = np.load(normal_path)

# XY 위치에 따라 작은 bump를 만든다. 이 bump는 머리의 앞/뒤 양쪽에 생길 수 있다.
# 같은 geometry vertex의 UV seam 양쪽에는 동일한 변위가 적용된다.
amplitude = .005  # 모델 단위. 모델이 m 단위일 때 2 mm다.
center = (vertices.min(0) + vertices.max(0)) / 2
extent = np.maximum(np.ptp(vertices, axis=0), 1e-6)
xy = (vertices[:, :2] - center[:2]) / extent[:2]
height_value = amplitude * np.exp(-((xy[:, 0] / .22)**2 + (xy[:, 1] / .22)**2))
vertex_offsets = vertex_normals * height_value[:, None]
displaced_vertices = vertices + vertex_offsets

# 움직인 표면에서는 normal도 다시 계산한다. 07과 같은 계산이다.
new_vertex_normals = np.zeros_like(vertices)
for face in faces:
    a, b, c = displaced_vertices[face]
    face_normal = np.cross(b - a, c - a)
    for vi in face:
        new_vertex_normals[vi] += face_normal
new_vertex_normals /= np.maximum(np.linalg.norm(new_vertex_normals, axis=1, keepdims=True), 1e-12)

h, w = mask.shape
displacement = np.zeros((h, w, 3))
displaced_position = np.zeros((h, w, 3))
displaced_normal = np.zeros((h, w, 3))
for row in range(h):
    for col in range(w):
        if not mask[row, col]:
            continue
        face = faces[face_index[row, col]]
        weights = barycentric[row, col]
        # 한 채널별 계산을 행렬 곱으로 적는다: [3] @ [3,3] → [3].
        displacement[row, col] = weights @ vertex_offsets[face]
        displaced_position[row, col] = weights @ displaced_vertices[face]
        normal = weights @ new_vertex_normals[face]
        displaced_normal[row, col] = normal / max(np.linalg.norm(normal), 1e-12)

error = np.abs(displaced_position - base_position - displacement)[mask].max()
print('P_displaced = P_base + D 최대 오차:', error)
points_base, points_new = base_position[mask], displaced_position[mask]
sample = np.arange(0, len(points_base), max(1, len(points_base) // 5000))
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(121)
image = ax.imshow(np.ma.array(np.linalg.norm(displacement, axis=-1), mask=~mask), cmap='viridis')
ax.set_title('Displacement magnitude (model units)')
fig.colorbar(image, ax=ax)
ax = fig.add_subplot(122, projection='3d')
ax.scatter(*points_base[sample].T, s=1, label='Base')
ax.scatter(*points_new[sample].T, s=1, label='Displaced')
ax.set_box_aspect(np.maximum(np.ptp(vertices, axis=0), 1e-6))
ax.set(xlabel='X', ylabel='Y', zlabel='Z', title='Same UV addresses; changed XYZ')
ax.legend()
out = root / 'outputs/canonical_uv_map/08'
out.mkdir(parents=True, exist_ok=True)
for name, values in [('displacement', displacement), ('position', displaced_position),
                     ('normal', displaced_normal), ('vertices', displaced_vertices)]:
    np.save(out / f'{name}.npy', values)
fig.tight_layout()
fig.savefig(out / 'displacement.png', dpi=150)
plt.close(fig)
# 실험: amplitude=0을 먼저 확인하고, .005로 확대해보자.
# 새 vertex/triangle을 추가하지 않았으므로 원래 mesh 해상도가 detail의 한계다.
