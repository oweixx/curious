"""07: 삼각형의 방향에서 vertex normal을 구하고 UV에 저장한다. 필요: 04, 05.

Object-space normal이다. Tangent-space normal texture나 shading은 아직 다루지 않는다.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
data_path = root / 'outputs/canonical_uv_map/04/flame_data.npz'
lookup_path = root / 'outputs/canonical_uv_map/05/lookup.npz'
if not data_path.exists() or not lookup_path.exists():
    raise FileNotFoundError('먼저 04와 05를 실행하세요.')
with np.load(data_path) as data:
    vertices, faces = data['vertices'], data['faces']
with np.load(lookup_path) as data:
    mask, face_index, barycentric = [data[k] for k in ('valid_mask', 'face_index', 'barycentric')]

# AB × AC는 삼각형에 수직이고 길이는 삼각형 면적의 두 배다.
# 정규화하지 않고 더하면 큰 삼각형이 더 기여하는 area-weighted normal이 된다.
vertex_normals = np.zeros_like(vertices)
for face in faces:
    a, b, c = vertices[face]
    face_normal = np.cross(b - a, c - a)
    for vertex_index in face:
        vertex_normals[vertex_index] += face_normal
length = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
vertex_normals /= np.maximum(length, 1e-12)

# UV 인접 픽셀로 normal을 계산하지 않는다. 원래 mesh의 연결관계를 사용했다.
height, width = mask.shape
normal_map = np.zeros((height, width, 3))
for row in range(height):
    for col in range(width):
        if not mask[row, col]:
            continue
        fi = face_index[row, col]
        wa, wb, wc = barycentric[row, col]
        na, nb, nc = vertex_normals[faces[fi]]
        normal = wa * na + wb * nb + wc * nc
        normal_map[row, col] = normal / max(np.linalg.norm(normal), 1e-12)

valid_lengths = np.linalg.norm(normal_map[mask], axis=1)
print('유효 normal의 길이 min/max:', valid_lengths.min(), valid_lengths.max())
# [-1,1]의 XYZ 방향을 [0,1] RGB로 바꾸는 것은 표시를 위한 변환일 뿐이다.
display = (normal_map + 1) / 2
display[~mask] = 0
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].imshow(display)
axes[0].set_title('Object-space normal: (N + 1) / 2')
image = axes[1].imshow(np.ma.array(normal_map[..., 2], mask=~mask),
                       cmap='coolwarm', vmin=-1, vmax=1)
axes[1].set_title('Normal Z component')
fig.colorbar(image, ax=axes[1])
out = root / 'outputs/canonical_uv_map/07'
out.mkdir(parents=True, exist_ok=True)
np.save(out / 'vertex_normals.npy', vertex_normals)
np.save(out / 'normal.npy', normal_map)
fig.tight_layout()
fig.savefig(out / 'normal.png', dpi=150)
plt.close(fig)
# 실험: cross의 두 입력을 반대로 하면 normal의 어떤 값이 달라질까?
# 보간한 normal을 다시 정규화하는 이유는 무엇일까?
