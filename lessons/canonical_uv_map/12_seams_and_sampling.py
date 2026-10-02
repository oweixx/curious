"""12: UV seam, 면적 왜곡, 해상도에 따른 sampling을 직접 관찰한다. 필요: 04.

하나의 geometry vertex에 여러 UV 주소가 붙는 경우를 찾고, 64/128 sampling을 비교한다.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

root = Path(__file__).resolve().parents[2]
data_path = root / 'outputs/canonical_uv_map/04/flame_data.npz'
if not data_path.exists():
    raise FileNotFoundError('먼저 04_read_flame.py를 실행하세요.')
with np.load(data_path) as data:
    vertices, faces, uv, uv_faces = [data[k] for k in ('vertices', 'faces', 'uv', 'uv_faces')]

# 1. 같은 geometry index에 연결된 texture index들을 모은다.
vertex_to_uv = [set() for _ in vertices]
for face, uv_face in zip(faces, uv_faces):
    for vi, ti in zip(face, uv_face):
        vertex_to_uv[vi].add(int(ti))
seam_vertices = [vi for vi, indices in enumerate(vertex_to_uv)
                 if len(indices) > 1 and np.ptp(uv[list(indices)], axis=0).max() > 1e-6]
if not seam_vertices:
    raise ValueError('이 template에는 분리된 UV 주소를 가진 vertex를 찾지 못했다.')
# UV 사이 거리가 큰 예를 골라 seam을 보기 쉽게 한다.
seam_vertex = max(seam_vertices, key=lambda vi: np.linalg.norm(np.ptp(uv[list(vertex_to_uv[vi])], axis=0)))
seam_uv = uv[sorted(vertex_to_uv[seam_vertex])]
print('Seam vertex 수:', len(seam_vertices))
print('한 geometry vertex:', seam_vertex, 'XYZ:', vertices[seam_vertex])
print('그 vertex의 여러 UV 주소:\n', seam_uv)

# 2. UV 면적 / 3D 면적이 클수록 UV uniform sampling의 표면 밀도가 커진다.
tri_uv, tri_3d = uv[uv_faces], vertices[faces]
uv_ab, uv_ac = tri_uv[:, 1] - tri_uv[:, 0], tri_uv[:, 2] - tri_uv[:, 0]
uv_area = .5 * np.abs(uv_ab[:, 0] * uv_ac[:, 1] - uv_ab[:, 1] * uv_ac[:, 0])
area_3d = .5 * np.linalg.norm(np.cross(tri_3d[:, 1] - tri_3d[:, 0],
                                     tri_3d[:, 2] - tri_3d[:, 0]), axis=1)
usable = (uv_area > 1e-15) & (area_3d > 1e-15)
density = uv_area[usable] / area_3d[usable]
relative_log_density = np.log10(density / np.median(density))
print('표면 sampling 상대 밀도 min/max:', (density / np.median(density)).min(),
      (density / np.median(density)).max())

# 3. 05의 계산을 두 해상도에서 반복한다. 실제 vertex/faces는 바꾸지 않는다.
counts = []
for resolution in [64, 128]:
    face_index = np.full((resolution, resolution), -1, dtype=int)
    for fi, triangle in enumerate(tri_uv):
        a, b, c = triangle
        ab, ac = b - a, c - a
        det = ab[0] * ac[1] - ab[1] * ac[0]
        if abs(det) < 1e-15:
            continue
        cols = triangle[:, 0] * resolution - .5
        rows = (1 - triangle[:, 1]) * resolution - .5
        for row in range(max(0, int(np.ceil(rows.min()))), min(resolution - 1, int(np.floor(rows.max()))) + 1):
            for col in range(max(0, int(np.ceil(cols.min()))), min(resolution - 1, int(np.floor(cols.max()))) + 1):
                if face_index[row, col] >= 0:
                    continue
                q = np.array([(col + .5) / resolution, 1 - (row + .5) / resolution]) - a
                wb = (q[0] * ac[1] - q[1] * ac[0]) / det
                wc = (ab[0] * q[1] - ab[1] * q[0]) / det
                if min(1 - wb - wc, wb, wc) >= -1e-10:
                    face_index[row, col] = fi
    mask = face_index >= 0
    sampled_faces = len(np.unique(face_index[mask]))
    counts.append([resolution, int(mask.sum()), sampled_faces])
    print('해상도 / 유효 픽셀 / sample된 face:', counts[-1])

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].scatter(*uv.T, s=.2, color='lightgray')
axes[0].scatter(*seam_uv.T, color='red', s=35)
axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='u', ylabel='v', title=f'Same 3D vertex {seam_vertex}, multiple UVs')
axes[0].set_aspect('equal')
patches = PolyCollection(tri_uv[usable], array=relative_log_density, cmap='coolwarm',
                         edgecolors='none', clim=(-2, 2))
axes[1].add_collection(patches)
axes[1].set(xlim=(0, 1), ylim=(0, 1), title='log10(surface density / median)')
axes[1].set_aspect('equal')
fig.colorbar(patches, ax=axes[1])
axes[2].bar(['64 x 64', '128 x 128'], [row[1] for row in counts])
axes[2].set(ylabel='Valid surface samples', title='More samples, same input geometry')
out = root / 'outputs/canonical_uv_map/12'
out.mkdir(parents=True, exist_ok=True)
np.savez(out / 'limitations.npz', seam_vertex=seam_vertex, seam_uv=seam_uv,
         vertex_position=vertices[seam_vertex], relative_log_density=relative_log_density,
         sampling_counts=np.array(counts))
fig.tight_layout()
fig.savefig(out / 'seams_and_sampling.png', dpi=150)
plt.close(fig)
# 실험/정리:
# - 같은 표면 지점이 UV의 양쪽 경계에 있다면 3x3 CNN은 그 이웃관계를 알 수 있을까?
# - 해상도를 올려도 FLAME에 없는 hair/teeth/주름의 geometry가 생기지는 않는다.
# - UV 면적 비율에 따라 loss를 보정할지, surface-area sampling할지 선택할 수 있다.
# - UV는 표면 parameterization이다. 부피 전체의 표현이나 카메라 visibility 정보는 아니다.
