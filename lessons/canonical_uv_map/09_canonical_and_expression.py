"""09: UV 주소가 고정되어 있어도 expression/pose에 따라 XYZ는 변할 수 있다.

필요: 04,05. FLAME2023의 expression basis 하나와 단순한 머리 전체 회전을 사용한다.
전체 FLAME LBS, 턱 articulation, tracking, inverse deformation은 구현하지 않는다.
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
    vertices, faces, shapedirs = [data[k] for k in ('vertices', 'faces', 'shapedirs')]
with np.load(lookup_path) as data:
    base_position, mask, face_index, barycentric = [data[k] for k in
                       ('position', 'valid_mask', 'face_index', 'barycentric')]

# 이번 FLAME2023 파일은 identity 300개 뒤에 expression 100개가 있다.
# 이 coefficient는 '미소 강도'처럼 이름 붙은 animation control이 아니라 PCA 계수다.
expression_index = 0
expression_coefficient = 2.0
if shapedirs.shape[2] != 400 or not 0 <= expression_index < 100:
    raise ValueError('이 예제는 300 identity + 100 expression인 FLAME2023를 사용한다.')
expressed_vertices = vertices + expression_coefficient * shapedirs[:, :, 300 + expression_index]

# 단순한 global 회전. 기본 0도에서 expression만 먼저 확인한 뒤 30도로 바꿔본다.
angle_degrees = 0.0
angle = np.deg2rad(angle_degrees)
rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])
posed_vertices = expressed_vertices @ rotation.T  # row-vector convention
expression_position = np.zeros_like(base_position)
posed_position = np.zeros_like(base_position)
h, w = mask.shape
for row in range(h):
    for col in range(w):
        if not mask[row, col]:
            continue
        face = faces[face_index[row, col]]
        weights = barycentric[row, col]
        expression_position[row, col] = weights @ expressed_vertices[face]
        posed_position[row, col] = weights @ posed_vertices[face]

# 05의 face_index/barycentric을 그대로 썼다. 다시 unwrap할 필요가 없다.
print('Expression에 의한 최대 위치 변화:', np.linalg.norm((expression_position - base_position)[mask], axis=1).max())
print('Global rotation에 의한 최대 위치 변화:', np.linalg.norm((posed_position - expression_position)[mask], axis=1).max())
fig = plt.figure(figsize=(12, 4))
all_vertices = np.concatenate([vertices, expressed_vertices, posed_vertices])
low, high = all_vertices.min(0) - .01, all_vertices.max(0) + .01
for col, (name, position) in enumerate([('Canonical', base_position),
                     ('Expression', expression_position), ('Expression + rotation', posed_position)], 1):
    ax = fig.add_subplot(1, 3, col, projection='3d')
    points = position[mask][::max(1, int(mask.sum()) // 6000)]
    ax.scatter(*points.T, s=1)
    ax.set_title(name)
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
    ax.set_box_aspect(high - low)
out = root / 'outputs/canonical_uv_map/09'
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / 'states.npz', canonical=base_position,
                    expression=expression_position, posed=posed_position, valid_mask=mask)
fig.tight_layout()
fig.savefig(out / 'canonical_vs_expression.png', dpi=150)
plt.close(fig)
# 실험: coefficient=0, angle=0일 때 기준 Position과 같아지는지 확인하자.
# Canonical은 선택한 기준 상태다. 동일한 UV atlas에 posed XYZ를 담을 수도 있다.
# Identity는 사람별로 달라도 된다. 이번 기준은 평균 identity / expression=0 / pose=0이다.
