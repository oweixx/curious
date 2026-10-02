"""05: FLAME 전체 표면을 UV Position Map으로 만든다. 먼저 04를 실행한다.

03처럼 삼각형을 순회한다. 차이는 각 UV 삼각형의 bounding box 안의 픽셀만 보는 것.
핵심 계산은 그대로 이 파일 안에 있다. GPU와 외부 rasterizer는 사용하지 않는다.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
input_path = root / 'outputs/canonical_uv_map/04/flame_data.npz'
if not input_path.exists():
    raise FileNotFoundError('먼저 04_read_flame.py를 실행하세요.')
with np.load(input_path) as data:
    vertices, faces, uv, uv_faces = [data[key] for key in ('vertices', 'faces', 'uv', 'uv_faces')]
height = width = 128  # 64부터 관찰해도 좋다. 올리면 CPU 반복문이 더 오래 걸린다.
face_index = np.full((height, width), -1, dtype=int)
barycentric = np.zeros((height, width, 3))
position = np.zeros((height, width, 3))
uv_grid = np.zeros((height, width, 2))
for row in range(height):
    for col in range(width):
        uv_grid[row, col] = [(col + .5) / width, 1 - (row + .5) / height]

for fi in range(len(faces)):
    triangle_uv = uv[uv_faces[fi]]
    a, b, c = triangle_uv
    ab, ac = b - a, c - a
    det = ab[0] * ac[1] - ab[1] * ac[0]
    if abs(det) < 1e-15:
        continue
    # UV → pixel center index. v 방향을 뒤집고, 중심 offset 0.5를 뺀다.
    cols = triangle_uv[:, 0] * width - .5
    rows = (1 - triangle_uv[:, 1]) * height - .5
    col_min = max(0, int(np.ceil(cols.min())))
    col_max = min(width - 1, int(np.floor(cols.max())))
    row_min = max(0, int(np.ceil(rows.min())))
    row_max = min(height - 1, int(np.floor(rows.max())))
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            if face_index[row, col] >= 0:
                continue  # 공유 경계는 첫 face에 배정. 겹치는 atlas는 지원하지 않는다.
            q = uv_grid[row, col] - a
            wb = (q[0] * ac[1] - q[1] * ac[0]) / det
            wc = (ab[0] * q[1] - ab[1] * q[0]) / det
            wa = 1 - wb - wc
            if min(wa, wb, wc) < -1e-10:
                continue
            face_index[row, col] = fi
            barycentric[row, col] = [wa, wb, wc]
            va, vb, vc = vertices[faces[fi]]
            position[row, col] = wa * va + wb * vb + wc * vc

valid_mask = face_index >= 0
print('Position shape / 유효 픽셀 수:', position.shape, valid_mask.sum())
print('XYZ 범위:', position[valid_mask].min(0), position[valid_mask].max(0))
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(valid_mask, cmap='gray')
axes[0].set_title('Valid mask')
for channel in range(3):
    values = np.ma.array(position[..., channel], mask=~valid_mask)
    image = axes[channel + 1].imshow(values, cmap='viridis')
    axes[channel + 1].set_title(['X', 'Y', 'Z'][channel])
    fig.colorbar(image, ax=axes[channel + 1], shrink=.7)
out = root / 'outputs/canonical_uv_map/05'
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / 'lookup.npz', position=position, valid_mask=valid_mask,
                    face_index=face_index, barycentric=barycentric, uv_grid=uv_grid)
fig.tight_layout()
fig.savefig(out / 'position_channels.png', dpi=150)
plt.close(fig)
# 실험: bounding box 없이 모든 face × 모든 pixel을 확인하면 계산량이 얼마나 늘어날까?
# 좌표계/모델 단위는 원본의 것을 유지한다. Position은 RGB나 camera depth가 아니다.
