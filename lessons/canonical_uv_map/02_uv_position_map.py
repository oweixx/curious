"""UV 주소 하나에서 픽셀 격자 전체로: 첫 Position Map 만들기.

01의 barycentric 계산을 for문으로 반복한다. 함수/클래스나 기존 src를 사용하지 않는다.
읽는 순서: 입력 삼각형 → 빈 배열 → 픽셀의 UV 주소 → 내부 판정 → XYZ 저장 → 시각화.
실행: python lessons/canonical_uv_map/02_uv_position_map.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 화면 없는 서버에서도 PNG로 관찰할 수 있다.
import matplotlib.pyplot as plt

# 1. 01과 같은 삼각형이다. 각 행의 3D 꼭짓점과 UV 꼭짓점이 대응한다.
vertices_3d = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
vertices_uv = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
a_3d, b_3d, c_3d = vertices_3d
a_uv, b_uv, c_uv = vertices_uv
edge_ab = b_uv - a_uv
edge_ac = c_uv - a_uv
determinant = edge_ab[0] * edge_ac[1] - edge_ab[1] * edge_ac[0]
if abs(determinant) < 1e-12:
    raise ValueError('UV 삼각형이 선/점으로 납작해졌다.')

# 2. 처음에는 8x8로 작게 만든다. 3은 RGB가 아니라 X/Y/Z 채널의 수다.
height, width = 8, 8
position_map = np.zeros((height, width, 3))
valid_mask = np.zeros((height, width), dtype=bool)
uv_grid = np.zeros((height, width, 2))

# 값이 0이라고 무효인 것은 아니다: [0, 0, 0]도 유효한 3D 위치일 수 있다.
# 그래서 표면에 대응되는 픽셀인지는 valid_mask에 따로 저장한다.

# 3. 각 픽셀의 중심에 UV 주소를 붙이고, 01의 계산을 반복한다.
for row in range(height):
    for col in range(width):
        # 이미지의 row는 아래로 증가하지만 UV의 v는 위로 증가하도록 정한다.
        # +0.5는 픽셀의 모서리 대신 중심을 sample한다는 뜻이다.
        u = (col + 0.5) / width
        v = 1.0 - (row + 0.5) / height
        point_uv = np.array([u, v])
        uv_grid[row, col] = point_uv
        from_a = point_uv - a_uv

        weight_b = (from_a[0] * edge_ac[1] - from_a[1] * edge_ac[0]) / determinant
        weight_c = (edge_ab[0] * from_a[1] - edge_ab[1] * from_a[0]) / determinant
        weight_a = 1.0 - weight_b - weight_c

        # 음수 가중치가 있으면 삼각형 밖이다. 그 픽셀에는 표면 위치를 저장하지 않는다.
        inside = min(weight_a, weight_b, weight_c) >= -1e-12
        if not inside:
            continue

        point_3d = weight_a * a_3d + weight_b * b_3d + weight_c * c_3d
        position_map[row, col] = point_3d
        valid_mask[row, col] = True

# 4. 배열에서 위치 정보를 다시 꺼낸다. 이것은 mesh가 아니라 표면 sample의 점들이다.
points_3d = position_map[valid_mask]
print('Position Map shape:', position_map.shape)
print('유효 픽셀 수:', valid_mask.sum(), '/', height * width)
print('row=4, col=2의 UV:', uv_grid[4, 2])
print('그 픽셀의 XYZ:', position_map[4, 2], '유효한가?', valid_mask[4, 2])
print('Z 채널 (삼각형 밖은 NaN으로 표시):')
print(np.where(valid_mask, position_map[:, :, 2], np.nan))

# 5. 주소의 배치, 저장된 값, 그 값에서 복원한 점을 나란히 본다.
# channel=0/1/2로 바꿔 X/Y/Z를 각각 관찰할 수 있다.
channel = 2
channel_name = ['X', 'Y', 'Z'][channel]
fig = plt.figure(figsize=(13, 4))
ax_uv = fig.add_subplot(131)
outline_uv = vertices_uv[[0, 1, 2, 0]]
ax_uv.plot(outline_uv[:, 0], outline_uv[:, 1], color='black')
ax_uv.scatter(uv_grid[~valid_mask, 0], uv_grid[~valid_mask, 1], color='lightgray', label='Outside')
ax_uv.scatter(uv_grid[valid_mask, 0], uv_grid[valid_mask, 1], color='royalblue', label='Valid')
ax_uv.set(xlabel='u', ylabel='v', title='UV pixel centers', xlim=(0, 1), ylim=(0, 1))
ax_uv.set_aspect('equal')
ax_uv.legend()

ax_map = fig.add_subplot(132)
colors = plt.get_cmap('viridis').copy()
colors.set_bad('black')
values = np.ma.array(position_map[:, :, channel], mask=~valid_mask)
image = ax_map.imshow(values, origin='upper', interpolation='nearest', cmap=colors)
ax_map.set(xlabel='Column', ylabel='Row', title=f'Position map: {channel_name} (black = invalid)')
fig.colorbar(image, ax=ax_map, label=f'{channel_name} coordinate')

ax_3d = fig.add_subplot(133, projection='3d')
outline_3d = vertices_3d[[0, 1, 2, 0]]
ax_3d.plot(outline_3d[:, 0], outline_3d[:, 1], outline_3d[:, 2], color='black')
ax_3d.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], color='royalblue')
ax_3d.set(xlabel='X', ylabel='Y', zlabel='Z', title='Points recovered from position map')
extent = np.ptp(vertices_3d, axis=0)
ax_3d.set_box_aspect(np.maximum(extent, 1e-6))
fig.tight_layout()

# 6. PNG는 숫자를 색으로 표시한 그림이다. 실제 XYZ 값은 .npy에 따로 저장한다.
output_dir = Path(__file__).resolve().parents[2] / 'outputs' / 'canonical_uv_map' / '02'
output_dir.mkdir(parents=True, exist_ok=True)
np.save(output_dir / 'position_map.npy', position_map)
np.save(output_dir / 'valid_mask.npy', valid_mask)
fig.savefig(output_dir / 'position_map.png', dpi=160)
plt.close(fig)
print('출력 위치:', output_dir)

# 직접 바꿔볼 것:
# - 8x8을 16x16으로 바꾸면 새 geometry가 생길까, 같은 표면을 더 촘촘히 sample할까?
# - C의 Z를 바꾸면 valid_mask와 Position Map 중 무엇이 달라질까?
# - v에서 1.0 - 를 제거하면 배열의 어느 방향이 뒤집힐까?
# - 삼각형 밖 픽셀도 보간하면 왜 표면이 아닌 점이 생길까?
#
# 다음 단계: 삼각형을 두 개로 늘리고 face index가 왜 필요한지 확인한다.
# 이후 FLAME의 실제 vertex/faces/UV를 읽어 같은 계산으로 확장한다.
