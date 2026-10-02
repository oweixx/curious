"""06: 같은 UV 주소에 XYZ 대신 RGB를 저장한다. 먼저 04, 05를 실행한다.

기본은 직접 만드는 체크무늬. 피부 texture를 복원하는 예제는 아니다.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

root = Path(__file__).resolve().parents[2]
lookup_path = root / 'outputs/canonical_uv_map/05/lookup.npz'
if not lookup_path.exists():
    raise FileNotFoundError('먼저 05_flame_position_map.py를 실행하세요.')
with np.load(lookup_path) as data:
    position, mask, uv_grid = [data[k] for k in ('position', 'valid_mask', 'uv_grid')]
height, width = mask.shape
texture = np.zeros((height, width, 3))
num_tiles = 16
color_a = np.array([.15, .45, .8])
color_b = np.array([.95, .75, .3])
for row in range(height):
    for col in range(width):
        if not mask[row, col]:
            continue
        u, v = uv_grid[row, col]
        tile_u = int(np.floor(u * num_tiles))
        tile_v = int(np.floor(v * num_tiles))
        texture[row, col] = color_a if (tile_u + tile_v) % 2 == 0 else color_b

# 선택: 이미 동일한 UV atlas로 펼쳐진 RGB 이미지를 사용한다.
# 얼굴 사진을 넣으면 올바른 texture가 되지 않는다. 사진 → UV는 별도 대응관계가 필요하다.
texture_path = None  # 예: root / 'assets/flame/texture.png'
if texture_path is not None:
    with Image.open(texture_path) as image:
        texture = np.asarray(image.convert('RGB').resize((width, height),
                             Image.Resampling.BILINEAR), dtype=float) / 255
    texture[~mask] = 0

# Position과 Texture의 같은 [row,col]을 동시에 읽으면 위치와 색이 연결된다.
points = position[mask]
colors = texture[mask]
sample = np.arange(0, len(points), max(1, len(points) // 8000))
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(121)
ax.imshow(texture)
ax.set_title('RGB values in UV space')
ax = fig.add_subplot(122, projection='3d')
ax.scatter(*points[sample].T, c=colors[sample], s=2)
ax.set(xlabel='X', ylabel='Y', zlabel='Z', title='Same UV addresses: position + color')
ax.set_box_aspect(np.maximum(np.ptp(points, axis=0), 1e-6))
out = root / 'outputs/canonical_uv_map/06'
out.mkdir(parents=True, exist_ok=True)
np.save(out / 'texture.npy', texture)
fig.tight_layout()
fig.savefig(out / 'texture.png', dpi=150)
plt.close(fig)
print('Texture shape:', texture.shape, '정의: RGB [0,1], 기본은 synthetic checker')
# 실험: num_tiles를 바꾸면 geometry의 위치도 바뀔까?
# Position과 Texture의 shape가 같아도 각 채널의 의미는 다르다.
