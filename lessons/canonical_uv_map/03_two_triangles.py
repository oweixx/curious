"""03: 삼각형이 둘이면 UV 픽셀은 어느 face에 속할까?

02의 반복문에 face 순회를 추가한다. 공유 경계는 먼저 만난 face를 택한다.
실행: python lessons/canonical_uv_map/03_two_triangles.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 사각형의 네 꼭짓점과 두 삼각형. 마지막 꼭짓점을 올려 접힌 표면을 만든다.
vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 1.]])
uv = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
faces = np.array([[0, 1, 2], [1, 3, 2]])
height = width = 16
face_index = np.full((height, width), -1, dtype=int)
barycentric = np.zeros((height, width, 3))
position = np.zeros((height, width, 3))

for row in range(height):
    for col in range(width):
        point_uv = np.array([(col + .5) / width, 1 - (row + .5) / height])
        for fi in range(len(faces)):
            a, b, c = uv[faces[fi]]
            ab, ac, q = b - a, c - a, point_uv - a
            det = ab[0] * ac[1] - ab[1] * ac[0]
            if abs(det) < 1e-12:
                continue
            wb = (q[0] * ac[1] - q[1] * ac[0]) / det
            wc = (ab[0] * q[1] - ab[1] * q[0]) / det
            wa = 1 - wb - wc
            if min(wa, wb, wc) < -1e-12:
                continue
            face_index[row, col] = fi
            barycentric[row, col] = [wa, wb, wc]
            va, vb, vc = vertices[faces[fi]]
            position[row, col] = wa * va + wb * vb + wc * vc
            break  # 같은 공유 경계를 두 face가 모두 포함할 수 있다.

valid_mask = face_index >= 0
print('Face 0/1의 픽셀 수:', (face_index == 0).sum(), (face_index == 1).sum())
print('픽셀 하나의 주소:', face_index[4, 3], barycentric[4, 3])
# UV 주소 → face 번호 → 세 vertex 번호 → barycentric 보간의 순서를 기억하자.
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].imshow(face_index, cmap='tab10', vmin=0, vmax=9, interpolation='nearest')
axes[0].set_title('Which triangle owns each pixel?')
image = axes[1].imshow(position[..., 2], cmap='viridis', interpolation='nearest')
axes[1].set_title('Z on a folded surface')
fig.colorbar(image, ax=axes[1])
out = Path(__file__).resolve().parents[2] / 'outputs/canonical_uv_map/03'
out.mkdir(parents=True, exist_ok=True)
np.savez(out / 'two_triangles.npz', position=position, face_index=face_index,
         barycentric=barycentric, valid_mask=valid_mask)
fig.tight_layout()
fig.savefig(out / 'two_triangles.png', dpi=150)
plt.close(fig)

# 실험: faces의 행 순서를 뒤집으면 경계의 face 번호와 XYZ는 각각 어떻게 달라질까?
# 마지막 vertex의 Z를 바꾸면 UV 주소와 face_index는 그대로일까?
