"""11: 직접 만든 다섯 map을 한 주소 체계로 모으고 3D 점으로 복원한다.

필요: 05,06,07,08,10. Position과 Normal은 08의 변위 적용 표면을 사용한다.
이 파일은 이미 배운 배열을 읽고 비교한다. 자동으로 앞 lesson을 실행하지 않는다.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go

root = Path(__file__).resolve().parents[2]
output_root = root / 'outputs/canonical_uv_map'
paths = {'texture': output_root / '06/texture.npy',
         'position': output_root / '08/position.npy',
         'displacement': output_root / '08/displacement.npy',
         'normal': output_root / '08/normal.npy',
         'feature': output_root / '10/feature.npy'}
lookup_path = output_root / '05/lookup.npz'
if not lookup_path.exists() or not all(p.exists() for p in paths.values()):
    raise FileNotFoundError('먼저 04→05→06→07→08→10 순서로 실행하세요.')
maps = {name: np.load(path) for name, path in paths.items()}
with np.load(lookup_path) as data:
    mask, base_position = data['valid_mask'], data['position']
    face_index, barycentric = data['face_index'], data['barycentric']
for name, values in maps.items():
    if values.shape[:2] != mask.shape:
        raise ValueError(f'{name}의 해상도가 다르다. 05 변경 후 후속 lesson을 다시 실행하세요.')
    print(name, values.shape, values.dtype)
error = np.abs(maps['position'] - base_position - maps['displacement'])[mask].max()
print('P = P_base + D 오차:', error)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, (name, values) in zip(axes.flat, maps.items()):
    if name == 'texture':
        display = values
    elif name == 'normal':
        display = (values + 1) / 2
    elif name == 'displacement':
        magnitude = np.linalg.norm(values, axis=-1)
        display = plt.get_cmap('viridis')(magnitude / max(magnitude[mask].max(), 1e-12))[..., :3]
    else:
        display = values[..., :3].copy()  # Position XYZ 또는 Feature 첫 3채널
        low, high = display[mask].min(0), display[mask].max(0)
        display = (display - low) / np.maximum(high - low, 1e-12)
        display = np.pad(display, ((0, 0), (0, 0), (0, 3 - display.shape[-1])))
    display = display.copy()
    display[~mask] = 0
    ax.imshow(display)
    ax.set_title(name)
    ax.axis('off')
axes.flat[-1].imshow(mask, cmap='gray')
axes.flat[-1].set_title('valid_mask')
axes.flat[-1].axis('off')
out = output_root / '11'
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / 'five_maps.npz', **maps, base_position=base_position,
                    valid_mask=mask, face_index=face_index, barycentric=barycentric)
fig.tight_layout()
fig.savefig(out / 'five_maps.png', dpi=150)
plt.close(fig)

# 점을 꺼내며 같은 주소의 RGB도 같이 꺼낸다. UV 격자의 이웃을 mesh face로 연결하지 않는다.
points = maps['position'][mask]
colors = maps['texture'][mask]
with (out / 'points.obj').open('w') as stream:
    stream.write('# UV position samples, no faces\n')
    np.savetxt(stream, points, fmt='v %.9g %.9g %.9g')
sample = np.arange(0, len(points), max(1, len(points) // 12000))
rgb = (np.clip(colors[sample], 0, 1) * 255).astype(int)
color_strings = [f'rgb({r},{g},{b})' for r, g, b in rgb]
points_display = points[sample]
viewer = go.Figure(go.Scatter3d(x=points_display[:, 0], y=points_display[:, 1],
                  z=points_display[:, 2], mode='markers', marker=dict(size=2, color=color_strings)))
viewer.update_layout(title='Points recovered from our UV maps', scene=dict(aspectmode='data'))
viewer.write_html(out / 'points.html', include_plotlyjs=True)
feature_info = json.loads((output_root / '10/feature_info.json').read_text())
(out / 'definitions.json').write_text(json.dumps(dict(
    layout='HWC; invalid pixels = 0; mask is bool', units='original model units',
    position='displaced canonical XYZ', displacement='object-space XYZ offsets from base',
    normal='object-space normal of displaced mesh', texture='procedural checker or matching RGB atlas',
    feature=feature_info, max_reconstruction_error=float(error)), indent=2) + '\n')
print('최종 결과물:', out)
# 실험: 다섯 map은 주소가 같다. 그러면 position[row,col]과 texture[row,col]을
# 동시에 바꾸는 것은 geometry/appearance를 각각 어떻게 편집하는 것일까?
