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
angle_degrees = 30.0
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

# UV 주소와 mask는 그대로다. 달라지는 것은 같은 [row,col]에 저장한 XYZ다.
# 상태별로 따로 min/max를 정하면 실제 변화가 색 범위 변화에 가려질 수 있다.
# 그래서 X/Y/Z 각각에 대해 세 상태가 같은 color scale을 사용하도록 한다.
states = [('Canonical', base_position), ('Expression', expression_position),
          ('Expression + rotation', posed_position)]
position_cmap = plt.get_cmap('viridis').copy()
position_cmap.set_bad('black')
fig, axes = plt.subplots(3, 3, figsize=(12, 11), layout='constrained')
for channel, channel_name in enumerate(['X', 'Y', 'Z']):
    value_min = min(position[mask, channel].min() for name, position in states)
    value_max = max(position[mask, channel].max() for name, position in states)
    if value_max == value_min:
        value_max = value_min + 1e-12
    for col, (name, position) in enumerate(states):
        values = np.ma.array(position[..., channel], mask=~mask)
        image = axes[channel, col].imshow(values, cmap=position_cmap,
                    vmin=value_min, vmax=value_max, origin='upper', interpolation='nearest')
        axes[channel, col].set_title(f'{name}: {channel_name}')
        axes[channel, col].set(xlabel='UV column', ylabel='UV row')
    fig.colorbar(image, ax=axes[channel, :].tolist(), label=f'{channel_name} (model units)')
fig.suptitle('UV Position Maps: shared color scale across states for each XYZ channel')
fig.savefig(out / 'uv_position_channels.png', dpi=150)
plt.close(fig)

# 작은 표정 변화는 절대 XYZ 그림에서 잘 안 보일 수 있어 차이도 직접 계산한다.
# 첫 열: 표정만의 효과. 둘째 열: expression 상태에 추가한 회전만의 효과.
expression_delta = expression_position - base_position
rotation_delta = posed_position - expression_position
changes = [('Expression - canonical', expression_delta),
           ('Rotated - expression', rotation_delta)]
signed_cmap = plt.get_cmap('coolwarm').copy()
signed_cmap.set_bad('black')
magnitude_cmap = plt.get_cmap('viridis').copy()
magnitude_cmap.set_bad('black')
fig, axes = plt.subplots(4, 2, figsize=(10, 14), layout='constrained')
for col, (name, delta) in enumerate(changes):
    # 표정과 회전의 크기는 다를 수 있다. 열별 범위는 따로, 한 열의 XYZ 범위는 동일하게.
    # Colorbar 숫자를 보고 크기를 비교한다. 빨강/파랑은 각 축의 +/- 방향이다.
    limit = max(np.abs(delta[mask]).max(), 1e-12)
    for channel, channel_name in enumerate(['X', 'Y', 'Z']):
        values = np.ma.array(delta[..., channel], mask=~mask)
        image = axes[channel, col].imshow(values, cmap=signed_cmap,
                        vmin=-limit, vmax=limit, origin='upper', interpolation='nearest')
        axes[channel, col].set_title(f'{name}: delta {channel_name}')
    fig.colorbar(image, ax=axes[:3, col].tolist(), label='Signed change (model units)')
    magnitude = np.linalg.norm(delta, axis=-1)
    magnitude_max = max(magnitude[mask].max(), 1e-12)
    image = axes[3, col].imshow(np.ma.array(magnitude, mask=~mask), cmap=magnitude_cmap,
                    vmin=0, vmax=magnitude_max, origin='upper', interpolation='nearest')
    axes[3, col].set_title(f'{name}: displacement magnitude')
    fig.colorbar(image, ax=axes[3, col], label='Length (model units)')
fig.suptitle('Changes at fixed UV addresses: each column has its own scale; black = invalid')
fig.savefig(out / 'uv_changes.png', dpi=150)
plt.close(fig)

# 실험: coefficient=0, angle=0일 때 기준 Position과 같아지는지 확인하자.
# - UV의 mask/주소는 유지되고, Position의 어느 채널이 변하는지 비교하자.
# - Y축 회전에서는 rotation_delta의 Y가 왜 거의 0일까?
# Canonical은 선택한 기준 상태다. 동일한 UV atlas에 posed XYZ를 담을 수도 있다.
# Identity는 사람별로 달라도 된다. 이번 기준은 평균 identity / expression=0 / pose=0이다.
