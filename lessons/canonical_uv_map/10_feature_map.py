"""10: HWC 배열을 신경망 입력으로 바꾸고 C채널 Feature Map을 만든다.

필요: 06,08. TRAIN_STEPS=0이면 초기화된 network의 출력만 관찰한다.
학습하지 않은 feature에는 semantic 의미가 없다. 선택적인 학습 목표는 입력 복원이다.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[2]
position_path = root / 'outputs/canonical_uv_map/08/position.npy'
normal_path = root / 'outputs/canonical_uv_map/08/normal.npy'
texture_path = root / 'outputs/canonical_uv_map/06/texture.npy'
lookup_path = root / 'outputs/canonical_uv_map/05/lookup.npz'
if not all(p.exists() for p in (position_path, normal_path, texture_path, lookup_path)):
    raise FileNotFoundError('먼저 05,06,07,08을 실행하세요.')
position, normal, texture = [np.load(p) for p in (position_path, normal_path, texture_path)]
with np.load(lookup_path) as lookup:
    mask = lookup['valid_mask']

# XYZ의 단위와 RGB의 범위가 달라 position만 중심/크기를 정규화한다.
center = (position[mask].min(0) + position[mask].max(0)) / 2
scale = max(np.ptp(position[mask], axis=0).max(), 1e-12)
normalized_position = (position - center) / scale
inputs_hwc = np.concatenate([normalized_position, normal, texture], axis=-1)
inputs_hwc[~mask] = 0  # 3+3+3=9채널. 빈 영역에는 관측값이 없다.

device = 'cuda:0'
# GPU 2에서 관찰하려면 위 값을 'cuda:0'으로 바꾸고 아래 명령으로 실행한다:
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python lessons/canonical_uv_map/10_feature_map.py
feature_channels = 8
TRAIN_STEPS = 1000  # 먼저 0에서 이해한 뒤, 선택적으로 100을 시도한다.
torch.manual_seed(7)
inputs = torch.tensor(inputs_hwc, dtype=torch.float32).permute(2, 0, 1)[None].to(device)
mask_tensor = torch.tensor(mask, dtype=torch.float32)[None, None].to(device)
print('HWC → NCHW:', inputs_hwc.shape, '→', tuple(inputs.shape))

# 1x1 convolution은 픽셀마다 같은 작은 MLP를 적용한다. UV 이웃을 섞지 않는다.
# CNN의 weight는 vertex별로 따로 있는 게 아니라 모든 UV 위치가 공유한다.
encoder = nn.Sequential(nn.Conv2d(9, 16, 1), nn.ReLU(),
                        nn.Conv2d(16, feature_channels, 1)).to(device)
decoder = nn.Conv2d(feature_channels, 9, 1).to(device)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=.01)
loss_history = []
for step in range(TRAIN_STEPS):
    feature = encoder(inputs) * mask_tensor
    reconstruction = decoder(feature)
    # 유효 표면의 9채널 복원을 학습한다. 빈 영역은 loss에서 제외한다.
    squared_error = (reconstruction - inputs)**2 * mask_tensor
    loss = squared_error.sum() / (mask_tensor.sum() * inputs.shape[1])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if step % 20 == 0:
        print('Step / reconstruction loss:', step, loss.item())

with torch.no_grad():
    features = encoder(inputs) * mask_tensor
    reconstruction = decoder(features)
    final_loss = (((reconstruction - inputs)**2 * mask_tensor).sum()
                  / (mask_tensor.sum() * inputs.shape[1])).item()
features_hwc = features[0].permute(1, 2, 0).cpu().numpy()
print('Feature shape:', features_hwc.shape, '입력 복원 MSE:', final_loss)
print('학습 여부:', TRAIN_STEPS > 0, '복원 학습은 identity/semantic supervision과 다르다.')


# Visualization
# 첫 3채널의 min/max를 RGB로 표시한다. 채널 번호에 코/눈 같은 의미가 붙는 건 아니다.
display = features_hwc[..., :3].copy()
low, high = display[mask].min(0), display[mask].max(0)
display = (display - low) / np.maximum(high - low, 1e-12)
display = np.pad(display, ((0, 0), (0, 0), (0, 3 - display.shape[-1])))
display[~mask] = 0
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].imshow(display)
axes[0].set_title(f'Feature: first 3 of {feature_channels} channels')
if loss_history:
    axes[1].plot(loss_history)
else:
    axes[1].text(.5, .5, 'TRAIN_STEPS = 0\nUntrained features', ha='center', va='center')
axes[1].set(xlabel='Step', ylabel='Masked reconstruction MSE')
out = root / 'outputs/canonical_uv_map/10'
out.mkdir(parents=True, exist_ok=True)
np.save(out / 'feature.npy', features_hwc)
torch.save({'encoder': {k: v.cpu() for k, v in encoder.state_dict().items()},
            'decoder': {k: v.cpu() for k, v in decoder.state_dict().items()}}, out / 'weights.pt')
(out / 'feature_info.json').write_text(json.dumps(dict(seed=7, channels=feature_channels,
                 train_steps=TRAIN_STEPS, reconstruction_mse=final_loss, device=device), indent=2) + '\n')
fig.tight_layout()
fig.savefig(out / 'feature.png', dpi=150)
plt.close(fig)
# 실험: seed/C/TRAIN_STEPS를 바꾸면 geometry와 feature 중 무엇이 달라질까?
# 다음 확장인 3x3 CNN은 seam을 넘는 정보 전달 문제가 있다. 12에서 확인한다.
