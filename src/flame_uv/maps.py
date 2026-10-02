"""Five different attributes, one shared UV atlas. All tensors are HWC."""

from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn

from .rasterize import interpolate, vertex_normals


class TinyFeatureEncoder(nn.Module):
    """Untrained pointwise CNN: 11 channels -> 32 -> C.

    A 1x1 kernel keeps this first exercise free of convolution across UV seams.
    These features have no learned semantic meaning until an objective is added.
    """

    def __init__(self, channels=16):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(11, 32, 1), nn.GELU(),
                                    nn.Conv2d(32, channels, 1))

    def forward(self, inputs, mask):
        return self.layers(inputs) * mask


def make_maps(mesh, lookup, device='cpu', amplitude=.002, channels=16, seed=7,
              texture_path: Path | None = None, offset_path: Path | None = None):
    base_vertices = torch.as_tensor(mesh.vertices, device=device)
    base_normals = vertex_normals(base_vertices, mesh.faces)
    if offset_path is None:
        # A geometry-space bump; seam duplicates of a vertex receive the same delta.
        center = (base_vertices.amin(0) + base_vertices.amax(0)) / 2
        extent = (base_vertices.amax(0) - base_vertices.amin(0)).clamp_min(1e-6)
        xy = (base_vertices[:, :2] - center[:2]) / extent[:2]
        height = amplitude * torch.exp(-((xy[:, 0] / .22)**2 + (xy[:, 1] / .22)**2))
        offsets = base_normals * height[:, None]
    else:
        offsets_np = np.load(offset_path, allow_pickle=False)
        if offsets_np.shape != mesh.vertices.shape or not np.isfinite(offsets_np).all():
            raise ValueError("Offsets .npy must have finite [V,3] values in model coordinates")
        offsets = torch.as_tensor(offsets_np, device=device, dtype=base_vertices.dtype)
    vertices = base_vertices + offsets
    base_position = interpolate(base_vertices, mesh.faces, lookup)
    displacement = interpolate(offsets, mesh.faces, lookup)
    position = interpolate(vertices, mesh.faces, lookup)
    normal = interpolate(vertex_normals(vertices, mesh.faces), mesh.faces, lookup)
    normal = torch.nn.functional.normalize(normal, dim=-1, eps=1e-12)
    mask = torch.as_tensor(lookup.mask, device=device)[..., None]
    uv = torch.as_tensor(lookup.uv_grid, device=device)
    height, width = lookup.mask.shape
    if texture_path:
        # Input must already use this atlas; this does not unwrap a face photograph.
        with Image.open(texture_path) as image:
            texture_np = np.asarray(image.convert('RGB').resize((width, height),
                                      Image.Resampling.BILINEAR), np.float32) / 255
        texture = torch.as_tensor(texture_np.copy(), device=device)
    else:
        checker = (torch.floor(uv[..., 0] * 16) + torch.floor(uv[..., 1] * 16)) % 2
        color_a = torch.tensor([.15, .45, .8], device=device)
        color_b = torch.tensor([.95, .75, .3], device=device)
        texture = color_a + checker[..., None] * (color_b - color_a)
    texture = texture * mask
    center = (base_vertices.amin(0) + base_vertices.amax(0)) / 2
    scale = (base_vertices.amax(0) - base_vertices.amin(0)).max().clamp_min(1e-6)
    features_in = torch.cat([(position - center) / scale, displacement / scale,
                             normal, uv], -1) * mask
    # Initialize on CPU without changing the user's global RNG or initializing CUDA.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        encoder = TinyFeatureEncoder(channels)
    encoder = encoder.to(device).eval()
    with torch.no_grad():
        feature = encoder(features_in.permute(2, 0, 1)[None],
                          mask.permute(2, 0, 1)[None]).squeeze(0).permute(1, 2, 0)
    maps = dict(texture=texture, position=position, displacement=displacement,
                normal=normal, feature=feature, base_position=base_position,
                valid_mask=mask[..., 0])
    return maps, vertices, encoder
