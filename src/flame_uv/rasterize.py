"""Naive UV rasterization: one triangle at a time, explicit barycentrics.

The address lookup runs on CPU once. Attribute interpolation is differentiable
PyTorch and runs on the selected device. Pixel centers use u=(col+.5)/W,
v=1-(row+.5)/H: OBJ v points up, image rows point down.
"""

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class UVLookup:
    face_index: np.ndarray     # [H, W]; -1 means no triangle
    barycentric: np.ndarray    # [H, W, 3]
    uv_grid: np.ndarray        # [H, W, 2]
    overlap_pixels: int

    @property
    def mask(self):
        return self.face_index >= 0


def rasterize_uv(uv, uv_faces, resolution):
    if resolution < 2:
        raise ValueError("Resolution must be >= 2")
    u, v = np.meshgrid((np.arange(resolution) + .5) / resolution,
                       1 - (np.arange(resolution) + .5) / resolution)
    grid = np.stack([u, v], -1)
    index = np.full((resolution, resolution), -1, np.int64)
    bary = np.zeros((resolution, resolution, 3), np.float32)
    overlaps = np.zeros_like(index, dtype=bool)
    pixel_uv = np.asarray(uv, np.float64).copy()
    pixel_uv[:, 0] = pixel_uv[:, 0] * resolution - .5
    pixel_uv[:, 1] = (1 - pixel_uv[:, 1]) * resolution - .5
    for fi, triangle in enumerate(pixel_uv[uv_faces]):
        a, b, c = triangle
        low = np.maximum(np.ceil(triangle.min(0)).astype(int), 0)
        high = np.minimum(np.floor(triangle.max(0)).astype(int), resolution - 1)
        if np.any(low > high):
            continue
        ab, ac = b - a, c - a
        determinant = ab[0] * ac[1] - ab[1] * ac[0]
        if abs(determinant) < 1e-12:
            continue
        cols, rows = np.meshgrid(np.arange(low[0], high[0] + 1),
                                 np.arange(low[1], high[1] + 1))
        qx, qy = cols - a[0], rows - a[1]
        wb = (qx * ac[1] - qy * ac[0]) / determinant
        wc = (ab[0] * qy - ab[1] * qx) / determinant
        weights = np.stack([1 - wb - wc, wb, wc], -1)
        inside = (weights >= -1e-8).all(-1)
        current = index[rows, cols]
        interior = (weights > 1e-7).all(-1)
        old_interior = (bary[rows, cols] > 1e-7).all(-1)
        overlaps[rows, cols] |= interior & old_interior & (current >= 0)
        take = inside & (current < 0)  # deterministic first-face tie at shared edges
        index[rows[take], cols[take]] = fi
        bary[rows[take], cols[take]] = weights[take].astype(np.float32)
    if not (index >= 0).any():
        raise ValueError("No valid UV pixels; check atlas or increase resolution")
    return UVLookup(index, bary, grid.astype(np.float32), int(overlaps.sum()))


def interpolate(attributes, faces, lookup):
    """[V,C] attribute -> [H,W,C] using the same correspondence for every map."""
    device = attributes.device
    face_ids = torch.as_tensor(lookup.face_index, device=device)
    weights = torch.as_tensor(lookup.barycentric, device=device, dtype=attributes.dtype)
    face_vertices = torch.as_tensor(faces, device=device)[face_ids.clamp_min(0)]
    values = (attributes[face_vertices] * weights[..., None]).sum(-2)
    return values * (face_ids >= 0)[..., None]


def vertex_normals(vertices, faces):
    """Area-weighted vertex normals using original mesh connectivity, not UV neighbors."""
    faces = torch.as_tensor(faces, device=vertices.device)
    a, b, c = vertices[faces].unbind(1)
    normals = torch.linalg.cross(b - a, c - a)
    accumulated = torch.zeros_like(vertices)
    for corner in range(3):
        accumulated.index_add_(0, faces[:, corner], normals)
    return torch.nn.functional.normalize(accumulated, dim=-1, eps=1e-12)
