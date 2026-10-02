"""CPU checks for UV conventions, seam indexing, geometry and gradients."""

import numpy as np
import pytest
import torch

from flame_uv.assets import Mesh, align_uv_faces, load_flame, read_uv_obj, synthetic_patch
from flame_uv.maps import make_maps
from flame_uv.rasterize import interpolate, rasterize_uv


def square():
    uv = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)
    faces = np.array([[0, 1, 2], [1, 3, 2]], np.int64)
    return uv, faces


def test_affine_geometry_and_v_axis():
    uv, faces = square()
    lookup = rasterize_uv(uv, faces, 16)
    assert lookup.mask.all()
    assert lookup.overlap_pixels == 0  # shared diagonal is not an overlapping island
    attrs = torch.tensor(np.column_stack([uv, 2 * uv[:, 0] + 3 * uv[:, 1]]))
    actual = interpolate(attrs, faces, lookup).numpy()
    expected = np.concatenate([lookup.uv_grid,
                               (2 * lookup.uv_grid[..., :1] + 3 * lookup.uv_grid[..., 1:])], -1)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert actual[0, 0, 1] > actual[-1, 0, 1]  # image row zero corresponds to high v
    np.testing.assert_allclose(lookup.barycentric.sum(-1), 1, atol=1e-6)


def test_rasterization_is_independent_of_uv_winding():
    uv, faces = square()
    lookup_a = rasterize_uv(uv, faces, 16)
    reverse = faces[:, ::-1].copy()
    lookup_b = rasterize_uv(uv, reverse, 16)
    attrs = torch.tensor(uv)
    torch.testing.assert_close(interpolate(attrs, faces, lookup_a),
                               interpolate(attrs, reverse, lookup_b))


def test_triangle_mask_and_attribute_gradients():
    uv, faces = square()
    lookup = rasterize_uv(uv, faces[:1], 16)
    assert lookup.mask.any() and not lookup.mask.all()
    attrs = torch.ones((4, 1), requires_grad=True)
    result = interpolate(attrs, faces[:1], lookup)
    assert torch.count_nonzero(result[~lookup.mask]) == 0
    result.sum().backward()
    assert attrs.grad[3].item() == 0  # vertex absent from this triangle
    assert attrs.grad[:3].min() > 0
    assert attrs.grad.sum().item() == pytest.approx(int(lookup.mask.sum()))


def test_overlapping_uv_islands_are_detected():
    uv, faces = square()
    lookup = rasterize_uv(uv, np.vstack([faces, faces]), 16)
    assert lookup.overlap_pixels > 0


def test_face_and_corner_order_alignment_preserves_seam_indices():
    _, faces = square()
    # Shared geometry vertices 1/2 have separate texture vertices in island 2.
    tex = np.array([[0, 1, 2], [3, 4, 5]], np.int64)
    order = [1, 0]
    corners = [2, 0, 1]
    aligned = align_uv_faces(faces, faces[order][:, corners], tex[order][:, corners])
    np.testing.assert_array_equal(aligned, tex)
    with pytest.raises(ValueError, match='topology'):
        align_uv_faces(np.array([[0, 1, 4], [1, 3, 2]]), faces, tex)


def test_obj_keeps_v_and_vt_separate(tmp_path):
    path = tmp_path / 'seams.obj'
    path.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\n'
                    'vt 0 0\nvt 1 0\nvt 0 1\nvt .5 .5\n'
                    'f -3/4 -2/2 -1/3 # different texture index\n')
    _, faces, _, uv_faces = read_uv_obj(path)
    np.testing.assert_array_equal(faces, [[0, 1, 2]])
    np.testing.assert_array_equal(uv_faces, [[3, 1, 2]])


def test_five_maps_geometry_identity_unit_normals_and_seed():
    mesh = synthetic_patch()
    lookup = rasterize_uv(mesh.uv, mesh.uv_faces, 24)
    maps, _, _ = make_maps(mesh, lookup, amplitude=.002, channels=5, seed=7)
    torch.testing.assert_close(maps['position'], maps['base_position'] + maps['displacement'])
    normals = torch.linalg.vector_norm(maps['normal'][lookup.mask], dim=-1)
    torch.testing.assert_close(normals, torch.ones_like(normals))
    assert maps['feature'].shape == (24, 24, 5)
    again, _, _ = make_maps(mesh, lookup, amplitude=.002, channels=5, seed=7)
    torch.testing.assert_close(maps['feature'], again['feature'])
    different, _, _ = make_maps(mesh, lookup, amplitude=.002, channels=5, seed=8)
    assert not torch.allclose(maps['feature'], different['feature'])
    zero, _, _ = make_maps(mesh, lookup, amplitude=0)
    assert torch.count_nonzero(zero['displacement']) == 0
    torch.testing.assert_close(zero['position'], zero['base_position'])


def test_vertex_offsets_follow_geometry_not_uv_neighbors(tmp_path):
    mesh = synthetic_patch()
    lookup = rasterize_uv(mesh.uv, mesh.uv_faces, 16)
    offset = np.tile(np.array([.001, -.002, .003], np.float32), (len(mesh.vertices), 1))
    path = tmp_path / 'offset.npy'
    np.save(path, offset)
    maps, _, _ = make_maps(mesh, lookup, offset_path=path)
    expected = torch.tensor([.001, -.002, .003]).expand(16, 16, 3)
    torch.testing.assert_close(maps['displacement'], expected)


def test_canonical_npz_shape_and_topology_validation(tmp_path):
    mesh = synthetic_patch(3)
    model = tmp_path / 'model.npz'
    atlas = tmp_path / 'uv.npz'
    shape = tmp_path / 'shape.npy'
    basis = np.zeros((len(mesh.vertices), 3, 300), np.float32)
    basis[:, 0, 0] = .01
    np.savez(model, v_template=mesh.vertices, f=mesh.faces, shapedirs=basis)
    np.savez(atlas, f=mesh.faces, vt=mesh.uv, ft=mesh.uv_faces)
    np.save(shape, np.array([2.], np.float32))
    loaded = load_flame(model, atlas, shape)
    np.testing.assert_allclose(loaded.vertices[:, 0], mesh.vertices[:, 0] + .02)
    np.testing.assert_array_equal(loaded.faces, mesh.faces)
    np.savez(atlas, vt=mesh.uv, ft=mesh.uv_faces)
    with pytest.raises(ValueError, match='must include f'):
        load_flame(model, atlas)


def test_cpu_maps_do_not_initialize_cuda():
    assert not torch.cuda.is_initialized()
    mesh = synthetic_patch(3)
    lookup = rasterize_uv(mesh.uv, mesh.uv_faces, 8)
    make_maps(mesh, lookup, device='cpu')
    assert not torch.cuda.is_initialized()
