"""Read a neutral FLAME mesh and align an existing UV atlas to its faces.

We use the canonical template directly; this is intentionally not a full FLAME
pose/expression/LBS implementation. Model files are supplied by the user.
"""

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray       # [V, 3], canonical model coordinates
    faces: np.ndarray          # [F, 3], geometry vertex indices
    uv: np.ndarray             # [T, 2], possibly duplicated at seams
    uv_faces: np.ndarray       # [F, 3], texture indices; same corner order as faces

    def validate(self):
        for name, array, width in [("vertices", self.vertices, 3),
                                   ("faces", self.faces, 3), ("uv", self.uv, 2),
                                   ("uv_faces", self.uv_faces, 3)]:
            if array.ndim != 2 or array.shape[1] != width or not len(array):
                raise ValueError(f"{name} must be nonempty [N, {width}], got {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains non-finite values")
        if self.faces.shape != self.uv_faces.shape:
            raise ValueError("Geometry faces and UV faces must have matching shapes")
        for name, faces, count in [("faces", self.faces, len(self.vertices)),
                                    ("uv_faces", self.uv_faces, len(self.uv))]:
            if not np.issubdtype(faces.dtype, np.integer):
                raise ValueError(f"{name} must contain integer indices")
            if faces.min() < 0 or faces.max() >= count:
                raise ValueError(f"{name} contains out-of-range indices")
        if self.uv.min() < -1e-6 or self.uv.max() > 1 + 1e-6:
            raise ValueError("This experiment expects an atlas inside [0, 1]^2")


def read_uv_obj(path: Path):
    """Preserve separate v/vt corner indices, including seam duplicates."""
    vertices, uv, faces, uv_faces = [], [], [], []
    with path.open() as stream:
        for line in stream:
            items = line.split('#', 1)[0].split()
            if not items:
                continue
            if items[0] == "v":
                vertices.append([float(x) for x in items[1:4]])
            elif items[0] == "vt":
                uv.append([float(x) for x in items[1:3]])
            elif items[0] == "f":
                if len(items) != 4:
                    raise ValueError("UV template must be a triangulated OBJ")
                geo, tex = [], []
                for item in items[1:]:
                    corner = item.split('/')
                    if len(corner) < 2 or not corner[1]:
                        raise ValueError("OBJ must include vt indices on every face corner")
                    vi, ti = int(corner[0]), int(corner[1])
                    if vi == 0 or ti == 0:
                        raise ValueError("OBJ indices cannot be zero")
                    geo.append(vi - 1 if vi > 0 else len(vertices) + vi)
                    tex.append(ti - 1 if ti > 0 else len(uv) + ti)
                faces.append(geo)
                uv_faces.append(tex)
    return (np.asarray(vertices, np.float32), np.asarray(faces, np.int64),
            np.asarray(uv, np.float32), np.asarray(uv_faces, np.int64))


def align_uv_faces(model_faces, template_faces, template_uv_faces):
    """Match triangles by geometry indices, then reorder their UV corners.

    Merely checking the number of triangles is insufficient: two templates can
    use different triangle/corner order, silently scrambling position maps.
    """
    if model_faces.shape != template_faces.shape or template_faces.shape != template_uv_faces.shape:
        raise ValueError("UV template and FLAME model have different triangle counts")
    lookup = {}
    for face, tex in zip(template_faces, template_uv_faces):
        key = tuple(sorted(map(int, face)))
        if len(set(key)) != 3 or key in lookup:
            raise ValueError("UV template has degenerate or duplicate geometry triangles")
        lookup[key] = dict(zip(map(int, face), map(int, tex)))
    aligned = []
    seen = set()
    for face in model_faces:
        key = tuple(sorted(map(int, face)))
        if key not in lookup or key in seen:
            raise ValueError("UV template does not match FLAME topology/vertex indexing")
        seen.add(key)
        aligned.append([lookup[key][int(vertex)] for vertex in face])
    return np.asarray(aligned, np.int64)


def load_flame(model_path: Path, uv_path: Path, shape_path: Path | None = None):
    for path in (model_path, uv_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing asset: {path}. See assets/flame/README.md")
    if model_path.suffix.lower() == '.npz':
        with np.load(model_path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
    else:
        # Only load trusted official/local model pickles, never untrusted downloads.
        try:
            with model_path.open('rb') as stream:
                data = pickle.load(stream, encoding='latin1')
        except ModuleNotFoundError as exc:
            raise ValueError("Legacy FLAME pickle requires an unavailable package "
                             f"({exc.name}). Run bash scripts/prepare_flame.sh or "
                             "use a converted canonical NPZ model.") from exc
    vertices = np.asarray(data['v_template'], np.float32).copy()
    faces = np.asarray(data['f'], np.int64)
    if shape_path is not None:
        beta = np.asarray(np.load(shape_path, allow_pickle=False), np.float32)
        if beta.ndim != 1 or not 1 <= len(beta) <= 300 or not np.isfinite(beta).all():
            raise ValueError("Shape must be a finite [K] .npy vector, 1 <= K <= 300")
        basis = np.asarray(data['shapedirs'], np.float32)
        if basis.shape[:2] != vertices.shape or basis.shape[2] < len(beta):
            raise ValueError("Model shapedirs is incompatible with the shape vector")
        vertices += np.einsum('vck,k->vc', basis[..., :len(beta)], beta)
    if uv_path.suffix.lower() == '.obj':
        template_vertices, template_faces, uv, uv_faces = read_uv_obj(uv_path)
        if len(template_vertices) != len(vertices):
            raise ValueError("UV OBJ and model have different geometry vertex counts")
        uv_faces = align_uv_faces(faces, template_faces, uv_faces)
    elif uv_path.suffix.lower() == '.npz':
        with np.load(uv_path, allow_pickle=False) as atlas:
            uv = np.asarray(atlas['vt'], np.float32)
            uv_faces = np.asarray(atlas['ft'], np.int64)
            if 'f' in atlas:
                uv_faces = align_uv_faces(faces, np.asarray(atlas['f'], np.int64), uv_faces)
            else:
                raise ValueError("UV NPZ must include f as well as vt/ft to verify topology. "
                                 "Use a matching triangulated UV OBJ instead.")
    else:
        raise ValueError("UV template must be .obj or .npz (f, vt, ft)")
    mesh = Mesh(vertices, faces, uv, uv_faces)
    mesh.validate()
    return mesh


def synthetic_patch(size=12):
    """Curved patch for CPU setup verification; NOT a FLAME head."""
    u, v = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    x, y = (u - .5) * .2, (v - .5) * .2
    vertices = np.stack([x, y, .025 * np.cos(x * 20) * np.cos(y * 20)], -1).reshape(-1, 3)
    faces = []
    for row in range(size - 1):
        for col in range(size - 1):
            a = row * size + col
            faces.extend([[a, a + 1, a + size], [a + 1, a + size + 1, a + size]])
    faces = np.asarray(faces, np.int64)
    mesh = Mesh(vertices.astype(np.float32), faces,
                np.stack([u, v], -1).reshape(-1, 2).astype(np.float32), faces.copy())
    mesh.validate()
    return mesh
