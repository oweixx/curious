"""Keep raw values separate from colors used to display those values."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MAP_NAMES = ('texture', 'position', 'displacement', 'normal', 'feature')


def display_colors(array, mask):
    low, high = array[mask].min(0), array[mask].max(0)
    return np.clip((array - low) / np.maximum(high - low, 1e-8), 0, 1) * mask[..., None]


def visualizations(maps):
    mask = maps['valid_mask']
    displacement_magnitude = np.linalg.norm(maps['displacement'], axis=-1)
    magnitude_scale = max(float(displacement_magnitude[mask].max()), 1e-12)
    return {
        'texture': maps['texture'],
        'position': display_colors(maps['position'], mask),
        'displacement': plt.get_cmap('viridis')(displacement_magnitude / magnitude_scale)[..., :3] * mask[..., None],
        'normal': (maps['normal'] + 1) / 2 * mask[..., None],
        'feature': display_colors(maps['feature'][..., :min(3, maps['feature'].shape[-1])], mask),
    }


def write_obj(path, vertices, faces, uv, uv_faces):
    with path.open('w') as stream:
        stream.write('# Canonical object coordinates; separate geometry/UV corner indices\n')
        for x, y, z in vertices:
            stream.write(f'v {x:.9g} {y:.9g} {z:.9g}\n')
        for u, v in uv:
            stream.write(f'vt {u:.9g} {v:.9g}\n')
        for face, tex in zip(faces + 1, uv_faces + 1):
            corners = ' '.join(f'{int(a)}/{int(b)}' for a, b in zip(face, tex))
            stream.write(f'f {corners}\n')


def export_results(out: Path, mesh, lookup, maps, vertices):
    out.mkdir(parents=True, exist_ok=True)
    arrays = {name: value.detach().cpu().numpy() for name, value in maps.items()}
    np.savez_compressed(out / 'maps.npz', **arrays)
    np.savez_compressed(out / 'correspondence.npz', face_index=lookup.face_index,
                        barycentric=lookup.barycentric, uv_grid=lookup.uv_grid)
    for name, value in arrays.items():
        np.save(out / f'{name}.npy', value)
    colors = visualizations(arrays)
    # Allow C=1/2: pad only the feature preview to RGB, never the raw feature tensor.
    colors['feature'] = np.pad(colors['feature'],
                               ((0, 0), (0, 0), (0, 3 - colors['feature'].shape[-1])))
    titles = ['Texture (UV checker or input atlas)', 'Position (XYZ scaled for display)',
              'Displacement (vector magnitude)', 'Normal (object-space XYZ)',
              'Feature (untrained; first 3 channels)', 'Valid UV mask']
    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for ax, name, title in zip(axes.flat, MAP_NAMES + ('valid_mask',), titles):
        ax.imshow(colors[name] if name in colors else arrays[name], cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
        if name in colors:
            Image.fromarray((np.clip(colors[name], 0, 1) * 255).astype(np.uint8)).save(out / f'{name}.png')
    fig.tight_layout()
    fig.savefig(out / 'overview.png', dpi=160)
    plt.close(fig)
    Image.fromarray(arrays['valid_mask'].astype(np.uint8) * 255).save(out / 'valid_mask.png')
    fig, ax = plt.subplots(figsize=(8, 8))
    triangles = mesh.uv[mesh.uv_faces]
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    ax.add_collection(LineCollection(edges, linewidths=.2, colors='royalblue'))
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='u', ylabel='v', title='UV atlas triangles (v points up)')
    ax.set_aspect('equal')
    fig.savefig(out / 'uv_layout.png', dpi=160)
    plt.close(fig)
    vertices_np = vertices.detach().cpu().numpy()
    write_obj(out / 'canonical_base.obj', mesh.vertices, mesh.faces, mesh.uv, mesh.uv_faces)
    write_obj(out / 'canonical_displaced.obj', vertices_np, mesh.faces, mesh.uv, mesh.uv_faces)
    points = arrays['position'][arrays['valid_mask']]
    # This OBJ is intentionally a point cloud. Grid adjacency is not mesh topology.
    with (out / 'uv_reconstructed_points.obj').open('w') as stream:
        stream.write('# Points reconstructed directly from valid UV position pixels\n')
        np.savetxt(stream, points, fmt='v %.9g %.9g %.9g')
    viewer = make_subplots(rows=1, cols=3, specs=[[{'type': 'scene'}] * 3],
                           subplot_titles=('Canonical base mesh', 'Displaced canonical mesh',
                                           'Points recovered from UV position map'))
    for col, verts in [(1, mesh.vertices), (2, vertices_np)]:
        viewer.add_trace(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                         i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
                         color='lightsteelblue', name=f'Mesh {col}', showscale=False), row=1, col=col)
    # Cap HTML size; raw exports retain every valid texel.
    pick = np.linspace(0, len(points) - 1, min(len(points), 16000), dtype=int)
    points = points[pick]
    color_strings = {}
    for name in MAP_NAMES:
        rgb = (np.clip(colors[name][arrays['valid_mask']][pick], 0, 1) * 255).astype(np.uint8)
        color_strings[name] = [f'rgb({r},{g},{b})' for r, g, b in rgb]
    viewer.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers',
                     marker=dict(size=2, color=color_strings['texture']), name='UV samples'), row=1, col=3)
    buttons = [dict(label=name, method='restyle',
                    args=[{'marker.color': [color_strings[name]]}, [2]]) for name in MAP_NAMES]
    viewer.update_layout(title='Shared UV addresses, five attributes — rotate each 3D view',
                          updatemenus=[dict(buttons=buttons, x=.78, y=1.12)],
                          height=650, showlegend=False)
    viewer.update_scenes(aspectmode='data', xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    viewer.write_html(out / 'viewer.html', include_plotlyjs=True, full_html=True)
    return arrays
