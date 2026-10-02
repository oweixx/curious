"""Explicit CLI; defaults to CPU. Only run_gpu2.sh selects physical GPU 2."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

from .assets import load_flame, synthetic_patch
from .export import export_results, MAP_NAMES
from .maps import make_maps
from .rasterize import rasterize_uv


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='FLAME canonical mesh -> five UV maps (no training)')
    parser.add_argument('--model', type=Path, default=Path('assets/flame/flame2023_canonical.npz'))
    parser.add_argument('--uv-template', type=Path, default=Path('assets/flame/head_template.obj'))
    parser.add_argument('--shape', type=Path, help='Optional identity coefficients [K], K <= 300 (.npy)')
    parser.add_argument('--texture', type=Path, help='Optional RGB image already using the same UV atlas')
    parser.add_argument('--offsets', type=Path, help='Optional object-space vertex displacement [V,3] (.npy)')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--amplitude', type=float, default=.002, help='Demo bump height in model units')
    parser.add_argument('--feature-channels', type=int, default=16)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda:0'])
    parser.add_argument('--output', type=Path, default=Path('outputs/flame_uv'))
    parser.add_argument('--check-assets', action='store_true', help='Validate model/atlas on CPU, then exit')
    parser.add_argument('--synthetic', action='store_true', help='Use a curved test patch, NOT FLAME')
    args = parser.parse_args()
    if args.resolution < 2 or args.feature_channels < 1 or not math.isfinite(args.amplitude):
        parser.error('Resolution >= 2, feature channels >= 1, and finite amplitude are required')
    if args.synthetic and args.check_assets:
        parser.error('--check-assets requires real assets; remove --synthetic')
    if args.synthetic and args.shape:
        parser.error('--shape is only supported for a FLAME model')
    try:
        mesh = synthetic_patch() if args.synthetic else load_flame(args.model, args.uv_template, args.shape)
        if args.check_assets:
            print(f'Assets OK: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles, '
                  f'{len(mesh.uv)} UV coordinates. GPU was not initialized.')
            return
        for path in (args.texture, args.offsets):
            if path is not None and not path.is_file():
                raise FileNotFoundError(f'Missing input: {path}')
        lookup = rasterize_uv(mesh.uv, mesh.uv_faces, args.resolution)
        if lookup.overlap_pixels:
            raise ValueError(f'Atlas has {lookup.overlap_pixels} overlapping interior pixels. '
                             'A single UV pixel cannot store two surface points; use a nonoverlapping atlas.')
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable. Use scripts/run_gpu2.sh or --device cpu.')
        maps, vertices, encoder = make_maps(mesh, lookup, device=args.device,
                                            amplitude=args.amplitude, channels=args.feature_channels,
                                            seed=args.seed, texture_path=args.texture, offset_path=args.offsets)
        arrays = export_results(args.output, mesh, lookup, maps, vertices)
        torch.save({key: value.cpu() for key, value in encoder.state_dict().items()},
                   args.output / 'feature_encoder.pt')
        identity_error = float(np.abs(arrays['position'] - arrays['base_position']
                                       - arrays['displacement'])[lookup.mask].max())
        sources = {key: {'path': str(path.resolve()), 'sha256': file_hash(path)}
                   for key, path in [('model', args.model), ('uv_template', args.uv_template),
                                     ('shape', args.shape), ('texture', args.texture), ('offsets', args.offsets)]
                   if path is not None and not (args.synthetic and key in ('model', 'uv_template'))}
        metadata = dict(source='synthetic curved patch (NOT FLAME)' if args.synthetic else 'FLAME canonical template',
                        sources=sources, device=args.device, resolution=args.resolution,
                        seed=args.seed, feature_channels=args.feature_channels,
                        canonical_state='zero expression, zero pose; optional identity shape',
                        texture='input UV atlas' if args.texture else 'procedural checker (not albedo)',
                        displacement='input vertex offsets' if args.offsets else 'procedural normal-direction bump',
                        amplitude=args.amplitude, units='original model units',
                        normal_space='object-space, displaced mesh, area-weighted smooth vertex normals',
                        feature='untrained pointwise CNN; no semantic supervision',
                        array_layout='HWC, float32; valid_mask is bool; invalid pixels are zero',
                        uv_convention='u=(col+0.5)/W; v=1-(row+0.5)/H',
                        valid_pixels=int(lookup.mask.sum()), overlap_pixels=lookup.overlap_pixels,
                        reconstruction_max_abs_error=identity_error,
                        map_shapes={name: list(arrays[name].shape) for name in MAP_NAMES})
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'Saved five UV maps to {args.output.resolve()}')
        print(f'Valid pixels: {lookup.mask.sum()} / {lookup.mask.size}; '
              f'P = P_base + D max error: {identity_error:.3g}')
        print('Open overview.png, uv_layout.png and viewer.html; raw values are in maps.npz.')
    except (FileNotFoundError, ValueError, KeyError, pickle.UnpicklingError) as exc:
        # Keep asset/setup errors actionable without hiding unexpected programming errors.
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == '__main__':
    main()
