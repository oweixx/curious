"""Run in isolated Python 3.10/NumPy 1.23/Chumpy environment, on CPU.

Extract only canonical geometry arrays needed for this lesson. Original model
files are never edited. Output is a plain NPZ loadable without pickle/Chumpy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.input.open('rb') as stream:
        model = pickle.load(stream, encoding='latin1')
    arrays = {}
    for key in ('v_template', 'f', 'shapedirs'):
        value = model[key]
        if hasattr(value, 'r'):
            value = value.r
        arrays[key] = np.asarray(value, dtype=np.int64 if key == 'f' else np.float32)
    if arrays['v_template'].ndim != 2 or arrays['v_template'].shape[1] != 3:
        raise ValueError('Unexpected v_template layout')
    if arrays['shapedirs'].shape[:2] != arrays['v_template'].shape:
        raise ValueError('Unexpected shapedirs layout')
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f'Non-finite {key}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    # Check conversion at the exact float32 values used by the main application.
    with np.load(args.output, allow_pickle=False) as archive:
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, archive[key])
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.with_suffix('.source.json').write_text(json.dumps({
        'source': str(args.input.resolve()), 'source_sha256': digest,
        'retained_arrays': {key: list(value.shape) for key, value in arrays.items()},
        'scope': 'canonical template/identity basis only, no pose/expression/LBS',
    }, indent=2) + '\n')
    print(f'Converted canonical arrays: {args.input} -> {args.output}')
    print({key: value.shape for key, value in arrays.items()})


if __name__ == '__main__':
    main()
