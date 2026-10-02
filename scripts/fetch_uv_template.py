"""Fetch the public DECA FLAME UV template at a pinned source revision."""

import hashlib
import json
from pathlib import Path
import urllib.request

SOURCE = ('https://raw.githubusercontent.com/yfeng95/DECA/'
          'a11554ae2a2b0f3998cf1fa94dd4db03babb34a2/data/head_template.obj')
EXPECTED_SHA256 = 'dd5bfbce75adb99b1963f43bca7ec3557bd4a7321f8fc515f4100599e80d99f2'


def main():
    destination = Path('assets/flame/head_template.obj')
    if destination.exists():
        print(f'Keeping existing UV template: {destination}')
        return
    with urllib.request.urlopen(SOURCE, timeout=60) as response:
        content = response.read()
    if not content.startswith(b'#') and not content.startswith(b'v '):
        raise ValueError('Unexpected OBJ response')
    if hashlib.sha256(content).hexdigest() != EXPECTED_SHA256:
        raise ValueError('UV template checksum does not match the pinned source')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    destination.with_suffix('.source.json').write_text(json.dumps({
        'source': SOURCE, 'sha256': hashlib.sha256(content).hexdigest(),
        'project': 'DECA (FLAME-based reconstruction, official author repository)',
        'project_license': 'https://github.com/yfeng95/DECA/blob/a11554ae2a2b0f3998cf1fa94dd4db03babb34a2/LICENSE',
    }, indent=2) + '\n')
    print(f'Downloaded public UV template: {destination}')


if __name__ == '__main__':
    main()
