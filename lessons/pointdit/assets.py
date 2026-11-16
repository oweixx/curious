"""공식 source/weights 준비. 5개 lesson과 별개인 자산 관리 helper.

repo root:
    python lessons/pointdit/assets.py check
    python lessons/pointdit/assets.py source
    python lessons/pointdit/assets.py dinov3
    python lessons/pointdit/assets.py pointdit --resolution 512

기존 HF login을 사용하며 token을 코드/명령행에 쓰지 않는다. DINOv3는 Meta의 접근 승인이 필요하다.
공식 HF safetensors를 Meta state-dict 이름으로 역변환하여 원본 DINOv3 code로 실행한다.
변환은 ViT-B/16만 지원한다. 이미 받은 Meta .pth도 --dino-weights로 바로 사용할 수 있다.
"""
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import torch

from common import ROOT, parser, save, sha256, write_json

DINO_REVISION = '6876159a11b4df116f30f667f8c9888617df0751'
DINO_REPO = ROOT/'.cache/dinov3'
DINO_WEIGHTS = ROOT/'weights/dinov3_vitb16_meta_format.pth'
POINTDIT_FILES = {256: 'pointditb-256-scenenet-nodinov3-58f0b231.pth',
                  512: 'pointditb-512-mixdata-nodinov3-1d42aacf.pth'}


def source():
    marker = DINO_REPO/'CURIOUS_REVISION'
    if marker.exists():
        if marker.read_text().strip() != DINO_REVISION:
            raise ValueError('DINO source revision differs; use a separate checkout')
        return DINO_REPO
    raw = urllib.request.urlopen(f'https://codeload.github.com/facebookresearch/dinov3/zip/{DINO_REVISION}', timeout=60).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for item in archive.infolist():
            rel = Path(*Path(item.filename).parts[1:])
            if item.is_dir() or not rel.parts:
                continue
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe archive path')
            p = DINO_REPO/rel; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(archive.read(item))
    marker.write_text(DINO_REVISION+'\n')
    return DINO_REPO


def meta_encoder(repo=DINO_REPO):
    if not (Path(repo)/'hubconf.py').is_file():
        raise FileNotFoundError('DINOv3 source missing. Run: python assets.py source')
    return torch.hub.load(str(repo), 'dinov3_vitb16', source='local', pretrained=False)


def hf_to_meta(state, template):
    """HF 공식 converter의 역방향. Q/K/V를 합치고 missing K bias는 고정 0으로 복원한다.

    Reference: transformers/models/dinov3_vit/convert_dinov3_vit_to_hf.py.
    이름만 맞추는 strict=False 로딩을 하지 않는다. 모든 학습 tensor와 shape를 검사한다.
    """
    state = {k.removeprefix('model.'): v for k, v in state.items()}
    used, result = set(), {}
    def take(key):
        used.add(key)
        return state[key]
    for key, expected in template.items():
        if key.endswith('bias_mask'):
            value = expected.clone()  # fixed [1,0,1] mask, not a learned tensor
        elif key == 'rope_embed.periods':
            value = expected.clone()  # fixed positional frequencies from the B/16 constructor
        elif key.startswith('blocks.'):
            idx, suffix = key.split('.', 2)[1:]
            prefix = f'layer.{idx}.'
            if suffix.startswith('attn.qkv.'):
                kind = suffix.rsplit('.', 1)[-1]
                q, v = take(prefix+f'attention.q_proj.{kind}'), take(prefix+f'attention.v_proj.{kind}')
                k_key = prefix+f'attention.k_proj.{kind}'
                k = take(k_key) if k_key in state else torch.zeros_like(q)
                if kind=='weight' and k_key not in state:
                    raise KeyError(k_key)
                value = torch.cat([q, k, v], 0)
            else:
                suffix = suffix.replace('attn.proj.', 'attention.o_proj.')
                suffix = suffix.replace('mlp.fc1.', 'mlp.up_proj.').replace('mlp.fc2.', 'mlp.down_proj.')
                suffix = re.sub(r'ls([12])\.gamma', r'layer_scale\1.lambda1', suffix)
                value = take(prefix+suffix)
        else:
            mapped = {'cls_token': 'embeddings.cls_token', 'mask_token': 'embeddings.mask_token',
                      'storage_tokens': 'embeddings.register_tokens'}.get(key, key)
            mapped = mapped.replace('patch_embed.proj.', 'embeddings.patch_embeddings.')
            value = take(mapped)
            if key == 'mask_token' and value.ndim==3:
                value = value.squeeze(1)
        if value.shape != expected.shape:
            raise ValueError(f'{key}: HF shape {value.shape} != Meta shape {expected.shape}')
        result[key] = value
    unused = set(state)-used
    if unused - {'rope_embeddings.inv_freq'}:
        raise ValueError(f'Unmapped HF tensors: {sorted(unused)}')
    return result


def download_dino():
    from huggingface_hub import hf_hub_download, HfApi
    from safetensors.torch import load_file
    source()
    repo = 'facebook/dinov3-vitb16-pretrain-lvd1689m'
    revision = HfApi().model_info(repo).sha
    kwargs = dict(repo_id=repo, revision=revision, local_dir=ROOT/'weights/dino_hf')
    config = json.loads(Path(hf_hub_download(filename='config.json', **kwargs)).read_text())
    if config['hidden_size']!=768 or config['num_hidden_layers']!=12 or config['patch_size']!=16:
        raise ValueError('Only the official LVD DINOv3 ViT-B/16 is supported')
    path = hf_hub_download(filename='model.safetensors', **kwargs)
    model = meta_encoder()
    converted = hf_to_meta(load_file(path), model.state_dict())
    model.load_state_dict(converted, strict=True)
    save(DINO_WEIGHTS, converted)
    write_json(DINO_WEIGHTS.with_suffix('.json'), dict(source=repo, revision=revision,
        original_safetensors_sha256=sha256(path), converted_sha256=sha256(DINO_WEIGHTS),
        source_revision=DINO_REVISION, conversion='HF -> official Meta B/16 state-dict; strict loaded'))
    print(DINO_WEIGHTS)


def download_pointdit(resolution):
    from huggingface_hub import hf_hub_download
    name = POINTDIT_FILES[resolution]
    path = hf_hub_download('haofeixu/pointdit', name, local_dir=ROOT/'weights')
    if not sha256(path).startswith(name.rsplit('-', 1)[-1].split('.')[0]):
        raise ValueError('Official PointDiT checksum mismatch')
    print(path)


def main():
    p = parser(__doc__); p.add_argument('stage', choices=['check', 'source', 'dinov3', 'pointdit'])
    p.add_argument('--resolution', type=int, choices=[256, 512], default=512)
    a = p.parse_args()
    if a.stage=='source':
        print(source())
    elif a.stage=='dinov3':
        download_dino()
    elif a.stage=='pointdit':
        download_pointdit(a.resolution)
    else:
        for path in [DINO_REPO/'hubconf.py', DINO_WEIGHTS, ROOT/'weights'/POINTDIT_FILES[a.resolution]]:
            print('OK' if path.is_file() else 'MISSING', path)


if __name__=='__main__':
    main()
