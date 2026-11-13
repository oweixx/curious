r"""03 — Full의 source를 block별 합으로 압축하고, two-phase 계산을 검증한다.

실행 (CPU):
    python 03_block_attnres.py

결과: outputs/03/grouping.png, source_cache.png, checks.json
block_size는 ATTN/MLP sublayer 수다. block_size=1이면 Full과 정확히 같아야 한다.
block_size>1은 모델 자체가 바뀐다. 반면 같은 Block 모델의 naive/two-phase는 같아야 한다.
수치 검증은 무작위의 nonzero query에서 forward와 backward 모두 수행한다.
source_cache 그림은 저장할 표현 수이며 전체 학습 VRAM이나 측정된 speedup이 아니다.
논문 pipeline-parallel cross-stage cache는 GUIDE에서 다루며 분산 구현은 포함하지 않는다.
"""

from dataclasses import replace
import copy
import math
import torch
from common import ROOT, parser, setup, plt, heatmap, savefig, write_json, recall_batch
from models import Config, RecallTransformer


def compare(left, right, keys, colors, right_two_phase=False):
    left.zero_grad(set_to_none=True); right.zero_grad(set_to_none=True)
    x = left(keys, colors)
    y = right(keys, colors, two_phase=right_two_phase)
    error = (x - y).abs().max().item()
    assert torch.allclose(x, y, atol=1e-9, rtol=1e-9), error
    # 동일한 스칼라 목적에 대한 모든 parameter gradient까지 비교한다.
    x.square().sum().backward(); y.square().sum().backward()
    gradient_error = 0.
    for (name, p), (other_name, q) in zip(left.named_parameters(), right.named_parameters()):
        assert name == other_name
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            gradient_error = max(gradient_error, (p.grad - q.grad).abs().max().item())
            assert torch.allclose(p.grad, q.grad, atol=1e-8, rtol=1e-8), name
    return dict(forward_max_error=error, gradient_max_error=gradient_error)


def main():
    p = parser(__doc__)
    p.add_argument("--out", type=type(ROOT), default=ROOT / "outputs/03")
    a = p.parse_args()
    setup()
    config = Config(variant="full", depth=3, dim=16, heads=2, pairs=4)
    full = RecallTransformer(config).double()
    # zero query만 검사하면 중요 오류를 놓칠 수 있으므로 norm/query도 무작위로 바꾼다.
    with torch.no_grad():
        for mixer in full.mixers:
            mixer.query.normal_(std=.3)
            mixer.norm.weight.uniform_(.5, 1.5)
    block_one = RecallTransformer(replace(config, variant="block", block_size=1)).double()
    block_one.load_state_dict(full.state_dict())
    keys, colors, _ = recall_batch(3, 4, torch.Generator().manual_seed(9))
    checks = {"block_size_1_equals_full": compare(full, block_one, keys, colors)}
    # 4로 나누어떨어지지 않는 L=6도 확인: 마지막 partial block을 빠뜨리지 않는다.
    for size in (1, 2, 4, 20):
        block = RecallTransformer(replace(config, variant="block", block_size=size)).double()
        block.load_state_dict(full.state_dict())
        checks[f"naive_equals_two_phase_size_{size}"] = compare(block, copy.deepcopy(block), keys, colors, right_two_phase=True)

    # softmax source -> 원래 branch의 계수로 펼친 그림. block 내부 가중치는 묶인다.
    sources = torch.randn(13, 1, 1, 16, dtype=torch.float64)
    mixer = full.mixers[-1]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3))
    for axis, size in zip(ax, (1, 2, 4)):
        groups = [sources[0]] + [sources[i:i+size].sum(0) for i in range(1, 13, size)]
        _, w = mixer(groups)
        expanded = torch.cat((w[:1, 0, 0], w[1:, 0, 0].repeat_interleave(size)))
        heatmap(axis, expanded.detach()[None], f"Source coefficient; block_size={size}", "embedding, ATTN0, MLP0, ...", "final output")
    savefig(fig, a.out / "grouping.png")
    depths = list(range(4, 129, 4))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(depths, [1 + x for x in depths], label="Full: 1 + L")
    for size in (4, 16):
        ax.plot(depths, [1 + math.ceil(x / size) for x in depths], label=f"Block: 1 + ceil(L/{size})")
    ax.set(xlabel="L = number of ATTN / MLP sublayers", ylabel="cached source vectors per token", title="Representation cache only; excludes autograd and token KV")
    ax.legend(); savefig(fig, a.out / "source_cache.png")
    write_json(a.out / "checks.json", checks)
    print(f"Full/Block 경계 및 two-phase forward/backward 동치 검증 통과 -> {a.out}")


if __name__ == "__main__":
    main()
