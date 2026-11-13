r"""01 — residual을 펼쳐 보면 '이전 출력들을 모두 계수 1로 더하기'다.

실행 (CPU):
    python 01_residual_stream.py

결과: outputs/01/accumulation.png, contributions.png, checks.json
여기서는 서로 독립인 random branch output으로 누적 현상만 분리해 관찰한다.
학습된 Transformer의 모든 layer가 독립이라는 주장도, baseline이 학습 불가능하다는 주장도 아니다.
질문: h가 커질 때 고정 크기의 새 branch가 h에 미치는 상대적 영향은?
      RMSNorm이 각 branch 입력을 정규화해도 residual stream 자체가 정규화되는가?
"""

from pathlib import Path
import torch
from common import ROOT, parser, setup, plt, savefig, write_json, heatmap


def main():
    p = parser(__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/01")
    a = p.parse_args()
    setup()
    outputs = torch.randn(65, 256, 64, dtype=torch.float64)
    # outputs[0]=embedding, outputs[1:]=f_i(h_i)의 역할. branch output의 대용 데이터.
    states = outputs.cumsum(dim=0)
    assert torch.allclose(states[-1], outputs[0] + outputs[1:].sum(0), atol=1e-12)
    magnitudes = states.square().mean((1, 2)).sqrt()
    relative = outputs[1:].square().mean((1, 2)).sqrt() / magnitudes[:-1]

    # RMSNorm 주변의 local sensitivity: 같은 perturbation을 넣었을 때 변화량.
    # residual의 identity gradient 전체를 측정하는 것이 아니다.
    noise = torch.randn_like(states)
    def norm(x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    eps = 1e-4
    sensitivity = ((norm(states + eps * noise) - norm(states)) / eps).square().mean((1, 2)).sqrt()
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    ax[0].plot(magnitudes, label="measured residual RMS")
    ax[0].plot(torch.arange(1, 66).sqrt(), "--", label="sqrt(number of terms)")
    ax[0].set(xlabel="branch count", ylabel="RMS", title="Independent increments accumulate"); ax[0].legend()
    ax[1].plot(range(1, 65), relative); ax[1].set(xlabel="branch count", ylabel="RMS(branch) / RMS(previous state)", title="Relative contribution shrinks")
    ax[2].plot(sensitivity); ax[2].set(xlabel="branch count", ylabel="local RMSNorm sensitivity", title="Normalized branch input changes less")
    savefig(fig, a.out / "accumulation.png")

    # 같은 source 목록에서 residual sum과 uniform attention average는 다르다.
    # AttnRes query=0의 초기값은 average이지 baseline과 같은 sum이 아니다.
    n = 9
    sum_coeff = torch.tril(torch.ones(n, n))
    average_coeff = sum_coeff / sum_coeff.sum(-1, keepdim=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    heatmap(ax[0], sum_coeff, "Residual sum: coefficient 1", "embedding + earlier branches", "destination depth", vmin=0, vmax=1)
    heatmap(ax[1], average_coeff, "Zero-query AttnRes: uniform average", "embedding + earlier branches", "destination depth", vmin=0, vmax=1)
    savefig(fig, a.out / "contributions.png")
    write_json(a.out / "checks.json", dict(unrolled_sum_error=float((states[-1] - outputs.sum(0)).abs().max()), controlled_independent_increments=True, training=False))
    print(f"residual 누적과 정규화 위치 확인 -> {a.out}")


if __name__ == "__main__":
    main()
