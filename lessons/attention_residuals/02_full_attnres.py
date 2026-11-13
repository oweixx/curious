r"""02 — token축에서 depth축으로: learned pseudo-query가 이전 출력들을 선택한다.

실행 (CPU, 실제 작은 학습 포함):
    python 02_full_attnres.py
    python 02_full_attnres.py --steps 200

결과: outputs/02/selection.png, learning.png, key_normalization.png, checks.json
이 파일에서는 backbone을 고정된 인공 source로 대신하고 AttnRes selector만 학습한다.
04는 token attention/MLP/embedding/selector를 모두 학습한다.
source[0] feature에 양의 marker를 가진 항목을 찾는 인공 과제로, 실제 layer 역할의 발견은 아니다.
질문: query는 입력에 무관한 Parameter인데 왜 attention weight는 입력마다 달라질까?
      key RMSNorm이 value의 크기까지 정규화하는가?
"""

import torch
from tqdm.auto import trange
from common import ROOT, parser, setup, plt, heatmap, savefig, write_json
from models import DepthAttention


def examples(batch, generator):
    sources = torch.randn(5, batch, 4, 16, generator=generator)
    chosen = torch.randint(5, (batch, 4), generator=generator)
    sources[..., 0] = -2
    b, t = torch.meshgrid(torch.arange(batch), torch.arange(4), indexing="ij")
    sources[chosen, b, t, 0] = 4
    target = sources[chosen, b, t]
    return sources, chosen, target


def main():
    p = parser(__doc__)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--out", type=type(ROOT), default=ROOT / "outputs/02")
    a = p.parse_args()
    if a.steps < 1:
        p.error("steps는 양수여야 합니다.")
    setup()
    training_rng = torch.Generator().manual_seed(101)
    fixed, correct, target = examples(32, torch.Generator().manual_seed(202))
    mixer = DepthAttention(16)
    first, uniform = mixer(list(fixed))
    assert torch.allclose(uniform, torch.full_like(uniform, .2))
    initial_mean_error = (first - fixed.mean(0)).abs().max().item()
    assert torch.allclose(first, fixed.mean(0), atol=5e-7)  # float32 reduction 순서 오차.
    before = uniform[:, 0].detach().T
    optimizer = torch.optim.Adam(mixer.parameters(), lr=.03)
    losses = []
    progress = trange(a.steps, desc="learn depth selection", dynamic_ncols=True)
    for _ in progress:
        sources, _, truth = examples(32, training_rng)
        output, _ = mixer(list(sources))
        loss = (output - truth).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        progress.set_postfix(mse=f"{loss.item():.4f}", refresh=False)
    with torch.no_grad():
        output, weights = mixer(list(fixed))
    accuracy = (weights.argmax(0) == correct).float().mean().item()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    heatmap(ax[0], before, "Before: query = 0", "source depth", "token", vmin=0, vmax=1)
    heatmap(ax[1], weights[:, 0].T, "After: same query, different token keys", "source depth", "token", vmin=0, vmax=1)
    heatmap(ax[2], torch.nn.functional.one_hot(correct[0], 5), "Desired source (toy labels)", "source depth", "token", vmin=0, vmax=1)
    savefig(fig, a.out / "selection.png")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(losses); ax[0].set(yscale="log", xlabel="optimizer step", ylabel="MSE", title="Selector learns from a loss")
    ax[1].bar(range(16), mixer.query.detach()); ax[1].set(xlabel="feature", title="Learned pseudo-query (shared over tokens)")
    savefig(fig, a.out / "learning.png")

    # key norm은 scale 편향을 막지만 value를 변환하지 않는다.
    scaled = fixed.clone(); scaled[1] *= 20
    with torch.no_grad():
        scaled_output, scaled_weights = mixer(list(scaled))
        raw_logits = torch.einsum("d,sbtd->sbt", mixer.query, fixed)
        raw_scaled = torch.einsum("d,sbtd->sbt", mixer.query, scaled)
    alpha_error = (weights - scaled_weights).abs().max().item()
    assert alpha_error < 1e-5
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    for axis, left, right, title in ((ax[0], weights, scaled_weights, "Key RMSNorm"), (ax[1], raw_logits.softmax(0), raw_scaled.softmax(0), "Ablation: raw keys")):
        axis.plot(left[:, 0, 0], "o-", label="original")
        axis.plot(right[:, 0, 0], "x--", label="source 1 multiplied by 20")
        axis.set(title=title, xlabel="source", ylabel="weight"); axis.legend(fontsize=8)
    savefig(fig, a.out / "key_normalization.png")
    write_json(a.out / "checks.json", dict(steps=a.steps, selection_accuracy=accuracy, fixed_mse=(output-target).square().mean().item(), initial_mean_error=initial_mean_error, key_scale_alpha_error=alpha_error, value_scale_changes_output=(output-scaled_output).abs().max().item(), trained="DepthAttention only; synthetic marker task"))
    print(f"selector 정확도 {accuracy:.1%} -> {a.out}")


if __name__ == "__main__":
    main()
