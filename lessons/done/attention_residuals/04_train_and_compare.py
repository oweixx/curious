r"""04 — 같은 과제를 세 연결 방식으로 실제 학습하고, 저장한 모델로 추론한다.

실행 (이 폴더에서; conda elite, 새 환경/데이터 다운로드 불필요):
    CUDA_VISIBLE_DEVICES=7 python 04_train_and_compare.py train --device cuda
    # CPU로 짧게 확인:
    python 04_train_and_compare.py train --device cpu --steps 100 --out outputs/smoke
    # 저장된 step부터 총 2000 step까지 이어서 (optimizer/data RNG도 복원):
    CUDA_VISIBLE_DEVICES=7 python 04_train_and_compare.py train --device cuda --resume --steps 2000
    # validation과 다른 seed로 만든 unseen 예제에서 checkpoint 추론:
    python 04_train_and_compare.py infer --checkpoint outputs/04/full/best.pt
    python 04_train_and_compare.py infer --checkpoint outputs/04/block/best.pt --two-phase
    # 선택 ablation: content-independent weight / key RMSNorm 제거:
    CUDA_VISIBLE_DEVICES=7 python 04_train_and_compare.py train --device cuda --variants static no_key_norm --out outputs/ablation
    # 저장된 history만 다시 비교 (재학습 없음):
    python 04_train_and_compare.py compare --out outputs/04

결과: outputs/04/{baseline,full,block}/0.png,100.png,200.png,...
      각 variant의 history.json, last.pt, best.pt / 전체 comparison.png, comparison.json
infer 결과: checkpoint 옆 inference/predictions.png, metrics.json
학습 루프는 이 파일의 train_one에 전부 있다. tqdm 외 step별 print는 없다.
모든 variant는 같은 backbone 초기값, minibatch 순서, validation, optimizer 설정을 쓴다.
과제: 매번 새로 주어지는 key-color 표를 읽고 query key의 색을 예측한다 (chance=1/8).
이미 있는 결과는 자동 덮어쓰지 않는다. 새 실험은 --out, 이어 학습은 --resume.
학습/검증/추론은 실제 network 계산이며 논문의 대규모 LLM 성능을 재현한 것은 아니다.
"""

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import trange

from common import ROOT, VARIANTS, parser, setup, plt, savefig, write_json, heatmap, recall_batch, draw_predictions, rms
from models import Config, matched_model, RecallTransformer


def device_of(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA를 사용할 수 없습니다. --device cpu로 실행하거나 현재 conda Torch를 확인하세요.")
    return device


@torch.no_grad()
def evaluate(model, batch, device, two_phase=False):
    model.eval()
    keys = batch[0]
    total_loss, correct = 0., 0
    for start in range(0, len(keys), 128):
        k, c, y = [x[start:start+128].to(device) for x in batch]
        logits = model(k, c, two_phase=two_phase)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
    return dict(loss=total_loss / len(keys), accuracy=correct / len(keys))


def inspect(model, batch, device, path, history, step, two_phase=False, gradients=True):
    """고정 예제를 관찰한다. train 검증에만 gradient probe를 쓰며 추론은 no_grad다."""
    model.eval()
    model.zero_grad(set_to_none=True)
    keys, colors, target = [x[:16].to(device) for x in batch]
    with torch.enable_grad() if gradients else torch.no_grad():
        logits, trace = model(keys, colors, capture=True, two_phase=two_phase)
        if gradients:
            F.cross_entropy(logits, target).backward()
    input_rms = [rms(x) for x in trace["inputs"]]
    gradient_rms = [rms(x.grad) if x.grad is not None else 0. for x in trace["inputs"]] if gradients else []
    branch_rms = [rms(x) for x in trace["outputs"]]
    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(grid[0, :2])
    draw_predictions(ax, keys[:8], colors[:8], target[:8], logits.argmax(-1)[:8])
    ax.set_title("Fixed examples: context changes the correct color", pad=15)
    ax = fig.add_subplot(grid[0, 2])
    if history["train"]:
        points = history["train"]
        # train은 매번 다른 minibatch, validation은 고정 집합이다. 실제 측정값을 그린다.
        ax.plot([p["step"] for p in points], [p["loss"] for p in points], alpha=.35, label="train minibatch CE")
    points = history["validation"]
    if points:
        ax.plot([p["step"] for p in points], [p["loss"] for p in points], "o-", label="fixed validation CE")
    ax.set(xlabel="optimizer step", ylabel="cross entropy", title="Learning curves")
    if points or history["train"]:
        ax.legend(fontsize=8)
    # token attention: sample 0의 query token에서 어떤 context token을 읽는가?
    ax = fig.add_subplot(grid[1, 0])
    token_weights = torch.stack([w[0, :, -1].mean(0).cpu() for w in trace["token_weights"]])
    heatmap(ax, token_weights, "Token attention: query row, head mean", "context token + query", "Transformer layer", vmin=0, vmax=1)
    ax = fig.add_subplot(grid[1, 1])
    if trace["depth_weights"]:
        rows = trace["depth_weights"]
        array = np.full((len(rows), max(w.shape[0] for w in rows)), np.nan)
        for i, w in enumerate(rows):
            array[i, :len(w)] = w[:, 0, -1].cpu().numpy()
        heatmap(ax, array, "Depth attention: query token (last row=head)", "embedding + branches / blocks", "ATTN / MLP sublayer + head", vmin=0, vmax=1)
    else:
        ax.text(.5, .5, "Baseline: all earlier outputs\naccumulate with coefficient 1", ha="center", va="center")
        ax.axis("off")
    ax = fig.add_subplot(grid[1, 2])
    ax.plot(input_rms, "o-", label="sublayer input RMS")
    ax.plot(branch_rms, "s-", label="branch output RMS")
    ax.set(xlabel="ATTN / MLP sublayer", ylabel="activation RMS", title="Same probe batch / loss reduction")
    ax.legend(fontsize=7, loc="upper left")
    if gradients:
        grad_axis = ax.twinx()
        grad_axis.plot(gradient_rms, "r.--", label="input gradient RMS")
        grad_axis.set_ylabel("input gradient RMS", color="r")
    fig.suptitle(f"{model.config.variant} | step {step} | task: synthetic key-color recall", fontsize=14)
    savefig(fig, path)
    model.zero_grad(set_to_none=True)
    return dict(input_rms=input_rms, branch_rms=branch_rms, input_gradient_rms=gradient_rms)


def checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_one(a, variant, device):
    config = Config(variant, a.depth, a.dim, a.heads, a.block_size, a.pairs)
    folder = a.out / variant
    folder.mkdir(parents=True, exist_ok=True)
    model = matched_model(config, a.seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    generator = torch.Generator().manual_seed(a.seed + 100)
    validation = recall_batch(512, a.pairs, torch.Generator().manual_seed(a.seed + 10000))
    train_config = dict(seed=a.seed, batch_size=a.batch_size, lr=a.lr, weight_decay=.01, warmup=50, grad_clip=1.)
    start, best = 0, float("inf")
    history = {"model_config": asdict(config), "train_config": train_config, "train": [], "validation": []}
    if a.resume:
        saved = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
        if saved["model_config"] != asdict(config) or saved["train_config"] != train_config:
            raise ValueError(f"{folder}: resume 시 model/seed/batch/lr 설정을 기존 학습과 동일하게 지정하세요.")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["data_rng"])
        start, best, history = saved["step"], saved["best_loss"], saved["history"]
        if a.steps < start:
            raise ValueError(f"--steps는 총 step입니다. 현재 checkpoint step={start} 이상을 지정하세요.")

    def validate_and_save(step):
        nonlocal best
        metrics = evaluate(model, validation, device)
        if not np.isfinite(metrics["loss"]):
            raise FloatingPointError(f"{variant} validation loss가 유한하지 않습니다.")
        history["validation"].append(dict(step=step, **metrics))
        diagnostics = inspect(model, validation, device, folder / f"{step}.png", history, step)
        improved = metrics["loss"] < best
        best = min(best, metrics["loss"])
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), model_config=asdict(config), train_config=train_config, step=step, best_loss=best, data_rng=generator.get_state(), history=history)
        checkpoint(folder / "last.pt", payload)
        if improved:
            checkpoint(folder / "best.pt", payload)
        write_json(folder / "history.json", history)
        write_json(folder / "diagnostics.json", dict(step=step, **diagnostics))
        return metrics

    if not a.resume:
        validate_and_save(0)
    progress = trange(start + 1, a.steps + 1, desc=variant, initial=start, total=a.steps, dynamic_ncols=True)
    for step in progress:
        model.train()
        keys, colors, target = [x.to(device) for x in recall_batch(a.batch_size, a.pairs, generator)]
        # 50 step warm-up 다음은 일정 LR. 모든 variant에 같은 작은 과제용 recipe를 적용.
        lr = a.lr * min(step / 50, 1.)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        begun = time.perf_counter()
        logits = model(keys, colors)
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{variant} step={step}: loss가 유한하지 않습니다.")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - begun
        history["train"].append(dict(step=step, loss=loss.item(), accuracy=(logits.argmax(-1) == target).float().mean().item(), lr=lr, gradient_norm=float(grad_norm), seconds=elapsed))
        if step % a.eval_every == 0 or step == a.steps:
            metrics = validate_and_save(step)
            progress.set_postfix(loss=f"{loss.item():.3f}", val_acc=f"{metrics['accuracy']:.1%}", refresh=False)
        else:
            progress.set_postfix(loss=f"{loss.item():.3f}", refresh=False)


def compare_runs(folder, variants):
    histories = {}
    for variant in variants:
        path = folder / variant / "history.json"
        if path.is_file():
            histories[variant] = json.loads(path.read_text())
    if not histories:
        raise FileNotFoundError(f"{folder}: 비교할 history.json이 없습니다.")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    summaries = {}
    for variant, history in histories.items():
        validation, training = history["validation"], history["train"]
        x = [row["step"] for row in validation]
        ax[0].plot(x, [row["loss"] for row in validation], "o-", label=variant)
        ax[1].plot(x, [row["accuracy"] for row in validation], "o-", label=variant)
        duration = [row["seconds"] * 1000 for row in training[10:]] or [0.]
        ax[2].bar(variant, np.median(duration))
        model = RecallTransformer(Config(**history["model_config"]))
        summaries[variant] = dict(last_validation=validation[-1], best_validation=min(validation, key=lambda r: r["loss"]), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), median_train_step_ms=float(np.median(duration)))
    ax[0].set(xlabel="optimizer step", ylabel="validation CE", title="Same data / backbone / recipe")
    ax[1].axhline(1/8, linestyle="--", color="gray", label="chance")
    ax[1].set(xlabel="optimizer step", ylabel="validation accuracy", ylim=(0, 1.03), title="Small task; no assumed winner")
    ax[2].set(ylabel="median train step (ms)", title="Naive implementation; excludes validation")
    for axis in ax[:2]:
        axis.legend(fontsize=8)
    ax[2].tick_params(axis="x", rotation=20)
    savefig(fig, folder / "comparison.png")
    write_json(folder / "comparison.json", summaries)


def infer(a):
    device = device_of(a.device)
    saved = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = RecallTransformer(Config(**saved["model_config"])).to(device)
    model.load_state_dict(saved["model"])
    if a.two_phase and model.config.variant != "block":
        raise ValueError("--two-phase는 block checkpoint로 실행하세요.")
    seed = a.seed if a.seed is not None else saved["train_config"]["seed"] + 20000
    test = recall_batch(2048, model.config.pairs, torch.Generator().manual_seed(seed))
    metrics = evaluate(model, test, device, a.two_phase)
    folder = a.out or a.checkpoint.parent / "inference"
    diagnostics = inspect(model, test, device, folder / "predictions.png", saved["history"], saved["step"], a.two_phase, gradients=False)
    write_json(folder / "metrics.json", dict(checkpoint=str(a.checkpoint.resolve()), step=saved["step"], test_seed=seed, examples=2048, two_phase=a.two_phase, **metrics, **diagnostics))
    print(f"{model.config.variant}: unseen recall accuracy={metrics['accuracy']:.2%}, CE={metrics['loss']:.4f} -> {folder}")


def main():
    p = parser(__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    train = sub.add_parser("train")
    train.add_argument("--device", default="cpu")
    train.add_argument("--out", type=Path, default=ROOT / "outputs/04")
    train.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS[:3]))
    train.add_argument("--steps", type=int, default=1000)
    train.add_argument("--eval-every", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--depth", type=int, default=6)
    train.add_argument("--dim", type=int, default=64)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--block-size", type=int, default=4)
    train.add_argument("--pairs", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--lr", type=float, default=.002)
    train.add_argument("--resume", action="store_true")
    inference = sub.add_parser("infer")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--device", default="cpu")
    inference.add_argument("--seed", type=int)
    inference.add_argument("--out", type=Path)
    inference.add_argument("--two-phase", action="store_true")
    comparison = sub.add_parser("compare")
    comparison.add_argument("--out", type=Path, default=ROOT / "outputs/04")
    comparison.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS[:3]))
    a = p.parse_args()
    seed = getattr(a, "seed", None)
    setup(42 if seed is None else seed)
    try:
        if a.stage == "train":
            if min(a.steps, a.eval_every, a.batch_size, a.lr) <= 0:
                raise ValueError("steps/eval-every/batch-size/lr은 양수여야 합니다.")
            if len(set(a.variants)) != len(a.variants):
                raise ValueError("variants를 중복 지정할 수 없습니다.")
            # 일부만 학습한 뒤 기존 결과와 충돌하지 않도록 모든 경로를 먼저 확인.
            for variant in a.variants:
                exists = (a.out / variant / "last.pt").exists()
                if exists and not a.resume:
                    raise ValueError(f"{a.out / variant}에 결과가 있습니다. --resume 또는 새로운 --out을 지정하세요.")
                if a.resume and not exists:
                    raise FileNotFoundError(f"resume checkpoint가 없습니다: {a.out / variant / 'last.pt'}")
            device = device_of(a.device)
            for variant in a.variants:
                train_one(a, variant, device)
                compare_runs(a.out, a.variants)
            print(f"학습과 고정 validation 비교 완료 -> {a.out / 'comparison.png'}")
        elif a.stage == "infer":
            infer(a)
        else:
            compare_runs(a.out, a.variants)
    except (ValueError, FileNotFoundError) as e:
        p.error(str(e))


if __name__ == "__main__":
    main()
