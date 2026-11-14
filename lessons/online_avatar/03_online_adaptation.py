"""03 — 관측 한 묶음 -> transient fitting -> persistent update -> 작은 replay memory.

repo root에서:
    CUDA_VISIBLE_DEVICES=7 python lessons/online_avatar/03_online_adaptation.py --seconds 2
lesson 폴더에서:
    CUDA_VISIBLE_DEVICES=7 python 03_online_adaptation.py --seconds 2 --max-steps 200
    CUDA_VISIBLE_DEVICES=7 python 03_online_adaptation.py --methods adapter_replay \
        --no-gate --out outputs/no_gate

출력: outputs/stream/{method}/{시각}.jpg, states/{시각}.pt, run.json.
읽는 순서: update_memory -> fit_lighting -> update_one_time -> run_method.
기본 비교: frozen / representation / adapter / adapter_replay.
고정된 초기 field는 전부 같다. 04의 평가 카메라 RGB는 이 파일에서 열지 않는다.
seconds는 관측별 update 상한이며 마지막 한 step만큼 초과할 수 있다. 실제 시간이 기록된다.
"""
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from common import (ROOT, arguments, cpu_state, finite_step, lesson, panels,
                    require_new, save_json, save_pt, setup, sha, timer)

L02 = lesson(2)


def update_memory(memory, t, capacity):
    """첫 anchor는 보존하고 나머지는 최근 관측. 학습된 memory policy가 아닌 명시적 baseline.

    읽기: update 이전 memory는 반드시 t보다 과거다.
    쓰기: current update가 끝난 다음 t를 넣는다. 미래 RGB/평가 RGB를 넣을 통로가 없다.
    """
    if t in memory:
        return memory
    result = memory + [t]
    if capacity == 1:
        return result[:1]
    if len(result) > capacity:
        result.pop(1)
    return result


def pixel_confidence(surface, raw, t, light):
    """현재 모델로 설명하기 어려운 픽셀을 영구 정보에 바로 저장하지 않는 heuristic.

    confidence는 detach한다. network가 weight를 낮춰서 loss를 회피하지 못한다.
    진짜 새 detail도 downweight할 수 있으므로 --no-gate ablation이 반드시 필요하다.
    """
    weights = {}
    with torch.no_grad():
        for c in surface.stream.inputs:
            out = surface.render(raw, t, c, light)
            error = (out["rgb"]-surface.stream.rgb(t, c)).abs().mean(0, keepdim=True)
            weights[c] = torch.exp(-error/.15).clamp_min(.05).detach()
    return weights


def fit_lighting(model, surface, t, light, start, seconds, steps):
    """이 짧은 최적화에서는 canonical raw를 detach한다. 조명 오차를 texture에 먼저 쓰지 않는다."""
    optimizer = torch.optim.Adam(light.parameters(), lr=.04)
    raw = model.raw().detach()
    done = 0
    for _ in range(steps):
        if timer()-start >= seconds:
            break
        loss = L02.photo_loss(surface, raw, t, light) + .003*light.coefficients.square().mean()
        finite_step(loss, light.parameters(), optimizer)
        done += 1
    light.requires_grad_(False)
    return done


def update_one_time(a, method, model, surface, t, lights, optimizer, memory):
    start = timer()
    light = L02.Lighting().to(surface.device)
    dynamic_steps = fit_lighting(model, surface, t, light, start, a.seconds, a.light_steps)
    lights[t] = light
    conf = surface.tracking["frames"][t]["confidence"]
    use_gate = method == "adapter_replay" and not a.no_gate
    use_replay = method == "adapter_replay" and not a.no_replay
    accepted = not use_gate or conf >= a.min_confidence
    weights = pixel_confidence(surface, model.raw().detach(), t, light) if use_gate and accepted else None
    steps, loss_value = 0, None
    if optimizer is not None and accepted:
        while steps < a.max_steps and timer()-start < a.seconds:
            raw = model.raw()
            loss = L02.photo_loss(surface, raw, t, light, pixel_weights=weights)
            # 한 번에 과거 한 묶음만 replay: buffer 크기가 커져도 step 비용이 폭증하지 않는다.
            if use_replay and memory:
                old = memory[steps % len(memory)]
                if old >= t:
                    raise AssertionError("memory에 현재/미래 관측이 들어갔다")
                loss = loss + a.replay_weight*L02.photo_loss(surface, raw, old, lights[old])
            loss = loss + L02.regularizer(raw, surface.data)
            finite_step(loss, model.trainable(), optimizer)
            steps += 1
            loss_value = float(loss.detach())
    elapsed = timer()-start
    raw = model.raw().detach()
    with torch.no_grad():
        offset, color = surface.decode(raw)
    return dict(t=t, steps=steps, light_steps=dynamic_steps, update_seconds=elapsed,
                budget_seconds=a.seconds, overrun_seconds=max(0., elapsed-a.seconds),
                stop="frozen" if optimizer is None else "gate" if not accepted else
                     "step_cap" if steps == a.max_steps else "time_cap",
                loss=loss_value, confidence=conf, accepted=accepted, memory_before=list(memory),
                tracking_seconds=surface.tracking["frames"][t]["tracking_seconds"],
                displacement_rms_mm=float(offset.square().sum(-1).mean().sqrt()*1000),
                light_norm=float(light.coefficients.norm()))


def save_snapshot(out, model, surface, lights, t, metadata):
    raw = model.raw().detach()
    save_pt(out / "states" / f"{t}.pt", dict(raw=raw.cpu(), seen_through=t,
        stream_sha256=surface.stream.fingerprint,
        lights={i: cpu_state(light) for i, light in lights.items() if i <= t}, metadata=metadata))
    with torch.no_grad():
        c = surface.stream.inputs[0]
        rendered = surface.render(raw, t, c, lights[t])
        delta, color = surface.decode(raw)
        panels(out / f"{t}.jpg", [surface.stream.rgb(t, c), rendered["rgb"],
               surface.uv(color), surface.uv((delta/.005+1)/2)],
               [f"input time {t}", "after update", "canonical appearance", "canonical XYZ residual"])


def run_method(a, method, device):
    # 방법마다 새 객체/같은 seed: optimizer와 replay state가 다음 baseline으로 새지 않는다.
    torch.manual_seed(7)
    stream, tracking, initial, surface = L02.load_components(a.work, device)
    out = a.out / method
    require_new(out / "run.json", a.overwrite)
    # overwrite 실행이 중단되어도 이전 complete 표시로 불완전한 새 결과를 평가하지 않는다.
    save_json(out / "run.json", dict(complete=False, method=method, stream_sha256=stream.fingerprint))
    mode = "adapter" if method == "adapter_replay" else method
    model = L02.Avatar(initial, mode, initial["rank"]).to(device)
    params = model.trainable()
    optimizer = torch.optim.Adam(params, lr=a.lr) if params else None
    lights = {}
    for t, state in enumerate(initial["lights"]):
        light = L02.Lighting().to(device)
        light.load_state_dict(state)
        lights[t] = light.requires_grad_(False)
    warm_start = timer()
    # CUDA/JIT first-call cost를 분리한다. 이 warmup은 weight update를 하지 않는다.
    probe = L02.Lighting().to(device)
    warmup_loss = L02.photo_loss(surface, model.raw().detach(), 0, probe)
    torch.autograd.grad(warmup_loss, probe.coefficients)
    warmup_seconds = timer()-warm_start
    memory = list(range(stream.prefix))[-a.memory:]
    if 0 not in memory:
        memory[0] = 0
    rows = []
    prefix_raw = model.raw().detach().cpu()
    # 동일한 initial raw인지 smoke에서 비교한다. 첫 snapshot은 prefix를 다 본 상태다.
    save_snapshot(out, model, surface, lights, stream.prefix-1, {"initial": True})
    torch.cuda.reset_peak_memory_stats(device)
    for t in tqdm(range(stream.prefix, stream.count), desc=method):
        row = update_one_time(a, method, model, surface, t, lights, optimizer, memory)
        if row["accepted"]:
            memory = update_memory(memory, t, a.memory)
        row["memory_after"] = list(memory)
        io_start = timer()
        save_snapshot(out, model, surface, lights, t, row)
        row["snapshot_seconds"] = timer()-io_start
        rows.append(row)
        # 완료 표시가 없어도 실행 중 동향을 볼 수 있게 매 관측 저장한다.
        save_json(out / "progress.json", rows)
    final = dict(method=method, stream_sha256=stream.fingerprint,
                 initial_sha256=sha(a.work / "initial.pt"), work=str(a.work.resolve()),
                 config={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                 initialization=initial["initialization"], trainable_parameters=sum(p.numel() for p in params),
                 warmup_seconds=warmup_seconds, peak_gpu_mib=torch.cuda.max_memory_allocated(device)/2**20,
                 prefix=stream.prefix, rows=rows, complete=True)
    # 최종 network/optimizer도 저장: raw map만 저장하면 adapter의 실제 학습 내용을 읽을 수 없다.
    save_pt(out / "checkpoint.pt", dict(model=cpu_state(model), mode=mode,
            optimizer=optimizer.state_dict() if optimizer else None, memory=memory,
            seen_through=stream.count-1, initial_raw=prefix_raw, metadata=final))
    save_json(out / "run.json", final)
    print(f"{method}: trainable={final['trainable_parameters']:,} -> {out}")


def run(a):
    if len(a.methods) != len(set(a.methods)):
        raise ValueError("methods에 중복 항목이 있다.")
    if a.seconds <= 0 or a.max_steps < 1 or a.light_steps < 0 or a.memory < 1 or a.lr <= 0:
        raise ValueError("seconds/max-steps/memory/lr 양수, light-steps>=0 필요")
    if not 0 <= a.min_confidence <= 1 or a.replay_weight < 0:
        raise ValueError("confidence는 [0,1], replay-weight>=0 필요")
    device = setup(a.device)
    for method in a.methods:
        run_method(a, method, device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    arguments(p)
    p.add_argument("--out", type=Path, default=ROOT / "outputs/stream")
    p.add_argument("--methods", nargs="+", choices=["frozen", "representation", "adapter", "adapter_replay"],
                   default=["frozen", "representation", "adapter", "adapter_replay"])
    p.add_argument("--seconds", type=float, default=2.)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--light-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=.01)
    p.add_argument("--memory", type=int, default=4)
    p.add_argument("--replay-weight", type=float, default=.5)
    p.add_argument("--min-confidence", type=float, default=.25)
    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--no-replay", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
