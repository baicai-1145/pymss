"""Per-stage device time of the 8 s forward pass, measured with NPU events.

Stage isolation (calling each child standalone) breaks under autocast, so this
records a start/end `torch.npu.Event` pair around every top-level child inside a
real forward. Device timestamps avoid host sync inflation, and the stage times
should sum to the total within a few percent.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freq_unet import FreqUNetConfig, build_model  # noqa: E402

STAGES = ("band_split", "enc62", "down62", "enc32", "down32", "enc16", "down16",
          "blocks", "up8", "dec16", "up16", "dec32", "up32", "dec62", "final_norm", "_estimate_masks")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--frames", type=int, default=690)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--precision", default="bf16", choices=("bf16", "32"))
    parser.add_argument("--out", type=Path, default=Path("/workspace/.tmp/profile.json"))
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401

    if args.precision == "32":
        torch.npu.conv.allow_hf32 = False
    cfg = FreqUNetConfig(n_fft=args.n_fft)
    model = build_model(cfg).to("npu:0").eval()
    f = args.n_fft // 2 + 1
    x = torch.randn(1, f * cfg.audio_channels, args.frames, 2, device="npu:0") * 0.1

    def autocast():
        return (
            torch.npu.amp.autocast(dtype=torch.bfloat16)
            if args.precision == "bf16"
            else torch.autocast("cpu", enabled=False)
        )

    # wall-clock total
    totals = []
    for _ in range(args.iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), autocast():
            model(x)
        torch.npu.synchronize()
        totals.append((time.perf_counter() - t0) * 1000)
    total_ms = statistics.median(totals)

    # device-side per-stage events
    spans: dict[str, list[float]] = {name: [] for name in STAGES}
    stack: dict[str, list[tuple[torch.npu.Event, torch.npu.Event | None]]] = {name: [] for name in STAGES}
    handles = []
    for name in STAGES:
        mod = model
        for part in name.split("."):
            mod = getattr(mod, part)

        def pre(_m, _inp, _name=name):
            start = torch.npu.Event(enable_timing=True)
            start.record()
            stack[_name].append((start, None))

        def post(_m, _inp, _out, _name=name):
            end = torch.npu.Event(enable_timing=True)
            end.record()
            start, _ = stack[_name].pop()
            stack[_name].append((start, end))

        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))

    forward_events = []
    for _ in range(args.iters):
        start = torch.npu.Event(enable_timing=True)
        start.record()
        with torch.no_grad(), autocast():
            model(x)
        end = torch.npu.Event(enable_timing=True)
        end.record()
        forward_events.append((start, end))
        for name in STAGES:
            for s, e in stack[name]:
                if e is not None:
                    spans[name].append(s.elapsed_time(e))
            stack[name].clear()
    torch.npu.synchronize()
    for h in handles:
        h.remove()

    dev_total = statistics.median([s.elapsed_time(e) for s, e in forward_events])
    rows = []
    for name in STAGES:
        if not spans[name]:
            continue
        ms = statistics.median(spans[name])
        rows.append({"stage": name, "ms": round(ms, 4), "share_pct": round(100 * ms / dev_total, 1)})
    measured = sum(r["ms"] for r in rows)
    mask_ms = next((r["ms"] for r in rows if r["stage"] == "_estimate_masks"), 0.0)
    payload = {
        "precision": args.precision,
        "frames": args.frames,
        "wall_total_ms": round(total_ms, 3),
        "device_total_ms": round(dev_total, 3),
        "sum_of_stages_ms": round(measured, 3),
        "stages": rows,
        "mask_estimator_ms": round(mask_ms, 4),
        "mask_estimator_share_pct": round(100 * mask_ms / dev_total, 1),
        "unaccounted_ms": round(dev_total - measured, 3),
    }
    print(json.dumps(payload, indent=2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
