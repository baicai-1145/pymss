#!/usr/bin/env python3
"""Measure vendor-device throughput for the shapes pymss models actually use.

Written after a BIREN session where `torch.conv2d` averaged 1.8 s per call and
98% of a 388 s inference, yet the numbers were unusable because two processes
shared the card and the CPU control group OOM-killed the notebook server.  The
script therefore enforces two rules:

* **single process** -- it refuses to run while another inference is active;
* **CPU control group only on small shapes** -- a full-resolution CPU conv needs
  tens of GB for im2col and will get the machine killed.

It reports the *steady-state* time separately from the first call, so a plan
recompilation cost is distinguishable from raw throughput.

Usage:
    python tools/cn_gpu_bench.py --device supa
    python tools/cn_gpu_bench.py --device supa --iters 20 --with-cpu
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time

import torch

# Shapes taken from the VR (UVR) path of pymss: full resolution spectrogram
# layers are the expensive ones, the low resolution layers are cheap.
SMALL_SHAPES = (
    ("(2,144,42,64)", (2, 144, 42, 64), 144, 144),
    ("(2,192,21,32)", (2, 192, 21, 32), 192, 192),
)
FULL_SHAPES = (
    ("(2,2,1025,860)", (2, 2, 1025, 860), 2, 32),
    ("(2,32,1025,860)", (2, 32, 1025, 860), 32, 32),
)
CPU_SAFE_BYTES = 512 * 2**20  # never build a CPU control tensors above this


def other_inferences_running() -> list[str]:
    """Return command lines of other pymss/inference processes."""
    try:
        out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return []
    hits = []
    for line in out.splitlines()[1:]:
        low = line.lower()
        if ("pymss" in low or "separator" in low) and "cn_gpu_bench" not in low and "ps -eo" not in low:
            hits.append(line.strip()[:110])
    return hits


def sync(device: torch.device) -> None:
    module = getattr(torch, device.type, None)
    for name in ("synchronize", "sync"):
        fn = getattr(module, name, None) if module is not None else None
        if callable(fn):
            try:
                fn(device)
            except TypeError:
                fn()
            return


def timeit(call, device: torch.device, iters: int) -> tuple[float, float, float]:
    """Return (first call ms, median ms, max ms) with a device sync around each call."""
    sync(device)
    first_start = time.perf_counter()
    call()
    sync(device)
    first_ms = (time.perf_counter() - first_start) * 1000

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        call()
        sync(device)
        samples.append((time.perf_counter() - start) * 1000)
    return first_ms, statistics.median(samples), max(samples)


def bench_conv(device: torch.device, label: str, shape, cin: int, cout: int, iters: int, variant: str) -> None:
    dtype = torch.bfloat16 if variant == "bf16" else torch.float32
    x = torch.randn(*shape, device=device, dtype=dtype)
    conv = torch.nn.Conv2d(cin, cout, 3, padding=1).to(device=device, dtype=dtype)
    if variant == "channels_last":
        x = x.to(memory_format=torch.channels_last)
        conv = conv.to(memory_format=torch.channels_last)
    with torch.no_grad():
        first, median, worst = timeit(lambda: conv(x), device, iters)
    print(f"conv {label:16s} {variant:13s} first={first:9.2f} ms  median={median:9.2f} ms  max={worst:9.2f} ms")


def bench_linear(device: torch.device, dim: int, iters: int) -> None:
    x = torch.randn(1024, dim, device=device)
    lin = torch.nn.Linear(dim, dim).to(device)
    with torch.no_grad():
        first, median, worst = timeit(lambda: lin(x), device, iters)
    print(f"linear [1024,{dim}]       fp32          first={first:9.2f} ms  median={median:9.2f} ms  max={worst:9.2f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="torch device type, e.g. supa / cuda / cpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--with-cpu", action="store_true", help="also time the CPU control group (small shapes only)")
    parser.add_argument("--full-res", action="store_true", help="include full-resolution shapes (device only)")
    parser.add_argument("--force", action="store_true", help="run even if another inference is active")
    args = parser.parse_args()

    busy = other_inferences_running()
    if busy and not args.force:
        print("refusing to measure: another inference is running (shared card numbers are meaningless)")
        for line in busy:
            print("  ", line)
        print("stop it, or pass --force")
        return 2

    device = torch.device(f"{args.device}:{args.device_index}" if args.device != "cpu" else "cpu")
    print(f"device={device}  iters={args.iters}  threads={torch.get_num_threads()}")

    shapes = list(SMALL_SHAPES) + (list(FULL_SHAPES) if args.full_res else [])
    for label, shape, cin, cout in shapes:
        for variant in ("fp32", "channels_last", "bf16"):
            try:
                bench_conv(device, label, shape, cin, cout, args.iters, variant)
            except Exception as exc:  # noqa: BLE001 - a vendor kernel may refuse the variant
                print(f"conv {label:16s} {variant:13s} FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")
    try:
        bench_linear(device, 1024, args.iters)
    except Exception as exc:  # noqa: BLE001
        print(f"linear FAILED {type(exc).__name__}: {str(exc).splitlines()[0][:90]}")

    if args.with_cpu:
        print("-- CPU control group (small shapes only) --")
        for label, shape, cin, cout in SMALL_SHAPES:
            elements = 1
            for dim in shape:
                elements *= dim
            if elements * 4 > CPU_SAFE_BYTES:
                print(f"conv {label:16s} skipped: {elements * 4 / 2**20:.0f} MiB input is too large for a CPU control")
                continue
            bench_conv(torch.device("cpu"), label, shape, cin, cout, max(3, args.iters // 3), "fp32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
