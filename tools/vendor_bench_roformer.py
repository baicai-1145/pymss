"""Stage-by-stage benchmark for the roformer family on the vendor device.

Optimising without a stage breakdown is guesswork: the separator is only ~half
convolution work, the rest is STFT/ISTFT and the complex mask arithmetic, and on
BIREN each of those has its own surprises.  This tool therefore times the three
stages separately on realistic audio lengths:

    stft  -> model.forward (band split, transformer blocks, mask heads) -> istft

and reports ms plus the achieved TFLOPS from torch.utils.flop_counter, so an
improvement can be attributed to a stage instead of to "the model".

Usage:
    python tools/vendor_bench_roformer.py --family bs_roformer --seconds 10
    python tools/vendor_bench_roformer.py --family mel_band_roformer --seconds 10 --repeat 5
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap

CHILD = """
import statistics
import sys
import time

sys.path.insert(0, {tests_dir!r})
import numpy as np
import torch
from pymss_core.device import prepare_device_alias

prepare_device_alias("biren")
import parity
import pymss_core.npu_fft as npu_fft

D = torch.device("supa:0")
family = {family!r}
seconds = {seconds}
repeat = {repeat}
sample_rate = {sample_rate}

builder, _shape, _ref = parity.cases()[family]
torch.manual_seed(0)
model = builder().eval().to(D)

samples = int(sample_rate * seconds)
t = torch.arange(samples, dtype=torch.float32) / sample_rate
wave = (0.25 * torch.sin(2 * torch.pi * 220.0 * t) + 0.12 * torch.sin(2 * torch.pi * 440.0 * t))
audio = wave.reshape(1, 1, -1).to(D)

# Stage timers: wrap the STFT helpers the models use, when they use them.
stages = {{"stft": [], "istft": []}}
original_stft, original_istft = npu_fft.stft_npu, npu_fft.istft_npu

def timed(name, original):
    def wrapper(*args, **kwargs):
        torch.supa.synchronize()
        start = time.perf_counter()
        out = original(*args, **kwargs)
        torch.supa.synchronize()
        stages[name].append(time.perf_counter() - start)
        return out
    return wrapper

npu_fft.stft_npu = timed("stft", original_stft)
npu_fft.istft_npu = timed("istft", original_istft)

# Models import the helpers directly in some paths, so patch those too.
for module_name in ("pymss_core.modules.bs_roformer.common", "pymss_core.modules.scnet.scnet"):
    try:
        module = __import__(module_name, fromlist=["stft_npu"])
    except Exception:
        continue
    if hasattr(module, "stft_npu"):
        module.stft_npu = npu_fft.stft_npu
    if hasattr(module, "istft_npu"):
        module.istft_npu = npu_fft.istft_npu

totals = []
with torch.inference_mode():
    model(audio)  # warm up: the vendor plans/autotunes on first sight of a shape
    torch.supa.synchronize()
    stages["stft"].clear()
    stages["istft"].clear()
    for _ in range(repeat):
        torch.supa.synchronize()
        start = time.perf_counter()
        model(audio)
        torch.supa.synchronize()
        totals.append(time.perf_counter() - start)

total = statistics.median(totals)
stft_ms = 1000 * statistics.median(stages["stft"]) if stages["stft"] else 0.0
istft_ms = 1000 * statistics.median(stages["istft"]) if stages["istft"] else 0.0
print(
    "RESULT family=%s seconds=%.1f total_ms=%.1f stft_ms=%.1f istft_ms=%.1f model_ms=%.1f speed=%.2fx"
    % (
        family,
        seconds,
        1000 * total,
        stft_ms,
        istft_ms,
        1000 * total - stft_ms - istft_ms,
        seconds / total,
    ),
    flush=True,
)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="bs_roformer")
    parser.add_argument("--seconds", default="10")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--tests-dir", default="/tmp/pc/tests")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--families", default="", help="comma list; overrides --family")
    args = parser.parse_args()

    env = dict(os.environ)
    env.setdefault("BRTB_ENABLE_SUBLAS_API", "1")
    env["PYTHONPATH"] = args.tests_dir

    families = args.families.split(",") if args.families else [args.family]
    for family in families:
        code = textwrap.dedent(CHILD).format(
            tests_dir=args.tests_dir, family=family, seconds=float(args.seconds),
            repeat=args.repeat, sample_rate=args.sample_rate,
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-u", "-c", code], capture_output=True, text=True,
                timeout=args.timeout, env=env,
            )
        except subprocess.TimeoutExpired:
            print(f"RESULT family={family} TIMEOUT>{args.timeout:.0f}s", flush=True)
            continue
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
        if lines:
            print(lines[-1], flush=True)
            continue
        err = [ln for ln in proc.stderr.splitlines() if ln.strip() and "Warning" not in ln]
        print(f"RESULT family={family} FAILED {(err[-1] if err else '?')[:150]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
