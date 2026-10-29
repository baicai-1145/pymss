"""Performance baseline for production-sized roformer models on the vendor device.

The correctness harnesses build parent-sized models (a few thousand parameters),
so their timings say nothing about a real separation: this tool builds the models
at their published sizes with random weights and measures the three stages that
matter separately.

    stft -> model.forward (band split, transformer blocks, mask heads) -> istft

Reported per configuration: stage milliseconds, GFLOPs from
``torch.utils.flop_counter`` on the same input, achieved TFLOPS, MFU against the
measured peaks, and a repeatability check (the vendor has produced different
numbers for identical inputs before).

Usage:
    python tools/vendor_perf.py --preset bs_roformer --seconds 10
    python tools/vendor_perf.py --preset mel_band_roformer --seconds 10 --bf16
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

import torch

sys.path.insert(0, {tests_dir!r})
from pymss_core.device import prepare_device_alias

prepare_device_alias("biren")
import pymss_core.npu_fft as npu_fft
from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
from pymss_core.modules.bs_roformer.mel_band_roformer import MelBandRoformer
from torch.utils.flop_counter import FlopCounterMode

D = torch.device("supa:0")
preset = {preset!r}
seconds = {seconds}
repeat = {repeat}
use_bf16 = {use_bf16}
profile = {profile}
peak = {peak}

# Published sizes: a real separation model, not the correctness harness's stand-in.
CONFIGS = {{
    "bs_roformer": dict(cls=BSRoformer, kwargs=dict(dim=384, depth=12, stereo=True, num_stems=1,
                                                    time_transformer_depth=1, freq_transformer_depth=1,
                                                    dim_head=64, heads=8, stft_n_fft=2048,
                                                    stft_hop_length=512, stft_win_length=2048)),
    "bs_roformer_small": dict(cls=BSRoformer, kwargs=dict(dim=192, depth=6, stereo=True, num_stems=1,
                                                         time_transformer_depth=1, freq_transformer_depth=1,
                                                         dim_head=64, heads=8)),
    "mel_band_roformer": dict(cls=MelBandRoformer, kwargs=dict(dim=384, depth=6, stereo=True, num_stems=1,
                                                               time_transformer_depth=1, freq_transformer_depth=1,
                                                               num_bands=60, dim_head=64, heads=8,
                                                               stft_n_fft=2048, stft_hop_length=512,
                                                               stft_win_length=2048)),
}}
torch.manual_seed(0)
model = CONFIGS[preset]["cls"](**CONFIGS[preset]["kwargs"]).eval()

samples = int(44100 * seconds)
wave = torch.randn(1, 2, samples) * 0.1

# FLOPs on the CPU for the same shapes: the counter is exact and does not care
# where it runs, and the CPU reference also gives the repeatability target.
with torch.inference_mode(), FlopCounterMode(display=False) as counter:
    model(wave)
flops = counter.get_total_flops()
params = sum(p.numel() for p in model.parameters())

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
import pymss_core.modules.bs_roformer.common as roformer_common
if hasattr(roformer_common, "stft_npu"):
    roformer_common.stft_npu = npu_fft.stft_npu
if hasattr(roformer_common, "istft_npu"):
    roformer_common.istft_npu = npu_fft.istft_npu

model = model.to(D)
x = wave.to(D)
amp = torch.supa.amp.autocast(dtype=torch.bfloat16) if use_bf16 else torch.no_grad()

# Optional per-stage profile.  Hooking only the top-level children misses the
# transformer stack (a ModuleList is iterated, its forward is never called), so leaf
# modules are timed and aggregated by class: that names the stage that actually
# costs instead of guessing.
profile = {profile}
stage_times = {{}}
if profile:
    def leaf_hooks(leaf):
        cls = type(leaf).__name__

        def before(_module, _inputs):
            try:
                torch.supa.synchronize()
            except Exception:
                return
            leaf._perf_start = time.perf_counter()

        def after(_module, _inputs, _output):
            start = getattr(leaf, "_perf_start", None)
            if start is None:
                return
            try:
                torch.supa.synchronize()
            except Exception:
                return
            stage_times.setdefault(cls, []).append(time.perf_counter() - start)

        leaf.register_forward_pre_hook(before)
        leaf.register_forward_hook(after)

    for module in model.modules():
        if not list(module.children()):
            leaf_hooks(module)
    # the model's own forward (stft, istft, band split assembly) is timed separately
    model.register_forward_pre_hook(lambda m, i: setattr(m, "_perf_start", time.perf_counter()))

totals = []
outputs = []
with torch.inference_mode():
    with amp:
        model(x)  # warm up: the vendor plans and autotunes on first sight of a shape
    torch.supa.synchronize()
    stages["stft"].clear()
    stages["istft"].clear()
    for _ in range(repeat):
        torch.supa.synchronize()
        start = time.perf_counter()
        with amp:
            out = model(x)
        torch.supa.synchronize()
        totals.append(time.perf_counter() - start)
        outputs.append(out.float().cpu())

total = statistics.median(totals)
stft = statistics.median(stages["stft"]) if stages["stft"] else 0.0
istft = statistics.median(stages["istft"]) if stages["istft"] else 0.0
spread = abs(outputs[0] - outputs[-1]).abs().max().item()
tflops = flops / total / 1e12
print(
    "PERF preset=%s params=%d flops=%.3e total_ms=%.1f stft_ms=%.1f istft_ms=%.1f "
    "tflops=%.3f mfu=%.2f%% repeat=%.2e speed=%.2fx"
    % (preset, params, flops, total * 1000, stft * 1000, istft * 1000,
       tflops, 100.0 * tflops / peak, spread, seconds / total),
    flush=True,
)
if profile and stage_times:
    rows = sorted(
        ((sum(v), len(v), k) for k, v in stage_times.items() if v), reverse=True
    )
    for stage_ms, count, stage in rows[:14]:
        print("STAGE %-30s %8.1f ms %5.1f%%  x%d" % (stage, stage_ms * 1000, 100.0 * stage_ms / total, count),
              flush=True)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="bs_roformer")
    parser.add_argument("--presets", default="", help="comma list; overrides --preset")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--profile", action="store_true", help="per-stage timing")
    parser.add_argument("--peak", type=float, default=0.0,
                        help="measured peak TFLOPS for MFU; 284.62 bf16 / 115.74 fp32")
    parser.add_argument("--tests-dir", default="/tmp/pc/tests")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    peak = args.peak or (284.62 if args.bf16 else 115.74)
    env = dict(os.environ)
    env.setdefault("BRTB_ENABLE_SUBLAS_API", "1")
    # Preserve SDK compiler modules (e.g. Biren br_generator) in child processes.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (args.tests_dir, env.get("PYTHONPATH", ""))))

    presets = args.presets.split(",") if args.presets else [args.preset]
    for preset in presets:
        code = textwrap.dedent(CHILD).format(
            tests_dir=args.tests_dir, preset=preset, seconds=args.seconds,
            repeat=args.repeat, use_bf16=args.bf16, peak=peak,
            profile=args.profile,
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-u", "-c", code], capture_output=True, text=True,
                timeout=args.timeout, env=env,
            )
        except subprocess.TimeoutExpired:
            print(f"PERF preset={preset} TIMEOUT>{args.timeout:.0f}s", flush=True)
            continue
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PERF")]
        if lines:
            print(lines[-1], flush=True)
            for stage in [ln for ln in proc.stdout.splitlines() if ln.startswith("STAGE")]:
                print("  " + stage[6:], flush=True)
            continue
        err = [ln for ln in proc.stderr.splitlines()
               if ln.strip() and "Warning" not in ln and "REGISTER_GENERATOR" not in ln]
        print(f"PERF preset={preset} FAILED {(err[-1] if err else '?')[:160]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
