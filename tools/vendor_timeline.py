"""Per-class time attribution for one model run, measured with GPU events.

Hooking modules with ``synchronize()`` serialises the pipeline and inflates whatever
is being measured -- an early profile reported RMSNorm as 20.8% of a roformer run
when ablation showed 0.6%.  GPU events cost one record per call and no sync, so the
numbers stay honest while still naming every class.

Usage:
    python tools/vendor_timeline.py --preset bs_roformer --seconds 10
    python tools/vendor_timeline.py --preset bs_roformer --seconds 10 --bf16
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap

CHILD = """
import sys

import torch

sys.path.insert(0, {tests_dir!r})
from pymss_core.device import prepare_device_alias

prepare_device_alias("biren")
import pymss_core.npu_fft as npu_fft
from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
from pymss_core.modules.bs_roformer.mel_band_roformer import MelBandRoformer

D = torch.device("supa:0")
use_bf16 = {use_bf16}

CONFIGS = {{
    "bs_roformer": dict(cls=BSRoformer, kwargs=dict(dim=384, depth=12, stereo=True, num_stems=1,
                                                    time_transformer_depth=1, freq_transformer_depth=1,
                                                    dim_head=64, heads=8)),
    "mel_band_roformer": dict(cls=MelBandRoformer, kwargs=dict(dim=384, depth=6, stereo=True, num_stems=1,
                                                               time_transformer_depth=1, freq_transformer_depth=1,
                                                               num_bands=60, dim_head=64, heads=8)),
}}

torch.manual_seed(0)
model = CONFIGS[{preset!r}]["cls"](**CONFIGS[{preset!r}]["kwargs"]).eval()
x = (torch.randn(1, 2, int(44100 * {seconds})) * 0.1)

records = {{}}
stages = {{"stft": [], "istft": []}}
original_stft, original_istft = npu_fft.stft_npu, npu_fft.istft_npu


def timed(name, original):
    def wrapper(*args, **kwargs):
        start = torch.supa.Event(enable_timing=True)
        end = torch.supa.Event(enable_timing=True)
        start.record()
        out = original(*args, **kwargs)
        end.record()
        stages[name].append((start, end))
        return out
    return wrapper


npu_fft.stft_npu = timed("stft", original_stft)
npu_fft.istft_npu = timed("istft", original_istft)


def instrument(module, label):
    def before(_module, _inputs):
        start = torch.supa.Event(enable_timing=True)
        module._tl_start = start
        start.record()

    def after(_module, _inputs, _output):
        start = getattr(module, "_tl_start", None)
        if start is None:
            return
        end = torch.supa.Event(enable_timing=True)
        end.record()
        records.setdefault(label, []).append((start, end))

    module.register_forward_pre_hook(before)
    module.register_forward_hook(after)


for name, module in model.named_modules():
    instrument(module, name)

model = model.to(D)
x = x.to(D)
amp = torch.supa.amp.autocast(dtype=torch.bfloat16) if use_bf16 else torch.no_grad()
with torch.inference_mode():
    with amp:
        model(x)  # warm up
    torch.supa.synchronize()
    records.clear()
    stages["stft"].clear()
    stages["istft"].clear()
    start_all = torch.supa.Event(enable_timing=True)
    end_all = torch.supa.Event(enable_timing=True)
    start_all.record()
    with amp:
        out = model(x)
    end_all.record()
    torch.supa.synchronize()

total = start_all.elapsed_time(end_all)

# Roll every module up to its top-level stage, and keep the deepest name as detail.
rows = []
for name, spans in records.items():
    elapsed = sum(s.elapsed_time(e) for s, e in spans)
    rows.append((elapsed, len(spans), name))
rows.sort(reverse=True)

total_ms = f"TOTAL {{total:.0f}} ms  preset={{{preset!r}}} bf16={{use_bf16}}"
print(total_ms, flush=True)
print(f"  stft   {{sum(s.elapsed_time(e) for s, e in stages['stft']):8.1f}} ms", flush=True)
print(f"  istft  {{sum(s.elapsed_time(e) for s, e in stages['istft']):8.1f}} ms", flush=True)
print("  --- by module, depth 2 ---", flush=True)
shown = 0
for elapsed, count, name in rows:
    if name.count(".") > 2:
        continue
    print(f"  {{name:44s}} {{elapsed:8.1f}} ms {{100.0 * elapsed / total:5.1f}}%  x{{count}}", flush=True)
    shown += 1
    if shown >= 18:
        break
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="bs_roformer")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tests-dir", default="/tmp/pc/tests")
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    env = dict(os.environ)
    env.setdefault("BRTB_ENABLE_SUBLAS_API", "1")
    # Preserve SDK compiler modules (e.g. Biren br_generator) in child processes.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (args.tests_dir, env.get("PYTHONPATH", ""))))
    code = textwrap.dedent(CHILD).format(
        tests_dir=args.tests_dir, preset=args.preset, seconds=args.seconds,
        use_bf16=args.bf16,
    )
    try:
        proc = subprocess.run([sys.executable, "-u", "-c", code], capture_output=True,
                              text=True, timeout=args.timeout, env=env)
    except subprocess.TimeoutExpired:
        print(f"TIMELINE TIMEOUT after {args.timeout:.0f}s")
        return 1
    for line in proc.stdout.splitlines():
        if line.startswith(("TOTAL", "  ")):
            print(line)
    if "TOTAL" not in proc.stdout:
        err = [ln for ln in proc.stderr.splitlines() if ln.strip() and "Warning" not in ln]
        print(f"TIMELINE FAILED {(err[-1] if err else '?')[:160]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
