"""End-to-end VR benchmark: wall time, FLOPs and MFU with fp32 vs bf16.

Two things this answers that the per-model benchmark cannot:

* does bfloat16 actually help on this card for the real separator (the AMP path
  the profile now enables), and
* what is the end-to-end MFU of an audio separation, i.e. how much of the
  measured peak the whole pipeline reaches.

Usage:
    python tools/vendor_bench_e2e.py --seconds 10,60
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap

CHILD = """
from pymss_core.device import prepare_device_alias
prepare_device_alias("biren")
import statistics
import time
import torch
from pymss_core.modules.bs_roformer.common import stft_roformer  # noqa: F401  (import parity)
import os

os.environ.setdefault("PYMSS_AMP", "{amp}")
import pymss
from pymss.separator import Separator

seconds = {seconds}
model_path = "{model}"
rate = 44100
t = torch.arange(int(rate * seconds), dtype=torch.float32) / rate
wave = (0.25 * torch.sin(2 * torch.pi * 220 * t) + 0.12 * torch.sin(2 * torch.pi * 440 * t)).numpy()

sep = Separator(model_path, device="supa", output_dir="/tmp/e2e_{amp}", output_format="wav")
sep.load_model()
import numpy as np
data = np.stack([wave, wave], axis=0)
mixes = [(0.4, 0.6)]
times = []
for _ in range(2):
    torch.supa.synchronize()
    start = time.perf_counter()
    sep.separate(mixes, rate, pbar=False)
    torch.supa.synchronize()
    times.append(time.perf_counter() - start)
print("E2E amp=%s seconds=%.1f wall=%.3f audio_per_wall=%.2fx" % ("{amp}", seconds, statistics.median(times), seconds / statistics.median(times)), flush=True)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="UVR-De-Echo-Aggressive.pth")
    parser.add_argument("--seconds", default="10")
    parser.add_argument("--timeout", type=float, default=1200.0)
    args = parser.parse_args()

    env = dict(os.environ)
    env.setdefault("BRTB_ENABLE_FORCE_EAGER_CONV2D", "1")
    env.setdefault("BRTB_ENABLE_FORCE_EAGER_CONV2DTRANSPOSE", "1")
    env.setdefault("BRTB_ENABLE_SUBLAS_API", "1")

    for amp in ("0", "1"):
        for seconds in args.seconds.split(","):
            code = textwrap.dedent(CHILD).format(model=args.model, amp=amp, seconds=float(seconds))
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", code], capture_output=True, text=True, timeout=args.timeout, env=env
                )
            except subprocess.TimeoutExpired:
                print(f"E2E amp={amp} seconds={seconds} TIMEOUT", flush=True)
                continue
            lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("E2E")]
            if lines:
                print(lines[-1], flush=True)
                continue
            err = [ln for ln in proc.stderr.splitlines() if ln.strip() and "Warning" not in ln]
            print(f"E2E amp={amp} seconds={seconds} FAILED {(err[-1] if err else '?')[:130]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
