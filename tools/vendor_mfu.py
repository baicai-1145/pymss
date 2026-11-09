"""Measure achieved TFLOPS and MFU on the vendor device.

Three numbers matter and only the first is published (and the published figures
disagree, 120 vs 170 TFLOPS for a BR106M), so everything else is measured:

1. ``peak``  -- achievable dense matmul TFLOPS, bf16 and fp32, from large GEMMs;
2. ``model`` -- the FLOPs a model actually performs, counted analytically by
   ``torch.utils.flop_counter`` on the CPU for the same shapes;
3. ``MFU``   -- achieved FLOPs/s divided by the measured peak, which is the only
   denominator that reflects this card with this driver.

Usage:
    python tools/vendor_mfu.py --device supa
    python tools/vendor_mfu.py --device supa --only bs_roformer,mdx23c
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import textwrap

#: Published figures, for reference only; the measured peak is what MFU uses.
PUBLISHED_BF16_TFLOPS = 120.0

GEMM_CHILD = """
from pymss_core.device import prepare_device_alias
prepare_device_alias("biren")
import statistics
import torch

D = torch.device("supa:0")
for dtype, label in ((torch.bfloat16, "bf16"), (torch.float32, "fp32")):
    best = 0.0
    for n in (2048, 4096, 8192):
        try:
            a = torch.randn(n, n, device=D, dtype=dtype)
            b = torch.randn(n, n, device=D, dtype=dtype)
            a @ b
            torch.supa.synchronize()
            times = []
            for _ in range(3):
                torch.supa.synchronize()
                start = torch.supa.Event(enable_timing=True)
                end = torch.supa.Event(enable_timing=True)
                start.record()
                c = a @ b
                end.record()
                torch.supa.synchronize()
                times.append(start.elapsed_time(end))
            ms = min(times)
            tflops = 2 * n ** 3 / (ms / 1000) / 1e12
            best = max(best, tflops)
            print("  %s %4d^3  %8.2f ms  %7.2f TFLOPS" % (label, n, ms, tflops), flush=True)
        except Exception as exc:
            print("  %s %4d^3  FAILED %s" % (label, n, str(exc).splitlines()[0][:60]), flush=True)
    print("PEAK %s %.2f" % (label, best), flush=True)
"""

MODEL_CHILD = """
from pymss_core.device import prepare_device_alias
prepare_device_alias("biren")
import statistics
import sys
import torch
sys.path.insert(0, "{tests_dir}")
import parity
from torch.utils.flop_counter import FlopCounterMode

name = "{name}"
builder, shape, _ = parity.cases()[name]
# parity's shapes are sized for correctness tests (64 samples for the roformer
# family), which makes both the FLOP count and the timing unrepresentative, so the
# benchmark uses real audio of --seconds length and keeps the channel count.
channels = shape[1] if len(shape) > 1 else 1
samples = int({sample_rate} * {seconds})
x = torch.randn(shape[0], channels, samples)

torch.manual_seed(0)
model_cpu = builder().eval()
with torch.inference_mode(), FlopCounterMode(display=False) as counter:
    model_cpu(x)
flops = counter.get_total_flops()

torch.manual_seed(0)
model = builder().eval().to(torch.device("supa:0"))
x_dev = x.to(torch.device("supa:0"))
USE_BF16 = {use_bf16}
amp = torch.supa.amp.autocast(dtype=torch.bfloat16) if USE_BF16 else torch.no_grad()
with torch.inference_mode():
    with amp:
        model(x_dev)
    torch.supa.synchronize()
    times = []
    for _ in range(3):
        torch.supa.synchronize()
        start = torch.supa.Event(enable_timing=True)
        end = torch.supa.Event(enable_timing=True)
        start.record()
        with amp:
            model(x_dev)
        end.record()
        torch.supa.synchronize()
        times.append(start.elapsed_time(end))
print("MODEL flops=%.3e ms=%.3f tflops=%.3f" % (flops, statistics.median(times), flops / (statistics.median(times) / 1000) / 1e12), flush=True)
"""


def _run(cmd_env: dict, code: str, timeout: float) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True, text=True, timeout=timeout, env=cmd_env,
        )
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    lines = [
        ln for ln in proc.stdout.splitlines()
        if ln.startswith(("MODEL", "PEAK", "  "))
    ]
    if not lines:
        err = [ln for ln in proc.stderr.splitlines() if ln.strip()]
        reason = err[-1][:120] if err else "no output"
        return proc.returncode or 1, reason
    return 0, "\n".join(lines)


def main() -> int:  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-dir", default="/tmp/pc/tests")
    parser.add_argument("--seconds", type=float, default=10.0, help="audio length per family")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--only", default="")
    parser.add_argument(
        "--skip-gemm",
        nargs="?",
        const="true",
        default="false",
        choices=["true", "false"],
        help="skip the matmul peak measurement",
    )
    parser.add_argument("--bf16", action="store_true", help="wrap the model forward in vendor bf16 autocast")
    args = parser.parse_args()

    env = dict(os.environ)
    env.setdefault("BRTB_ENABLE_FORCE_EAGER_CONV2D", "1")
    env.setdefault("BRTB_ENABLE_FORCE_EAGER_CONV2DTRANSPOSE", "1")
    env.setdefault("BRTB_ENABLE_SUBLAS_API", "1")
    # Preserve SDK compiler modules (e.g. Biren br_generator) in child processes.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (args.tests_dir, env.get("PYTHONPATH", ""))))

    peaks = {}
    if args.skip_gemm != "true":
        print("=== achievable dense matmul", flush=True)
        code, out = _run(env, GEMM_CHILD, args.timeout)
        print(out, flush=True)
        for line in out.splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[0] == "PEAK":
                peaks[parts[1]] = float(parts[2])

    sys.path.insert(0, args.tests_dir)
    import parity  # noqa: E402

    names = list(parity.cases())
    if args.only:
        names = [n for n in names if n in args.only.split(",")]

    peak_key = "bf16" if args.bf16 else "fp32"
    peak_used = peaks.get(peak_key) or (PUBLISHED_BF16_TFLOPS / (1 if args.bf16 else 2))
    print(
        "\n=== per-model (MFU against measured %s peak %.1f TFLOPS)"
        % (peak_key, peak_used),
        flush=True,
    )
    print(f"{'family':22s} {'GFLOPs':>10s} {'ms':>9s} {'TFLOPS':>8s} {'MFU':>7s}", flush=True)
    width = max(len(n) for n in names)
    for name in names:
        code, out = _run(
            env,
            MODEL_CHILD.format(tests_dir=args.tests_dir, name=name, use_bf16=str(args.bf16),
                               sample_rate=args.sample_rate, seconds=args.seconds),
            args.timeout,
        )
        if not out.startswith("MODEL"):
            print(f"{name:{width}s} FAILED {out}", flush=True)
            continue
        parts = dict(p.split("=") for p in out.split()[1:])
        flops, ms, tflops = float(parts["flops"]), float(parts["ms"]), float(parts["tflops"])
        print(
            f"{name:{width}s} {flops / 1e9:10.2f} {ms:9.3f} {tflops:8.3f} {tflops / peak_used:6.2%}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
