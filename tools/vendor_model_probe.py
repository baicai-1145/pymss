"""Run every model family from tests/parity.cases() on the vendor device vs CPU.

The parity harness already builds a minimal model of each family with a fixed
input shape, which makes it the cheapest possible device-coverage tripwire: no
weight download, no config plumbing, and the CPU result is the reference.

Each family runs in its own subprocess, because a missing kernel raises while a
bad plan can terminate() the interpreter.

Usage:
    python tools/vendor_model_probe.py --device supa
    python tools/vendor_model_probe.py --device supa --trace --only htdemucs
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap

#: Records every aten op so the last line names the operator that actually failed.
TRACE_CLASS = """
import torch


class _OpTrace(torch.overrides.TorchFunctionMode):
    def __init__(self, path):
        self.path = path

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        shapes = ",".join(str(tuple(a.shape)) for a in list(args)[:3] if isinstance(a, torch.Tensor))
        with open(self.path, "a") as fh:
            fh.write(func.__name__ + " " + shapes + "\\n")
        return func(*args, **kwargs)
"""

CHILD = """
import torch
{prelude}
import sys
sys.path.insert(0, "{tests_dir}")
import parity
{trace_class}

TRACE_PATH = "{trace_path}"
name = "{name}"
builder, shape, _mlx = parity.cases()[name]
torch.manual_seed(0)
model = builder().eval()
x = torch.randn(*shape)

with torch.inference_mode():
    y_ref = model(x).float()

device = torch.device("{device}")
model = model.to(device)
x_dev = x.to(device)
with torch.inference_mode():
    if TRACE_PATH:
        with _OpTrace(TRACE_PATH):
            y_dev = model(x_dev).float()
    else:
        y_dev = model(x_dev).float()
{sync}

y_dev_cpu = y_dev.cpu()
if y_dev_cpu.shape != y_ref.shape:
    print("SHAPE " + str(tuple(y_dev_cpu.shape)) + " vs " + str(tuple(y_ref.shape)), flush=True)
    raise SystemExit(0)
diff = (y_dev_cpu - y_ref).abs()
scale = y_ref.abs().max().item() or 1.0
nan = int(torch.isnan(y_dev_cpu).sum())
print(
    "OK max_err=%.3e rel=%.2e nan=%d shape=%s" % (diff.max().item(), diff.max().item() / scale, nan, tuple(y_ref.shape)),
    flush=True,
)
"""


def _scrub(line: str) -> str:
    """Drop the vendor's build path, which is longer than the message itself."""
    line = " ".join(line.split())
    if "csrc/" in line:
        _head, _sep, tail = line.rpartition("csrc/")
        parts = tail.split(None, 1)
        line = parts[1] if len(parts) > 1 else tail
    if "Reason:" in line:
        line = line[line.index("Reason:"):]
    return line[:170]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="supa")
    parser.add_argument(
        "--prelude",
        default=(
            "from pymss_core.device import prepare_device_alias\n"
            "prepare_device_alias('biren')  # imports torch_br, sets device, installs shims"
        ),
    )
    parser.add_argument("--sync", default="torch.supa.synchronize()")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--tests-dir", default="/tmp/pc/tests")
    parser.add_argument("--only", default="")
    parser.add_argument("--trace", action="store_true", help="record every aten op to find the culprit")
    args = parser.parse_args()

    sys.path.insert(0, args.tests_dir)
    import parity  # noqa: E402

    names = list(parity.cases())
    if args.only:
        names = [n for n in names if n in args.only.split(",")]

    env = dict(os.environ)
    env.setdefault("BRTB_ENABLE_FORCE_EAGER_CONV2D", "1")
    env.setdefault("BRTB_ENABLE_FORCE_EAGER_CONV2DTRANSPOSE", "1")
    env["PYTHONPATH"] = args.tests_dir

    width = max(len(n) for n in names)
    for name in names:
        trace_path = f"/tmp/lastop_{name}.log" if args.trace else ""
        if trace_path and os.path.exists(trace_path):
            os.remove(trace_path)
        code = textwrap.dedent(CHILD).format(
            prelude=args.prelude,
            name=name,
            device=args.device,
            sync=args.sync,
            tests_dir=args.tests_dir,
            trace_class=TRACE_CLASS,
            trace_path=trace_path,
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, timeout=args.timeout, env=env
            )
        except subprocess.TimeoutExpired:
            print(f"{name:{width}s} TIMEOUT>{args.timeout:.0f}s", flush=True)
            continue
        ok = [ln for ln in proc.stdout.splitlines() if ln.startswith(("OK", "SHAPE"))]
        if ok:
            print(f"{name:{width}s} {ok[-1]}", flush=True)
            continue
        err = [ln for ln in proc.stderr.splitlines() if ln.strip()]
        verdict = "CRASH" if proc.returncode not in (0, 1) else "FAIL"
        interesting = [
            ln
            for ln in err
            if ("Reason:" in ln or "RuntimeError" in ln or "TypeError" in ln or "not implemented" in ln.lower())
            and "backtrace" not in ln
        ]
        detail = " | ".join(_scrub(ln) for ln in interesting[-3:]) or (_scrub(err[-1]) if err else "?")
        if trace_path and os.path.exists(trace_path):
            ops = [ln.strip() for ln in open(trace_path).read().splitlines() if ln.strip()]
            detail += "  <-- last ops: " + " ; ".join(ops[-2:])
        print(f"{name:{width}s} {verdict} {detail}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
