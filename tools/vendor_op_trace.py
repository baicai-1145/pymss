"""Per-operator trace for one model family on the vendor device.

A single forward is not enough to trust a family: BIREN has failed on the second
run of the same input, and its errors are asynchronous, so the Python traceback
usually names an innocent later operator.  This tool therefore

* keeps a rolling list of the last operators (name, shapes, dtypes) so a crash
  still shows what was actually running;
* times every call and prints anything slower than ``--slower-than``;
* stops itself after ``--cap`` seconds instead of waiting out a ten-minute
  planner stall;
* prints the entire vendor message, which is where the real reason lives.

One family per process (the caller runs it in a subprocess), because a vendor
abort kills the whole interpreter.

Usage:
    python tools/vendor_op_trace.py --only scnet --repeat 2 --cap 120
    python tools/vendor_op_trace.py --only legacy_v3_demucs --cap 120
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap

CHILD = """
import sys, time
sys.path.insert(0, {tests_dir!r})
from pymss_core.device import prepare_device_alias
prepare_device_alias("biren")
import torch
import parity

name = {name!r}
repeat = {repeat}
slower_than = {slower_than}
cap = {cap}

builder, shape, _ = parity.cases()[name]
torch.manual_seed(0)
model = builder().eval().to(torch.device("supa:0"))
x = torch.randn(*shape, device=torch.device("supa:0"))
print("### %s input=%s" % (name, tuple(x.shape)), flush=True)

recent = []
t0 = time.perf_counter()

class Trace(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        tensors = [a for a in list(args) if isinstance(a, torch.Tensor)][:3]
        recent.append((func.__name__, [tuple(t.shape) for t in tensors],
                       [str(t.dtype).replace("torch.", "") for t in tensors]))
        if len(recent) > 8:
            recent.pop(0)
        start = time.perf_counter()
        out = func(*args, **(kwargs or {{}}))
        try:
            torch.supa.synchronize()
        except Exception:
            pass
        spent = time.perf_counter() - start
        if spent > slower_than:
            print("  SLOW %7.2fs %-20s %s at %6.1fs"
                  % (spent, func.__name__, [tuple(t.shape) for t in tensors], time.perf_counter() - t0), flush=True)
        if time.perf_counter() - t0 > cap:
            raise TimeoutError("cap %.0fs" % cap)
        return out

for run in range(repeat):
    recent.clear()
    try:
        with torch.inference_mode(), Trace():
            y = model(x)
        torch.supa.synchronize()
        planes = torch.view_as_real(y) if y.is_complex() else y
        print("  run %d OK absmax=%.6f nonzero=%d" % (run, float(planes.abs().max()), int((planes != 0).sum())), flush=True)
    except Exception as exc:
        print("  run %d FAILED after %.1fs" % (run, time.perf_counter() - t0), flush=True)
        for line in str(exc).splitlines():
            if line.strip():
                print("    " + line.strip()[:200], flush=True)
        print("  last operators:", flush=True)
        for op, shapes, dtypes in recent[-6:]:
            print("    %-18s %s %s" % (op, shapes, dtypes), flush=True)
        break
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="")
    parser.add_argument("--tests-dir", default="/tmp/pc/tests")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--slow", dest="slower_than", type=float, default=1.0)
    parser.add_argument("--cap", type=float, default=120.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    sys.path.insert(0, args.tests_dir)
    import parity  # noqa: E402

    names = [n for n in parity.cases() if n in args.only.split(",")] if args.only else list(parity.cases())
    import os

    env = dict(os.environ)
    env.setdefault("BRTB_ENABLE_SUBLAS_API", "1")
    env["PYTHONPATH"] = args.tests_dir
    for name in names:
        code = textwrap.dedent(CHILD).format(
            tests_dir=args.tests_dir, name=name, repeat=args.repeat,
            slower_than=args.slower_than, cap=args.cap,
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-u", "-c", code], capture_output=True, text=True,
                timeout=args.timeout, env=env,
            )
            out = proc.stdout
            if not out.strip():
                out = "\n".join(proc.stderr.strip().splitlines()[-4:])
        except subprocess.TimeoutExpired:
            out = "TIMEOUT>%.0fs (the child was killed)" % args.timeout
        print(out.strip(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
