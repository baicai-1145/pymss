"""Verify every model family against a CPU reference, on the device, twice.

A single device run is not enough evidence: BIREN has already been caught
returning silent zeros, so the acceptance criteria are

* ``rel`` -- max |device - cpu| / max |cpu|, small enough to call it correct;
* ``repeat`` -- max |run1 - run2| on the device, which catches nondeterminism
  that would otherwise look like a tolerance issue;
* ``nonzero`` -- a model that outputs only zeros is a failure even when it runs.

Usage:
    python tools/vendor_model_verify.py --only htdemucs,legacy_tasnet
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
import torch
import sys
sys.path.insert(0, "{tests_dir}")
import parity

name = "{name}"
builder, shape, _ = parity.cases()[name]
torch.manual_seed(0)
model_cpu = builder().eval()
x = torch.randn(*shape)

with torch.inference_mode():
    ref = model_cpu(x).float()

torch.manual_seed(0)
model = builder().eval().to(torch.device("supa:0"))
x_dev = x.to(torch.device("supa:0"))
outs = []
with torch.inference_mode():
    for _ in range(2):
        outs.append(model(x_dev).float().cpu())
torch.supa.synchronize()

a, b = outs
if a.shape != ref.shape:
    print("SHAPE %s vs %s" % (tuple(a.shape), tuple(ref.shape)), flush=True)
    raise SystemExit(0)
scale = ref.abs().max().item() or 1.0
rel = (a - ref).abs().max().item() / scale
repeat = (a - b).abs().max().item() / scale
nonzero = int((a != 0).sum())
total = a.numel()
print(
    "rel=%.2e repeat=%.2e nonzero=%d/%d%s"
    % (rel, repeat, nonzero, total, "" if nonzero else "  <-- SILENT ZEROS"),
    flush=True,
)
"""


def _scrub(line: str) -> str:
    line = " ".join(line.split())
    if "csrc/" in line:
        _head, _sep, tail = line.rpartition("csrc/")
        parts = tail.split(None, 1)
        line = parts[1] if len(parts) > 1 else tail
    if "Reason:" in line:
        line = line[line.index("Reason:"):]
    return line[:150]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests-dir", default="/tmp/pc/tests")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--only", default="")
    parser.add_argument("--verbose", type=int, default=0, help="print N stderr lines of a failing child")
    args = parser.parse_args()

    sys.path.insert(0, args.tests_dir)
    import parity  # noqa: E402

    names = list(parity.cases())
    if args.only:
        wanted = args.only.split(",")
        names = [n for n in names if n in wanted]

    env = dict(os.environ)
    env.setdefault("BRTB_ENABLE_FORCE_EAGER_CONV2D", "1")
    env.setdefault("BRTB_ENABLE_FORCE_EAGER_CONV2DTRANSPOSE", "1")
    env.setdefault("BRTB_ENABLE_SUBLAS_API", "1")
    env["PYTHONPATH"] = args.tests_dir

    width = max(len(n) for n in names)
    ok = 0
    for name in names:
        code = textwrap.dedent(CHILD).format(tests_dir=args.tests_dir, name=name)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code], capture_output=True, text=True, timeout=args.timeout, env=env
            )
        except subprocess.TimeoutExpired:
            print(f"{name:{width}s} TIMEOUT>{args.timeout:.0f}s", flush=True)
            continue
        good = [ln for ln in proc.stdout.splitlines() if ln.startswith(("rel=", "SHAPE"))]
        if good:
            line = good[-1]
            if line.startswith("rel="):
                ok += 1
            print(f"{name:{width}s} {line}", flush=True)
            continue
        err = [ln for ln in proc.stderr.splitlines() if ln.strip()]
        reason = next(
            (ln for ln in reversed(err) if "Reason:" in ln or "Error" in ln or "Exception" in ln),
            err[-1] if err else "?",
        )
        print(f"{name:{width}s} FAILED {_scrub(reason)}", flush=True)
        if args.verbose:
            # The interesting part of a vendor failure is usually the few lines
            # above the last one (the operator name and its shape).
            for line in err[-args.verbose:]:
                print(f"    {line.strip()[:190]}", flush=True)
    print(f"--- {ok}/{len(names)} ran end to end", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
