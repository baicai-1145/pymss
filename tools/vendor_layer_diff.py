"""Find the submodule where the device output first diverges from the CPU reference.

Comparing only the final audio hides where a model goes wrong -- scnet ends up 5.8%
off while its STFT is exact to 7e-07, so the drift happens inside the network.  The
two runs happen in separate processes (a CPU forward poisons the first device
forward on BIREN), each dumping per-module summary statistics, and the summaries
are compared afterwards: the first module whose statistics leave tolerance is the
one to look at.

Statistics instead of tensors keep the dumps small enough to compare across
processes (`--max-bytes` bounds them further).

Usage:
    python tools/vendor_layer_diff.py --family scnet
    python tools/vendor_layer_diff.py --family scnet --tol 1e-3 --max-bytes 4000000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap

CHILD = """
import json
import sys

import torch

mode = "{mode}"
sys.path.insert(0, {tests_dir!r})
import parity
from pymss_core.device import prepare_device_alias

prepare_device_alias("biren")

name = {name!r}
builder, shape, _ = parity.cases()[name]
stats = {{}}
order = []
budget = {{"left": {max_bytes}}}


def summarise(tensor):
    if not isinstance(tensor, torch.Tensor):
        return None
    planes = torch.view_as_real(tensor.detach()) if tensor.is_complex() else tensor.detach()
    planes = planes.float()
    if planes.numel() * 4 > budget["left"]:
        return None
    budget["left"] -= planes.numel() * 4
    host = planes.cpu()
    return {{
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "sum": float(host.double().sum()),
        "absmax": float(host.abs().max()),
        "absmean": float(host.abs().mean()),
        "std": float(host.std()) if host.numel() > 1 else 0.0,
    }}


def hook(module, _inputs, output):
    if not hasattr(module, "_diff_index"):
        return
    info = summarise(output)
    if info is not None:
        stats[module._diff_name] = info
        order.append(module._diff_name)


saved = None
if mode == "device":
    saved = torch.load("{input_path}")
    torch.manual_seed(0)
    model = builder().eval().to(torch.device("supa:0"))
    x = saved["input"].to(torch.device("supa:0"))
else:
    torch.manual_seed(0)
    model = builder().eval()
    x = torch.randn(*shape)
    torch.save({{"input": x}}, "{input_path}")

for index, (module_name, module) in enumerate(model.named_modules()):
    if module_name:
        module._diff_index = index
        module._diff_name = module_name
        module.register_forward_hook(hook)

with torch.inference_mode():
    out = model(x)
if mode == "device":
    torch.supa.synchronize()
final = summarise(out)
with open("{out_path}", "w") as handle:
    json.dump({{"order": order, "stats": stats, "final": final}}, handle)
print("DONE %s %d modules" % (mode, len(order)), flush=True)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="scnet")
    parser.add_argument("--tests-dir", default="/tmp/pc/tests")
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--max-bytes", type=int, default=4_000_000)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    env = dict(os.environ)
    env.setdefault("BRTB_ENABLE_SUBLAS_API", "1")
    env["PYTHONPATH"] = args.tests_dir
    tmp = tempfile.gettempdir()
    input_path = os.path.join(tmp, f"layerdiff_{args.family}_input.pt")
    dumps = {}
    for mode in ("cpu", "device"):
        out_path = os.path.join(tmp, f"layerdiff_{args.family}_{mode}.json")
        code = textwrap.dedent(CHILD).format(
            mode=mode, tests_dir=args.tests_dir, name=args.family,
            input_path=input_path, out_path=out_path, max_bytes=args.max_bytes,
        )
        proc = subprocess.run(
            [sys.executable, "-u", "-c", code], capture_output=True, text=True,
            timeout=args.timeout, env=env,
        )
        if not os.path.exists(out_path):
            err = [ln for ln in proc.stderr.splitlines() if ln.strip() and "Warning" not in ln]
            print(f"{mode} run failed: {(err[-1] if err else '?')[:160]}")
            return 1
        with open(out_path) as handle:
            dumps[mode] = json.load(handle)

    cpu, dev = dumps["cpu"], dumps["device"]
    shared = [n for n in cpu["order"] if n in dev["stats"]]
    print(f"compared {len(shared)} modules (tol={args.tol:g})")
    worst = []
    for module_name in shared:
        a, b = cpu["stats"][module_name], dev["stats"][module_name]
        scale = max(abs(a["absmax"]), 1e-12)
        err = abs(a["sum"] - b["sum"]) / max(abs(a["sum"]), 1e-12)
        err = min(err, abs(a["absmean"] - b["absmean"]) / max(abs(a["absmean"]), 1e-12))
        worst.append((err, module_name, a["absmax"], b["absmax"], a["shape"]))

    worst.sort(reverse=True)
    print(f"{'rel':>10s}  module / absmax cpu -> device")
    for err, module_name, absmax_cpu, absmax_dev, shape in worst[:12]:
        flag = "  <-- diverges" if err > args.tol else ""
        print(f"{err:10.2e}  {module_name[:56]:56s} {absmax_cpu:9.4f} -> {absmax_dev:9.4f} {shape}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
