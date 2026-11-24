"""Phase F drivers: are the triton-ascend fused kernels correct and faster than eager?

For each kernel: compile status, eager vs fused latency, and the numerical delta
against the exact eager expression that the model path uses.

    python kernels_test.py --dtype fp16 --out /workspace/.tmp/kernels.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from npu_kernels_bandunet import (  # noqa: E402
    bandsplit_norm_scale,
    fused_gated_out,
    fused_rope,
    triton_status,
)


def timeit(fn, device, iters=30, warmup=5) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "npu":
        torch.npu.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if device.type == "npu":
            torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def delta(out, ref) -> tuple[float, float]:
    d = (out.float() - ref.float()).abs()
    return float(d.max()), float(d.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt().clamp_min(1e-12))


def check_bandsplit(device, dtype) -> dict:
    """Real band-split shape: 62 bands, per-band feature widths 1..60, dim=192."""
    n_bands, dim = 62, 64  # rows = bands*t, one band-width per launch in production
    for n_feat in (16, 32, 60):
        rows = n_bands * 64
        x = torch.randn(rows, n_feat, device=device, dtype=dtype)
        gamma = torch.randn(n_feat, device=device, dtype=dtype).abs() + 0.5

        def eager():
            return torch.nn.functional.normalize(x, dim=-1) * (n_feat ** 0.5) * gamma

        ref, out = eager(), bandsplit_norm_scale(x, gamma)
        mx, rr = delta(out, ref)
        entry = {
            "kernel": "bandsplit_norm_scale",
            "n_feat": n_feat,
            "rows": rows,
            "max_abs": mx,
            "rel_rms": rr,
            "eager_ms": round(timeit(eager, device), 5),
            "fused_ms": round(timeit(lambda: bandsplit_norm_scale(x, gamma), device), 5),
        }
        entry["speedup"] = round(entry["eager_ms"] / entry["fused_ms"], 3) if entry["fused_ms"] else None
        yield entry


def check_rope(device, dtype) -> dict:
    rows, d = 690 * 8, 64
    half = d // 2
    t = torch.randn(rows, d, device=device, dtype=dtype)
    cos = torch.randn(1, 1, 1, half, device=device, dtype=dtype) * 0.1
    sin = torch.randn(1, 1, 1, half, device=device, dtype=dtype) * 0.1

    def eager():
        even = t[..., ::2] * cos - t[..., 1::2] * sin
        odd = t[..., 1::2] * cos + t[..., ::2] * sin
        return torch.cat((even, odd), dim=-1)

    ref = eager()
    out = fused_rope(t, cos.reshape(-1), sin.reshape(-1))
    mx, rr = delta(out, ref)
    entry = {
        "kernel": "fused_rope",
        "rows": rows,
        "max_abs": mx,
        "rel_rms": rr,
        "eager_ms": round(timeit(eager, device), 5),
        "fused_ms": round(timeit(lambda: fused_rope(t, cos.reshape(-1), sin.reshape(-1)), device), 5),
    }
    entry["speedup"] = round(entry["eager_ms"] / entry["fused_ms"], 3) if entry["fused_ms"] else None
    return entry


def check_gated_out(device, dtype) -> dict:
    b, h, t, d = 2, 2, 690, 64
    attn = torch.randn(b, h, t, d, device=device, dtype=dtype)
    gates = torch.randn(b, h, t, device=device, dtype=dtype)
    eager = attn * gates.sigmoid().unsqueeze(-1)
    out = fused_gated_out(attn.clone(), gates)
    mx, rr = delta(out, eager)
    entry = {
        "kernel": "fused_gated_out",
        "shape": [b, h, t, d],
        "max_abs": mx,
        "rel_rms": rr,
        "eager_ms": round(timeit(lambda: attn * gates.sigmoid().unsqueeze(-1), device), 5),
        "fused_ms": round(timeit(lambda: fused_gated_out(attn.clone(), gates), device), 5),
    }
    entry["speedup"] = round(entry["eager_ms"] / entry["fused_ms"], 3) if entry["fused_ms"] else None
    return entry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="fp16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--out", type=Path, default=Path("/workspace/.tmp/kernels.json"))
    args = parser.parse_args(argv)

    status = triton_status()
    print("triton_status:", json.dumps(status), flush=True)

    device = torch.device("cpu")
    try:
        import torch_npu

        if torch.npu.is_available():
            device = torch.device("npu:0")
    except Exception:
        pass
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    results = {"status": status, "device": str(device), "dtype": args.dtype, "kernels": []}
    for fn in (check_bandsplit, check_gated_out, check_rope):
        try:
            produced = fn(device, dtype)
            rows = list(produced) if hasattr(produced, "__iter__") and not isinstance(produced, dict) else [produced]
        except Exception as exc:
            rows = [{"kernel": fn.__name__, "error": f"{type(exc).__name__}: {exc}"}]
        for row in rows:
            row["triton_path_taken"] = bool(status["available"])
            results["kernels"].append(row)
            print(json.dumps(row), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
