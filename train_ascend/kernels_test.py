"""Phase F drivers: does triton-ascend work here, and are the fused kernels correct/faster?

On this box triton-ascend publishes cp39/cp310/cp311 wheels only while the
interpreter is CPython 3.12.16, so the kernels cannot be compiled. This script is
still the deliverable: it (a) prints the probed status, (b) when triton is
available, compares each fused kernel against its eager reference and times
both, and (c) always verifies that the *fallback* path used in production is
numerically equivalent to the plain eager expression, so the fused slot can be
enabled later without further work.
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
    fused_bandsplit_linear,
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


def check_gated_out(device, dtype) -> dict:
    b, h, t, d = 2, 2, 690, 64
    attn = torch.randn(b, h, t, d, device=device, dtype=dtype)
    gates = torch.randn(b, h, t, device=device, dtype=dtype)
    eager = attn * gates.sigmoid().unsqueeze(-1)
    fused = fused_gated_out(attn.clone(), gates)
    diff = (fused.float() - eager.float()).abs()
    return {
        "kernel": "attention_gated_out",
        "max_abs": float(diff.max()),
        "rel_rms": float(diff.pow(2).mean().sqrt() / eager.float().pow(2).mean().sqrt()),
        "eager_ms": round(timeit(lambda: attn * gates.sigmoid().unsqueeze(-1), device), 4),
        "fused_ms": round(timeit(lambda: fused_gated_out(attn.clone(), gates), device), 4),
    }


def check_rope(device, dtype) -> dict:
    n, d = 690 * 8, 64
    q = torch.randn(n, d, device=device, dtype=dtype)
    k = torch.randn(n, d, device=device, dtype=dtype)
    half = d // 2
    cos = torch.randn(n, d, device=device, dtype=dtype) * 0.1
    sin = torch.randn(n, d, device=device, dtype=dtype) * 0.1

    def eager(x):
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cos + rotated * sin

    q_ref, k_ref = eager(q), eager(k)
    q_f, k_f = fused_rope(q.clone(), k.clone(), cos, sin)
    diff = max(float((q_f.float() - q_ref.float()).abs().max()), float((k_f.float() - k_ref.float()).abs().max()))
    rms = float((q_f.float() - q_ref.float()).pow(2).mean().sqrt() / q_ref.float().pow(2).mean().sqrt())
    return {
        "kernel": "rope",
        "max_abs": diff,
        "rel_rms": rms,
        "eager_ms": round(timeit(lambda: eager(q), device), 4),
        "fused_ms": round(timeit(lambda: fused_rope(q.clone(), k.clone(), cos, sin), device), 4),
    }


def check_bandsplit(device, dtype) -> dict:
    bands, feat, dim = 62, 92, 192
    x = torch.randn(bands, feat, device=device, dtype=dtype)
    weight = torch.randn(dim, feat, device=device, dtype=dtype) * 0.02
    bias = torch.zeros(dim, device=device, dtype=dtype)

    def eager():
        normed = torch.nn.functional.normalize(x, dim=-1) * (feat ** 0.5)
        return torch.nn.functional.linear(normed, weight, bias)

    ref = eager()
    out = fused_bandsplit_linear(x, weight, bias)
    diff = (out.float() - ref.float()).abs()
    return {
        "kernel": "bandsplit_linear",
        "max_abs": float(diff.max()),
        "rel_rms": float(diff.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()),
        "eager_ms": round(timeit(eager, device), 4),
        "fused_ms": round(timeit(lambda: fused_bandsplit_linear(x, weight, bias), device), 4),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
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
            row = fn(device, dtype)
        except Exception as exc:
            row = {"kernel": fn.__name__, "error": f"{type(exc).__name__}: {exc}"}
        row["triton_path_taken"] = bool(status["available"])
        results["kernels"].append(row)
        print(json.dumps(row), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
