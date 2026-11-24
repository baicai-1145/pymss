"""Prove triton-ascend really compiles and runs kernels on this NPU.

Reports the version matrix, the triton target backend, and runs the canonical
vector-add tutorial kernel plus a fused `x*y+z` chain on the device, checking
results against the eager torch op.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import time
from pathlib import Path

import torch


def versions() -> dict:
    info = {}
    for name in ("torch", "torch-npu", "triton", "triton-ascend"):
        try:
            info[name] = md.version(name)
        except Exception:
            info[name] = None
    info["cann"] = open("/usr/local/Ascend/ascend-toolkit/latest/version.cfg").read().splitlines()[0] if Path(
        "/usr/local/Ascend/ascend-toolkit/latest/version.cfg").exists() else None
    try:
        import torch_npu  # noqa: F401

        info["npu_available"] = bool(torch.npu.is_available())
        info["npu_count"] = int(torch.npu.device_count())
        info["npu_name"] = torch.npu.get_device_name(0) if info["npu_available"] else None
    except Exception as exc:
        info["npu_error"] = repr(exc)
    try:
        import triton

        target = triton.runtime.driver.active.get_current_target()
        info["triton_backend"] = getattr(target, "backend", None)
        info["triton_arch"] = getattr(target, "arch", None)
        info["triton_version_attr"] = triton.__version__
    except Exception as exc:
        info["triton_target_error"] = repr(exc)
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1 << 20)
    parser.add_argument("--out", type=Path, default=Path("/workspace/.tmp/triton_check.json"))
    args = parser.parse_args(argv)

    info = versions()
    print(json.dumps(info, indent=2), flush=True)

    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    @triton.jit
    def fma_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
        z = tl.load(z_ptr + offs, mask=mask).to(tl.float32)
        tl.store(out_ptr + offs, x * y + z, mask=mask)

    results = {"versions": info}
    n = args.size
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(n, device="npu:0", dtype=dtype)
        y = torch.randn(n, device="npu:0", dtype=dtype)
        out = torch.empty_like(x)
        grid = (triton.cdiv(n, 4096),)
        t0 = time.perf_counter()
        add_kernel[grid](x, y, out, n, BLOCK=4096)
        torch.npu.synchronize()
        first_ms = (time.perf_counter() - t0) * 1000
        ref = x + y
        diff = (out - ref).abs()
        entry = {
            "kernel": "vector_add",
            "dtype": str(dtype).replace("torch.", ""),
            "compiled": True,
            "first_call_ms": round(first_ms, 2),
            "max_abs": float(diff.max()),
            "rel_rms": float(diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-12)),
        }
        # steady-state timing
        times = []
        for _ in range(20):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            add_kernel[grid](x, y, out, n, BLOCK=4096)
            torch.npu.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        entry["steady_ms"] = round(sorted(times)[len(times) // 2], 4)
        entry["eager_ms"] = None
        t0 = time.perf_counter()
        for _ in range(20):
            torch.npu.synchronize()
            t1 = time.perf_counter()
            x + y
            torch.npu.synchronize()
            times.append((time.perf_counter() - t1) * 1000)
        entry["eager_ms"] = round(sorted(times[-20:])[10], 4)
        results.setdefault("vector_add", []).append(entry)
        print(json.dumps(entry), flush=True)

    x = torch.randn(n, device="npu:0", dtype=torch.float16)
    y = torch.randn(n, device="npu:0", dtype=torch.float16)
    z = torch.randn(n, device="npu:0", dtype=torch.float16)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, 4096),)
    fma_kernel[grid](x, y, z, out, n, BLOCK=4096)
    torch.npu.synchronize()
    ref = x.float() * y.float() + z.float()
    d = (out.float() - ref).abs()
    fma = {"kernel": "fma_chain", "compiled": True, "max_abs": float(d.max()),
           "rel_rms": float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-12))}
    results["fma_chain"] = fma
    print(json.dumps(fma), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
