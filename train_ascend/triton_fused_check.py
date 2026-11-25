"""Triton fusion for the U-Net blocks (C-3): correctness and measured cost.

Two kernels are provided in ``triton_blocks.py``:

* ``norm_dwconv``  – RMSNorm over the channel axis + depthwise conv along the band axis,
  forward in Triton, analytic backward (dx/dgamma/dw/db) in Triton + one torch pass.
* ``fused_silu_mul`` – ``silu(a) * g`` in one kernel.

This script checks both against PyTorch and times them at the real training shapes
(M = batch*frames = 16560, F = 62 bands, C = 192).

Note on precision: with ``torch.npu.conv.allow_hf32 = True`` (the training default) the
vendored ``Conv1d`` itself deviates from an fp32 shift-sum by 2.9e-04, so the comparison is
done with HF32 disabled to expose the kernels' own error.

    python triton_fused_check.py --out results/triton_fused.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def sync():
    torch.npu.synchronize()


def bench(fn, iters=8, warmup=3) -> float:
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(ts), 2)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    d = a.float() - b.float()
    return float(d.pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt().clamp_min(1e-12))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--frames", type=int, default=690)
    parser.add_argument("--bands", type=int, default=62)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--kernel", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("results/triton_fused.json"))
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401

    import triton_blocks as tb

    dev = torch.device("npu:0")
    torch.manual_seed(0)
    M, Fb, C, K = args.batch * args.frames, args.bands, args.dim, args.kernel
    payload: dict = {"shape": {"M": M, "F": Fb, "C": C, "K": K}, "triton_available": tb.available()}

    x = torch.randn(M, Fb, C, device=dev)
    gamma = torch.randn(C, device=dev)
    w = torch.randn(C, 1, K, device=dev) * 0.1
    bias = torch.randn(C, device=dev)

    def torch_fwd(xx=x, gg=gamma, ww=w, bb=bias):
        n = F.normalize(xx, dim=-1) * (C ** 0.5) * gg
        return F.conv1d(n.transpose(1, 2), ww, bb, padding=K // 2, groups=C).transpose(1, 2)

    prev = torch.npu.conv.allow_hf32
    torch.npu.conv.allow_hf32 = False
    ref = torch_fwd().detach()
    y = tb.norm_dwconv(x, gamma, w, bias).detach()
    payload["norm_dwconv"] = {"fwd_rel_rms_hf32_off": rel(y, ref)}

    g = torch.randn_like(ref)
    x1 = x.clone().requires_grad_(True); g1 = gamma.clone().requires_grad_(True)
    w1 = w.clone().requires_grad_(True); b1 = bias.clone().requires_grad_(True)
    tb.norm_dwconv(x1, g1, w1, b1).backward(g)
    x2 = x.clone().requires_grad_(True); g2 = gamma.clone().requires_grad_(True)
    w2 = w.clone().requires_grad_(True); b2 = bias.clone().requires_grad_(True)
    torch_fwd(x2, g2, w2, b2).backward(g)
    payload["norm_dwconv"].update({
        "dx_rel_rms": rel(x1.grad, x2.grad),
        "dgamma_rel_rms": rel(g1.grad, g2.grad),
        "dw_rel_rms": rel(w1.grad, w2.grad),
        "db_rel_rms": rel(b1.grad, b2.grad),
    })
    payload["norm_dwconv"]["torch_fwd_ms"] = bench(torch_fwd)
    payload["norm_dwconv"]["triton_fwd_ms"] = bench(lambda: tb.norm_dwconv(x, gamma, w, bias))

    def torch_fwb():
        a = x.clone().requires_grad_(True); gg = gamma.clone().requires_grad_(True)
        ww = w.clone().requires_grad_(True); bb = bias.clone().requires_grad_(True)
        torch_fwd(a, gg, ww, bb).backward(g)

    def triton_fwb():
        a = x.clone().requires_grad_(True); gg = gamma.clone().requires_grad_(True)
        ww = w.clone().requires_grad_(True); bb = bias.clone().requires_grad_(True)
        tb.norm_dwconv(a, gg, ww, bb).backward(g)

    payload["norm_dwconv"]["torch_fwd_bwd_ms"] = bench(torch_fwb, 5)
    payload["norm_dwconv"]["triton_fwd_bwd_ms"] = bench(triton_fwb, 5)
    torch.npu.conv.allow_hf32 = prev

    # silu(gate) * value, the elementwise half of the SwiGLU stage
    a = torch.randn(M, Fb, C, device=dev)
    v = torch.randn(M, Fb, C, device=dev)
    r = F.silu(a) * v
    t = tb.fused_silu_mul(a, v)
    payload["silu_mul"] = {
        "rel_rms": rel(t, r),
        "torch_ms": bench(lambda: F.silu(a) * v),
        "triton_ms": bench(lambda: tb.fused_silu_mul(a, v)),
    }

    payload["verdict"] = {
        "norm_dwconv": "triton " + (
            "slower" if payload["norm_dwconv"]["triton_fwd_ms"] > payload["norm_dwconv"]["torch_fwd_ms"]
            else "faster") + " than torch",
        "reason": "UB on this device allows only (BF=8, BC=256) tiles, so the kernel needs "
                  "M*ceil(F/BF) = 132k programs with 7 masked loads each; the fixed per-program "
                  "cost dominates (~0.85 us/program). Larger tiles fail to compile with "
                  "'ub overflow, requires 1957888 bits while 1769472 bits available'.",
    }

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
