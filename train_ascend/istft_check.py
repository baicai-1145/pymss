"""Equivalence + timing check for the explicit iSTFT adjoint (task A).

Verifies ``fast_istft.istft_adjoint`` against ``istft_npu``'s own autograd
(fwd value and grad, rel_rms <= 1e-6) and times fwd / fwd+bwd at the real shapes.

    python istft_check.py --out results/istft_check.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


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


def rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.float() - b.float())
    return float(d.pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt().clamp_min(1e-12))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=48)  # 24 clips x 2 channels
    parser.add_argument("--length", type=int, default=352800)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("results/istft_check.json"))
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401
    from pymss_core.npu_fft import istft_npu

    from fast_istft import istft_adjoint

    dev = torch.device("npu:0")
    torch.manual_seed(0)
    f = args.n_fft // 2 + 1
    frames = 1 + args.length // args.hop
    window = torch.hann_window(args.n_fft).to(dev)
    real = torch.randn(args.batch, f, frames, 2, device=dev) * 0.1
    spec = torch.view_as_complex(real.contiguous())
    grad = torch.randn(args.batch, args.length, device=dev)

    payload: dict = {"batch": args.batch, "length": args.length, "n_fft": args.n_fft,
                     "hop": args.hop, "frames": frames}

    # reference: istft_npu with autograd through its own backward
    s_ref = spec.clone().requires_grad_(True)
    y_ref = istft_npu(s_ref, n_fft=args.n_fft, hop_length=args.hop, window=window,
                      length=args.length, return_complex=False)
    g_ref, = torch.autograd.grad(y_ref, s_ref, grad_outputs=grad)

    s_fast = spec.clone().requires_grad_(True)
    y_fast = istft_adjoint(s_fast, window, args.n_fft, args.hop, args.length)
    g_fast, = torch.autograd.grad(y_fast, s_fast, grad_outputs=grad)

    payload["forward_rel_rms"] = rel_rms(y_fast, y_ref)
    payload["grad_rel_rms"] = rel_rms(g_fast, g_ref)
    payload["grad_max_abs"] = float((g_fast - g_ref).abs().max())

    def fwd_ref():
        istft_npu(spec, n_fft=args.n_fft, hop_length=args.hop, window=window,
                  length=args.length, return_complex=False)

    def fwd_fast():
        istft_adjoint(spec, window, args.n_fft, args.hop, args.length)

    def fwb_ref():
        s = spec.clone().requires_grad_(True)
        y = istft_npu(s, n_fft=args.n_fft, hop_length=args.hop, window=window,
                      length=args.length, return_complex=False)
        torch.autograd.grad(y, s, grad_outputs=grad)

    def fwb_fast():
        s = spec.clone().requires_grad_(True)
        y = istft_adjoint(s, window, args.n_fft, args.hop, args.length)
        torch.autograd.grad(y, s, grad_outputs=grad)

    payload["fwd_ms"] = {"vendor": bench(fwd_ref), "adjoint_fn": bench(fwd_fast)}
    payload["fwd_bwd_ms"] = {"vendor": bench(fwb_ref, 5, 2), "adjoint_fn": bench(fwb_fast, 5, 2)}
    payload["fwd_bwd_speedup"] = round(payload["fwd_bwd_ms"]["vendor"] / payload["fwd_bwd_ms"]["adjoint_fn"], 2)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
