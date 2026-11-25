"""Per-module timing at the real training shapes (bs24, 690 frames, 62 bands).

Times forward and forward+backward for every block of the U-Net so fusion work can be
ranked by measured cost instead of guesswork.

    python bench_blocks.py --batch 24 --out results/blocks_bf16.json
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


def bench(fn, iters=10, warmup=3) -> float:
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
    return round(statistics.median(ts), 3)


def kernel_count(fn) -> dict:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        fn()
    kernels = sum(e.count for e in prof.key_averages() if e.self_device_time_total > 0)
    aten = sum(e.count for e in prof.key_averages() if e.key.startswith("aten::"))
    return {"device_kernels": kernels, "aten_calls": aten}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--frames", type=int, default=690)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--out", type=Path, default=Path("results/blocks_bf16.json"))
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401
    from freq_unet import FreqUNetConfig, build_model

    dev = torch.device("npu:0")
    cfg = FreqUNetConfig(n_fft=args.n_fft)
    model = build_model(cfg).to(dev).train()  # training path, NOT the eval/grouped one
    f = args.n_fft // 2 + 1
    b, t = args.batch, args.frames
    d62, d32, d16, ds = cfg.dims

    stft_repr = torch.randn(b, f * cfg.audio_channels, t, 2, device=dev) * 0.1
    feats = stft_repr.permute(0, 2, 1, 3).reshape(b, t, f * cfg.audio_channels * 2)

    x62 = torch.randn(b, t, 62, d62, device=dev) * 0.1
    x32 = torch.randn(b, t, 32, d32, device=dev) * 0.1
    x16 = torch.randn(b, t, 16, d16, device=dev) * 0.1
    x8t = torch.randn(b, 8, t, ds, device=dev) * 0.1      # temporal blocks work (B, F, T, C)
    x8 = torch.randn(b, t, 8, ds, device=dev) * 0.1
    y62 = torch.randn(b, t, 62, d62, device=dev) * 0.1

    blk = model.blocks[0]
    cases = {
        "band_split": (model.band_split, (feats,)),
        "enc62": (model.enc62, (x62,)),
        "down62": (model.down62, (x62,)),
        "enc32": (model.enc32, (x32,)),
        "down32": (model.down32, (x32,)),
        "enc16": (model.enc16, (x16,)),
        "down16": (model.down16, (x16,)),
        "msc": (blk.msc, (x8t,)),
        "freq_attn": (blk.freq, (x8t,)),
        "up8+dec16": (lambda z, s: model.dec16(model.up8(z, s)), (x8, x16)),
        "up16+dec32": (lambda z, s: model.dec32(model.up16(z, s)), (torch.randn(b, t, 16, d16, device=dev) * 0.1, x32)),
        "up32+dec62": (lambda z, s: model.dec62(model.up32(z, s)), (torch.randn(b, t, 32, d32, device=dev) * 0.1, x62)),
        "mask_head": (model._estimate_masks, (y62,)),
    }

    import contextlib

    ac = (torch.npu.amp.autocast(dtype=torch.bfloat16) if args.precision == "bf16"
          else contextlib.nullcontext())
    payload = {"config": {"batch": b, "frames": t, "n_fft": args.n_fft, "precision": args.precision},
               "blocks": {}}
    for name, (mod, inputs) in cases.items():
        def fwd(mod=mod, inputs=inputs):
            with torch.no_grad(), ac:
                return mod(*inputs)

        fwd_ms = bench(fwd)

        def fwd_bwd(mod=mod, inputs=inputs):
            ins = [i.clone().requires_grad_(True) if torch.is_floating_point(i) else i for i in inputs]
            with ac:
                out = mod(*ins)
            out.float().pow(2).mean().backward()

        try:
            fwb_ms = bench(fwd_bwd, 5, 2)
        except Exception as exc:  # noqa: BLE001
            fwb_ms = None
            payload["blocks"].setdefault(name, {})["fwd_bwd_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        kc = kernel_count(fwd)
        payload["blocks"].setdefault(name, {}).update(
            {"fwd_ms": fwd_ms, "fwd_bwd_ms": fwb_ms, **kc})
        print(f"{name:14s} fwd {fwd_ms:8.2f} ms  fwd+bwd {fwb_ms if fwb_ms is None else f'{fwb_ms:8.2f}'} ms"
              f"  kernels {kc['device_kernels']:5d}  aten {kc['aten_calls']:5d}", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
