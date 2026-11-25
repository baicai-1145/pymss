"""Compare the three ways to run the 62-band Linear groups (C-1 / C-2).

  per_band : torch.stack([to_feature(slice) ...])            -- pymss_core training path
  grouped7 : one bmm per contiguous equal-width run          -- pymss_core experimental path
  padded   : one padded bmm for all 62 bands                 -- this repo (packed_bands.py)

Each is verified against the per-band path and timed forward and forward+backward at the
real training shapes.

    python bench_packing.py --batch 24 --out results/packing.json
"""

from __future__ import annotations

import argparse
import json
import os
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


def kernel_stats(fn) -> dict:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
        fn()
    kernels = sum(e.count for e in prof.key_averages() if e.self_device_time_total > 0)
    aten = sum(e.count for e in prof.key_averages() if e.key.startswith("aten::"))
    return {"device_kernels": kernels, "aten_calls": aten}


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    d = a.float() - b.float()
    return float(d.pow(2).mean().sqrt() / b.float().pow(2).mean().sqrt().clamp_min(1e-12))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--frames", type=int, default=690)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--out", type=Path, default=Path("results/packing.json"))
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401

    import packed_bands as pb
    from freq_unet import FreqUNetConfig, build_model
    from pymss_core.modules.bs_roformer.bands import BandSplit, MaskEstimator

    dev = torch.device("npu:0")
    torch.manual_seed(0)
    cfg = FreqUNetConfig(n_fft=2048)
    model = build_model(cfg).to(dev).train()
    b, t = args.batch, args.frames
    feats = torch.randn(b, t, 4100, device=dev) * 0.1
    y62 = torch.randn(b, t, 62, 192, device=dev) * 0.1
    ac = (torch.npu.amp.autocast(dtype=torch.bfloat16) if args.precision == "bf16"
          else __import__("contextlib").nullcontext())

    per_band_bs = BandSplit.forward
    per_band_mask = MaskEstimator.forward
    payload: dict = {"config": {"batch": b, "frames": t, "precision": args.precision}, "variants": {}}

    def measure(name: str) -> None:
        def bs_fwd():
            with torch.no_grad(), ac:
                return model.band_split(feats)

        def bs_fwb():
            x = feats.clone().requires_grad_(True)
            with ac:
                out = model.band_split(x)
            out.float().pow(2).mean().backward()
            return x.grad

        def mh_fwd():
            with torch.no_grad(), ac:
                return model._estimate_masks(y62)

        def mh_fwb():
            x = y62.clone().requires_grad_(True)
            with ac:
                out = model._estimate_masks(x)
            out.float().pow(2).mean().backward()
            return x.grad

        def guarded(fn, *a, **k):
            try:
                return fn(*a, **k)
            except Exception as exc:  # noqa: BLE001
                return f"{type(exc).__name__}: {str(exc)[:120]}"

        row = {
            "band_split_fwd_ms": guarded(bench, bs_fwd),
            "band_split_fwd_bwd_ms": guarded(bench, bs_fwb, 5),
            "mask_fwd_ms": guarded(bench, mh_fwd),
            "mask_fwd_bwd_ms": guarded(bench, mh_fwb, 5),
            "band_split_kernels": guarded(kernel_stats, bs_fwd),
            "mask_kernels": guarded(kernel_stats, mh_fwd),
        }
        with torch.no_grad(), ac:
            row["_bs_out"] = guarded(lambda: model.band_split(feats).float().clone())
            row["_mh_out"] = guarded(lambda: model._estimate_masks(y62).float().clone())
        payload["variants"][name] = row
        print(name, {k: v for k, v in row.items() if not k.startswith("_")}, flush=True)

    os.environ.pop("PYMSS_CORE_EXPERIMENTAL_TRAIN_GROUPED_BANDS", None)
    os.environ.pop("PYMSS_CORE_EXPERIMENTAL_DEEP_MASK_GROUPING", None)
    BandSplit.forward = per_band_bs
    MaskEstimator.forward = per_band_mask
    measure("per_band")

    os.environ["PYMSS_CORE_EXPERIMENTAL_TRAIN_GROUPED_BANDS"] = "1"
    os.environ["PYMSS_CORE_EXPERIMENTAL_DEEP_MASK_GROUPING"] = "1"
    measure("grouped7")
    os.environ.pop("PYMSS_CORE_EXPERIMENTAL_TRAIN_GROUPED_BANDS", None)
    os.environ.pop("PYMSS_CORE_EXPERIMENTAL_DEEP_MASK_GROUPING", None)

    pb.enable_packed_band_split(True)
    pb.enable_padded_mask(True)
    try:
        measure("padded")
    finally:
        pb.enable_packed_band_split(False)
        pb.enable_padded_mask(False)

    base = payload["variants"]["per_band"]
    for name in ("grouped7", "padded"):
        v = payload["variants"].get(name)
        if not v:
            continue
        if isinstance(v["_bs_out"], torch.Tensor):
            v["band_split_out_rel"] = rel(v["_bs_out"], base["_bs_out"])
        if isinstance(v["_mh_out"], torch.Tensor):
            v["mask_out_rel"] = rel(v["_mh_out"], base["_mh_out"])
        for key, base_key in (("band_split_speedup_fwd", "band_split_fwd_ms"),
                              ("band_split_speedup_fwd_bwd", "band_split_fwd_bwd_ms"),
                              ("mask_speedup_fwd", "mask_fwd_ms"),
                              ("mask_speedup_fwd_bwd", "mask_fwd_bwd_ms")):
            mine, theirs = v[base_key], base[base_key]
            if isinstance(mine, (int, float)) and isinstance(theirs, (int, float)):
                v[key] = round(theirs / mine, 2)
    for v in payload["variants"].values():
        for k in [k for k in v if k.startswith("_")]:
            del v[k]

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if "speedup" in kk or "rel" in kk}
                      for k, v in payload["variants"].items()}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
