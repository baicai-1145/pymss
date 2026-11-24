"""8 s inference latency on NPU (bf16 + fp32) vs the G4 reference (11.691 ms eager bf16)."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freq_unet import FreqUNetConfig, build_model  # noqa: E402

G4_REF_MS = 11.691


def bench(model, spec, device, precision, iters=50, warmup=10) -> dict:
    times = []
    model.eval()
    with torch.no_grad():
        for i in range(warmup + iters):
            if device.type == "npu":
                torch.npu.synchronize()
            t0 = time.perf_counter()
            if precision == "bf16":
                with torch.npu.amp.autocast(dtype=torch.bfloat16):
                    model(spec)
            else:
                model(spec)
            if device.type == "npu":
                torch.npu.synchronize()
            dt = (time.perf_counter() - t0) * 1000
            if i >= warmup:
                times.append(dt)
    return {
        "device": str(device),
        "precision": precision,
        "iters": iters,
        "median_ms": round(statistics.median(times), 3),
        "mean_ms": round(statistics.mean(times), 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "p90_ms": round(sorted(times)[int(0.9 * len(times))], 3),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--length", type=int, default=352800)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("/workspace/.tmp/infer_bench.json"))
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401

    cfg = FreqUNetConfig(n_fft=args.n_fft)
    model = build_model(cfg)
    frames = args.length // args.hop + 1
    f = args.n_fft // 2 + 1
    spec = torch.randn(1, f * cfg.audio_channels, frames, 2, device="npu:0") * 0.1

    results = {"params": sum(p.numel() for p in model.parameters()), "frames": frames, "g4_ref_ms": G4_REF_MS}
    for precision in ("32", "bf16"):
        model.to("npu:0")
        results[precision] = bench(model, spec, torch.device("npu:0"), precision, iters=args.iters)
        results[precision]["vs_g4"] = round(results[precision]["median_ms"] / G4_REF_MS, 2)
        print(json.dumps(results[precision]), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
