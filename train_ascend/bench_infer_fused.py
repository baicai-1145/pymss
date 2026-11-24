"""8 s inference latency: baseline vs the two math-preserving NPU optimizations.

Configurations, all numerically equivalent to the eager reference path:
  A  `grouped`     - unmodified model (BandSplit/MaskEstimator grouped bmm path)
  B  `per_band`    - grouped forward disabled: 62 per-band Linear calls instead of
                     one grouped bmm (same algebra, different reduction order)
  C  `per_band+rope` - B plus the triton-ascend RoPE kernel replacing
                     `rotate_qk_fast_bnhd`

Each configuration is timed in 3 independent repetitions; the reported value is
the median of the three medians (run-to-run spread on this box is ~5-8%).
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

from freq_unet import FreqUNetConfig, build_model  # noqa: E402
from npu_kernels_bandunet import fused_rope, triton_status  # noqa: E402

G4_REF_MS = 11.691
REPEATS = 3


def bench_once(model, spec, precision: str, iters: int) -> float:
    def ac():
        return (
            torch.npu.amp.autocast(dtype=torch.bfloat16)
            if precision == "bf16"
            else torch.autocast("cpu", enabled=False)
        )

    with torch.no_grad():
        for _ in range(10):
            with ac():
                model(spec)
    torch.npu.synchronize()
    times = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), ac():
            model(spec)
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--length", type=int, default=352800)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("/workspace/.tmp/infer_bench_fused.json"))
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401
    import pymss_core.modules.bs_roformer.transformer as core_transformer

    cfg = FreqUNetConfig(n_fft=args.n_fft)
    frames = args.length // args.hop + 1
    f = args.n_fft // 2 + 1
    spec = torch.randn(1, f * cfg.audio_channels, frames, 2, device="npu:0") * 0.1
    status = triton_status()

    payload = {"params": None, "frames": frames, "g4_ref_ms": G4_REF_MS, "triton": status,
               "repeats": REPEATS, "note": "median of 3 independent medians"}
    for precision in ("32", "bf16"):
        if precision == "32":
            torch.npu.conv.allow_hf32 = False
        model = build_model(cfg).to("npu:0").eval()
        payload["params"] = sum(p.numel() for p in model.parameters())

        a = [bench_once(model, spec, precision, args.iters) for _ in range(REPEATS)]
        model.band_split.use_grouped_forward = False
        model._estimate_masks.use_grouped_forward = False
        b = [bench_once(model, spec, precision, args.iters) for _ in range(REPEATS)]

        original = core_transformer.rotate_qk_fast_bnhd

        def patched(rotary_embed, q, k):
            cos, sin = core_transformer.cached_rotary_cos_sin(rotary_embed, q.shape[1], q.device, q.dtype)
            half = q.shape[-1] // 2
            cos_v, sin_v = cos.reshape(-1)[:half], sin.reshape(-1)[:half]
            q2 = fused_rope(q.reshape(-1, q.shape[-1]).contiguous(), cos_v, sin_v).reshape(q.shape)
            k2 = fused_rope(k.reshape(-1, k.shape[-1]).contiguous(), cos_v, sin_v).reshape(k.shape)
            return q2, k2

        core_transformer.rotate_qk_fast_bnhd = patched
        try:
            c = [bench_once(model, spec, precision, args.iters) for _ in range(REPEATS)]
        finally:
            core_transformer.rotate_qk_fast_bnhd = original

        entry = {
            "grouped_ms": round(statistics.median(a), 3),
            "per_band_ms": round(statistics.median(b), 3),
            "per_band_rope_ms": round(statistics.median(c), 3),
            "grouped_runs": [round(x, 3) for x in a],
            "per_band_runs": [round(x, 3) for x in b],
            "per_band_rope_runs": [round(x, 3) for x in c],
        }
        entry["per_band_speedup"] = round(entry["grouped_ms"] / entry["per_band_ms"], 3)
        entry["total_speedup_vs_baseline"] = round(entry["grouped_ms"] / entry["per_band_rope_ms"], 3)
        entry["vs_g4"] = round(entry["per_band_rope_ms"] / G4_REF_MS, 2)
        payload[precision] = entry
        print(json.dumps({precision: entry}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
