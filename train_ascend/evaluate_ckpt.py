"""Evaluate a saved checkpoint (last.ckpt / best.pt / ema-<step>.pt) on the frozen protocol.

    python evaluate_ckpt.py --ckpt /workspace/runs/overnight/best.pt --out results/best_eval.json

Prints median/mean SI-SDR with a per-song bootstrap 95% CI and the quiet-window leakage,
and (with --latency) the 8 s inference latency of exactly that checkpoint.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import bootstrap_by_song, build_protocol, evaluate as evaluate_protocol  # noqa: E402
from freq_unet import FreqUNetConfig, build_model  # noqa: E402
from train import EMA, MusdbChunkDataset, scan_musdb  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/musdb18hq"))
    parser.add_argument("--chunk-size", type=int, default=352800)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401

    dev = torch.device("npu:0")
    cfg = FreqUNetConfig(n_fft=args.n_fft)
    model = build_model(cfg).to(dev)
    payload = torch.load(args.ckpt, map_location=dev, weights_only=False)
    step = int(payload.get("step", -1))
    if "model" in payload:                      # last.ckpt
        model.load_state_dict(payload["model"])
        weights = "raw"
    else:                                       # best.pt / ema-<step>.pt
        model.load_state_dict(payload["ema"]["shadow"], strict=False)
        weights = "ema"
    model.eval()

    dataset = MusdbChunkDataset(scan_musdb(args.data_root, splits=("train",)), args.chunk_size,
                                args.n_fft, args.hop)
    test_tracks = scan_musdb(args.data_root, splits=("test",))
    tracks_by_song = {t["mixture"].parent.name: t for t in test_tracks}
    protocol = build_protocol(test_tracks, args.chunk_size, 56)
    window = torch.hann_window(args.n_fft).to(device=dev, dtype=torch.float32)

    t0 = time.time()
    res = evaluate_protocol(model, dataset, tracks_by_song, protocol, dev, args.n_fft, args.hop,
                            args.chunk_size, window, args.precision)
    stats = bootstrap_by_song(res["windows"])
    leak = [r["leakage_rms"] for r in res["windows"] if not np.isnan(r["leakage_rms"])]
    stats["leakage_rms_median"] = float(np.median(leak)) if leak else None
    out = {"ckpt": str(args.ckpt), "step": step, "weights": weights, "eval_s": round(time.time() - t0, 1),
           **stats, "windows": res["windows"]}

    if args.latency:
        static = torch.randn(1, (args.n_fft // 2 + 1) * cfg.audio_channels, 690, 2, device=dev) * 0.1

        def run():
            with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.bfloat16):
                return model(static)

        for _ in range(5):
            run()
        torch.npu.synchronize()
        ts = []
        for _ in range(20):
            torch.npu.synchronize()
            t = time.perf_counter()
            run()
            torch.npu.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        out["infer_8s_ms"] = round(statistics.median(ts), 3)
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU]) as prof:
            run()
        out["device_kernels"] = sum(e.count for e in prof.key_averages() if e.self_device_time_total > 0)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "windows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
