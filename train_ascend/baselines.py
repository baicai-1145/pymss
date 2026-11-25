"""Reference baselines on the frozen 56-window protocol (B0 / B1).

B0: the mixture itself is the vocal estimate (no model at all).
B1: the model at initialisation (step 0), i.e. what an untrained net scores.
B2: the published G4 "H2 50k" number (9.406 dB) - reported here only as an external
    reference; our protocol (56 fixed windows, 4 relative positions per track, MUSDB18-HQ
    test) is not the one that number was produced with.

Window list + sha1 come from results/eval_windows.json so every number in the report is
tied to one immutable window set.

    python baselines.py --out results/baselines.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import (  # noqa: E402
    bootstrap_by_song,
    build_protocol,
    load_window,
    evaluate as evaluate_protocol,
)
from freq_unet import FreqUNetConfig, build_model  # noqa: E402
from train import MusdbChunkDataset, scan_musdb, si_sdr, split_fc  # noqa: E402

FRAME = 2048
QUIET_RMS = 0.005


def leakage(pred_wav: torch.Tensor, tgt_wav: torch.Tensor) -> tuple[float, float]:
    n = (pred_wav.shape[-1] // FRAME) * FRAME
    t = tgt_wav[..., :n].reshape(-1, n // FRAME, FRAME)
    p = pred_wav[..., :n].reshape(-1, n // FRAME, FRAME)
    quiet = t.pow(2).mean(-1).sqrt() < QUIET_RMS
    if not bool(quiet.any()):
        return float("nan"), 0.0
    return float(p.pow(2).mean(-1).sqrt()[quiet].mean()), float(quiet.float().mean())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/musdb18hq"))
    parser.add_argument("--chunk-size", type=int, default=352800)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("results/baselines.json"))
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401
    from pymss_core.npu_fft import istft_npu

    dev = torch.device("npu:0")
    cfg = FreqUNetConfig(n_fft=args.n_fft)
    dataset = MusdbChunkDataset(scan_musdb(args.data_root, splits=("train",)), args.chunk_size,
                                args.n_fft, args.hop)
    test_tracks = scan_musdb(args.data_root, splits=("test",))
    tracks_by_song = {t["mixture"].parent.name: t for t in test_tracks}
    protocol = build_protocol(test_tracks, args.chunk_size, 56)
    manifest = json.loads((Path(__file__).resolve().parent / "results" / "eval_windows.json").read_text())
    same = (json.dumps([[w["index"], w["song"], w["offset"]] for w in protocol]).encode())
    import hashlib

    sha1 = hashlib.sha1(same).hexdigest()
    if manifest["sha1"] != sha1:
        raise SystemExit(f"protocol mismatch: {manifest['sha1']} != {sha1}")
    window = torch.hann_window(args.n_fft).to(device=dev, dtype=torch.float32)

    # ---- B0: mixture as the vocal estimate
    b0_rows = []
    for entry in protocol:
        track = tracks_by_song[entry["song"]]
        target, mixture = load_window(track, entry["offset"], args.chunk_size, dataset)
        target, mixture = target.unsqueeze(0).to(dev), mixture.unsqueeze(0).to(dev)
        tgt_c, mix_c = torch.view_as_complex(target.contiguous()), torch.view_as_complex(mixture.contiguous())
        flat_m = split_fc(mix_c, dataset.channels)
        if False:  # pragma: no cover
            pass
        mix_wav = istft_npu(flat_m, n_fft=args.n_fft, hop_length=args.hop, window=window,
                            length=args.chunk_size, return_complex=False).reshape(1, dataset.channels, -1)
        tgt_wav = istft_npu(split_fc(tgt_c, dataset.channels), n_fft=args.n_fft, hop_length=args.hop,
                            window=window, length=args.chunk_size, return_complex=False).reshape(1, dataset.channels, -1)
        leak, qfrac = leakage(mix_wav, tgt_wav)
        b0_rows.append({"index": entry["index"], "song": entry["song"], "offset": entry["offset"],
                        "si_sdr": float(si_sdr(mix_wav, tgt_wav).mean()),
                        "leakage_rms": leak, "quiet_frac": qfrac})
    b0 = bootstrap_by_song(b0_rows)
    leak = [r["leakage_rms"] for r in b0_rows if not np.isnan(r["leakage_rms"])]
    b0["leakage_rms_median"] = float(np.median(leak)) if leak else None

    # ---- B1: untrained model (same init as training step 0)
    torch.manual_seed(0)
    model = build_model(cfg).to(dev).eval()
    b1 = evaluate_protocol(model, dataset, tracks_by_song, protocol, dev, args.n_fft, args.hop,
                           args.chunk_size, window, "bf16")
    b1_stats = bootstrap_by_song(b1["windows"])
    leak1 = [r["leakage_rms"] for r in b1["windows"] if not np.isnan(r["leakage_rms"])]
    b1_stats["leakage_rms_median"] = float(np.median(leak1)) if leak1 else None

    payload = {
        "protocol_sha1": sha1, "n_windows": len(protocol),
        "b0_mixture_as_vocals": b0,
        "b1_untrained_model": b1_stats,
        "b2_external": {
            "value_db": 9.406,
            "source": "published G4 'H2 50k' result",
            "comparable": False,
            "why": "different hardware, different architecture/recipe, and the evaluation "
                   "protocol of that number is not this 56-window list; treat as a reference "
                   "point only",
        },
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v for k, v in payload.items() if k != "b1_untrained_model"}, indent=2))
    print("B1", json.dumps(b1_stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
