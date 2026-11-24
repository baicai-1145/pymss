"""Phase G evaluation protocol: 56 deterministic 8 s windows from the MUSDB test split.

Protocol (frozen; every later comparison must reuse the same window list):
  * windows are sorted test tracks x fixed offsets, 56 of them;
  * the (song, offset) list is written next to the results and must be reused
    verbatim (`--protocol-in`), never regenerated with other parameters;
  * metrics: per-window SI-SDR of the predicted vocals, median, paired
    bootstrap 95% CI resampled **by song**, and leakage on near-silent windows
    (vocal RMS < 0.005) kept out of the SI-SDR statistic.
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freq_unet import FreqUNetConfig, build_model  # noqa: E402
from train import MusdbChunkDataset, scan_musdb, si_sdr, split_fc  # noqa: E402

N_WINDOWS = 56


# Deterministic relative positions inside each track. Fractions (not absolute
# offsets) so the protocol survives any re-download with identical durations,
# and they avoid the instrumental intro that an "offset = 1 chunk" rule hits.
WINDOW_FRACTIONS = (0.25, 0.45, 0.65, 0.85)


def build_protocol(test_tracks: list[dict[str, Path]], chunk: int, n_windows: int = N_WINDOWS) -> list[dict]:
    protocol = []
    for i in range(n_windows):
        track = test_tracks[i % len(test_tracks)]
        with wave.open(str(track["mixture"]), "rb") as handle:
            total = handle.getnframes()
        frac = WINDOW_FRACTIONS[(i // len(test_tracks)) % len(WINDOW_FRACTIONS)]
        offset = int(max(0, min(total - chunk, total * frac)))
        protocol.append({"index": i, "song": track["mixture"].parent.name, "offset": int(offset), "fraction": frac})
    return protocol


def load_window(track: dict[str, Path], offset: int, chunk: int, dataset: MusdbChunkDataset) -> tuple[torch.Tensor, torch.Tensor]:
    from train import _read_wav

    mixture = torch.from_numpy(_read_wav(track["mixture"], offset, chunk, dataset.channels))
    target = torch.from_numpy(_read_wav(track["vocals"], offset, chunk, dataset.channels))
    return dataset._stft_fc(target), dataset._stft_fc(mixture)


@torch.no_grad()
def evaluate(model, dataset, tracks_by_song, protocol, device, n_fft, hop, chunk, window, precision="32") -> dict:
    model.eval()
    rows = []
    for entry in protocol:
        track = tracks_by_song[entry["song"]]
        target, mixture = load_window(track, entry["offset"], chunk, dataset)
        target = target.unsqueeze(0).to(device)
        mixture = mixture.unsqueeze(0).to(device)
        with torch.autocast(device_type="npu", dtype=torch.bfloat16) if precision == "bf16" else torch.autocast("cpu", enabled=False):
            mask = model(mixture)
        mix_c = torch.view_as_complex(mixture.contiguous())
        tgt_c = torch.view_as_complex(target.contiguous())
        pred_c = mix_c * torch.view_as_complex(mask.float().contiguous())
        from pymss_core.npu_fft import istft_npu

        def to_wav(x):
            flat = split_fc(x, dataset.channels)
            return istft_npu(flat, n_fft=n_fft, hop_length=hop, window=window, length=chunk, return_complex=False)

        pred_wav, tgt_wav = to_wav(pred_c), to_wav(tgt_c)
        sdr = float(si_sdr(pred_wav, tgt_wav).mean())
        # quiet-window leakage: fraction of predicted energy in windows where vocals are silent
        frame = 2048
        n = (chunk // frame) * frame
        t = tgt_wav[..., :n].reshape(-1, n // frame, frame)
        p = pred_wav[..., :n].reshape(-1, n // frame, frame)
        tgt_rms = t.pow(2).mean(-1).sqrt()
        prd_rms = p.pow(2).mean(-1).sqrt()
        quiet = tgt_rms < 0.005
        leakage = float(prd_rms[quiet].mean()) if bool(quiet.any()) else float("nan")
        rows.append({"index": entry["index"], "song": entry["song"], "offset": entry["offset"],
                     "si_sdr": sdr, "leakage_rms": leakage, "quiet_frac": float(quiet.float().mean())})
    return {"windows": rows}


def bootstrap_by_song(rows: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    songs = sorted({r["song"] for r in rows})
    by_song = {s: np.array([r["si_sdr"] for r in rows if r["song"] == s]) for s in songs}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(songs, size=len(songs), replace=True)
        means.append(np.mean(np.concatenate([by_song[s] for s in pick])))
    means = np.array(means)
    return {
        "n_songs": len(songs),
        "median_si_sdr": float(np.median([r["si_sdr"] for r in rows])),
        "mean_si_sdr": float(np.mean([r["si_sdr"] for r in rows])),
        "ci95_by_song": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/musdb18hq"))
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--precision", default="bf16", choices=("32", "bf16"))
    # EMA with decay 0.999 is still ~0.905 * init weights after 100 steps, so a short
    # smoke run must be scored with the raw weights; long runs use ema.
    parser.add_argument("--weights", default="ema", choices=("ema", "raw"))
    parser.add_argument("--chunk-size", type=int, default=352800)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("/workspace/runs/eval_protocol.json"))
    parser.add_argument("--protocol-out", type=Path, default=Path("/workspace/runs/eval_protocol_windows.json"))
    args = parser.parse_args(argv)

    test_tracks = scan_musdb(args.data_root, splits=("test",))
    if not test_tracks:
        raise SystemExit(f"no test tracks under {args.data_root}")
    protocol = build_protocol(test_tracks, args.chunk_size)
    args.protocol_out.parent.mkdir(parents=True, exist_ok=True)
    args.protocol_out.write_text(json.dumps({"n_windows": len(protocol), "windows": protocol}, indent=2))

    device = torch.device("cpu")
    try:
        import torch_npu

        if torch.npu.is_available():
            device = torch.device("npu:0")
    except Exception:
        pass

    cfg = FreqUNetConfig(n_fft=args.n_fft)
    model = build_model(cfg).to(device)
    if args.ckpt:
        payload = torch.load(args.ckpt, map_location=device, weights_only=False)
        if args.weights == "ema" and "ema" in payload:
            from train import EMA

            ema = EMA(model)
            ema.load_state_dict(payload["ema"])
            ema.apply_to(model)
        else:
            model.load_state_dict(payload.get("model", payload))
    model.eval()

    tracks_by_song = {t["mixture"].parent.name: t for t in test_tracks}
    dataset = MusdbChunkDataset(test_tracks, args.chunk_size, args.n_fft, args.hop)
    window = torch.hann_window(args.n_fft).to(device=device, dtype=torch.float32)
    result = evaluate(model, dataset, tracks_by_song, protocol, device, args.n_fft, args.hop, args.chunk_size, window, args.precision)
    stats = bootstrap_by_song(result["windows"])
    active = [r for r in result["windows"] if r["quiet_frac"] < 0.5]
    stats["n_active_windows"] = len(active)
    stats["median_si_sdr_active"] = float(np.median([r["si_sdr"] for r in active])) if active else None
    stats["n_near_silent_windows"] = len(result["windows"]) - len(active)
    quiet = [r["leakage_rms"] for r in result["windows"] if r["leakage_rms"] == r["leakage_rms"]]
    payload = {
        "device": str(device),
        "precision": args.precision,
        "ckpt": str(args.ckpt) if args.ckpt else None,
        "weights": args.weights,
        "protocol": str(args.protocol_out),
        "stats": stats,
        "leakage_rms_mean": float(np.mean(quiet)) if quiet else None,
        "windows": result["windows"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"leakage_rms_mean={payload['leakage_rms_mean']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
