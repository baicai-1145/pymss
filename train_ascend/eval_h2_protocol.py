"""Paired evaluation against the H2 archived 56-window protocol.

The earlier reports compared our numbers to H2's 9.4 dB as if the window sets were the same.
They are not: H2's archive is 56 windows over 45 test songs, ours was 56 windows over 50
songs, so 9.406 vs 2.964 was never a paired comparison. This module evaluates any checkpoint
on *exactly* H2's 56 windows (song|offset|h2_median_dB, from results/h2_56_windows.txt) and
reports per-window deltas against the archived H2 values.

Song names in the archive are sometimes truncated ("M", "Triviul feat"), so resolution is
exact-match first, then unique-prefix match, and every resolution is printed for review.

    python eval_h2_protocol.py --ckpt /workspace/runs/overnight2/best.pt --out results/h2_eval_best.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freq_unet import FreqUNetConfig, build_model  # noqa: E402
from train import MusdbChunkDataset, scan_musdb, si_sdr, split_fc  # noqa: E402

HERE = Path(__file__).resolve().parent
WINDOW_FILE = HERE / "results" / "h2_56_windows.txt"
FRAME = 2048
QUIET_RMS = 0.005


def load_h2_windows(path: Path = WINDOW_FILE) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        song, offset, db = line.rsplit("|", 2)
        rows.append({"song": song, "offset": int(offset), "h2_si_sdr": float(db)})
    if len(rows) != 56:
        raise SystemExit(f"expected 56 windows, got {len(rows)}")
    return rows


# The archive truncates two names; "M" is genuinely ambiguous by prefix (Moosmusic,
# Motor Tapes, Mu all match) but the list is alphabetical, so "M" can only be the entry
# that sorts before "Moosmusic" -- and that track does appear separately, as do the others.
ALIASES = {"M": "M.E.R.C. Music - Knockout"}


def resolve_tracks(windows: list[dict], test_tracks: list[dict]) -> tuple[dict, list[str]]:
    """song name -> track dict, with exact-then-unique-prefix matching."""
    by_name = {t["mixture"].parent.name: t for t in test_tracks}
    resolved, notes = {}, []
    for w in windows:
        name = w["song"]
        if name in ALIASES:
            resolved[name] = by_name[ALIASES[name]]
            notes.append(f"{name!r} -> {ALIASES[name]!r} (alias)")
            continue
        if name in by_name:
            resolved[name] = by_name[name]
            continue
        cands = [n for n in by_name if n.startswith(name)]
        if len(cands) != 1:
            raise SystemExit(f"cannot resolve '{name}': {len(cands)} candidates {cands}")
        resolved[name] = by_name[cands[0]]
        notes.append(f"{name!r} -> {cands[0]!r}")
    return resolved, notes


def leakage(pred_wav: torch.Tensor, tgt_wav: torch.Tensor) -> float:
    n = (pred_wav.shape[-1] // FRAME) * FRAME
    t = tgt_wav[..., :n].reshape(-1, n // FRAME, FRAME)
    p = pred_wav[..., :n].reshape(-1, n // FRAME, FRAME)
    quiet = t.pow(2).mean(-1).sqrt() < QUIET_RMS
    if not bool(quiet.any()):
        return float("nan")
    return float(p.pow(2).mean(-1).sqrt()[quiet].mean())


def bootstrap_by_song(values: dict[str, list[float]], n_boot: int = 4000, seed: int = 0) -> dict:
    songs = sorted(values)
    rng = np.random.default_rng(seed)
    meds, means = [], []
    for _ in range(n_boot):
        pick = rng.choice(len(songs), size=len(songs), replace=True)
        flat = np.concatenate([np.atleast_1d(values[songs[i]]) for i in pick])
        meds.append(np.median(flat))
        means.append(np.mean(flat))
    return {"ci95_median": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))],
            "ci95_mean": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]}


def load_model(ckpt: Path, dev, n_fft: int):
    cfg = FreqUNetConfig(n_fft=n_fft)
    model = build_model(cfg).to(dev)
    payload = torch.load(ckpt, map_location=dev, weights_only=False)
    if "model" in payload:
        model.load_state_dict(payload["model"])
        kind = "raw"
    else:
        model.load_state_dict(payload["ema"]["shadow"], strict=False)
        kind = "ema"
    model.eval()
    return model, int(payload.get("step", -1)), kind, cfg


def h2_protocol(chunk: int, test_tracks: list[dict]) -> tuple[list[dict], dict, dict]:
    """The H2 56-window list in the shape evaluate.evaluate() expects.

    Returns (protocol, tracks_by_song, meta) where every protocol entry carries the resolved
    track name plus the archived H2 value, so callers can report paired deltas directly.
    """
    windows = load_h2_windows()
    tracks, notes = resolve_tracks(windows, test_tracks)
    by_name = {t["mixture"].parent.name: t for t in test_tracks}
    protocol = []
    for i, w in enumerate(windows):
        track = tracks[w["song"]]
        protocol.append({"index": i, "song": track["mixture"].parent.name,
                         "offset": w["offset"], "h2_si_sdr": w["h2_si_sdr"],
                         "archive_name": w["song"]})
    import hashlib

    blob = json.dumps([[w["song"], w["offset"], w["h2_si_sdr"]] for w in windows])
    meta = {"name": "h2_56", "windows": len(windows), "songs": len({p["song"] for p in protocol}),
            "sha1": hashlib.sha1(blob.encode()).hexdigest(), "chunk": chunk,
            "file": str(WINDOW_FILE), "notes": notes}
    return protocol, by_name, meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/musdb18hq"))
    parser.add_argument("--chunk-size", type=int, default=352800)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401
    from pymss_core.npu_fft import istft_npu

    dev = torch.device("npu:0")
    windows = load_h2_windows()
    test_tracks = scan_musdb(args.data_root, splits=("test",))
    tracks, notes = resolve_tracks(windows, test_tracks)
    dataset = MusdbChunkDataset(test_tracks[:1], args.chunk_size, args.n_fft, args.hop)
    window = torch.hann_window(args.n_fft).to(device=dev, dtype=torch.float32)

    model, step, kind, cfg = load_model(args.ckpt, dev, args.n_fft)
    from train import _read_wav

    rows = []
    for i, w in enumerate(windows):
        track = tracks[w["song"]]
        mix = torch.from_numpy(_read_wav(track["mixture"], w["offset"], args.chunk_size, dataset.channels))
        tgt = torch.from_numpy(_read_wav(track["vocals"], w["offset"], args.chunk_size, dataset.channels))
        mix_t = dataset._stft_fc(mix).unsqueeze(0).to(dev)
        tgt_t = dataset._stft_fc(tgt).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.bfloat16):
            mask = model(mix_t)
        mix_c = torch.view_as_complex(mix_t.contiguous())
        tgt_c = torch.view_as_complex(tgt_t.contiguous())
        pred_c = mix_c * torch.view_as_complex(mask.float().contiguous())

        def to_wav(x):
            return istft_npu(split_fc(x, dataset.channels), n_fft=args.n_fft, hop_length=args.hop,
                             window=window, length=args.chunk_size, return_complex=False)

        pred_wav, tgt_wav = to_wav(pred_c), to_wav(tgt_c)
        sdr = float(si_sdr(pred_wav, tgt_wav).mean())
        rows.append({"index": i, "song": w["song"], "offset": w["offset"],
                     "mine": sdr, "h2": w["h2_si_sdr"], "delta": sdr - w["h2_si_sdr"],
                     "leakage_rms": leakage(pred_wav, tgt_wav)})
        if not args.quiet:
            print(f"  [{i:02d}] {w['song'][:38]:38s} off {w['offset']:>9} mine {sdr:7.3f} "
                  f"h2 {w['h2_si_sdr']:7.3f} delta {sdr - w['h2_si_sdr']:+7.3f}", flush=True)

    mine = np.array([r["mine"] for r in rows])
    h2 = np.array([r["h2"] for r in rows])
    delta = mine - h2
    by_song_mine: dict[str, list[float]] = {}
    by_song_delta: dict[str, list[float]] = {}
    for r in rows:
        by_song_mine.setdefault(r["song"], []).append(r["mine"])
        by_song_delta.setdefault(r["song"], []).append(r["delta"])
    leak = [r["leakage_rms"] for r in rows if not np.isnan(r["leakage_rms"])]

    summary = {
        "ckpt": str(args.ckpt), "step": step, "weights": kind,
        "protocol": "H2 archived 56 windows (45 songs)",
        "protocol_file": str(WINDOW_FILE),
        "windows": len(rows), "songs": len({r["song"] for r in rows}),
        "name_resolution": notes,
        "mine": {"median": float(np.median(mine)), "mean": float(mine.mean()),
                 **bootstrap_by_song(by_song_mine)},
        "h2": {"median": float(np.median(h2)), "mean": float(h2.mean())},
        "paired_delta": {"median": float(np.median(delta)), "mean": float(delta.mean()),
                         **bootstrap_by_song(by_song_delta),
                         "win": int((delta > 0).sum()), "loss": int((delta < 0).sum()),
                         "ties": int((delta == 0).sum())},
        "leakage_rms_median": float(np.median(leak)) if leak else None,
        "rows": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
