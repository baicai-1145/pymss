"""Why is the mean negative: which half of the 56 windows breaks, and did v3 fix it?

For two checkpoints (v1 reference, v3) on the exact H2 56 windows:

  * per-window SI-SDR for both, paired delta,
  * split each model's windows at the median into a good half and a bad half,
  * characterise the bad half: song, vocal RMS, mixture RMS, vocal/mixture energy ratio,
    spectral centroid of the mixture,
  * count how many of v1's bad windows v3 improves by >1 dB and how many it loses by >1 dB.

    python v3_vs_v1_diag.py --v1 /workspace/runs/overnight2/ema-19000.pt \
                           --v3 /workspace/runs/v3/overnight/ema-10000.pt \
                           --out results/v3_vs_v1_diag.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_h2_protocol import (WINDOW_FILE, load_h2_windows, load_model,  # noqa: E402
                              resolve_tracks)
from train import MusdbChunkDataset, _read_wav, scan_musdb, si_sdr, split_fc  # noqa: E402


def features(mix: torch.Tensor, voc: torch.Tensor, sr: int = 44100) -> dict:
    """Cheap per-window descriptors of the mixture and of the vocal stem."""
    mono_mix = mix.mean(0)
    mono_voc = voc.mean(0)
    mix_rms = float(mono_mix.pow(2).mean().sqrt())
    voc_rms = float(mono_voc.pow(2).mean().sqrt())
    spec = torch.fft.rfft(mono_mix * torch.hann_window(mono_mix.numel()), dim=-1).abs()
    freqs = torch.fft.rfftfreq(mono_mix.numel(), 1.0 / sr)
    centroid = float((spec * freqs).sum() / spec.sum().clamp_min(1e-9))
    return {"mix_rms": mix_rms, "vocal_rms": voc_rms,
            "vocal_to_mix_db": 20.0 * float(np.log10(max(voc_rms, 1e-9) / max(mix_rms, 1e-9))),
            "centroid_hz": centroid,
            "vocal_peak": float(mono_voc.abs().max())}


def run_checkpoint(ckpt: Path, arch: str, windows, tracks, dataset, dev, window) -> list[dict]:
    from pymss_core.npu_fft import istft_npu

    model, step, kind, _ = load_model(ckpt, dev, 2048, arch)
    rows = []
    for i, w in enumerate(windows):
        track = tracks[w["song"]]
        mix = torch.from_numpy(_read_wav(track["mixture"], w["offset"], dataset.chunk_size, dataset.channels))
        voc = torch.from_numpy(_read_wav(track["vocals"], w["offset"], dataset.chunk_size, dataset.channels))
        mix_t = dataset._stft_fc(mix).unsqueeze(0).to(dev)
        voc_t = dataset._stft_fc(voc).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.npu.amp.autocast(dtype=torch.bfloat16):
            mask = model(mix_t)
        pred_c = torch.view_as_complex(mix_t.contiguous()) * torch.view_as_complex(mask.float().contiguous())

        def to_wav(x):
            return istft_npu(split_fc(x, dataset.channels), n_fft=2048, hop_length=512, window=window,
                             length=dataset.chunk_size, return_complex=False)

        sdr = float(si_sdr(to_wav(pred_c), to_wav(torch.view_as_complex(voc_t.contiguous()))).mean())
        row = {"index": i, "song": w["song"], "offset": w["offset"], "h2": w["h2_si_sdr"], "si_sdr": sdr}
        row.update(features(mix, voc))
        rows.append(row)
        print(f"  [{arch} {i:02d}] {w['song'][:36]:36s} {sdr:7.3f} dB", flush=True)
    del model
    torch.npu.empty_cache()
    return rows, step, kind


def half_stats(rows: list[dict], key: str = "si_sdr") -> dict:
    vals = np.array([r[key] for r in rows])
    med = float(np.median(vals))
    bad = [r for r, v in zip(rows, vals) if v <= med]
    good = [r for r, v in zip(rows, vals) if v > med]
    def agg(group):
        if not group:
            return {}
        return {"n": len(group), "median": float(np.median([g[key] for g in group])),
                "songs": sorted({g["song"] for g in group}),
                "mix_rms_median": float(np.median([g["mix_rms"] for g in group])),
                "vocal_rms_median": float(np.median([g["vocal_rms"] for g in group])),
                "vocal_to_mix_db_median": float(np.median([g["vocal_to_mix_db"] for g in group])),
                "centroid_hz_median": float(np.median([g["centroid_hz"] for g in group]))}
    return {"split_at": med, "good_half": agg(good), "bad_half": agg(bad)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v1-arch", default="v1")
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--v3-arch", default="v3")
    parser.add_argument("--out", type=Path, default=Path("results/v3_vs_v1_diag.json"))
    args = parser.parse_args(argv)

    import torch_npu  # noqa: F401

    dev = torch.device("npu:0")
    windows = load_h2_windows()
    test_tracks = scan_musdb(Path("/workspace/musdb18hq"), splits=("test",))
    tracks, _ = resolve_tracks(windows, test_tracks)
    dataset = MusdbChunkDataset(test_tracks[:1], 352800, 2048, 512)
    window = torch.hann_window(2048).to(dev, torch.float32)

    v1_rows, v1_step, _ = run_checkpoint(args.v1, args.v1_arch, windows, tracks, dataset, dev, window)
    v3_rows, v3_step, _ = run_checkpoint(args.v3, args.v3_arch, windows, tracks, dataset, dev, window)

    paired = []
    for a, b in zip(v1_rows, v3_rows):
        assert (a["song"], a["offset"]) == (b["song"], b["offset"])
        paired.append({"song": a["song"], "offset": a["offset"], "h2": a["h2"],
                       "v1": a["si_sdr"], "v3": b["si_sdr"], "delta": b["si_sdr"] - a["si_sdr"],
                       "mix_rms": a["mix_rms"], "vocal_rms": a["vocal_rms"],
                       "vocal_to_mix_db": a["vocal_to_mix_db"], "centroid_hz": a["centroid_hz"]})
    d = np.array([p["delta"] for p in paired])
    v1_med = float(np.median([p["v1"] for p in paired]))
    v1_bad = {p["song"] + str(p["offset"]) for p in paired if p["v1"] <= v1_med}
    fixed = [p for p in paired if p["song"] + str(p["offset"]) in v1_bad]
    out = {
        "protocol": str(WINDOW_FILE),
        "windows": len(paired),
        "songs": len({p["song"] for p in paired}),
        "v1": {"ckpt": str(args.v1), "step": v1_step,
               "median": float(np.median([p["v1"] for p in paired])),
               "mean": float(np.mean([p["v1"] for p in paired])), **half_stats(v1_rows)},
        "v3": {"ckpt": str(args.v3), "step": v3_step,
               "median": float(np.median([p["v3"] for p in paired])),
               "mean": float(np.mean([p["v3"] for p in paired])), **half_stats(v3_rows)},
        "paired": {"delta_median": float(np.median(d)), "delta_mean": float(d.mean()),
                   "improved_gt_1db": int((d > 1).sum()), "worse_gt_1db": int((d < -1).sum()),
                   "improved_any": int((d > 0).sum()), "worse_any": int((d < 0).sum())},
        "v1_bad_half": {"n": len(fixed),
                        "fixed_gt_1db": int(sum(1 for p in fixed if p["delta"] > 1)),
                        "worse_gt_1db": int(sum(1 for p in fixed if p["delta"] < -1)),
                        "delta_median": float(np.median([p["delta"] for p in fixed])),
                        "delta_mean": float(np.mean([p["delta"] for p in fixed]))},
        "rows": paired,
    }
    # which features correlate with a broken window (v1)?
    if len(paired) >= 8:
        v1_vals = np.array([p["v1"] for p in paired])
        corr = {}
        for f in ("mix_rms", "vocal_rms", "vocal_to_mix_db", "centroid_hz"):
            fv = np.array([p[f] for p in paired])
            corr[f] = float(np.corrcoef(fv, v1_vals)[0, 1])
        out["v1_feature_correlation"] = corr
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
