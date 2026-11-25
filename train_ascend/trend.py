"""Read the overnight run's eval-*.json files and decide what phase 4 should do next.

Thresholds (from the task brief):
  * median improves by >= 0.02 dB at two consecutive evaluation points -> keep training
  * two consecutive points with < 0.02 dB improvement -> plateau -> tune one variable
  * train loss falls while validation does not -> capacity bound, record evidence only

    python trend.py --run /workspace/runs/overnight
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("/workspace/runs/overnight"))
    parser.add_argument("--plateau-db", type=float, default=0.02)
    args = parser.parse_args(argv)

    evals = []
    for p in sorted(args.run.glob("eval-*.json"), key=lambda q: int(q.stem.split("-")[1])):
        evals.append(json.loads(p.read_text()))
    if not evals:
        print("no evals yet")
        return 0

    print(f"{'step':>7} {'raw med':>8} {'raw mean':>9} {'ema med':>8} {'ema mean':>9} "
          f"{'raw ci95':>18} {'ema ci95':>18} {'leak raw':>9} {'leak ema':>9}")
    for e in evals:
        r, m = e["raw"], e["ema"]
        ci_r = f"[{r['ci95_by_song'][0]:.2f},{r['ci95_by_song'][1]:.2f}]"
        ci_m = f"[{m['ci95_by_song'][0]:.2f},{m['ci95_by_song'][1]:.2f}]"
        print(f"{e['step']:>7} {fmt(r['median_si_sdr']):>8} {fmt(r['mean_si_sdr']):>9} "
              f"{fmt(m['median_si_sdr']):>8} {fmt(m['mean_si_sdr']):>9} {ci_r:>18} {ci_m:>18} "
              f"{fmt(r.get('leakage_rms_median'), 4):>9} {fmt(m.get('leakage_rms_median'), 4):>9}")

    # train loss trend from train.jsonl (loss_mrstft is the noisiest term; use `loss`)
    losses = []
    tj = args.run / "train.jsonl"
    if tj.exists():
        for line in tj.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "loss" in rec:
                losses.append((rec["step"], rec["loss"]))
    if losses:
        tail = losses[-200:]
        first = sum(v for _, v in tail[:50]) / max(1, len(tail[:50]))
        last = sum(v for _, v in tail[-50:]) / max(1, len(tail[-50:]))
        print(f"\ntrain loss: {first:.4f} -> {last:.4f} over steps {tail[0][0]}..{tail[-1][0]}")

    if len(evals) >= 3:
        med = [e["raw"]["median_si_sdr"] for e in evals]
        d1, d2 = med[-2] - med[-3], med[-1] - med[-2]
        verdict = ("plateau: two consecutive points below the 0.02 dB gate -> tune one variable"
                   if d1 < args.plateau_db and d2 < args.plateau_db
                   else "still rising: continue training")
        print(f"\ndeltas {d1:+.3f} / {d2:+.3f} dB -> {verdict}")
        if losses and last <= first and d2 < args.plateau_db:
            print("train loss is falling while validation is flat -> capacity bound (record only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
