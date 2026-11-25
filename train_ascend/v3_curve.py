"""Assemble results/v3_curve.json : v1 vs v3 at every evaluation point, both protocols.

v1 per-1000-step numbers come from the existing runs/overnight2 history (no rerun); v3's come
from runs/v3_fromscratch/, which evaluates both the frozen own protocol (sha1 8a9e6137) and the
H2 56-window protocol at every 1000-step point (--eval-protocol both).

    python v3_curve.py --out results/v3_curve.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

COLS = ("median_si_sdr", "mean_si_sdr", "ci95_by_song", "leakage_rms_median")


def pick(stats: dict) -> dict:
    return {k: stats.get(k) for k in COLS}


def load_run(directory: Path, tag: str) -> dict:
    """{step: {"raw": {...}, "ema": {...}}} from eval*.json files in a run directory."""
    out = {}
    for path in sorted(glob.glob(str(directory / "eval*.json"))):
        name = Path(path).stem
        proto = "own"
        step_part = name.split("-")[-1]
        if name.startswith("eval_h2-"):
            proto = "h2"
        elif name.startswith("eval_own-"):
            proto = "own"
        if not step_part.isdigit():
            continue
        data = json.loads(Path(path).read_text())
        entry = out.setdefault(int(step_part), {}).setdefault(proto, {})
        entry["raw"] = pick(data["raw"])
        entry["ema"] = pick(data["ema"])
        entry["eval_s"] = data.get("eval_s")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-run", type=Path, default=Path("/workspace/runs/overnight2"))
    parser.add_argument("--v3-run", type=Path, default=Path("/workspace/runs/v3_fromscratch"))
    parser.add_argument("--out", type=Path, default=Path("results/v3_curve.json"))
    args = parser.parse_args(argv)

    v1 = load_run(args.v1_run, "v1")
    v3 = load_run(args.v3_run, "v3")
    steps = sorted(set(v1) | set(v3))

    rows = []
    for step in steps:
        row = {"step": step}
        for tag, src in (("v1", v1), ("v3", v3)):
            for proto in ("own", "h2"):
                node = src.get(step, {}).get(proto)
                if node:
                    row[f"{tag}_{proto}"] = node
        rows.append(row)

    h2_ref = {"5000": {"median": 5.066, "mean": 3.727}, "10000": {"median": 6.818, "mean": 5.520},
              "15000": {"median": 7.742, "mean": 6.412}, "20000": {"median": 7.972, "mean": 7.038},
              "30000": {"median": 8.742, "mean": 7.805}, "40000": {"median": 9.046, "mean": 8.224},
              "50000": {"median": 9.406, "mean": 8.719}}
    payload = {"protocols": {"own": "frozen 56 windows / 50 songs, sha1 8a9e6137561b (legacy, NOT paired with H2)",
                             "h2": "H2 archived 56 windows / 45 songs (paired)"},
               "v1_run": str(args.v1_run), "v3_run": str(args.v3_run),
               "rows": rows, "h2_reference_curve": h2_ref}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    def cell(node, key):
        if not node:
            return "     -"
        v = node[key]["median_si_sdr"]
        return f"{v:6.3f}" if v is not None else "     -"

    print(f"{'step':>6} {'v1own raw':>9} {'v1own ema':>9} {'v3own raw':>9} {'v3own ema':>9} "
          f"{'v1h2 ema':>9} {'v3h2 ema':>9}")
    for row in rows:
        print(f"{row['step']:>6} "
              f"{cell(row.get('v1_own'), 'raw'):>9} {cell(row.get('v1_own'), 'ema'):>9} "
              f"{cell(row.get('v3_own'), 'raw'):>9} {cell(row.get('v3_own'), 'ema'):>9} "
              f"{cell(row.get('v1_h2'), 'ema'):>9} {cell(row.get('v3_h2'), 'ema'):>9}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
