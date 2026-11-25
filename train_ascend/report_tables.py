"""Generate the markdown tables for the OVERNIGHT_SUMMARY section from on-disk artifacts.

CPU only: reads runs/*/eval-*.json, results/*.json and prints markdown. Nothing here touches
the NPU, so it is safe to run while training owns the device.

    python report_tables.py --run /workspace/runs/overnight > /workspace/.tmp/tables.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_evals(run: Path) -> list[dict]:
    out = []
    for p in sorted(run.glob("eval-*.json"), key=lambda q: int(q.stem.split("-")[1])):
        out.append(json.loads(p.read_text()))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("/workspace/runs/overnight"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    evals = load_evals(args.run)
    print("### 训练曲线（冻结 56 窗协议）\n")
    print("| step | raw 中位 | raw 均值 | raw 95% CI | EMA 中位 | EMA 均值 | EMA 95% CI | 近静音泄漏(raw) | 近静音泄漏(EMA) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for e in evals:
        r, m = e["raw"], e["ema"]
        print(f"| {e['step']} | {r['median_si_sdr']:.3f} | {r['mean_si_sdr']:.3f} | "
              f"[{r['ci95_by_song'][0]:.2f}, {r['ci95_by_song'][1]:.2f}] | "
              f"{m['median_si_sdr']:.3f} | {m['mean_si_sdr']:.3f} | "
              f"[{m['ci95_by_song'][0]:.2f}, {m['ci95_by_song'][1]:.2f}] | "
              f"{r.get('leakage_rms_median') or float('nan'):.4f} | "
              f"{m.get('leakage_rms_median') or float('nan'):.4f} |")

    bl = args.results / "baselines.json"
    best = None
    bestp = args.run / "best.pt"
    if bestp.exists():
        import torch

        best = torch.load(bestp, map_location="cpu", weights_only=False).get("meta")
    print("\n### 与 baseline 对比（同一 56 窗）\n")
    print("| 系统 | 中位 SI-SDR | 均值 | 95% CI(按歌 bootstrap) | 说明 |")
    print("|---|---|---|---|---|")
    if bl.exists():
        d = json.loads(bl.read_text())
        b0, b1 = d["b0_mixture_as_vocals"], d["b1_untrained_model"]
        print(f"| B0 混合当人声 | {b0['median_si_sdr']:.3f} | {b0['mean_si_sdr']:.3f} | "
              f"[{b0['ci95_by_song'][0]:.2f}, {b0['ci95_by_song'][1]:.2f}] | 什么都不做 |")
        print(f"| B1 未训练模型 | {b1['median_si_sdr']:.3f} | {b1['mean_si_sdr']:.3f} | "
              f"[{b1['ci95_by_song'][0]:.2f}, {b1['ci95_by_song'][1]:.2f}] | 第 0 步权重 |")
        print(f"| B2 G4 H2-50k 公开值 | {d['b2_external']['value_db']:.3f} | — | — | "
              f"不同硬件/结构/协议，{('可直接对比' if d['b2_external']['comparable'] else '仅参考')} |")
    if evals:
        last = evals[-1]
        print(f"| 本模型 last (step {last['step']}) raw | {last['raw']['median_si_sdr']:.3f} | "
              f"{last['raw']['mean_si_sdr']:.3f} | "
              f"[{last['raw']['ci95_by_song'][0]:.2f}, {last['raw']['ci95_by_song'][1]:.2f}] | 最新权重 |")
        print(f"| 本模型 last (step {last['step']}) EMA | {last['ema']['median_si_sdr']:.3f} | "
              f"{last['ema']['mean_si_sdr']:.3f} | "
              f"[{last['ema']['ci95_by_song'][0]:.2f}, {last['ema']['ci95_by_song'][1]:.2f}] | EMA |")
    if best:
        e = next((x for x in evals if x["step"] == best["step"]), None)
        if e:
            print(f"| **本模型 best (step {best['step']}, {best['which']})** | "
                  f"{e[best['which']]['median_si_sdr']:.3f} | {e[best['which']]['mean_si_sdr']:.3f} | "
                  f"[{e[best['which']]['ci95_by_song'][0]:.2f}, "
                  f"{e[best['which']]['ci95_by_song'][1]:.2f}] | best.pt |")
    print(f"\nbest 检查点: {bestp}  meta={json.dumps(best)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
