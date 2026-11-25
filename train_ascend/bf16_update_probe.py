"""Is a bf16 master weight safe for AdamW at lr 5e-4?

The failure mode is an update that rounds to zero: |dw| < ulp(w)/2 ~ |w| * 2**-9 = |w| * 0.00195.
With lr 5e-4 and AdamW's normalised step |dw| ~ lr, and |w| ~ 0.03 for Xavier fan_in 192, the ratio
is 1.7% of |w| = 4x the half-ulp, so updates should survive. This measures it on real weights.

    python bf16_update_probe.py --steps 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def one_run(w0: torch.Tensor, grads: list[torch.Tensor], *, bf16: bool, kahan: bool,
            lr: float = 5e-4, wd: float = 0.01):
    """Run AdamW on one tensor, bf16 storage with fp32 state (or fp32 reference)."""
    ref = w0.clone()
    w = w0.clone().to(torch.bfloat16) if bf16 else w0.clone()
    m = torch.zeros_like(w0)
    v = torch.zeros_like(w0)
    comp = torch.zeros_like(w0)          # Kahan compensation, fp32
    b1, b2, eps = 0.9, 0.999, 1e-8
    rounded_away = 0
    total = 0
    for step, g in enumerate(grads, start=1):
        m.mul_(b1).add_(g, alpha=1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)
        bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
        denom = (v / bc2).sqrt().add_(eps)
        delta = (-lr / bc1) * (m / denom) - lr * wd * w.float()
        if bf16:
            if kahan:
                comp = comp + delta
                new = w.float() + comp
                w_new = new.to(torch.bfloat16)
                applied = w_new.float() - w.float()
                comp = comp - applied
            else:
                w_new = (w.float() + delta).to(torch.bfloat16)
            nonzero = delta != 0
            rounded_away += int(((w_new.float() - w.float() == 0) & nonzero).sum())
            total += int(nonzero.sum())
            w = w_new
        else:
            w = w + delta.to(w.dtype)
    out = {"final": w.float(), "rounded_away": rounded_away, "total_updates": total}
    return out, ref


def main(argv=None) -> int:
    import torch_npu  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("results/bf16_update_probe.json"))
    args = parser.parse_args(argv)

    dev = torch.device("npu:0")
    torch.manual_seed(0)
    results = {}
    # real weight shapes from the model: 192x192 projection + 192x768 gate + a 62-band linear
    for name, shape, fan_in in (("proj_192x192", (192, 192), 192),
                                ("gate_192x768", (768, 192), 192),
                                ("band_132x192", (192, 132), 132)):
        bound = (6.0 / (fan_in + shape[0])) ** 0.5          # Xavier-ish init bound
        w0 = (torch.rand(shape, generator=torch.Generator().manual_seed(1)) * 2 - 1) * bound
        w0 = w0.to(dev)
        g = torch.Generator().manual_seed(2)
        # grad scale: training grad_norm ~0.5 over ~600 tensors -> per-tensor rms ~0.5
        grads = [(torch.randn(shape, generator=g) * 0.5).to(dev) for _ in range(args.steps)]
        ref, w_fp32 = one_run(w0, grads, bf16=False, kahan=False)
        entry = {"shape": list(shape), "init_abs_mean": float(w0.abs().mean())}
        for tag, bf16, kahan in (("bf16", True, False), ("bf16_kahan", True, True),
                                 ("fp32", False, False)):
            got, _ = one_run(w0, grads, bf16=bf16, kahan=kahan)
            d = got["final"] - ref["final"]
            entry[tag] = {
                "weight_rel_rms_vs_fp32_ref": float(d.pow(2).mean().sqrt()
                                                    / ref["final"].pow(2).mean().sqrt()),
                "max_abs_diff": float(d.abs().max()),
            }
            if bf16:
                entry[tag]["rounded_away_fraction"] = (got["rounded_away"]
                                                       / max(got["total_updates"], 1))
        results[name] = entry
        print(f"{name:14s} bf16 rel {entry['bf16']['weight_rel_rms_vs_fp32_ref']:.3e} "
              f"rounded {entry['bf16']['rounded_away_fraction']*100:.4f}% "
              f"| kahan rel {entry['bf16_kahan']['weight_rel_rms_vs_fp32_ref']:.3e}", flush=True)

    worst_rounded = max(v["bf16"]["rounded_away_fraction"] for v in results.values())
    worst_rel = max(v["bf16"]["weight_rel_rms_vs_fp32_ref"] for v in results.values())
    verdict = {
        "steps": args.steps,
        "worst_rounded_away_fraction": worst_rounded,
        "worst_weight_rel_rms_vs_fp32": worst_rel,
        "threshold_rounded": 0.01,
        "use_kahan": worst_rounded > 0.01,
        "recommendation": ("bf16 master weights are safe without compensation"
                           if worst_rounded <= 0.01 else
                           "rounding loss exceeds 1% - use the Kahan variant"),
        "shapes": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(verdict, indent=2))
    print(json.dumps({k: v for k, v in verdict.items() if k != "shapes"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
