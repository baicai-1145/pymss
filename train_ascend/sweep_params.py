"""Parameter budget sweep: find the width config closest to the reference 20,961,844 behind a 21.63M cap."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freq_unet import FreqUNetConfig, build_model, param_count  # noqa: E402

CAP = 21_630_000
REF = 20_961_844


def main() -> int:
    rows = []
    for d_mid, mlp, n_temporal in itertools.product((192, 256, 320, 384, 448, 512), (384, 512, 768, 1024), (4,)):
        cfg = FreqUNetConfig(dims=(192, d_mid, d_mid, 128), mlp_dim=mlp, n_temporal_blocks=n_temporal)
        model = build_model(cfg)
        n = param_count(model)
        rows.append((abs(n - REF), n, d_mid, mlp, n_temporal))
    rows.sort()
    print(f"{'params':>12} {'delta':>10} {'(M)':>7}  dims=(192,d,d,128) mlp  blocks")
    for delta, n, d, mlp, nb in rows[:12]:
        print(f"{n:>12,d} {n - REF:>+10,d} {n / 1e6:>7.2f}  d={d:<4} mlp={mlp:<5} blocks={nb}"
              f"{'  OVER CAP' if n > CAP else ''}")
    print(f"\ncap={CAP:,} ref={REF:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
