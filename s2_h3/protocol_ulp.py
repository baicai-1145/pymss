"""Measure the sample-level difference of this port's decoded data against the Colab per-sample reference (TASK §3).

protocol_check.py rejects a window when a sample differs by more than 1 float16 ulp, where the ulp is `np.spacing(want)`.
For a reference sample of exactly 0 that ulp is 6e-8, i.e. 1/512 of one PCM16 LSB, so any one-LSB difference reads as
512 ulp.  This script reports the same comparison in absolute terms - the PCM16 LSB, which is the data's own quantum and
what the model actually sees - so the lead can judge the gate with both numbers.

    S2_REF_DIR=<clone of s2-h3-ref>/ref python protocol_ulp.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
REF = Path(os.environ['S2_REF_DIR'])
LSB = 1.0 / 32768          # one PCM16 step: the resolution of the wav files and of everything downstream


def windows(tracks):
    import random
    for i in range(64):
        epoch, slot = divmod(i, len(tracks))
        name, n = tracks[slot]
        off = random.Random(0xC0FFEE + epoch * 100003 + slot).randint(0, max(0, n - 352800))
        yield name, off, min(352800, n - off)


def main():
    tracks = json.loads((ROOT / 'test.json').read_text())
    ref = {(r['name'], r['off']) for r in json.loads((ROOT / 'protocol_ref.json').read_text())['windows']}
    coords = {tuple(c): i for i, c in enumerate(json.loads((REF / 'coords.json').read_text()))}
    ref_arr = np.concatenate([np.load(REF / f'windows_{x}.npy') for x in 'ab'])   # (56, 2, 2, 352800)
    rows = []
    for name, off, length in windows(tracks):
        if (name, off) not in ref:      # the 64 protocol slots only overlap the 56 reference windows
            continue
        order = coords[(name, off)]
        stacked = np.stack([np.asarray(ref_arr[order, s].T, dtype=np.float32) for s in (0, 1)])  # (2, n, 2)
        for s, stem in enumerate(('mixture', 'vocals')):
            arr = sf.read(ROOT / 'data' / 'test' / name / (stem + '.wav'), start=off, frames=length,
                          dtype='float32', always_2d=True)[0]
            want = stacked[s]
            got = arr.astype(np.float16).astype(np.float32)
            delta = np.abs(got - want)
            diff = delta != 0
            # absolute difference in units of the PCM16 step, and the same thing as a fraction of the f16 ulp
            lsb = delta / LSB
            ratio = delta / np.spacing(want).astype(np.float32)
            rows.append(dict(name=name, off=off, stem=stem, frac=float(diff.mean()), max_lsb=float(lsb.max()),
                             max_ratio=float(ratio.max()), differing=int(diff.sum()), samples=int(diff.size),
                             at_zero=int(((want == 0) & diff).sum()),
                             p999_lsb=float(np.quantile(lsb[diff], 0.999)) if diff.any() else 0.0))
    frac = np.array([r['frac'] for r in rows])
    lsb = np.array([r['max_lsb'] for r in rows])
    print(f'{len(rows)} window/stem pairs (56 windows x 2 stems)', flush=True)
    print(f'samples differing: {100 * frac.mean():.4f}% mean, {100 * frac.max():.4f}% max (gate allows 2%)', flush=True)
    print(f'max |difference| overall: {lsb.max():.3f} PCM16 LSB; per-pair 99.9th pct <= {max(r["p999_lsb"] for r in rows):.3f} LSB',
          flush=True)
    print(f'pairs whose max difference exceeds 1 PCM16 LSB: {int((lsb > 1.0).sum())} of {len(rows)}', flush=True)
    print(f'differing samples sitting on a reference value of exactly 0: {sum(r["at_zero"] for r in rows)} of '
          f'{sum(r["differing"] for r in rows)}', flush=True)
    print(f'pairs that fail protocol_check simply because a reference sample is 0: '
          f'{sum(1 for r in rows if r["max_ratio"] > 1.01 and r["max_lsb"] <= 1.0)}', flush=True)
    print(f'worst pairs by max |difference|:', flush=True)
    for r in sorted(rows, key=lambda r: -r['max_lsb'])[:5]:
        print(f'  {r["max_lsb"]:5.2f} LSB  {100 * r["frac"]:.4f}% differ  {r["name"][:34]:36s} {r["stem"]}', flush=True)
    print(f'worst pairs by f16-ulp ratio:', flush=True)
    for r in sorted(rows, key=lambda r: -r['max_ratio'])[:5]:
        print(f'  {r["max_ratio"]:7.1f} ulp = {r["max_lsb"]:.2f} LSB  {r["name"][:30]:32s} {r["stem"]}', flush=True)
    (ROOT / 'profile' / 'data_gate_ulp.json').write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
