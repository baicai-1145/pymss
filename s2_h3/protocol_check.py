"""Is this port's data the Colab protocol's data?

The 56 evaluation windows are derived from test.json exactly the way stage.score() derives them.  Two references:

* `protocol_ref.json` (public): every window's offset, the CRC32 of its float16-cast PCM (both stems) and its energies.
  Same offsets means the track lengths are the same; same CRC means bit-identical samples.
* `S2_REF_DIR=<clone of the private s2-h3-ref>/ref` (optional): the 56 windows themselves, float16-rounded exactly like
  stage.score() feeds the model.  A window that is not bit-identical is then measured instead of merely rejected: every
  sample may differ by at most one PCM16 LSB (plus the float16 rounding of the reference itself), at at most 2% of the
  samples.  Near zero this pins the difference to one LSB; at full scale float16 cannot resolve below ~16 LSB, which the
  rounding term expresses.

Prints PROTOCOL_PASS only when all 56 windows are accounted for.
"""
import json
import os
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf


LSB = np.float32(2.0 ** -15)   # one PCM16 step for audio in [-1, 1)


def windows(tracks):
    """The 64 protocol slots, exactly as stage.score() derives them: (track, offset, length)."""
    for i in range(64):
        epoch, slot = divmod(i, len(tracks)); name, n = tracks[slot]
        off = random.Random(0xC0FFEE + epoch * 100003 + slot).randint(0, max(0, n - 352800)); length = min(352800, n - off)
        yield name, off, length


def load(root, name, stem, off, length):
    return sf.read(root / 'data' / 'test' / name / (stem + '.wav'), start=off, frames=length, dtype='float32', always_2d=True)[0]


def crc(array):   # stage.score(): PCM16 -> float32 -> float16
    return zlib.crc32(array.T.astype('<f2').tobytes())


def main(root=None):
    root = Path(root or Path(__file__).resolve().parent)
    tracks = json.loads((root / 'test.json').read_text())
    train = json.loads((root / 'train.json').read_text())
    reference = json.loads((root / 'protocol_ref.json').read_text())['windows']
    ref = {(r['name'], r['off']): r for r in reference}
    print('tracks: train', len(train), 'test', len(tracks), '| reference windows', len(ref), flush=True)

    samples = None
    ref_dir = os.environ.get('S2_REF_DIR')
    if ref_dir:
        order = {(name, off): i for i, (name, off) in enumerate(json.loads((Path(ref_dir) / 'coords.json').read_text()))}
        samples = np.concatenate([np.load(Path(ref_dir) / f'windows_{p}.npy') for p in 'ab'])
        assert samples.shape == (56, 2, 2, 352800) and set(order) == set(ref), samples.shape
        print('per-sample reference:', ref_dir, flush=True)

    exact, measured, failed = [], [], []
    for name, off, length in windows(tracks):
        r = ref.get((name, off))
        if r is None:
            continue
        mix, vox = load(root, name, 'mixture', off, length), load(root, name, 'vocals', off, length)
        if crc(mix) == r['crc_mixture'] and crc(vox) == r['crc_vocals']:
            exact.append((name, off)); continue
        got = np.stack([mix.T, vox.T])
        rel = {k: round(abs(float(np.square(g.astype(np.float64)).sum()) - r[k]) / r[k], 9)
               for k, g in (('energy_mixture', got[0]), ('energy_vocals', got[1]))}
        if samples is None:
            failed.append((name, off, float('inf'), float('nan'), rel, 'CRC differs and there is no per-sample reference (set S2_REF_DIR)')); continue
        want = samples[order[(name, off)]]
        if got.shape != want.shape:
            failed.append((name, off, float('inf'), float('nan'), {}, 'shape %s vs %s' % (got.shape, want.shape))); continue
        delta = np.abs(got.astype(np.float16).astype(np.float32) - want.astype(np.float16).astype(np.float32))
        ulp = np.spacing(want).astype(np.float32)           # float16 resolution at the reference value
        excess = delta - (LSB + 1.01 * ulp)                 # one PCM16 LSB plus the reference's own float16 rounding
        ratio, frac = float((excess / (LSB + ulp)).max()), float((delta != 0).mean())
        why = ('beyond one LSB' if ratio > 0.05 else '') + (' and ' if ratio > 0.05 and frac > 0.02 else '') + ('more than 2% of samples' if frac > 0.02 else '')
        (measured if not why else failed).append((name, off, ratio, frac, rel, why))

    reached = set(exact) | {(f[0], f[1]) for f in measured} | {(f[0], f[1]) for f in failed}
    missing = sorted(set(ref) - reached)
    print('bit-identical:', len(exact), '| within 1 PCM16 LSB:', len(measured), '| failed:', len(failed),
          '| never reached:', len(missing), flush=True)
    if measured:
        print('within-1-LSB windows: differing samples %.4f..%.4f%% (median %.4f%%)' % (
            100 * min(m[3] for m in measured), 100 * max(m[3] for m in measured), 100 * float(np.median([m[3] for m in measured]))), flush=True)
    for name, off, ratio, frac, rel, why in sorted(failed, key=lambda m: -m[2])[:12]:
        print('FAIL', name, off, '| %s | excess over 1 LSB = %.3f of (1 LSB + float16 ulp), %.4f%% of samples differ |' % (why, ratio, frac * 100), rel, flush=True)
    for item in missing[:8]:
        print('MISSING', item, flush=True)
    ok = not missing and not failed and len(exact) + len(measured) == len(ref) == 56
    print('PROTOCOL_PASS' if ok else 'PROTOCOL_FAIL', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
