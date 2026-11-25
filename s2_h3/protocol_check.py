"""Is this port's data the Colab protocol's data?

The 56 evaluation windows are derived from test.json exactly the way stage.score() derives them.  Two references:

* `protocol_ref.json` (public): every window's offset, the CRC32 of its float16-cast PCM (both stems) and its energies.
  Same offsets means the track lengths are the same; same CRC means bit-identical samples.
* `S2_REF_DIR=<clone of the private s2-h3-ref>/ref` (optional): the 56 windows themselves, float16-rounded exactly like
  stage.score() feeds the model.  A window that is not bit-identical is then measured instead of merely rejected.  It
  passes when no sample differs by more than 16 PCM16 LSB (one float16 step at full scale, i.e. below the resolution of
  the evaluation itself) and the energy of the difference is at most 1e-9 of the window's energy (-90 dB).

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
MAX_LSB = 16                    # one float16 step at full scale: the eval itself cannot resolve finer
MAX_ENERGY = 1e-9               # difference energy / reference energy per window (-90 dB)


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
        max_lsb, frac = float(delta.max() / LSB), float((delta != 0).mean())
        rel_e = float(np.square(delta.astype(np.float64)).sum() / np.square(want.astype(np.float64)).sum())
        why = ('a sample differs by more than 16 LSB' if max_lsb > MAX_LSB else '') + (' and ' if max_lsb > MAX_LSB and rel_e > MAX_ENERGY else '') + ('difference energy above 1e-9 of the reference' if rel_e > MAX_ENERGY else '')
        (measured if not why else failed).append((name, off, max_lsb, frac, rel_e, rel, why))

    reached = set(exact) | {(f[0], f[1]) for f in measured} | {(f[0], f[1]) for f in failed}
    missing = sorted(set(ref) - reached)
    print('bit-identical:', len(exact), '| within tolerance:', len(measured), '| failed:', len(failed),
          '| never reached:', len(missing), flush=True)
    if measured:
        print('within tolerance: max |diff| per window %.2f..%.2f LSB | difference energy / reference energy %.2e..%.2e | '
              'differing samples %.4f..%.4f%% (median %.4f%%)' % (
                  min(m[2] for m in measured), max(m[2] for m in measured), min(m[4] for m in measured), max(m[4] for m in measured),
                  100 * min(m[3] for m in measured), 100 * max(m[3] for m in measured), 100 * float(np.median([m[3] for m in measured]))), flush=True)
    for name, off, max_lsb, frac, rel_e, rel, why in sorted(failed, key=lambda m: -m[4])[:12]:
        print('FAIL', name, off, '| %s | max |diff| %.2f LSB, %.4f%% of samples differ, difference energy %.2e of reference |' % (why, max_lsb, frac * 100, rel_e), rel, flush=True)
    for item in missing[:8]:
        print('MISSING', item, flush=True)
    ok = not missing and not failed and len(exact) + len(measured) == len(ref) == 56
    print('rule: no sample differs by more than %d LSB (%.2e); difference energy <= %.0e of the reference energy' % (MAX_LSB, MAX_LSB * LSB, MAX_ENERGY), flush=True)
    print('PROTOCOL_PASS' if ok else 'PROTOCOL_FAIL', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
