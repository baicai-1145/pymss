"""Is this port's data the Colab protocol's data?

The 56 evaluation windows are derived from test.json exactly the way stage.score() derives them.  protocol_ref.json holds,
for the Colab runs (2026-11-24/25), every window's offset and the CRC32 of its float16-cast PCM (both stems).  Same offsets
means the track lengths are the same; same CRCs means the decoded samples are bit-identical.  Prints PROTOCOL_PASS only
when all 56 windows match.
"""
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

root = Path(__file__).resolve().parent
tracks = json.loads((root / 'test.json').read_text())
reference = json.loads((root / 'protocol_ref.json').read_text())['windows']
ref = {(r['name'], r['off']): r for r in reference}
train = json.loads((root / 'train.json').read_text())
print('tracks: train', len(train), 'test', len(tracks), '| reference windows', len(ref), flush=True)


def load(name, stem, off, length):
    return sf.read(root / 'data' / 'test' / name / (stem + '.wav'), start=off, frames=length, dtype='float32', always_2d=True)[0]


def crc(array):   # stage.score(): PCM16 -> float32 -> float16
    return zlib.crc32(array.T.astype('<f2').tobytes())


matched, mismatches = [], []
for i in range(64):
    epoch, slot = divmod(i, len(tracks)); name, n = tracks[slot]
    off = random.Random(0xC0FFEE + epoch * 100003 + slot).randint(0, max(0, n - 352800)); length = min(352800, n - off)
    r = ref.get((name, off))
    if r is None:
        continue
    mix, vox = load(name, 'mixture', off, length), load(name, 'vocals', off, length)
    got = dict(length=length, crc_mixture=crc(mix), crc_vocals=crc(vox),
               energy_mixture=float(np.square(mix.astype(np.float64)).sum()), energy_vocals=float(np.square(vox.astype(np.float64)).sum()))
    if all(got[k] == r[k] for k in ('length', 'crc_mixture', 'crc_vocals')):
        matched.append((name, off)); continue
    rel = {k: round(abs(got[k] - r[k]) ** 0.5 / max(r[k], 1e-12) ** 0.5, 6) for k in ('energy_mixture', 'energy_vocals')}
    mismatches.append((name, off, {k: (got[k], r[k]) for k in got if got[k] != r[k]}, rel))

missing = sorted(set(ref) - set(matched) - {(n, o) for n, o, *_ in mismatches})
print('matched (bit-identical):', len(matched), '| mismatched:', len(mismatches), '| reference windows never reached:', len(missing), flush=True)
for name, off, diff, rel in mismatches[:12]:
    print('MISMATCH', name, off, 'fields', list(diff), 'rel. energy difference', rel, flush=True)
for item in missing[:8]:
    print('MISSING', item, flush=True)
ok = len(matched) == len(ref) == 56 and not mismatches and not missing
print('PROTOCOL_PASS' if ok else 'PROTOCOL_FAIL', flush=True)
sys.exit(0 if ok else 1)
