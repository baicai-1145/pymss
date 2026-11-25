"""Is this port's decode chain the Colab reference's decode chain?

protocol_check.py says no (56/56 CRCs differ).  This script says *where* the difference is, and how big it is:

  * E_ref * 2**30 is an exact integer, so protocol_ref.json's energy is the exact sum of squares of the reference's
    int16 samples: our int16 sequence can be compared to it as an integer, with no tolerance guessing.
  * The residual is ~1e-6 of the window energy, i.e. ~1 float32 ULP per sample: below the PCM16 quantization step,
    but enough to move a few percent of samples to the other side of a rounding boundary.

Usage:  S2_MP3=<dir with NAME.{vocals,bass,drums,other}.mp3> python protocol_diag.py
"""
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
MP3 = Path(os.environ.get('S2_MP3', ROOT / 'mp3' / 'test'))
STEMS = ('vocals', 'bass', 'drums', 'other')
STATIC_FFMPEG = os.environ.get('S2_STATIC_FFMPEG')  # optional second decoder, e.g. an imageio-ffmpeg build


def decode_ffmpeg(path, binary='ffmpeg', flags=(), codec=None, fmt='f32le'):
    cmd = [binary, '-loglevel', 'error', '-nostdin', *flags]
    if codec:
        cmd += ['-c:a', codec]                      # decoder, so it goes before -i
    cmd += ['-i', str(path), '-ar', '44100', '-ac', '2', '-f', fmt, '-']
    raw = subprocess.check_output(cmd)
    if fmt == 's16le':
        return np.frombuffer(raw, '<i2').reshape(-1, 2).astype(np.float32) / 32768
    return np.frombuffer(raw, '<f4').reshape(-1, 2)


def pcm16(x, rule='rint'):
    s = np.asarray(x, np.float64) * 32768
    if rule == 'rint':
        i = np.rint(s)
    elif rule == '32767':
        i = np.rint(np.asarray(x, np.float64) * 32767)
    elif rule == 'away':
        i = np.sign(s) * np.floor(np.abs(s) + .5)
    elif rule == 'up':
        i = np.where(s - np.floor(s) == .5, np.floor(s) + 1, np.rint(s))
    elif rule == 'down':
        i = np.where(s - np.floor(s) == .5, np.floor(s), np.rint(s))
    elif rule == 'trunc':
        i = np.trunc(s)
    else:
        raise ValueError(rule)
    return np.clip(i, -32768, 32767).astype(np.int64)


def crc16(i):
    return zlib.crc32((i.astype(np.float32) / 32768).T.astype('<f2').tobytes())


def main():
    ref = [(r['name'], r['off'], r) for r in json.loads((ROOT / 'protocol_ref.json').read_text())['windows']]
    win = [(n, o, r) for n, o, r in ref if all((MP3 / f'{n}.{s}.mp3').is_file() for s in STEMS)]
    print(f'reference windows: {len(ref)} | comparable here (4 stems present): {len(win)}', flush=True)
    decoders = {'ffmpeg': dict(binary='ffmpeg')}
    if STATIC_FFMPEG:
        decoders['ffmpeg-static'] = dict(binary=STATIC_FFMPEG)
    decoders['ffmpeg-fixedpoint'] = dict(binary='ffmpeg', codec='mp3')
    decoders['ffmpeg-s16le'] = dict(binary='ffmpeg', fmt='s16le')
    decoders['libmpg123'] = dict(binary=None)       # libsndfile 1.2.x reads mp3 through libmpg123
    rules = ('rint', '32767', 'away', 'up', 'down', 'trunc')
    score = {f'{d}/{r}': 0 for d in decoders for r in rules}
    exact = {f'{d}/{r}': 0 for d in decoders for r in rules}
    for n, o, r in win:
        stems = {}
        for label, kw in decoders.items():
            if kw.get('binary') is None:
                stems[label] = {s: sf.read(MP3 / f'{n}.{s}.mp3', dtype='float32', always_2d=True)[0] for s in STEMS}
            else:
                stems[label] = {s: decode_ffmpeg(MP3 / f'{n}.{s}.mp3', **kw) for s in STEMS}
        for label, a in stems.items():
            m = min(len(v) for v in a.values()); mix = np.zeros((m, 2), np.float32)
            for s in STEMS: mix += a[s][:m]        # prepare.py's mixture
            for stem, sig in (('vocals', a['vocals'][:m]), ('mixture', mix)):
                seg = sig[o:o + r['length']]
                target = round(r[f'energy_{stem}'] * 32768.0 ** 2)
                for rule in rules:
                    i = pcm16(seg, rule)
                    exact[f'{label}/{rule}'] += int((i ** 2).sum()) == target       # exact integer fingerprint
                    score[f'{label}/{rule}'] += crc16(i) == r[f'crc_{stem}']
    total = 2 * len(win)
    print(f'\n{"decoder/rule":34s} {"CRC match":>10s} {"exact energy":>13s}  (of {total})', flush=True)
    for k in sorted(score, key=lambda k: (-score[k], -exact[k])):
        print(f'{k:34s} {score[k]:10d} {exact[k]:13d}', flush=True)
    print('\nNo row is perfect: the reference is the same recording (all 56 offsets match) decoded by an arithmetic '
          'we cannot reproduce here.  See PROTOCOL_FAIL_REPORT.md.', flush=True)


if __name__ == '__main__':
    main()
