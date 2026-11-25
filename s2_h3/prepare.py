"""MUSDB18-HQ (Viol2000/musdb18hq MP3 stems) -> PCM16 WAV tree, identical to the archived prepare.py."""
import json, os, shutil, subprocess, sys, tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

root = Path(os.environ.get('S2_ROOT', '/content/s2')); stems = ('vocals', 'bass', 'drums', 'other')
for split in ('test', 'train'):
    dst_root = root / 'data' / split
    if (root / f'{split}.json').is_file():
        print('skip', split, flush=True); continue
    src = root / 'mp3' / split; src.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download('Viol2000/musdb18hq', split + '.tar', repo_type='dataset')
    with tarfile.open(archive) as tar:
        for m in tar:
            if m.isfile() and m.name.lower().endswith('.mp3'):
                with (src / Path(m.name).name).open('wb') as dst:
                    shutil.copyfileobj(tar.extractfile(m), dst)
    os.remove(archive)
    names = sorted({p.name.rsplit('.', 2)[0] for p in src.glob('*.mp3')})
    tracks = [n for n in names if all((src / (n + '.' + s + '.mp3')).exists() for s in stems)]

    def decode(n):
        dst = dst_root / n; dst.mkdir(parents=True, exist_ok=True)
        arrays = [np.frombuffer(subprocess.check_output(['ffmpeg', '-loglevel', 'error', '-nostdin', '-i', str(src / (n + '.' + s + '.mp3')), '-ar', '44100', '-ac', '2', '-f', 'f32le', '-']), dtype='<f4').reshape(-1, 2) for s in stems]
        length = min(map(len, arrays)); mix = np.zeros((length, 2), np.float32)
        for s, a in zip(stems, arrays):
            mix += a[:length]; sf.write(dst / (s + '.wav'), a[:length], 44100, subtype='PCM_16')
        sf.write(dst / 'mixture.wav', mix, 44100, subtype='PCM_16')
        return [n, int(length)]

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(decode, tracks))
    assert len(rows) >= (99 if split == 'train' else 50), (split, len(rows))
    (root / f'{split}.json').write_text(json.dumps(rows))
    shutil.rmtree(src)
    print('DECODE_OK', split, len(rows), 'tracks', round(sum(r[1] for r in rows) / 44100 / 3600, 2), 'hours', flush=True)
(root / 'DATA_READY').touch()
print('DATA_READY', flush=True)
