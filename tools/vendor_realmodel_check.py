"""Real-model validation across every catalog model type on the SUPA device.

One representative checkpoint per model_type, downloaded from ModelScope by
catalog name, loaded through MSSeparator with device='biren' (SUPA shims
active), and run on a deterministic 10 s stereo mix.  Each model runs in its
own subprocess so a hard crash cannot kill the batch; results are appended as
one JSON line per model plus a REALMODEL summary.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

#: one well-known representative per catalog model_type (fallback: first entry)
PREFERRED = {
    'bs_roformer': ['model_bs_roformer_ep3179_sdr_12.9755.ckpt', 'BS-Roformer_LargeV1.ckpt'],
    'mel_band_roformer': ['denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',
                          'mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt'],
    'mdx23c': ['MDX23C-8KFFT-InstVoc_HQ.ckpt'],
    'htdemucs': ['HTDemucs4.th'],
    'scnet': ['model_scnet_ep_36_sdr_10.0891.ckpt'],
    'apollo': ['apollo_model.ckpt'],
    'bandit': ['model_bandit_plus_dnr_sdr_11.47.ckpt'],
    'bandit_v2': ['checkpoint-eng_state_dict.ckpt'],
    'vr': ['UVR-De-Echo-Normal.pth'],
    'legacy_demucs': ['0d19c1c6-0f06f20e.th'],
    'legacy_tasnet': ['tasnet.th'],
}

WORKER = r'''
import json, sys, time, traceback
import numpy as np

name, model_dir = sys.argv[1], sys.argv[2]
result = {"name": name, "ok": False}
try:
    from pymss_core.device import prepare_device_alias
    prepare_device_alias("biren")
    import torch
    import torch.nn.functional as F
    _linear = F.linear
    def debug_linear(input, weight, bias=None, *args, **kwargs):
        try:
            return _linear(input, weight, bias, *args, **kwargs)
        except RuntimeError as error:
            print(f"LINEAR_FAIL shape={tuple(input.shape)} dtype={input.dtype} "
                  f"w={tuple(weight.shape)} wdtype={weight.dtype} err={str(error)[:120]}", flush=True)
            raise
    F.linear = debug_linear
    from pymss import MSSeparator
    separator = MSSeparator.from_model_name(
        name, model_dir=model_dir, download=True, source="modelscope", device="biren")
    model = getattr(separator, "model", None)
    device = str(next(p.device for p in model.parameters())) if model is not None and hasattr(model, "parameters") else "n/a"
    result["device"] = device
    result["model_type"] = getattr(separator, "model_type", None) or getattr(getattr(separator, "config", None), "model_type", None)
    sr = 44100
    seconds = 10
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    mix = np.stack([0.4*np.sin(2*np.pi*220*t) + 0.2*np.sin(2*np.pi*880*t),
                    0.35*np.sin(2*np.pi*330*t) + 0.25*np.sin(2*np.pi*660*t)])
    rng = np.random.default_rng(0)
    mix += 0.02 * rng.standard_normal(mix.shape).astype(np.float32)
    start = time.perf_counter()
    stems = separator.separate(mix, pbar=False)
    result["separate_ms"] = round((time.perf_counter() - start) * 1000, 1)
    if isinstance(stems, dict):
        report = {}
        for stem, wave in stems.items():
            wave = np.asarray(wave)
            report[stem] = {"shape": list(wave.shape),
                            "finite": bool(np.isfinite(wave).all()),
                            "rms": round(float(np.sqrt(np.mean(np.square(wave.astype(np.float64))))), 6)}
        result["stems"] = report
        result["ok"] = bool(report) and all(v["finite"] for v in report.values())
    else:
        wave = np.asarray(stems)
        result["stems"] = {"output": {"shape": list(wave.shape), "finite": bool(np.isfinite(wave).all())}}
        result["ok"] = bool(np.isfinite(wave).all())
except SystemExit as exit_error:
    result["error"] = f"SystemExit: {exit_error}"
except Exception as error:
    result["error"] = "".join(traceback.format_exception_only(type(error), error))[:300]
    result["traceback_tail"] = traceback.format_exc().splitlines()[-3:]
print("REALMODEL_SINGLE " + json.dumps(result), flush=True)
'''


def resolve_candidates(catalog_path):
    data = json.loads(Path(catalog_path).read_text())
    models = data.get('models', data)
    entries = models.items() if isinstance(models, dict) else [(m.get('name'), m) for m in models]
    by_type = {}
    for name, meta in entries:
        by_type.setdefault(meta.get('model_type'), []).append(name)
    chosen = []
    for model_type, preferences in PREFERRED.items():
        names = by_type.get(model_type, [])
        pick = next((p for p in preferences if p in names), names[0] if names else None)
        if pick:
            chosen.append((model_type, pick))
        else:
            print(f'REALMODEL_MISSING_TYPE {model_type}', flush=True)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', default=None)
    parser.add_argument('--model-dir', default='/tmp/code/pymss_models')
    parser.add_argument('--output', default='.tmp/br-realmodels.json')
    parser.add_argument('--only', default=None, help='run a single model name')
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()

    catalog = args.catalog
    if catalog is None:
        root = Path(__file__).resolve().parent.parent
        catalog = root / 'pymss' / 'resources' / 'model_catalog.json'
    chosen = resolve_candidates(catalog)
    if args.only:
        chosen = [(t, n) for t, n in chosen if n == args.only or t == args.only]

    results = []
    for model_type, name in chosen:
        print(f'REALMODEL_BEGIN {model_type} {name}', flush=True)
        start = time.perf_counter()
        try:
            completed = subprocess.run([sys.executable, '-u', '-c', WORKER, name, args.model_dir],
                                       capture_output=True, text=True, timeout=args.timeout, cwd=str(Path(__file__).resolve().parent.parent))
            payload = None
            for line in completed.stdout.splitlines():
                if line.startswith('REALMODEL_SINGLE '):
                    payload = json.loads(line[len('REALMODEL_SINGLE '):])
            if payload is None:
                payload = {'name': name, 'ok': False,
                           'error': (completed.stderr or completed.stdout).splitlines()[-1][:300] if (completed.stderr or completed.stdout) else f'exit {completed.returncode} no output',
                           'returncode': completed.returncode}
            else:
                payload['returncode'] = completed.returncode
        except subprocess.TimeoutExpired:
            payload = {'name': name, 'ok': False, 'error': f'timeout after {args.timeout}s'}
        payload['model_type'] = model_type
        payload['wall_s'] = round(time.perf_counter() - start, 1)
        results.append(payload)
        print('REALMODEL ' + json.dumps(payload), flush=True)

    summary = {'total': len(results),
               'ok': sum(1 for r in results if r.get('ok')),
               'failed': [r['name'] for r in results if not r.get('ok')]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'summary': summary, 'results': results}, indent=2, ensure_ascii=False))
    print('REALMODEL_SUMMARY ' + json.dumps(summary), flush=True)
    print('REALMODEL_ARTIFACT ' + str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
