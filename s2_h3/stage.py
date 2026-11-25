"""5k-step segmented training + 56-window eval + backup.  Mirrors the archived run_s_aw.py / stage.py."""
import json, math, os, random, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from entry import ROOT, CFG, DEVICE, ACCELERATOR, autocast, build, ARMS

ARM = os.environ.get('ARM', 'S2'); assert ARM in ARMS, ARM
OUT = ROOT / 'out' / ARM


def score(arm, step, path):
    m = build(arm); cp = torch.load(path, map_location='cpu', weights_only=False)
    assert cp['metadata']['global_step'] == step and cp['metadata']['ema_active']
    m.load_state_dict(cp['state_dict'], strict=True); m = m.to(DEVICE).eval()
    tracks = json.loads((ROOT / 'test.json').read_text()); rows = []
    with torch.inference_mode():
        for i in range(64):
            epoch, slot = divmod(i, len(tracks)); name, n = tracks[slot]
            off = random.Random(0xC0FFEE + epoch * 100003 + slot).randint(0, max(0, n - 352800)); length = min(352800, n - off)
            def load(stem):
                arr = sf.read(ROOT / 'data' / 'test' / name / (stem + '.wav'), start=off, frames=length, dtype='float32', always_2d=True)[0]
                return torch.from_numpy(arr.astype('float16').astype('float32').T.copy()).to(DEVICE)
            mix, vox = load('mixture'), load('vocals')
            if vox.square().mean().sqrt() < 3e-3:
                continue
            with autocast():
                out = m(mix[None])[0].float()
            r = vox - vox.mean(-1, keepdim=True); p = out - out.mean(-1, keepdim=True)
            s = (p * r).sum(-1, keepdim=True) * r / (r.square().sum(-1, keepdim=True) + 1e-8)
            v = (10 * torch.log10((s.square().sum(-1) + 1e-8) / ((p - s).square().sum(-1) + 1e-8))).mean().item()
            assert np.isfinite(v); rows.append([name, off, v])
    assert rows
    return dict(arm=arm, step=step, windows=len(rows), songs=len({r[0] for r in rows}), median_dB=float(np.median([r[2] for r in rows])), rows=rows)


def publish(files, step):
    status = {}
    try:
        subprocess.run(['rsync', '-az', '-e', 'ssh -p 23 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30', str(OUT) + '/', 'root@106.75.1.18:/root/bsdiag/s2_out/' + ARM + '/'], check=True, timeout=3600, capture_output=True)
        status['rsync'] = True; print('RSYNC_BACKUP_OK', ARM, step, flush=True)
    except Exception as exc:
        status['rsync'] = repr(exc)[:200]; print('RSYNC_BACKUP_FAILED', status['rsync'], flush=True)
    try:
        from huggingface_hub import get_token, HfApi
        tok = get_token()
        if tok:
            api = HfApi(token=tok)
            for f in files:
                status['commit'] = api.upload_file(repo_id='baicai1145/pymss-s2-adamw', repo_type='model', path_in_repo=f'{ARM}/{Path(f).name}', path_or_fileobj=str(f), commit_message=f'{ARM} step {step}').oid
            print('HF_BACKUP_OK', ARM, step, status['commit'], flush=True)
    except Exception as exc:
        print('HF_BACKUP_FAILED', repr(exc)[:120], flush=True)
    (OUT / f'hf-{step}.json').write_text(json.dumps(status))


def main(step):
    assert (ROOT / 'DATA_READY').is_file() and 'PREFLIGHT_PASS' in (ROOT / 'check.log').read_text()
    d = ROOT / 'runs' / ARM; d.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    ck, ema, ev = d / f'resume-{step}.ckpt', d / f'ema-{step}.pt', d / f'eval-{step}.json'
    mark = OUT / f'hf-{step}.json'
    if mark.is_file() and ck.is_file() and ev.is_file() and ema.is_file():
        print('STAGE_SKIP_DONE', ARM, step, flush=True); return
    last = d / 'bs_roformer' / 'checkpoints' / 'last.ckpt'
    if not (ck.is_file() and ema.is_file()):
        before = 0 if not last.exists() else int(torch.load(last, map_location='cpu', weights_only=False, mmap=True)['global_step'])
        assert max(0, step - 5000) <= before <= step, (ARM, step, before)
        if before < step:
            cmd = [sys.executable, '-u', 's2_aw_entry.py', 'train', '--model-type', 'bs_roformer', '--config-path', str(CFG), '--data-root', str(ROOT / 'data'), '--results-path', str(d),
                   '--seed', '0', '--batch-size', '24', '--num-workers', '12', *(['--pin-memory'] if DEVICE.type == 'cuda' else []), '--steps-per-epoch', '1000', '--max-steps', str(step), '--val-tracks', '0',
                   '--accelerator', ACCELERATOR, '--devices', '1', '--precision', 'bf16-mixed', '--fused-kernel-backend', os.environ.get('H2_FUSED_BACKEND', 'g4' if DEVICE.type == 'cuda' else 'off'), '--cuda-graph-mode', 'off', '--disable-progress-bar',
                   '--train-log-interval', '50', '--loss-history-interval', '50', '--log-every-n-steps', '50', '--export-core-checkpoint', str(ema)]
            if before:
                cmd += ['--resume-checkpoint', str(d / f'resume-{before}.ckpt')]
            t0 = time.time()
            with (d / f'train-{step}.log').open('a', buffering=1) as log:
                subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, ARM=ARM), stdout=log, stderr=subprocess.STDOUT, check=True)
            (d / f'time-{step}.json').write_text(json.dumps(dict(seconds=time.time() - t0, steps=step - before, steps_per_min=(step - before) / (time.time() - t0) * 60)))
        choices = []
        for path in (d / 'bs_roformer' / 'checkpoints').glob('last*.ckpt'):
            c = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            if c['global_step'] == step:
                assert int(c['state_dict']['ema_model.n_averaged']) == step and c['lr_schedulers'][0]['last_epoch'] == step
                g = c['optimizer_states'][0]['param_groups'][0]
                assert math.isclose(g['lr'], 5e-4 * .9 ** (step // 40000)) and g['weight_decay'] == .01
                choices.append(path)
            del c
        assert len(choices) == 1, choices
        tmp = ck.with_suffix('.partial'); shutil.copyfile(choices[0], tmp); os.replace(tmp, ck)
    if not ev.is_file():
        z = score(ARM, step, ema); z.update(optimizer='adamw', initial_lr=5e-4, weight_decay=0.01, lr_gamma=0.9, lr_step=40000)
        ev.write_text(json.dumps(z))
    files = []
    for f in (ema, ev, ck, d / f'time-{step}.json', d / f'train-{step}.log'):
        if f.is_file():
            dst = OUT / f.name
            if not dst.is_file() or f.suffix in ('.json', '.log'):
                shutil.copyfile(f, dst)
            files.append(dst)
    if not mark.is_file() and os.environ.get('LOCAL_ARTIFACTS_ONLY') != '1':
        publish(files, step)
    print('STAGE_OK', ARM, step, json.loads(ev.read_text())['median_dB'], flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
