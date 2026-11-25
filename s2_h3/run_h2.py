"""Run one 5k H2 stage inside the Colab kernel's execution session."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(os.environ.get('S2_ROOT', '/content/s2'))
ARM = os.environ.get('H2_ARM', 'H2')
assert ARM in ('H2', 'H2M', 'H2MG', 'H2N128', 'BandUNet', 'H3')
ENV = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1', ARM=ARM,
           LOCAL_ARTIFACTS_ONLY='1', HF_HUB_DISABLE_PROGRESS_BARS='1')


def run(label, *args):
    print(label, flush=True)
    with (ROOT / f'{label}.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-u', *args], cwd=ROOT, env=ENV,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            try:
                status = process.wait(timeout=20)
                break
            except subprocess.TimeoutExpired:
                print(f'ACTIVE {label}', flush=True)
    if status:
        print((ROOT / f'{label}.log').read_text()[-4000:], flush=True)
        raise RuntimeError(f'{label} exited with {status}')


steps = tuple(map(int, os.environ.get('H2_STEPS', '5000').split(',')))
if steps[0] == 5000:
    if os.environ.get('S2_SKIP_PIP') != '1':   # a pinned environment (e.g. the Ascend box) installs its own
        run('dependencies', '-m', 'pip', 'install', '-q', 'pytorch-lightning', 'soundfile')
    run('check', 'entry.py', 'check')
    run('prepare', 'prepare.py')
else:
    assert (ROOT / 'out' / ARM / f'resume-{steps[0] - 5000}.ckpt').is_file()
for step in steps:
    label = f'{ARM}-stage-{step}'
    run(label, 'stage.py', str(step))
    print((ROOT / f'{label}.log').read_text().strip(), flush=True)
    print(f'H2_STAGE_READY {step}', flush=True)
