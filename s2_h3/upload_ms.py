"""Upload finished EMA checkpoints to ModelScope (TASK: token via MODELSCOPE_API_TOKEN, never committed).

Repo id is recorded in STATUS.md.  Uploads are resumable/pointless-if-unchanged: a file already present with the same
size is skipped, so re-running after a retry is cheap.  Weights only - never data (the git repo stays weight-free).
"""
import os
import sys
from pathlib import Path

from modelscope.hub.api import HubApi

ROOT = Path(__file__).resolve().parent
OWNER = os.environ.get('MS_OWNER', 'baicai1145')   # the token's namespace (HubApi().whoami())
NAME = os.environ.get('MS_NAME', 's2-h3-ascend-ema')
TOKEN = os.environ['MODELSCOPE_API_TOKEN']


def repo_id(model='H3'):  # one repo for the run family, one subdir per arm
    return f'{OWNER}/{NAME}'


def main(steps, arm='H3', force=False):
    api = HubApi()
    api.login(TOKEN)
    rid = repo_id(arm)
    try:
        api.create_model(rid, visibility=5, license='Apache License 2.0',
                         chinese_name=f'S2 {arm} EMA checkpoints (Ascend 950PR)',
                         description='EMA checkpoints of the segmented 50k run. Weights only.')
        print('MS_REPO_CREATED', rid, flush=True)
    except Exception as exc:  # already exists is the common case
        print('MS_REPO_EXISTS_OR_WARN', rid, repr(exc)[:160], flush=True)
    done = []
    for step in steps:
        src = ROOT / 'out' / arm / f'ema-{step}.pt'
        if not src.is_file():
            print('MS_SKIP_MISSING', src, flush=True)
            continue
        target = f'{arm}/ema-{step}.pt'
        try:
            existing = api.get_model_files(rid, recursive=True) or []
            sizes = {f['Path']: f.get('Size') for f in existing}
            if not force and sizes.get(target) == src.stat().st_size:
                print('MS_SKIP_PRESENT', target, src.stat().st_size, flush=True)
                done.append(target); continue
        except Exception:  # listing can fail before the first upload
            pass
        try:
            api.upload_file(repo_id=rid, path_or_fileobj=str(src), path_in_repo=target)
            print('MS_UPLOAD_OK', rid, target, src.stat().st_size, flush=True)
            done.append(target)
        except Exception as exc:
            print('MS_UPLOAD_FAIL', target, repr(exc)[:200], flush=True)
    print('MS_SUMMARY', rid, len(done), 'files', done, flush=True)
    return 0 if len(done) == len(steps) else 1


if __name__ == '__main__':
    steps = [int(s) for s in sys.argv[1:]] or [50000]
    sys.exit(main(steps, arm=os.environ.get('MS_ARM', 'H3'), force=bool(os.environ.get('MS_FORCE'))))
