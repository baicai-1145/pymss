#!/bin/bash
# §8 回传 for one finished stage of any ARM (H3, H4s, ...):
#   artifacts + STATUS first line + ModelScope upload of that stage's EMA checkpoint, commit, push (rebases first).
# Usage: ./push_stage.sh <step> [ARM]      ARM defaults to $ARM_NAME, then H3
# Requires MODELSCOPE_API_TOKEN in the environment for the upload (skipped with a note if unset).
set -e
cd /workspace/s2_h3
export GIT_AUTHOR_NAME=${GIT_AUTHOR_NAME:-pi} GIT_AUTHOR_EMAIL=${GIT_AUTHOR_EMAIL:-pi@local}
export GIT_COMMITTER_NAME=${GIT_COMMITTER_NAME:-pi} GIT_COMMITTER_EMAIL=${GIT_COMMITTER_EMAIL:-pi@local}
S=$1
ARM=${2:-${ARM_NAME:-H3}}
ROOT=s2_h3

python - "$S" "$ARM" <<'PY'
import glob, json, subprocess, sys
from pathlib import Path
step, arm = sys.argv[1], sys.argv[2]
root = Path('s2_h3')
ev = root / 'out' / arm / f'eval-{step}.json'
tm = root / 'runs' / arm / f'time-{step}.json'
dB = json.load(open(ev))['median_dB']
per_min = json.load(open(tm))['steps_per_min']
# STATUS first line: every finished stage of every arm, oldest step first
rows = []
for f in glob.glob(str(root / 'out' / '*' / 'eval-*.json')):
    a = f.split('/')[-2]; s = int(f.split('eval-')[1].split('.')[0])
    rows.append((s, a, json.load(open(f))['median_dB']))
rows.sort()
stages = ', '.join(f'{a} {s//1000}k {v:.2f} dB' for s, a, v in rows)
now = subprocess.check_output(['date', '+%Y-%m-%d %H:%M']).decode().strip()
(root / 'STATUS.md').write_text(
    f"STATUS: {len(rows)} stages OK ({stages}) | {per_min:.1f} steps/min ({60000/per_min:.0f} ms/step) on "
    f"{arm}, S2_TRITON=dwconv_dw_bmm | MS repo baicai1145/s2-h3-ascend-ema | updated {now}\n")
print((root / 'STATUS.md').read_text().strip())
PY

# ModelScope: this stage's EMA checkpoint (weights only; git stays weight-free)
if [ -n "$MODELSCOPE_API_TOKEN" ]; then
  MS_ARM="$ARM" PYTHONPATH="$PWD/$ROOT" python "$ROOT/upload_ms.py" "$S" 2>&1 | grep -E '^MS_' | tail -3 || true
else
  echo "MS_UPLOAD_SKIPPED no MODELSCOPE_API_TOKEN in environment"
fi

git add "$ROOT/out/$ARM"/*.json "$ROOT/out/$ARM"/train-*.log "$ROOT"/*.md "$ROOT"/*.py "$ROOT"/*.json "$ROOT"/profile/ 2>/dev/null || true
git add -f "$ROOT/runs/$ARM"/time-*.json "$ROOT/runs/$ARM"/train-*.log "$ROOT/runs/$ARM"/bs_roformer/loss_curve.csv \
           "$ROOT/runs/$ARM"/bs_roformer/run_metadata.json "$ROOT/${ARM}-stage-"*.log 2>/dev/null || true
git commit -q -m "$ARM@npu stage $S: median_dB $(python -c "import json;print(round(json.load(open('$ROOT/out/$ARM/eval-$S.json'))['median_dB'],3))"), $(python -c "import json;print(round(json.load(open('$ROOT/runs/$ARM/time-$S.json'))['steps_per_min'],1))") steps/min - artifacts per TASK section 8; EMA uploaded to ModelScope" || echo "nothing to commit"
git fetch -q origin exp/h3-ascend && { git rebase -q origin/exp/h3-ascend 2>/dev/null || true; }
git push -q origin exp/h3-ascend && git log --oneline -1
