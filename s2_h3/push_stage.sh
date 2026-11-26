#!/bin/bash
# §8 回传 for one finished stage: artifacts + STATUS first line, commit, push (rebases onto upstream first).
# Usage: ./push_stage.sh <step>   (run only after that stage's STAGE_OK)
set -e
cd /workspace/s2_h3
export GIT_AUTHOR_NAME=pi GIT_AUTHOR_EMAIL=pi@local GIT_COMMITTER_NAME=pi GIT_COMMITTER_EMAIL=pi@local
S=$1
python - "$S" <<'PY'
import json, glob, sys
from pathlib import Path
step = sys.argv[1]
d = json.load(open(f's2_h3/out/H3/eval-{step}.json'))
t = json.load(open(f's2_h3/runs/H3/time-{step}.json'))
done = sorted(int(p.split('eval-')[1].split('.')[0]) for p in glob.glob('s2_h3/out/H3/eval-*.json'))
dbs = {s: round(json.load(open(f's2_h3/out/H3/eval-{s}.json'))['median_dB'], 2) for s in done}
import subprocess
now = subprocess.check_output(['date', '+%Y-%m-%d %H:%M']).decode().strip()
Path('s2_h3/STATUS.md').write_text(
    f"STATUS: 50k run {len(done)}/10 stages OK ({', '.join(f'{s//1000}k {v} dB' for s, v in dbs.items())}) | "
    f"{t['steps_per_min']:.1f} steps/min ({60000/t['steps_per_min']:.0f} ms/step), S2_TRITON=dwconv_dw_bmm | "
    f"200-step control 0.0076% | updated {now}\n")
print(Path('s2_h3/STATUS.md').read_text().strip())
PY
git add s2_h3/out/H3/*.json s2_h3/out/H3/train-*.log s2_h3/*.md s2_h3/*.py s2_h3/*.json s2_h3/profile/ 2>/dev/null || true
git add -f s2_h3/runs/H3/time-*.json s2_h3/runs/H3/train-*.log s2_h3/runs/H3/bs_roformer/loss_curve.csv \
            s2_h3/runs/H3/bs_roformer/run_metadata.json s2_h3/H3-stage-*.log s2_h3/run_h2.log 2>/dev/null || true
git commit -q -m "H3@npu stage $S: median_dB $(python -c "import json;print(round(json.load(open('s2_h3/out/H3/eval-$S.json'))['median_dB'],3))"), $(python -c "import json;print(round(json.load(open('s2_h3/runs/H3/time-$S.json'))['steps_per_min'],1))") steps/min - artifacts per TASK section 8" || echo "nothing to commit"
git fetch -q origin exp/h3-ascend && git rebase -q origin/exp/h3-ascend 2>/dev/null || true
git push -q origin exp/h3-ascend && git log --oneline -1
