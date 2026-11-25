#!/bin/bash
# Supervisor for the overnight run: restart train_long.py on any exit, at most MAX_RESTARTS.
# Usage: setsid nohup ./supervise.sh /workspace/runs/overnight 10000 [max_restarts] [arch] > ... &
#   arch: v1 (depthwise band blocks) or v3 (window attention); default v1.
DIR=${1:-/workspace/runs/overnight}
STEPS=${2:-10000}
MAX_RESTARTS=${3:-20}
ARCH=${4:-v1}
PROTO=${5:-both}
P=/usr/local/python3.12.16/bin/python
mkdir -p "$DIR"
n=0
while [ $n -lt $MAX_RESTARTS ]; do
  echo "[supervise] $(date -Is) launch #$n" | tee -a "$DIR/supervise.log"
  cd /workspace/pymss/train_ascend || exit 1
  $P train_long.py --steps "$STEPS" --save-dir "$DIR" --resume auto --arch "$ARCH" \
      --batch-size 24 --precision bf16 --fast-loss --num-workers 4 --eval-protocol "$PROTO" \
      >> "$DIR/stdout.log" 2>&1
  rc=$?
  echo "[supervise] $(date -Is) exit rc=$rc" | tee -a "$DIR/supervise.log"
  # normal completion: summary.json written with steps >= STEPS
  if [ -f "$DIR/summary.json" ]; then
    done_steps=$($P -c "import json;print(json.load(open('$DIR/summary.json'))['steps'])" 2>/dev/null)
    if [ "${done_steps:-0}" -ge "$STEPS" ]; then
      echo "[supervise] finished $done_steps steps" | tee -a "$DIR/supervise.log"
      exit 0
    fi
  fi
  n=$((n+1))
  sleep 20
done
echo "[supervise] gave up after $MAX_RESTARTS restarts" | tee -a "$DIR/supervise.log"
exit 1
