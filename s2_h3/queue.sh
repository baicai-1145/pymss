#!/bin/bash
# Segmented S2 training queue.  Launched detached from the Colab kernel via subprocess.Popen(start_new_session=True).
set -uo pipefail
cd /content/s2
export PYTHONPATH=/content/s2 PYTHONUNBUFFERED=1 ARM="${ARM:-S2}"
log(){ echo "[$(date +%H:%M:%S)] $*"; }
log "START arm=$ARM host=$(hostname) cpus=$(nproc) mem=$(free -g | awk '/Mem:/{print $2}')G"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
python -m pip install -q -U pytorch-lightning soundfile > pip.log 2>&1 || { log PIP_FAILED; tail -5 pip.log; exit 1; }
python -c "import torch, pytorch_lightning as pl, soundfile; print('torch', torch.__version__, torch.cuda.is_available(), 'pl', pl.__version__, 'sf', soundfile.__version__)" || { log IMPORT_FAILED; exit 1; }
python prepare.py || { log PREPARE_FAILED; exit 1; }
log DATA_OK
python entry.py check > check.log 2>&1 || { log CHECK_FAILED; tail -20 check.log; exit 1; }
log "CHECK_OK $(grep -c '^arm' check.log) arms"
for step in 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000; do
  if python stage.py $step; then log "STAGE_DONE $step"; else log "STAGE_FAILED $step"; tail -30 runs/$ARM/train-$step.log; exit 1; fi
done
log ALL_DONE
