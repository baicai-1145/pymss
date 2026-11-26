"""R1b: is the batched (grouped) band forward correct AND faster in *training*?

Runs one fixed forward+backward (saving output + every gradient) and then times the real training step, so three
configs can be compared pairwise:

  python r1b_bands.py <tag>            # tag = base | grouped | grouped_deep  (env set by the caller)

Writes `.tmp/r1b_<tag>.npz` (outputs, per-parameter grad, step ms, op counts).  `r1b_compare.py` diffs the tags.
Note `should_use_grouped_forward` reads the env per call, but each tag still gets a fresh process so the grouped
parameter caches (`_get_group_params`, `_layer_plan`) cannot leak between configs.
"""
import json
import os
import statistics
import sys
import time

os.environ.setdefault('S2_DEVICE', 'npu')
os.environ.setdefault('ARM', 'H4s')
sys.path.insert(0, '/workspace/s2_h3/s2_h3')

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn  # noqa: E402

import pymsst.training.module as tm  # noqa: E402
from entry import DEVICE, autocast, build  # noqa: E402
from pymsst.training.losses import masked_loss  # noqa: E402
import triton_npu  # noqa: E402
from pymss_core.modules.bs_roformer import bands  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else 'base'
GROUPED = bands.experimental_train_grouped_bands()
DEEP = bands.experimental_deep_mask_grouping()
print(f'R1B tag={TAG} grouped_env={GROUPED} deep_env={DEEP}', flush=True)

torch.manual_seed(0)
model = build('H4s').to(DEVICE)
names = [x for x in os.environ.get('S2_TRITON', '').replace(' ', '').split(',') if x]
if names:
    print('R1B triton', triton_npu.install(model, names), flush=True)
model.train()

B, S = 24, 131584
gen = torch.Generator(device='cpu').manual_seed(1234)
mix = ((torch.rand(B, 2, S, generator=gen) * 2 - 1) * 0.15).to(DEVICE)
tgt = ((torch.rand(B, 2, S, generator=gen) * 2 - 1) * 0.08).to(DEVICE)

# --- one fixed step: output + gradients (no optimizer step, so weights stay comparable across tags) ---
model.zero_grad(set_to_none=True)
with autocast():
    y = model(mix)
    loss = masked_loss(y, tgt, q=0.95, coarse=True)
loss.backward()
store = {'y': y.detach().float().cpu().numpy().astype(np.float32),
         'loss': np.array([float(loss.detach())], dtype=np.float64)}
for n, p in model.named_parameters():
    if p.grad is not None:
        store['g:' + n] = p.grad.detach().float().cpu().numpy().astype(np.float32)
print(f'R1B grads_present={sum(1 for k in store if k.startswith("g:"))}', flush=True)
model.zero_grad(set_to_none=True)

# --- count CPU-side launches for one training step ---
from torch.profiler import ProfilerActivity, profile  # noqa: E402
with profile(activities=[ProfilerActivity.CPU]) as prof:
    with autocast():
        loss = masked_loss(model(mix), tgt, q=0.95, coarse=True)
    loss.backward()
launches = sum(e.count for e in prof.key_averages())
model.zero_grad(set_to_none=True)
print(f'R1B launches={launches}', flush=True)

# --- timed training step (optimizer + EMA, as in the real loop) ---
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=.01, **tm._optimizer_backend_kwargs('default'))
ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))


def step():
    opt.zero_grad(set_to_none=True)
    with autocast():
        loss = masked_loss(model(mix), tgt, q=0.95, coarse=True)
    loss.backward()
    opt.step()
    ema.update_parameters(model)


for _ in range(10):
    step()
torch.npu.synchronize()
times = []
for _ in range(20):
    t0 = time.perf_counter()
    step()
    torch.npu.synchronize()
    times.append(time.perf_counter() - t0)
ms = statistics.median(times) * 1000
hbm = torch.npu.max_memory_allocated() / 2 ** 30
print(f'R1B step_ms={ms:.1f} steps_per_s={1000/ms:.3f} hbm_gb={hbm:.1f}', flush=True)

store['meta'] = np.array([ms, launches, float(loss.detach()), hbm], dtype=np.float64)
np.savez(f'.tmp/r1b_{TAG}.npz', **store)
print(f'R1B saved .tmp/r1b_{TAG}.npz', flush=True)
