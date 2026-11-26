import os, sys, time, statistics
os.environ['S2_DEVICE']='npu'; os.environ['ARM']='H4s'
sys.path.insert(0,'/workspace/s2_h3/s2_h3')
import torch, torch_npu
from entry import DEVICE, autocast, build
from pymsst.training.losses import masked_loss
import triton_npu, npu_dropout
torch.manual_seed(0)
m = build('H4s').to(DEVICE).train()
triton_npu.install(m, ['dwconv_dw_ntc'])
seed = npu_dropout.seed_buffer(DEVICE)
n = npu_dropout.install(m, seed); print('P2A dropout patched', n, flush=True)
params=[p for p in m.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=.01)
mix=(torch.rand(24,2,131584)*2-1).mul_(0.15).to(DEVICE); tgt=(torch.rand(24,2,131584)*2-1).mul_(0.08).to(DEVICE)
def step():
    opt.zero_grad(set_to_none=True)
    with autocast(): loss = masked_loss(m(mix), tgt, q=0.95, coarse=True)
    loss.backward(); opt.step(); return loss
for i in range(4):
    seed.add_(1); l = step()
torch.npu.synchronize(); print(f'P2A eager loss {float(l):.6f}', flush=True)
ts=[]
for _ in range(8):
    seed.add_(1); t0=time.perf_counter(); step(); torch.npu.synchronize(); ts.append(time.perf_counter()-t0)
print(f'P2A eager_step {statistics.median(ts)*1000:.1f} ms (no graph)', flush=True)
# stateless dropout must equal nn.Dropout statistically and be deterministic given the seed
x = torch.randn(4096, device=DEVICE, dtype=torch.bfloat16)
mask_kept = float((npu_dropout.dropout(x, 0.1, seed) != 0).float().mean())
print(f'P2A dropout keep-frac {mask_kept:.3f} (expect ~0.900)', flush=True)
s2 = torch.zeros(1, device=DEVICE, dtype=torch.int64)
print('P2A same-seed repeat identical:', torch.equal(npu_dropout.dropout(x,0.1,s2), npu_dropout.dropout(x,0.1,s2)), flush=True)
try:
    graph = torch.npu.NPUGraph()
    g = torch.npu.graph(graph)
    with g:
        seed.add_(1); loss_g = step()
    torch.npu.synchronize()
    print('P2A GRAPH capture OK (full fwd+bwd+opt.step)', flush=True)
    ts=[]
    for _ in range(8):
        seed.add_(1); t0=time.perf_counter(); graph.replay(); torch.npu.synchronize(); ts.append(time.perf_counter()-t0)
    print(f'P2A graph_step {statistics.median(ts)*1000:.1f} ms', flush=True)
except Exception as e:
    print(f'P2A GRAPH FAIL {type(e).__name__}: {str(e)[:300]}', flush=True)
