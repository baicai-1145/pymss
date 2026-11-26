"""MSC layout change: same output, same gradients, fewer transposes."""
import os, sys, time, torch
os.environ.setdefault('S2_DEVICE', 'cpu'); os.environ['ARM'] = 'H3'
sys.path.insert(0, os.environ.get('S2_ROOT', '/Volumes/2T/repos/pymss/.tmp/colab_s2'))
import torch.nn.functional as F
import entry
from entry import MSC

def old_forward(self, x):
    b, t, f, d = x.shape
    u, v = self.proj_in(x).chunk(2, dim=-1)
    g = self.gate_drop(u * F.silu(v))
    hid = g.shape[-1]
    g = g.permute(0, 2, 1, 3).reshape(b * f, t, hid).transpose(1, 2)
    g = sum(c(g) for c in self.convs)
    g = g.transpose(1, 2).reshape(b, f, t, hid).permute(0, 2, 1, 3)
    return self.drop(self.proj_out(g))

torch.manual_seed(0)
m = MSC(dim=192, mult=4, dilations=(1,), k=31, dropout=0.0, gate_dropout=False).eval()
x = torch.randn(1, 690, 62, 192)
with torch.no_grad():
    new = m(x)
    MSC.forward, new_fwd = old_forward, MSC.forward
    old = m(x)
    MSC.forward = new_fwd
print('forward  max|Δ| %.3e   ref scale %.3f' % ((new - old).abs().max(), old.abs().max()))
g = torch.randn_like(old)
def grads(out):
    m.zero_grad(); out.backward(g, retain_graph=False)
    return {n: p.grad.clone() for n, p in m.named_parameters()}
m.zero_grad(); out_new = m(x); gn = grads(out_new)
MSC.forward = old_forward
m.zero_grad(); out_old = m(x); go = grads(out_old)
MSC.forward = new_fwd
print('backward max|Δ| per tensor:', {n: float('%.2e' % (gn[n] - go[n]).abs().max()) for n in gn})
def bench(fn, n=10):
    for _ in range(3): fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1000
with torch.no_grad():
    print('ms: new %.2f  old %.2f' % (bench(lambda: m(x)), (MSC.forward.__set__ if False else None) or bench(lambda: old_forward(m, x))))
