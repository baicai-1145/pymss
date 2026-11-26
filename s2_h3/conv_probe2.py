"""Same conv, same shape: model-captured tensor vs a freshly allocated one."""
import os, statistics, sys, time, torch, torch.nn.functional as F
os.environ.setdefault('S2_DEVICE', 'cpu')
torch.set_num_threads(int(os.environ.get('S2_THREADS', 8)))
sys.path.insert(0, os.environ.get('S2_ROOT', '/home/baicai1145/pymss-bench/.tmp/cpu24/fast'))
import entry
m = entry.build(os.environ.get('ARM', 'H3')).eval()
x = torch.randn(1, 2, 352800)

def med(fn, n=9, warm=2):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)

g = {}
def grab(mod, inp):
    if not g: g.update(x=inp[0].detach().clone(), w=mod.weight.detach().clone(), b=mod.bias.detach().clone(), pad=mod.padding[0])
h = None
for mod in m.modules():
    if isinstance(mod, torch.nn.Conv1d) and mod.groups == mod.in_channels and mod.kernel_size[0] == 31:
        h = mod.register_forward_pre_hook(grab); break
with torch.no_grad(): m(x)
h.remove()
gx, w, b, pad = g['x'], g['w'], g['b'], g['pad']
fresh = torch.randn_like(gx)
clone = gx.clone()
print('model tensor ptr %% 4096 = %d | fresh ptr %% 4096 = %d | clone ptr %% 4096 = %d'
      % (gx.data_ptr() % 4096, fresh.data_ptr() % 4096, clone.data_ptr() % 4096))
with torch.no_grad():
    for name, t in (('model-captured', gx), ('clone of it', clone), ('fresh randn', fresh)):
        ms = med(lambda t=t: F.conv1d(t, w, b, padding=pad, groups=384))
        ms2 = med(lambda t=t: F.conv2d(t.unsqueeze(2).contiguous(memory_format=torch.channels_last), w.unsqueeze(2), b, padding=(0, pad), groups=384))
        print('  %-16s conv1d %6.2f ms | channels_last 2d %6.2f ms' % (name, ms, ms2))
    # is it the weight?
    w2 = torch.randn_like(w)
    print('  model tensor, random weight  %6.2f ms' % med(lambda: F.conv1d(gx, w2, b, padding=pad, groups=384)))
