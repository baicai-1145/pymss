"""Why is the in-model depthwise conv ~49 ms when the same shape runs in ~1-3 ms in isolation?"""
import os, statistics, sys, time, torch, torch.nn.functional as F
os.environ.setdefault('S2_DEVICE', 'cpu')
torch.set_num_threads(int(os.environ.get('S2_THREADS', 8)))
sys.path.insert(0, os.environ.get('S2_ROOT', '/home/baicai1145/pymss-bench/.tmp/cpu24/fast'))
import entry
ARM = os.environ.get('ARM', 'H3')
m = entry.build(ARM).eval()
x = torch.randn(1, 2, int(os.environ.get('S2_SAMPLES', 352800)))

def med(fn, n=12, warm=3):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)

# capture the real tensor the model feeds into its conv
grabbed = {}
def grab(mod, inp):
    grabbed.setdefault('x', inp[0].detach().clone())
    grabbed.setdefault('w', mod.weight.detach().clone())
    grabbed.setdefault('b', mod.bias.detach().clone())
    grabbed.setdefault('pad', mod.padding[0])
    grabbed.setdefault('groups', mod.groups)
h = None
for mod in m.modules():
    if isinstance(mod, torch.nn.Conv1d) and mod.groups == mod.in_channels and mod.kernel_size[0] == 31:
        h = mod.register_forward_pre_hook(grab); break
with torch.no_grad(): m(x)
h.remove()
gx, w, b, pad, g = grabbed['x'], grabbed['w'], grabbed['b'], grabbed['pad'], grabbed['groups']
print('captured conv input', tuple(gx.shape), 'contiguous', gx.is_contiguous(), '| weight', tuple(w.shape), '| pad', pad)
with torch.no_grad():
    ref = F.conv1d(gx, w, b, padding=pad, groups=g)
    xc = gx.contiguous()
    print('  as-given  (conv1d)          %7.2f ms' % med(lambda: F.conv1d(gx, w, b, padding=pad, groups=g)))
    print('  contiguous copy + conv1d    %7.2f ms' % med(lambda: F.conv1d(xc, w, b, padding=pad, groups=g)))
    print('  the contiguous() step alone %7.2f ms' % med(lambda: gx.contiguous()))
    # delete-then-time: is it weight-packing on first use per unique weight?
    w2 = w.clone()
    print('  fresh weight clone (1st)    %7.2f ms' % med(lambda: F.conv1d(xc, w2, b, padding=pad, groups=g), n=1, warm=0))
    print('  same weight again           %7.2f ms' % med(lambda: F.conv1d(xc, w2, b, padding=pad, groups=g)))
    # how much of the 49 ms is the conv vs its output allocation
    print('  2-D channels_last form      %7.2f ms' % med(lambda: F.conv2d(xc.unsqueeze(2).contiguous(memory_format=torch.channels_last), w.unsqueeze(2), b, padding=(0, pad), groups=g)))
