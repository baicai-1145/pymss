"""Reference output + time for the current code; later runs compare against it."""
import os, sys, time, torch
os.environ.setdefault('S2_DEVICE', 'cpu')
torch.set_num_threads(int(os.environ.get('S2_THREADS', 8)))
sys.path.insert(0, os.environ.get('S2_ROOT', '/home/baicai1145/pymss-bench/.tmp/cpu24/fast'))
import entry
ARM = os.environ.get('ARM', 'H3')
N = int(os.environ.get('S2_SAMPLES', 352800))
mode, path = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'ref.npz')
torch.manual_seed(1234)
m = entry.build(ARM).eval()
x = torch.randn(1, 2, N)
def once():
    with torch.no_grad():
        return m(x)
if mode == 'save':
    o = once()
    ts = []
    for _ in range(2): once()
    for _ in range(5):
        t0 = time.perf_counter(); once(); ts.append(time.perf_counter() - t0)
    ts.sort()
    torch.save({'out': o, 'x': x}, path)
    print('saved %s | out %.4f | median %.1f ms' % (path, float(o.abs().mean()), ts[len(ts)//2]*1000))
else:
    ref = torch.load(path, weights_only=False)
    o = once()
    d = (o - ref['out']).abs()
    scale = ref['out'].abs().max()
    ts = []
    for _ in range(2): once()
    for _ in range(5):
        t0 = time.perf_counter(); once(); ts.append(time.perf_counter() - t0)
    ts.sort()
    print('max|Δ| %.3e  (ref scale %.3e, rel %.2e) | median %.1f ms' % (d.max(), scale, d.max()/scale, ts[len(ts)//2]*1000))
