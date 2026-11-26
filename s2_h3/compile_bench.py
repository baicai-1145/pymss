"""Eager vs torch.compile for the whole model, both measured in one process, outputs compared."""
import os, statistics, sys, time, torch
os.environ.setdefault('S2_DEVICE', 'cpu')
torch.set_num_threads(int(os.environ.get('S2_THREADS', 8)))
sys.path.insert(0, os.environ.get('S2_ROOT', '/home/baicai1145/pymss-bench/.tmp/cpu24/fast'))
import entry
m = entry.build(os.environ.get('ARM', 'H3')).eval()
x = torch.randn(1, 2, int(os.environ.get('S2_SAMPLES', 352800)))

def med(fn, warm=2, n=5):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)

with torch.no_grad():
    ref = m(x)
    e1 = med(lambda: m(x))
print('eager      %8.1f ms' % e1, flush=True)
t0 = time.time()
try:
    cm = torch.compile(m, dynamic=False)
    with torch.no_grad():
        out = cm(x)
    print('compiled   %8.1f ms   (first call incl. compile %.0fs)' % (med(lambda: cm(x)), time.time() - t0), flush=True)
    d = (out - ref).abs()
    print('max|Δ| %.3e   rel %.2e   (ref scale %.3e)' % (d.max(), d.max() / ref.abs().max(), ref.abs().max()))
    with torch.no_grad():
        e2 = med(lambda: m(x))
    print('eager again %7.1f ms  -> speedup %.2fx' % (e2, e2 / med(lambda: cm(x))))
except Exception as exc:
    print('COMPILE_FAIL', type(exc).__name__, str(exc)[:300])
