"""Ablate the inference fast-path switches in ONE process, round-robin, so CPU frequency drift hits every
variant equally.  Each variant's output is compared against the all-off reference."""
import itertools, os, statistics, sys, time, torch
os.environ.setdefault('S2_DEVICE', 'cpu')
torch.set_num_threads(int(os.environ.get('S2_THREADS', 8)))
sys.path.insert(0, os.environ.get('S2_ROOT', '/home/baicai1145/pymss-bench/.tmp/cpu24/fast'))
import entry
from pymss_core.modules.bs_roformer.transformer import FAST
ARM = os.environ.get('ARM', 'H3')
KEYS = list(FAST)
combos = [c for c in itertools.product((False, True), repeat=len(KEYS)) if c[0] or not any(c[1:])]
combos = [(False,) * len(KEYS)] + [(k == i) for i in range(len(KEYS)) for k in range(len(KEYS)) if k == i]
combos = [[k == i for k in range(len(KEYS))] for i in range(len(KEYS))] + [[True] * len(KEYS)]
combos = [[False] * len(KEYS)] + combos
def apply(c):
    for k, v in zip(KEYS, c): FAST[k] = bool(v)
    return '+'.join(k for k, v in zip(KEYS, c) if v) or 'baseline'
torch.manual_seed(1234)
m = entry.build(ARM).eval()
x = torch.randn(1, 2, int(os.environ.get('S2_SAMPLES', 352800)))
@torch.no_grad()
def run():
    return m(x)
ref = None; out, times = {}, {i: [] for i in range(len(combos))}
for rep in range(7):
    for i, c in enumerate(combos):
        apply(c)
        t0 = time.perf_counter(); o = run(); dt = (time.perf_counter() - t0) * 1000
        if rep == 0:
            ref = o if i == 0 else ref
            out[i] = o
        if rep > 1: times[i].append(dt)
print('%-26s %9s %9s  %s' % ('variant', 'median ms', 'vs base', 'max|Δ|'))
base = statistics.median(times[0])
for i, c in enumerate(combos):
    d = '' if i == 0 else '%.2fx' % (base / statistics.median(times[i]))
    diff = '' if i == 0 else '%.2e' % (out[i] - ref).abs().max()
    print('%-26s %9.1f %9s  %s' % (apply(c), statistics.median(times[i]), d, diff))
