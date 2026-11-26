"""DRAM streaming bandwidth: the working set is 2 x 400 MB, far past the box's 128 MB of L3."""
import os, statistics, time, torch
torch.set_num_threads(int(os.environ.get('S2_THREADS', 8)))
N = 100_000_000                      # 400 MB per tensor
a, b = torch.randn(N), torch.randn(N)
y = torch.empty_like(a)
def samples(fn, n=6):
    for _ in range(2): fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); out.append((time.perf_counter() - t0) * 1000)
    return out
for name, fn, bytes_ in (('copy (r+w)', lambda: y.copy_(a), 2 * N * 4),
                         ('add  (2r+w)', lambda: torch.add(a, b, out=y), 3 * N * 4),
                         ('sum  (r)', lambda: torch.sum(a), N * 4)):
    s = samples(fn)
    ms = statistics.median(s)
    print('  %-11s %7.1f ms  %6.1f GB/s   samples %s'
          % (name, ms, bytes_ / ms / 1e6, ' '.join('%.0f' % v for v in s)))
