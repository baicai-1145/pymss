"""Can zig cc serve as inductor's C++ compiler?  If yes, torch.compile can fuse the elementwise chains."""
import os, time, torch
print('torch', torch.__version__, '| CC =', os.environ.get('CC'), '| CXX =', os.environ.get('CXX'))
def f(a, b):
    return ((a * b).relu() + a).sum(-1)
x1, x2 = torch.randn(2000, 2000), torch.randn(2000, 2000)
ref = f(x1, x2)
try:
    t0 = time.time()
    g = torch.compile(f, dynamic=False)
    out = g(x1, x2)
    print('COMPILE_OK max|Δ| %.2e  (first call %.1fs)' % ((out - ref).abs().max(), time.time() - t0))
except Exception as exc:
    msg = str(exc)
    import re
    errs = [l for l in msg.splitlines() if re.search(r'error|fatal|undefined reference|cannot find|not found', l, re.I)]
    print('COMPILE_FAIL', type(exc).__name__)
    for l in errs[:8]: print('   ', l[:180])
