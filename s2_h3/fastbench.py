"""Pick the fast inference implementation by measurement, not by guesswork (TASK-fast §1).

CPU only, 8 cores / FP32 — the same protocol the RTF numbers use.  Every candidate is compared to the current
implementation twice: max|output difference| (must be ~0) and time (must be lower, else it is rejected).

    python fastbench.py            # everything
    python fastbench.py conv       # just the depthwise conv candidates
"""
import json, os, sys, time
import torch

os.environ.setdefault('S2_DEVICE', 'cpu')
threads = int(os.environ.get('S2_THREADS', 8))
torch.set_num_threads(threads)
try:
    os.sched_setaffinity(0, set(range(threads)))
except Exception:
    pass

sys.path.insert(0, os.environ.get('S2_ROOT', '/workspace/s2_h3/s2_h3'))
import entry
ARM = os.environ.get('ARM', 'H3').upper() if os.environ.get('ARM', 'H3') != 'A' else 'A'

SAMPLES = 131584                      # 8 s chunk the RTF numbers are quoted per
CH = 2
T, F, D, HID, K = 690, 62, 192, 384, 31
OUT = {}

def bench(fn, warmup=3, reps=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] * 1000.0      # median ms

def rnd(*shape):
    return torch.randn(*shape)

# ---------------------------------------------------------------- depthwise conv candidates
def conv_candidates():
    import torch.nn.functional as F
    x = rnd(1 * F_, HID, T)              # (b*f, hid, t) — what MSC feeds the conv today
    w = rnd(HID, 1, K); b = rnd(HID)
    ref = F.conv1d(x, w, b, padding=K // 2, groups=HID)
    out = {}

    out['conv1d (current)'] = (bench(lambda: F.conv1d(x, w, b, padding=K // 2, groups=HID)), None)

    # depthwise as 2-D with an empty height, channels_last: oneDNN's fast depthwise path
    x4 = x.unsqueeze(2).contiguous(memory_format=torch.channels_last)      # (n, c, 1, t)
    w4 = w.unsqueeze(2)                                                    # (c, 1, 1, k)
    def conv2d_cl():
        return F.conv2d(x4, w4, b, padding=(0, K // 2), groups=HID).reshape(x.shape)
    out['conv2d channels_last'] = (bench(conv2d_cl), (conv2d_cl() - ref).abs().max().item())

    # shifted multiply-add over a contiguous time slab (no im2col, no layout change)
    def shift_add():
        y = torch.zeros_like(x)
        for k in range(K):
            lo, hi = k - K // 2, k - K // 2 + T
            xs = x[:, :, max(lo, 0):min(hi, T)]
            y[:, :, max(-lo, 0):max(-lo, 0) + xs.shape[-1]] += w[:, 0, k, None] * xs
        return y + b[None, :, None]
    out['shift-add loop'] = (bench(shift_add), (shift_add() - ref).abs().max().item())

    # same arithmetic, blocked over time so each block stays in an L3 slice (the EPYC has 16 MB per CCX)
    for bt in (32, 64, 128):
        def shift_add_blocked(BT=bt):
            y = torch.empty_like(x)
            for t0 in range(0, T, BT):
                t1 = min(t0 + BT, T)
                acc = torch.zeros(x[:, :, t0:t1].shape)
                for k in range(K):
                    a0 = t0 + (k - K // 2)                       # first source frame for this tap
                    d0 = max(0, -a0)                            # skip outputs whose source is off the start
                    d1 = min(t1 - t0, T - a0)                   # ... and those off the end
                    if d1 > d0:
                        acc[:, :, d0:d1] += w[:, 0, k, None] * x[:, :, a0 + d0:a0 + d1]
                y[:, :, t0:t1] = acc
            return y + b[None, :, None]
        out['shift-add blocked t=%d' % bt] = (bench(shift_add_blocked), (shift_add_blocked() - ref).abs().max().item())
    return ref, out

# ---------------------------------------------------------------- model-level candidates
def segment_times(model, x):
    """Time every top-level block of the real forward with hooks."""
    times, handles = {}, []
    def mk(name):
        def hook(m, inp, outp):
            t0 = time.perf_counter()
            def done():
                times[name] = times.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0
            done()
        return hook
    names = {}
    for i, conv in enumerate(getattr(model, 'convs', [])):
        names[id(conv)] = f'temporal[{i}]'
    for i, layer in enumerate(getattr(model, 'layers', [])):
        names[id(layer[1])] = f'freq[{i}]'
    for attr in ('band_split', 'final_norm'):
        m = getattr(model, attr, None)
        if m is not None: names[id(m)] = attr
    for mod in model.modules():
        if id(mod) in names:
            handles.append(mod.register_forward_hook(mk(names[id(mod)])))
    with torch.inference_mode():
        model(x)
    for h in handles: h.remove()
    return times

def model_candidates(x):
    out = {}
    base = entry.build(ARM).eval()
    with torch.inference_mode():
        ref = base(x).clone()
    out['baseline'] = (bench(lambda: base(x), 2, 5), None)

    # 1. channels_last memory format: lets oneDNN pick its preferred blocked layout everywhere
    cl = entry.build(ARM).eval()
    with torch.inference_mode():
        cl = cl.to(memory_format=torch.channels_last) if any(t.dim() == 4 for t in cl.parameters()) else cl
        try:
            o = cl(x); d = (o - ref).abs().max().item()
            out['channels_last'] = (bench(lambda: cl(x), 2, 5), d)
        except Exception as exc:
            out['channels_last'] = (None, repr(exc)[:120])

    # 2. torch.compile: inductor fuses the elementwise chains and may emit a direct depthwise conv
    try:
        cp = torch.compile(entry.build(ARM).eval(), dynamic=False)
        with torch.inference_mode():
            o = cp(x); d = (o - ref).abs().max().item()
            out['torch.compile'] = (bench(lambda: cp(x), 1, 3), d)
    except Exception as exc:
        out['torch.compile'] = (None, repr(exc)[:200])

    out['_segments_baseline'] = segment_times(base, x)

    # 3. where the time actually goes, by operator (self time), not by module
    from torch.profiler import profile, ProfilerActivity
    with torch.inference_mode():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=False) as prof:
            base(x)
    rows = sorted(((e.self_cpu_time_total, e.key) for e in prof.key_averages()), reverse=True)
    out['_top_ops'] = [(round(ms / 1000.0, 1), k) for ms, k in rows[:14]]
    return ref, out

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    global F_
    F_ = 1
    print(f'device cpu, {threads} threads, {torch.__version__}, arm {ARM}', flush=True)
    if what in ('all', 'conv'):
        print('\n== depthwise conv, (62, 384, 690), k=31, groups=384 ==', flush=True)
        ref, res = conv_candidates()
        for name, (ms, diff) in res.items():
            print('  %-22s %8.2f ms   max|Δ| %s' % (name, ms, 'ref' if diff is None else '%.2e' % diff), flush=True)
        OUT['conv'] = res
    if what in ('all', 'model'):
        x = rnd(1, CH, SAMPLES)
        print('\n== whole forward, 1 x 2 x %d (8 s) ==' % SAMPLES, flush=True)
        ref, res = model_candidates(x)
        seg = res.pop('_segments_baseline')
        top = res.pop('_top_ops')
        OUT['top_ops'] = top
        for name, (ms, diff) in res.items():
            print('  %-22s %8.2f ms   max|Δ| %s' % (name, 'n/a' if ms is None else '%.2f' % ms, 'ref' if diff is None else diff), flush=True)
        print('\n  top operators by self time (ms):', flush=True)
        for ms, k in top:
            print('    %-52s %7.1f' % (k[:52], ms), flush=True)
        print('\n  segments of the baseline (ms):', flush=True)
        for k, v in sorted(seg.items(), key=lambda kv: -kv[1]):
            print('    %-14s %7.1f' % (k, v), flush=True)
        OUT['model'] = {k: v for k, v in res.items()}
        OUT['segments'] = seg
    json.dump(OUT, open(os.environ.get('S2_BENCH_OUT', 'fastbench.json'), 'w'), indent=1, default=str)
    print('\nwrote', os.environ.get('S2_BENCH_OUT', 'fastbench.json'), flush=True)
