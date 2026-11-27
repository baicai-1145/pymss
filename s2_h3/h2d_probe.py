import os, sys, time, statistics
os.environ['S2_DEVICE']='npu'
import torch, torch_npu
D='npu'
shape=(24,2,131584); nbytes=24*2*131584*4
print(f'H2D tensor {nbytes/1e6:.1f} MB', flush=True)
pageable = (torch.rand(*shape)*2-1).mul_(0.1)
pinned = pageable.pin_memory()
tgt = torch.empty(shape, device=D)
def t(fn, reps=10):
    for _ in range(3): fn()
    torch.npu.synchronize(); ts=[]
    for _ in range(reps):
        t0=time.perf_counter(); fn(); torch.npu.synchronize(); ts.append(time.perf_counter()-t0)
    return statistics.median(ts)*1000
ms_page = t(lambda: tgt.copy_(pageable, non_blocking=False))
ms_pin  = t(lambda: tgt.copy_(pinned, non_blocking=False))
ms_pin_nb = t(lambda: tgt.copy_(pinned, non_blocking=True))
side = torch.npu.Stream()
def stream_copy():
    with torch.npu.stream(side):
        tgt.copy_(pinned, non_blocking=True)
    cur = torch.npu.current_stream(); cur.wait_stream(side)
ms_stream = t(stream_copy)
print(f'H2D pageable_blocking {ms_page:.2f} ms = {nbytes/1e6/ms_page:.2f} GB/s', flush=True)
print(f'H2D pinned_blocking   {ms_pin:.2f} ms = {nbytes/1e6/ms_pin:.2f} GB/s', flush=True)
print(f'H2D pinned_nb         {ms_pin_nb:.2f} ms', flush=True)
print(f'H2D side_stream       {ms_stream:.2f} ms', flush=True)
# overlap: a representative 40 ms of device work while the copy runs
M=torch.randn(2048,2048,device=D,dtype=torch.bfloat16); W=torch.randn(2048,2048,device=D,dtype=torch.bfloat16)
def work(): torch.matmul(M,W)
ms_work = t(work, reps=5)
def serial():
    tgt.copy_(pinned, non_blocking=False); work()
def overlapped():
    with torch.npu.stream(side):
        tgt.copy_(pinned, non_blocking=True)
    work()
    cur=torch.npu.current_stream(); cur.wait_stream(side)
ms_serial = t(serial, reps=5); ms_over = t(overlapped, reps=5)
print(f'OVERLAP work {ms_work:.1f} + copy {ms_pin:.1f} = serial {ms_serial:.1f} | overlapped {ms_over:.1f} | saved {ms_serial-ms_over:.1f} ms', flush=True)
