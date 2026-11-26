# Triton on the 950PR

One row per kernel: target op, fp32 rel_rms (out / grad), bf16 rel. error, ms eager vs Triton at the training shape,
adopted or not and why.  Admission = numbers **and** speed **and** whole-step loss (TASK §6).

Shapes are the real training shapes: dwconv `(b*f, hid, t) = (1488, 384, 258)`, k=31, groups=384; silu_mul
`(1488, 258, 384)`.  Harness `triton_admit.py` (numerics vs eager autograd, ABBA speed, then the H3 step loss), raw
numbers in `profile/triton_admit.json`.  Kernel source: `triton_npu.py`.  Installed only from `s2_aw_entry.py` via
`S2_TRITON`; `stage.py`'s 56-window evaluation always runs eager.

## Adopted: `dwconv_dw` — the depthwise-conv weight gradient

| | |
|---|---|
| target | `Conv3DBackpropFilterV2` = 496 ms of the 1169 ms step (42 %), the profile's rank-1 hotspot |
| kernel | `_dw_dw`: one program per (channel block, row split), 3-D `(c, k, t)` tile, fp32 accumulate, atomic add |
| config | `bc=1, bt=128, bk=32, ns=4` (env `S2_TRITON_{BC,BT,BK,NS}`) |
| bf16 numerics | 1.7e-3 rel_rms vs eager autograd (limit 1e-2) — **pass** |
| fp32 numerics | 5.7e-07 vs a float64 reference (limit 1e-5) — **pass**; 2.9e-04 vs *eager*, because eager is 2.9e-04 off the float64 truth itself (see below) |
| speed | **62.7 ms vs 127.1 ms eager (2.03x)**, ABBA, 20 reps, median — **pass** |
| whole step | H3 forward+loss with the kernel installed: 0.00110775 vs eager 0.00110775, rel 0.0e+00 (limit 1e-4) — **pass** |

The fp32 gate needs a note: comparing against eager autograd for this reduction (1488 x 258 = 384k terms) measures
*eager's* accumulation error, not the kernel's.  Same inputs, three implementations:

| implementation | rel_rms vs float64 |
|---|---|
| eager autograd (`Conv3DBackpropFilterV2`, FP32) | 2.91e-04 |
| this Triton kernel (fp32 accumulate) | **5.74e-07** |

i.e. the kernel is ~500x closer to the exact reduction than the eager kernel it replaces.  Both numbers are in
`profile/triton_admit.json` (`vs_float64`, `eager_vs_float64`), and `triton_admit.py` prints them next to each row.

## Rejected (recorded, not installed)

| target | kernel | fp32 rel | bf16 rel | eager ms | Triton ms | why rejected |
|---|---|---|---|---|---|---|
| dwconv **forward** (rank 1) | `_dw_fwd`, 3-D tile | — | — | 3.7 | does not compile | UB overflow: the tile is `(c, k=31, t)` = 31x the output; 442 KB needed / 221 KB available |
| dwconv forward, tap loop | `_dwconv_fwd`, K unrolled / dynamic | — | — | 3.7 | 1485.9 / 1375.3 | 370x slower: re-reads the row per tap, and 214k tiny programs cost more than the arithmetic |
| dwconv **dX** (33 ms/step) | `_dw_dx` + loop variant | — | — | 8.2 | compile fail / 680.5 | same two failure modes as forward |
| full conv in Triton (fwd+dX+dW together) | `depthwise_conv1d(stages=STAGES)` | 2.9e-04 | 8.3e-06 | 357.6 / 140.1 | 247.3 / 236.0 | loses to eager once forward and dX are included (0.59-1.45x) |
| `u * silu(v)` gate (rank 2, 191 ms/step chain) | `_silu_mul`, 1-D, 8192-elem blocks | 0.0 | 2.9e-3 | 12.9 / 8.5 | 18.2 / 11.2 | numbers pass, speed does not (0.71x fp32, 0.76x bf16) |
| STFT / ISTFT front end (rank 3) | not attempted | | | 7.1 ms | | profile puts it at 0.6 % of the step |
| `torch.quantile` in `masked_loss` (rank 4) | not attempted | | | 1.1 ms | | profile puts it at 0.1 % of the step |

Why the forward/dX kernels lose, in one line: both are memory-bound with arithmetic intensity ~1, so the only win is
loading each sample once and reusing it across the 31 taps — and the UB (221 KB ≈ 28k fp32 elements) cannot hold a
`(c, k, t)` tile large enough to amortise the 1100+ programs that a smaller tile would need.  The dW kernel escapes this
because its reduction is over rows, so one program can loop over all of `n` with a small tile and still do 12M MACs.
For reference, the existing kernels in `train_ascend/triton_blocks.py` are fast for exactly the same reason: they fuse
over a 62-element band axis with K=3, not K=31 over a 147M-element tensor.

## 200-step reference runs (TASK §7.1 / §7.2)

Same command, same `--seed 0`, `runs/H3-ref-eager` vs `runs/H3-ref-triton` (`S2_TRITON=dwconv_dw`), 200 steps each,
curves in `runs/H3-ref-*/loss_curve.csv`, summary in `profile/ref200.json`:

| | eager | Triton `dwconv_dw` |
|---|---|---|
| loss step 1 → 200 | 0.0052292 → 0.0016331 | 0.0052292 → 0.0016332 |
| ms/step, 20 timed steps, median | 1111.8 (0.899 steps/s) | **844.6 (1.184 steps/s)** |
| step-to-step loss difference | — | mean 0.005 %, max 0.019 % (limit 1 %) |

So the kernel is worth **1.32x end-to-end** (267 ms of the step), and the loss trajectory is unchanged to 5e-5 relative.

## Numerics note for the next kernel here

Two identical `model(mixture)` calls differ by 1.7e-7 relative on this device (NPU FFT/reduction order), so
"bit-identical to eager" is unattainable; compare with `rel_rms` against the task's limits (1e-5 fp32, 1e-2 bf16), and
for large reductions also against a float64 reference, otherwise you are measuring the eager kernel's error.

## §10 speed sprint (exclusive NPU, 2026-11-25 22:38 -> 00:40)

Rounds in the order TASK §10 asks, each measured whole-step on an idle NPU (`profile/step_round{1,2,3}.json`, step rate
from `.tmp/steprate.py` = 10 warmup + 20 timed steps, median):

| round | change | step ms | vs eager 1112.6 | profile |
|---|---|---|---|---|
| start | Triton `dwconv_dw` (v1) | 845.3 | 1.32x | 829.4 ms, backward 621 |
| 1 | dW via §10(b) cube bmm, **FP32** | 668.3 | 1.67x | 829.4 -> 720.7 (`BatchMatMulV3` FP32 85.4 -> bf16 23.2) |
| 1 | dW bmm in **bf16** (fp32 accumulate on `dw`) | 604.9 | 1.84x | 720.7 -> 656.8 |
| 2 | h2d: `--pin-memory` on NPU (stage.py) | (in profile) | | h2d 42.1 ms, pageable 2.43 -> pinned 0.54 ms/tensor |
| 3 | RMSNorm: fused `F.rms_norm` in training too | | | `RealDiv FLOAT32` 42.5 ms gone, `Cast` 95 -> 29 ms |

**Round 1 detail — both paths measured, (b) wins, and (a) cannot be fixed:**

| variant | ms/conv | note |
|---|---|---|
| eager `Conv3DBackpropFilterV2` | 122.9 | baseline |
| Triton v1 `_dw_dw` (1 row, K taps, 31x traffic) | 62.7 | admitted in round 1, still in the tree |
| Triton v2a `_dw_dw2` (row blocks x time slices in grid, 4608 programs, BR=4/BKC=16) | 89.0-107.0 | §10(a): **slower** — more programs did not help; the extra `axis=0` reduction per tile costs more than the latency it hides |
| Triton v2a, other tiles (BR 2/4/8, BT 64/128/256, BKC 8/16/32) | 89-107 or UB overflow | swept, best 89.0 |
| **cube bmm, `(n*c)` batch, bf16, fp32 accumulate** (§10(b)) | **17.8** | adopted: `dwconv_dw_bmm` |
| same, 31-pass elementwise instead of im2col | 30.5 | |

Why no Triton cube kernel exists for this op: `dW[c,k] = sum_t DY[c,t] X[c,t+k-15]` needs the reduction over `t` to be the
matmul's K dimension, which leaves M=1 per channel; any M>1 would mix channels (dW[c,k] for c' != c), and the x operand
is shifted by both `t` and `k`, so it cannot be factored into a single `(M,K)x(K,N)` product. `tl.dot` has no batched
form, so the cube path is only reachable through `torch.bmm` — i.e. **the adopted dW kernel is an aclnn/cube GEMM, not
Triton**, exactly the §10(b) option. The Triton kernels stay in the tree and remain selectable (`S2_TRITON=dwconv_dw`),
which costs 240 ms/step.

**Round 3 detail — the remaining hotspots (`profile/step_round3.json`), after the three rounds:**

| ms | calls | op | verdict |
|---|---|---|---|
| 74.6 | 186 | `Mul` inside the dW im2col chunk loop | inherent to im2col (`AsStrided` 43 ms alongside) |
| 61.6 | 12 | RoPE `Mul` (`apply_rotary_emb_fast`: `t[...,::2]*cos` etc., q and k x 6 blocks) | **identified, not changed**: rewriting the interleaved form to cut 4 strided muls needs a convention change and a fresh control run; left for the next segment boundary |
| 43.0 | 2232 | `AsStrided` (im2col copies) | inherent |
| 29.7 | 723 | `Mul` bf16 | distributed |
| 29.1 | 24 | `Conv2DV2` (the 4 conv forwards, 1.2 ms each) | Triton fwd does not compile in UB at this shape (round 1) |
| 20.5 | 258 | `Add` bf16 | distributed |
| 16.0+11.0+10.7 | | `MatMulV3` bf16 (attention) | aclnn is faster than anything rotatable here |
| 15.2 | 75 | `DropOut` | aclnn |

Stop reason: 3 rounds done, all admitted, but the step is 604.9 ms rather than the 450 ms goal — the remaining items are
either inherent to im2col (118 ms combined), need a RoPE convention change (61.6 ms), or are distributed small
elementwise ops whose only fix is fewer dispatches. Per TASK §10's stop rule this is the point to launch and keep the
rest for segment boundaries.

## Final kernel list for the 50k run

`S2_TRITON=dwconv_dw_bmm` (the depthwise-conv weight gradient on the cube, 4 convs) + fused `F.rms_norm` in the model
(not a kernel, an op-count reduction) + `--pin-memory` on the NPU dataloader. Everything else is the eager path.
200-step control against the eager curve: **mean 0.0076 %, max 0.032 %** (limit 1 %), `runs/H3-ref-triton2`
(re-run after the RMSNorm fix; the earlier 0.055 % figure was measured while the norm gain was detached from the graph).

## H4s speed sprint, round 1 (stage 5000 -> 10000 window, exclusive NPU)

Baseline on H4s: **370.0 ms/step, 46,912 CPU launches/step**, HBM 17.2 GB, NPU util 76 % (real stage: 374 ms/step).

| round | change | ms/step | verdict |
|---|---|---|---|
| R1 | vendor-FFT silent-zero probe cached per `(n_fft, device)` (`npu_fft.py`), so the 2 host syncs/step run once | **367.4** | **admitted**: small (−2.6 ms) but removes the graph-break precondition |
| R1b | `PYMSS_CORE_EXPERIMENTAL_TRAIN_GROUPED_BANDS=1` (+ `..._DEEP_MASK_GROUPING=1`) | 387.7 / 388.6 | **rejected**: 18 ms *slower* with worst grad rel 1.3e-2; launches drop 46.9k -> 32.7k but the grouped ops are individually slower, so launch count is not the binding constraint here |
| R2a | `make_graphed_callables` on the whole mask core | — | **blocked**: `aclnnMatmul ... error code 161002`. Bisected: `band_split` captures, every 62-band frequency mixer does not (inner dim 62 is not 32B-aligned; eager accepts it, ACLGraph does not) |
| R2b | capture the 6 time-axis transformers only (`npu_graphs.py`, opt-in `S2_GRAPHS=time`) | 358.7 vs 358.4 | **neutral, not enabled**: fwd bit-identical (rel 0.0) and all 6 capture, but zero gain — that branch is compute-bound, not dispatch-bound. Also freezes dropout RNG (max rate 0.1), so it would need an explicit decision |
| R3a | `torch.optim.AdamW(..., foreach=True)` | 366.8 = 366.8 | no effect on this device |
| R3b | `NpuFusedAdamW` | — | fails `set_to_none is not supported in fused optimizers`; `issubclass(NpuFusedAdamW, torch.optim.AdamW)` is False so `s2_aw_entry.py:43` would need relaxing. Not pursued: the optimizer is not a bottleneck |
| R3c | EMA as one `_foreach_lerp_` | 363.3 | −3.5 ms, **not adopted** (changes the EMA code path for ~1 % gain; revisit only if EMA export parity is verified) |
| R5 | H2D pinned `non_blocking` | **1.10 ms** (pageable 2.75) | the old 59 ms profile figure was an artifact of unpinned buffers in the profiler, not a training cost |

**Real bug fixed along the way** (`h4.py`): `band_up = nn.Parameter(q.T.clone())` produced 4 non-contiguous
parameters (stride `(1, 62)`), which breaks any multi-tensor optimizer path (`Tensors to combine must be contiguous`);
now `q.T.contiguous()`. Also cached `StridedPooledKVAttention._centers` (per-step `torch.arange`).

**Conclusion for the ~10 steps/s target:** not reachable by configuration on this shape. The step is compute-bound at
76 % util with 46.9k launches, and the two structural levers both fail for concrete reasons: the frequency branch's
62-dim matmul cannot be ACLGraph-captured (161002), and the time branch that *can* be captured gains nothing. Getting
from 367 ms to ~100 ms needs fewer FLOPs (smaller L/k on L1, lower band rate) or a hand-written UB-resident conv — an
algorithmic change, not a wiring change.
