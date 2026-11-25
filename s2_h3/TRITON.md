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
