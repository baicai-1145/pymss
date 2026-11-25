# Triton on the 950PR

One row per kernel: target op, fp32 rel_rms (out / grad), bf16 rel. error, ms eager vs Triton at the training shape,
adopted or not and why.  Admission = numbers **and** speed **and** whole-step loss (TASK §6).  Shapes are the real
training shapes: dwconv `(b*f, hid, t) = (1488, 384, 258)`, k=31, groups=384, bf16 and fp32; silu_mul `(1488, 258, 384)`.
Harness: `triton_admit.py` (numerics vs eager autograd, ABBA speed, then the H3 step loss).  Raw: `profile/triton_admit.json`.

## Verdict: no kernel adopted, training starts eager

| target (profile rank) | kernel | fp32 rel_rms out / grad | bf16 rel. err | eager ms | Triton ms | verdict |
|---|---|---|---|---|---|---|
| 1. dwconv **dW** (496 ms/step, rank 1) | `_dw_dw`, 3-D (c,k,t) tile, row-split + atomic, bc=2/bt=256 | 6.0 (wrong) | — | 122.9 | 73.9 | **rejected**: slower than the 30.4 ms torch reduction below, and numerically wrong (atomics on a 2-D tile; `AutoBlockify` is disabled by the compiler for this kernel) |
| 1. dwconv fwd (rank 1) | `_dw_fwd`, 3-D (c,k,t) tile | — | — | 3.7 | does not compile | **rejected**: UB overflow (442 KB needed / 221 KB available at bc=8/bt=128; the tile must hold K=31 taps) |
| 1. dwconv fwd, tap loop instead of a tile | `_dwconv_fwd`, K unrolled / dynamic loop | — | — | 3.7 | 1485.9 / 1375.3 | **rejected**: 370x slower — each pass re-reads the row, and 214k tiny programs cost more than the arithmetic |
| 1. dwconv dX (33 ms/step) | `_dw_dx` + loop variant | — | — | 8.2 | compile fail / 680.5 | **rejected**: same two failure modes as fwd |
| 2. `u * silu(v)` gate (191 ms/step chain) | `_silu_mul`, 1-D, 8192-elem blocks | 0.0 | 2.9e-3 | 10.4 | 15.7 | **rejected**: numbers pass, speed does not (0.66x; 0.73x in bf16 — 7.2 → 9.9 ms) |
| 3. STFT / ISTFT front end | — not attempted | | | 7.1 | | deprioritised by the profile (7 ms of 1169) |
| 4. `torch.quantile` in `masked_loss` | — not attempted | | | 1.1 | | deprioritised by the profile |

Why Triton loses here, in one paragraph: every kernel in this list is memory-bound with a small arithmetic intensity, so
the win would have to come from *fewer passes over the row*, and that is exactly what the UB forbids — a `(c, k, t)`
tile is 31x larger than the output it produces, and 221 KB of UB holds ~28k fp32 elements, which forces either
1100 programs each holding one tap (launch-bound: measured 0.37 s of pure overhead) or a 2-D tile per tap (31 passes,
which is what the vendor conv already does in 3.7 ms).  The existing kernels in `train_ascend/triton_blocks.py` are
fast for the same reason these are not: they fuse over a 62-element band axis with K=3, not K=31 taps of a 147M-element
tensor.

## Measured but *not* Triton, therefore not enabled (worth a decision)

Replacing only the conv *backward* with two torch calls is 4x cheaper than autograd's choice of kernel, and is
mathematically identical (same padding, groups, fp32 accumulate):

| stage | eager autograd | torch implementation | speedup |
|---|---|---|---|
| dW | `Conv3DBackpropFilterV2` 122.9 ms | 31 shifted masked reductions (`stack([(dy*pad(x))[..., j:j+t].sum((0,2)) for j in range(31)], -1)`) 30.4 ms | 4.0x |
| dX | `Conv3DBackpropInputV2` 8.2 ms | `F.conv1d(dy, weight.flip(-1), padding=15, groups=384)` 3.6 ms | 2.3x |
| fwd | `Conv2DV2` 3.7 ms | `F.conv1d` 3.8 ms | 1.0x |

That is ~97 ms of the 1169 ms step (8 %) for one function swap, with no Triton involved.  It is left **off** by default
because TASK §6 admits *kernels* under the two gates and this is not a kernel; `triton_npu.py` already routes those two
stages through torch when the Triton stage is not selected, so enabling it is a one-line change in the wrapper if the
lead wants it (`stages=('dwconv_dw',)` with the Triton branch disabled).

## Numerics note that matters for any future kernel here

Two identical `model(mixture)` calls on this device differ by 1.7e-7 relative (NPU FFT/reduction order), so a
"bit-identical to eager" test is unattainable; the harness compares with `rel_rms` and the limits the task sets
(1e-5 fp32, 1e-2 bf16).  Recording it here so the next person does not chase a phantom mismatch.
