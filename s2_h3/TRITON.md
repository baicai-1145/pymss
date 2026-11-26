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

## H4s speed sprint, round 2 (compute-first, driven by step_h4s_baseline profile)

Attribution first: `with_stack` gives no Python frames for NPU events on this torch_npu, but the call counts are
decisive — `RealDiv_FLOAT32` **143 calls/step** is the 62 BandSplit norms taking the fp32 `F.normalize` fallback
(`transformer.py` only used the fused kernel for fp16/bf16, and STFT hands BandSplit fp32), and the single **12.0 ms
`Slice` (1 call/step)** is `repeat_interleave(stride,1)[:, :t]` materialising a 4x tensor and cropping it.

| milestone | step ms | vs 367.4 baseline | launches/step | hbm |
|---|---|---|---|---|
| R1 baseline (FFT probe cached) | 367.4 | 1.00x | 46,912 | 17.2 GB |
| + `npu_rms_norm` for **all** dtypes (kills the 62-norm fp32 fallback) + `block_mean` without fp32 `.mean` | **295.9** | 1.24x | 44,390 | 13.8 GB |
| + `StaticBandMixer` as 8 real GEMMs instead of a broadcast matmul lowering to 12,480 tiny ones | **296.5** | 1.24x | 44,465 | 13.8 GB |
| + `npu_rotary_mul` (gated `S2_NPU_ROPE`, **off**: see below) | 277.4 | 1.32x | 43,045 | 14.5 GB |

Honest verdicts:
- **RMSNorm + block_mean: admitted.** −71.5 ms, deterministic eval-mode parity: loss rel **4.1e-06**, every weight
  gradient <= 1.6e-02 (bf16 level); the 3.4e-02 outliers are norm-gain params, and four gammas that previously
  received **no** gradient at all now do (the old eval path detached them).
- **StaticBandMixer GEMM rewrite: kept but neutral** (296.5 vs 295.9 — within noise). It is still worth keeping because
  it removes `flat` tiny matmuls (a launch-count and ACLGraph problem), not device time.
- **`npu_rotary_mul`: ADMITTED after fixing my own cache bug (see below).** 295.9 -> **278.0 ms/step**. It measured −19 ms (277.4) and the loss matched to 2e-8, but
  `convs.4.query.{weight,bias}` gradients came out at rel 1.3: `StridedPooledKVAttention` runs q at stride 2 and kv at
  stride 4, so one cos/sin table cannot serve both, and the fused kernel silently accepted the mismatch instead of
  raising. Left behind the `S2_NPU_ROPE` flag (default off) until it is wired per-stride with a per-tensor table.
- **`S2_TRITON_BMM_CHUNK` sweep: chunk=8 wins.** 8 -> 278.1, 64 -> 295.0, 128 -> 294.0, 256/512 no better, so the
  "fewer, larger chunks" hypothesis is wrong on this shape (bigger chunks lose the L2 reuse that makes 8 cheap).

### The RoPE "parity failure" was a use-after-free in my cache, not the kernel

`_INTERLEAVED_CACHE` was keyed on `(cos.data_ptr(), dtype)` and held no reference to `cos`. `StridedPooledKVAttention`
passes `cos.index_select(1, q_pos)` — a temporary. It is freed, and the allocator hands the same block back for
`cos.index_select(1, kv_pos)`, so keys were rotated with the **query** positions: the rel-1.3 `convs.4.query.*`
gradients. Fix: the interleaved tables are cached **on the rotary module** keyed by `(seq_len, device, dtype)` (the
cache owns them), positions are selected *after* interleaving, and `npu_rope()` takes the module + `pos`. Verified with
`S2_NPU_ROPE=1`: loss rel **4.88e-06** vs clean `a2bc621`, `convs.4.query` no longer in the top gradient deltas, worst
remaining is the fp32 BandSplit norm gain (5.4e-02, the old path detached it). Also restored the CPU fp32
`F.rms_norm` fastpath that round 2 had dropped.

## Win 1 — `npu_swiglu` in MSC (measured, not yet default)

`S2_NPU_SWIGLU=1` on the H4s step: **274.2 -> 234.3 ms/step (-39.9 ms, -14.5 %)**, HBM 14.5 -> 14.0 GB, loss identical
to 8 decimals (rel 2.37e-06). The mechanism is the one measured in the audit: `chunk(2, -1)` returns non-contiguous
views and the elementwise mul on those views is ~17x slower under autocast than the fused op on the whole tensor.
Implemented as a one-time in-place swap of the two halves of `proj_in.weight`/`bias` in `_make_npu_swiglu`, so
`npu_swiglu` (`silu(first) * second`) computes exactly MSC's `first * silu(second)`.

**Why it is still off by default:** the swap makes `proj_in` a *permuted* parameterisation, so `dL/dW` rows come back
permuted (observed: rel 1.42 on exactly `proj_in.weight`/`bias`, everything else <= 1.6e-02, loss rel 2e-06). That is
correct for a fresh run, but resuming `resume-5000.ckpt` would leave Adam's `exp_avg`/`exp_avg_sq` and the EMA shadow in
the *old* row order while the weights are in the new one. Enabling it for the resume therefore requires permuting those
four tensors identically at load (a ~10-line hook, exact by construction). Not enabled until that hook exists.

## Win 3 — fused optimizer + flat EMA (`npu_opt.py`, opt-in `S2_OPT=fused`)

Measured on the H4s step: AdamW + `AveragedModel` **278.7 ms** -> `FixedNpuFusedAdamW` + `FlatEMA` **272.8 ms (-5.9 ms)**.
Smaller than the audit's -16 ms because H4s's `optimizer+ema` segment is only 19.2 ms in the profile and most of that is
AveragedModel's Python loop, which `FlatEMA` (one `lerp_` on `get_combined_params()`) does replace. Two API facts
found while wiring it: `get_combined_params()` returns a *list* whose entries are `None` until the first `step()`, and
`NpuFusedAdamW` is not an `AdamW` subclass, so `s2_aw_entry.py`'s callback assertion now accepts either type.

## Win 2 — channels-last NTC Triton dW (`S2_TRITON=dwconv_dw_ntc`), ADMITTED

Kernel is V3's `_dw_ntc_tiled` verbatim (persistent grid, UB-resident `(BT+K-1, BC)` tile, `ext.extract_slice` walking
contiguous rows, partial sums in `(NS,K,C)` reduced by one torch sum, no atomics). Two independent measurements:

| what | result |
|---|---|
| standalone on L1 `(N=1488,T=258,C=384,K=31)`, BT=128 BC=64 NS=9, grid=(54,) | **7.65 ms** vs **18.17 ms** for `dwconv_dw_bmm` |
| accuracy vs float64 on 32 samples | **rel 1.73e-07** |
| whole training step | **272.8 -> 262.8 ms** (launches 44.4k -> **28,353**; 186 launches -> 1 per conv) |

Caveat recorded from the run: the NTC tensors are built with `x.transpose(1,2)` / `grad_out.transpose(1,2)`, i.e. two
copies per conv rather than the zero-copy NTC layout V3 recommends (build `x_ntc` once in MSC and reuse it for fwd).
Removing those two transposes is the next increment on this win.

## Win 1 ADMITTED as default (with Win 2): 223.0 ms/step

Gated on the RESUME tier after the lead's `4aa3f2d` fix (the resume assert read `swiglu_permuted` off the
LightningModule instead of `module.model`, so the prepermuted path always looked unswapped):

| check | result |
|---|---|
| `SWIGLU_PREPARE` | `bound=4 prepermuted=True swapped=True` |
| `SWIGLU_CONVENTION` / `RESUME_VERIFIED` | `swapped` / `5000 EMA 5000 Adam [5000]` |
| **per-step loss gate** vs `profile/h4s_resume5000_ref200.json` (200 steps, 5001->5200) | **max 0.2018 % at step 5188, mean 0.0271 %** (limit 1 %) |
| export sanity (`_unswap_swiglu` on the saved swapped ckpt) | **PASS** — as-saved cos(block0, original) 0.0023 -> after unswap **0.9969**, and unswap is the exact inverse permutation |
| **Win 1 + Win 2 step time** | **223.0 ms, 4.48 steps/s, 28,257 launches, HBM 14.1 GB** |

Gate-tier correction (the lead's): resumes are gated against a reference frozen from the *same* resume point, not
against the fresh-run curve — `profile/h4s_ref200.json` is steps 1-200 with loss ~0.0052, while step 5000 sits at
~0.00065, so comparing them is meaningless (the first attempt scored a bogus 87.6 %). Also `loss_curve.csv` needs
`--loss-history-interval 1` to carry one row per step; the default 25 produced 8 rows.

## Win 5 — SDPA in BlockSlowBandAttention: REVERTED (opt-in `S2_SDPA=1`), not admitted

Manual `softmax(q@k^T*scale) -> @v` replaced by `F.scaled_dot_product_attention` with V zero-padded to a 32 B multiple
(`v_dim` is 16) and sliced back; padded columns multiply the same softmax weights so they contribute exactly zero. The
manual path stays as the fallback when SDPA raises. Measured **-6.5 ms** (223.0 -> 216.5), launches 28,257 -> 28,221,
HBM 13.5 GB — better than the -3.3 ms V3 projected for the inner attention alone.

## Literal export round-trip (closes the open flag)

`runs/swiglu5233/ema-5200.pt` (81,232,409 B) written through `export_core_checkpoint` from the swapped EMA shadow:
`convs.1.proj_in.weight` comes back in the ORIGINAL row order, `cos(block0, original)` **0.9957** vs **0.0026** for the
swapped source, and the file is the exact inverse permutation. So the disk path and the in-memory `state_dict` path
agree — CPU/H4s inference of exported weights is untouched.

### Why Win 5 was reverted

The measured speed is real (**-6.5 ms**, 223.0 -> **216.5**) but its 200-step resume-tier gate **failed**: max per-step
relative loss deviation **8.3343 % at step 5171, mean 1.7080 %** (limit 1 %). My own rule says a failed gate is not an
admitted milestone, so `S2_SDPA` now defaults **off** and the milestone baseline is **223.0 ms/step**.

Cause is RNG order, not arithmetic. `F.scaled_dot_product_attention` fuses the attention dropout, so it draws from the
Philox stream differently than `F.dropout(a, p)` on the manual softmax matrix; every downstream dropout mask shifts and
the two 200-step curves are different realisations, not different functions. Deterministic eval-mode check (all dropout
off, identical inputs, identical seed):

| check | result |
|---|---|
| loss, SDPA vs manual | **rel 2.93e-06** |
| 555 gradients | worst **2.18e-02** on `band_split.to_features.0.0.gamma` — the same fp32-norm artifact that reads 3.4e-02 / 5.4e-02 in the already-admitted comparisons |

So the gate as written cannot judge any change that perturbs dropout consumption. To admit Win 5 one of these is needed
and each is a rule/semantics decision, not engineering: (a) accept deterministic eval parity as the gate for RNG-order
changes, (b) drop attention dropout (a semantic change), or (c) freeze a new reference *with* `S2_SDPA=1` and gate later
changes against it, admitting SDPA itself on the eval parity above.

## P2-a — stateless-seed dropout: works, full-graph capture still BLOCKED

`npu_dropout.py` draws the mask on-device from a 1-element int64 NPU tensor (Philox-style hash of element index and
seed), writes `x * mask * scale` **and** the mask, and reuses that mask in backward, so it reproduces `nn.Dropout`
exactly while consuming no host RNG. Verified: keep-fraction **0.898** at p=0.1 over 4096 elements, identical output on
repeated calls with the same seed, and the eager step is unchanged (**219.6 ms**), i.e. it is not a slowdown.

Two findings from the capture attempts:

1. Patching `nn.Dropout` modules is **not** sufficient — the attention paths call `F.dropout(a, p)` directly, and that
   functional call still aborted on `NPUGeneratorImpl::philox_engine_inputs`. `install()` now wraps
   `torch.nn.functional.dropout` itself, which covers both call styles.
2. Even then the capture fails, and it fails on **forward alone** (so it is not backward, the optimizer, or the loss):

```
RuntimeError: Refactor this op to use NPUGeneratorImpl::philox_npu_state. Cannot call
NPUGeneratorImpl::philox_engine_inputs during NPU graph capture. npuStreamCaptureStatus: Active
```

So a third RNG consumer remains inside the model forward. Ruled out: `nn.Dropout` modules (25 patched),
`torch.nn.functional.dropout` (wrapped), every `self.dropout(...)` site (all are `nn.Dropout` instances), SDPA (off by
default after the Win 5 revert). Remaining suspects are native ops that request the default generator internally, which
the per-module capture bisect (the technique that localised the 161002 mixer error in four runs) will pin down.

## P2-a — full-core NPUGraph capture: ACHIEVED, but NEUTRAL (219.5 vs 220.9 ms)

The third Philox consumer was **not** Win 5 and not `BlockSlowBandAttention`: it was
`h4.py StridedPooledKVAttention.forward` calling `F.scaled_dot_product_attention(..., dropout_p=self.dropout ...)`
unconditionally (a plain float, invisible to both the `nn.Dropout` and the `F.dropout` wrappers). Located by the RNG
state-advance probe, not by capture bisect: forward hooks + generator-state snapshots around one eager forward, three
hits (`convs.0`, `convs.4`, root `ArmModel`).

**Semantics choice (deliberate): exact.** Dropping on `out` would be a different function, so the pooled-KV path now
runs the manual `softmax(q k^T * d^-0.5) -> stateless dropout on the scores -> @ v`, with SDPA only when dropout is
inactive. Identical distribution and identical placement; only the RNG order changes, which is what tier 2 covers.

With that, `torch.npu.graph` captures the **whole step** (fwd + bwd + `opt.step`) and replays it:

| | ms |
|---|---|
| eager step (fwd+bwd+opt) | 220.9 |
| **NPUGraph replay of the same step** | **219.5** |

So launch reduction is now worth **~1.4 ms**, not the projected 50-80 ms: at 223 ms the step is **device-bound**, and the
earlier 94 ms "dispatch gap" was mostly the unpinned H2D plus profiler overhead, both already fixed. P2-a as a lever is
exhausted — the remaining distance to 105 ms has to come from device time (the `freq[*].ffn` 29.4 ms block, `time[1]` MSC
34 ms, `mask_head` 13.9 ms), not from capturing launches.

## `expand_blocks` — bare `repeat_interleave + crop` replaced by one gather: NEUTRAL in ms, real in HBM

The largest single op in the 219 ms profile was a **3-call, 11.9 ms `Slice`**, i.e. the `repeat_interleave(stride,1)[:, :t]`
crop that materialises a `stride`-times larger tensor and then copies the kept part. Replacing it with
`out.index_select(1, arange(t)//stride)` is a **pure gather — bit-exact by construction**, so it needs no tier-1 run, and
it removes the 4x intermediate. Five sites (h3.py:117, h4.py:61/152/170/235).

Measured: **223.0 -> 222.5 ms** (within noise) but **HBM 14.1 -> 13.7 GB**. So the `Slice` was not the leak its 11.9 ms
suggested — the gather pays its own copy — and this lands as a memory win, not a speed win. Kept because it is exact and
strictly less traffic; not claimed as a milestone.
