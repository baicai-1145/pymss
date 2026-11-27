# Win 8 increment 1 — the 58 ms is real work, not bubbles

Async timeline over **20 clean steps** with nothing instrumented inside the step (one `torch.profiler` NPU trace,
`cat=kernel` events only, 92,200 kernels, all on a single stream so `sum(dur)` is exact busy time).

## Result

| quantity | value |
|---|---|
| kernel-busy per step | **216.2 ms** |
| timeline span per step | 216.7 ms |
| **busy fraction** | **99.8 %** |
| total gap over 20 steps | **8.9 ms** = **0.45 ms/step** |
| kernels per step (device) | 4,610 |
| largest single gap | 0.16 ms |

**Deciding number: busy/span = 99.8 %.** There are no device bubbles to schedule away; the largest gap in twenty steps is
0.16 ms. So the ~58 ms that `sum(self_device_ms)` left unattributed is **real device work that the per-op ranking
undercounts**, and the ranking approach stands.

Cross-check that closes the budget: 216.2 ms kernel + ~6 ms host = **222.1 ms wall**, which is the measured step. The
`self_device_ms` sum (163.5 ms) is therefore a *ranking of a subset*, not a budget — it drops ~52 ms, most likely
sub-threshold ops and the per-op self-time accounting.

## The correct ranking (complete, from the full trace)

| family | ms/step | calls/step |
|---|---|---|
| **Add** | **31.1** | 73 |
| **MatMulV** | **21.8** | 708 |
| **Transpose** | **19.4** | 250 |
| **Slice** | **15.2** | 405 |
| **Cast** | **15.0** | 1,303 |
| DropOutV | 11.9 | 56 |
| RmsNormGrad | 10.6 | 81 |
| `_dw_ntc_tiled` | 10.0 | 4 |
| Conv | 9.6 | 8 |
| ReduceSum | 9.1 | 268 |
| BatchMatMulV | 8.0 | 69 |
| ConcatD | 6.6 | 20 |

Top individual kernels: `Slice` 15.2, `Cast` 15.0, `Add_FLOAT32_8` 10.9, `_dw_ntc_tiled` 10.0, `RmsNormGrad` 9.0,
`Conv2DV2` 8.8, `Add_FLOAT32_9` 8.3, `DropOutV3` 8.1, `MatMulV3_bf16` 8.0, `Add_BFLOAT16_FLOAT32_8` 7.7.

Two corrections to the earlier picture: **Add is the single largest family at 31.1 ms over only 73 calls** (0.43 ms per
call — these are full-tensor passes, not launch-bound), and **Transpose at 19.4 ms / 250 calls** did not appear at all in
the `self_device_ms` top-10, so layout churn is a bigger lever than that ranking suggested.

**Conclusion: device-bound at 216 ms with 99.8 % occupancy. Kernel fusion / layout elimination is the path; scheduling is
not.** Next increments per the brief: `_dw_ntc_tiled` transposes (10.0 ms kernel, 2 transposes per conv), then the Cast
family after answering the autocast question.

## Increment 3 — the Cast question, answered before writing any kernel

Cast is **15.0 ms over 1,303 calls (0.0115 ms/call)** — launch-cheap. Producer attribution:

| cast block | ms | pattern |
|---|---|---|
| after `Add_FLOAT32` -> before `ZerosLike_2_BYTES` | 4.2 | structural, not dtype churn |
| `Cast -> Cast` chains | 3.2 / 2.2 | structural |
| before bf16 `MatMulV3` | **1.9** | genuine autocast churn |
| before `DropOutV3Grad` | 1.0 | dropout mask dtype |

**So dtype pinning (Win 6) would remove at most ~2-6 ms of the 15.0**, and the rest is structural — not worth a
checkpoint-dialect change, and not worth a fused Cast kernel at 0.0115 ms/call. Cast is closed as a target.

## Increment 3b — where the leverage actually is: layout chains feeding Add

Attributing the biggest family by its neighbours instead of by name:

| family | ms/step | calls | ms/call | dominant chain |
|---|---|---|---|---|
| **Add** | **32.5** | 628 | 0.052 | after `ViewCopy` 9.3, after `GatherV2` **8.4**, after `BroadcastTo` 6.9 |
| Transpose | 19.4 | 250 | 0.078 | runs into `Slice` (3.2) and `RotaryPositionEmbeddingGrad` |
| Slice | 15.2 | 405 | 0.038 | **after `Transpose` 12.0 -> before `ViewCopy` 12.0** |

One chain dominates: **`Transpose -> Slice -> ViewCopy -> Add`, plus `GatherV2 -> Add` (8.4) and
`BroadcastTo -> Add` (6.9)** — roughly **37 ms/step** of copy-then-add that a single fused
"gather/broadcast + add" kernel collapses into one pass, with no intermediate tensor. That is the next target, and it is
the fused `block_broadcast_add` the brief anticipated, now with a measured address instead of a guess.

## Increment 4 — fused `gather_add` built and measured: 1.66x on the op, but the op is too small to wire

`triton_npu.gather_add(residual, src, idx)` computes `residual + src.index_select(1, idx)` in a single pass, so the
`GatherV2 -> Add` pattern has no expanded intermediate in HBM. Two bugs found while bringing it up, both mine:
`base` omitted the time index, and `BLOCK` must be the next power of two above the whole `(f, d)` row (11,904 for the
freq blocks) — a smaller BLOCK silently writes only part of each row.

| | ms | |
|---|---|---|
| `res + expand_blocks(src, s, t)` | 0.46 | baseline |
| `gather_add` | **0.28** | **1.66x, bit-exact (rel 0.00e+00)** |

At 0.46 ms per site and roughly 8 strided sites, the total is **~1.5 ms against a 222 ms step**, i.e. below run-to-run
noise, and wiring it needs an optional `residual` argument through the generic transformer loop plus three block classes.
Measured and **not wired** — the kernel stays in `triton_npu.py` for when a real batch of layout chains can be fused in
one pass, but on its own it does not pay for the plumbing risk.

The 37 ms attributed to `copy-then-add` is therefore **not** one fusable op: it is spread over rotary `index_select`,
`block_mean`'s slice+expand padding, and the strided residual expansions, each worth well under a millisecond. That
reframes the next step: the remaining device time has to come from the *conv/MSC* path and the matmuls, not from layout.

## Increment 5 — conv/MSC path: kernel config is already optimal, and the "no copies" idea loses

**Config sweep** (`_dw_ntc_tiled` on the four real shapes, 8 configs of BC/NS/BT): the shipped `bc=64, ns=9, bt=128` is
within **0.01 ms** of the best everywhere — L1 (N1488,T258,C384,K31) 7.67 vs best 7.66, L2 1.53, L3/L5 0.19 each,
**9.58 ms total**, matching the 10.0 ms the trace attributes to it. No headroom in tuning.

**Stride-aware kernel**: I made the loads take explicit strides so a strided `(N,C,T)->(N,T,C)` view could be read without
`.contiguous()`, removing both copies. It is **bit-exact** (rel 0.0) and **slower**:

| input | ms |
|---|---|
| contiguous NTC | **7.66** |
| strided view (no copies) | **14.63** |
| contiguous + the two copies | **8.76** |

Stride-T loads coalesce badly, so the copies are the cheaper path. Dispatch stays on `.contiguous()`. The kernel keeps
the stride parameters because they are what let this be measured instead of assumed.

**Net for the increment: 0 ms.** Conv/MSC as it stands: `_dw_ntc_tiled` 9.6, `Conv2DV2` fwd 9.6, `MatMulV` 21.8 — and
each of the three has now had its obvious lever measured and closed (config, layout, and — for matmuls — nothing to fuse
without changing the module structure). Final confirmed step on the corrected path: **223.4 ms** (ntc) vs **238.4 ms**
(bmm), i.e. the corrected ntc win is **14.9 ms**.
