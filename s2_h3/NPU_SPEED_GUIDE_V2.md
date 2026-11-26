# H4s Ascend 950PR Speed Sprint — Concrete Addendum (Verified Against Installed Wheel)

Static & API-level verification against the live 950PR container (`torch_npu 2.13.0`, `triton-ascend 3.6.0`, `H4s`) surfaced **5 concrete blockers/shortcuts** to save sprint cycles.

---

## 1. Real Bug Blocking `NpuFusedAdamW` (Fix Before R3)

- `h4.py` L42 & L79 create `self.band_up = nn.Parameter(q.T.clone())`. In PyTorch, `.clone()` defaults to `memory_format=torch.preserve_format`, leaving **4 non-contiguous parameters** with stride `(1, 62)`: `convs.{0,2,3,5}.band_up`.
- `torch_npu.optim.NpuFusedAdamW.step()` calls `npu_combine_tensors(...)`, which **immediately crashes** with:
  `RuntimeError: Tensors to combine must be contiguous.`
- **Fix**:
  1. Change `q.T.clone()` -> `q.T.contiguous()` in `h4.py` L42 & L79 (math/values identical, only memory layout).
  2. Guard before creating `NpuFusedAdamW`: `for p in model.parameters(): if not p.is_contiguous(): p.data = p.data.contiguous()` (also safe when resuming from `resume-5000.ckpt`).
- **Second blocker in `s2_aw_entry.py` L43**:
  - `NpuFusedAdamW` MRO is `['NpuFusedAdamW', 'NpuFusedOptimizerBase', 'Optimizer', 'object']` — `issubclass(NpuFusedAdamW, torch.optim.AdamW)` is **`False`**!
  - Relax the startup assert in `s2_aw_entry.py` L43 to `isinstance(opt, (torch.optim.AdamW, torch_npu.optim.NpuFusedAdamW))`.
- **Optimizer & EMA recipe**:
  - Benchmark BOTH `torch.optim.AdamW(..., foreach=True)` (zero resume risk, `tm._optimizer_backend_kwargs('foreach')` already exists) AND `NpuFusedAdamW` (verify `load_state_dict` from `resume-5000.ckpt` + 20-step parity).
  - If `NpuFusedAdamW` passes, note `opt.get_combined_params()` exposes a single flat fp32 parameter buffer — EMA can optionally run as **one** `flat_ema.lerp_(flat_p, 1 - 0.999)` kernel instead of 550 per-param kernels (verify `ema_model.n_averaged` and checkpoint `state_dict` parity).

---

## 2. R2 Graph Capture — Beat Dynamo Graph Breaks With `torch.npu.make_graphed_callables`

- `torch.compile(backend='npugraphs' / 'inductor' / 'npugraph_ex')` is likely to hit Dynamo graph breaks on:
  1. Custom `autograd.Function` in `triton_npu.py` (`_DWConv`) — Dynamo stops unless wrapped with `@torch.library.custom_op`.
  2. `stft_npu` / `istft_npu` (`torch.fft.rfft`, `view_as_complex`).
  3. Dict caches (`cached_rotary_cos_sin`, `RMSNorm._gamma_dtype_cache`).
  4. `h4.py` L86 `StridedPooledKVAttention._centers()` creating `torch.arange(0, frames, stride, device=x.device)` every step -> **cache `(frames, stride, device)` in a dict on the module** (also removes per-step op).
- **High-leverage alternative / fallback already verified in `torch_npu 2.13.0`**:
  - `torch.npu.make_graphed_callables(core_module, sample_args, num_warmup_iters=3)`.
  - Captures at the **NPU stream level** (both forward AND backward as one autograd node), so it captures **custom `autograd.Function`, Triton kernels, `BandSplit` (62 bands), `MaskEstimator` (62 bands), `RMSNorm`, and `SlowBandAttention`** in a single ACLGraph replay with **zero Dynamo graph breaks**!
  - **Boundary**: wrap `model._forward_mask_core` (100% real-valued, fixed shape `(24, 258, 1025, 4)` -> `(24, 1, 258, 4100)`), NOT STFT/ISTFT first. Make sure all parameters used inside `_forward_mask_core` are registered submodules on the wrapped `nn.Module` so `make_graphed_callables` captures all parameter grads cleanly. Verify dropout RNG / 200-step loss curve <= 1%.
  - **Why R1b (`TRAIN_GROUPED_BANDS=1`) was 18 ms slower**: `bands.py:55` disables `_group_cache` in training and runs `torch.stack` across 62 weights/biases/gammas every step + backprops through `stack`. Capturing the ungrouped 62-band loop inside `make_graphed_callables` eliminates the Python launch overhead with zero extra parameter copies.

---

## 3. R3 `npu_rotary_mul` — Verified Layout & Exactness

- Verified live on 950PR (`B=1488, T=129, H=2, D=64`, bf16):
  - `apply_rotary_emb_fast` returns `[even | odd]` halves (`rel=1.39` vs `npu_rotary_mul`).
  - `torch_npu.npu_rotary_mul(x, cos2, sin2, rotary_mode='interleave')` returns interleaved `[e0, o0, e1, o1, ...]` (`rel=0.0028` bf16 vs interleaved reference).
  - Because **only Q and K are rotated** (never V) and both receive the exact same head-dim permutation, `Q @ K^T` and attention output/gradients are **mathematically identical**.
- **Implementation detail**: cache `cos2 = cos.repeat_interleave(2, -1).contiguous()` and `sin2 = sin.repeat_interleave(2, -1).contiguous()` inside `cached_rotary_cos_sin` (so `repeat_interleave` runs once, not per step). Pass `(B, T, H, D)` + `(1, T, 1, D)` directly.

---

## 4. R4 Triton-Ascend 3.6.0 — Verified APIs, Alignment & Shape Priority

- **Shape priority on H4s**: only **L1** is large (`N=1488, C=384, T=258, K=31`). L2 is 4x smaller (`N=768, T=129, K=15`), and L3/L5 are 16x smaller (`N=768, T=65, K=9`). Optimize L1 (and L2) first!
- **Immediate 0-code baseline**: sweep `S2_TRITON_BMM_CHUNK=64, 128, 256, 512` (H4s uses only 17.3 GB HBM; `chunk=8` launches 744 `AsStrided + BatchMatMulV3` loops; `chunk=256` cuts that to ~24).
- **Verified `triton.language.extra.cann.extension` APIs in 3.6.0**:
  - `ext.extract_slice(ful, offsets=(0, k), sizes=(BC, T), strides=(1, 1))`
  - `ext.insert_slice(ful, sub, offsets=(0, k), sizes=(BC, 1), strides=(1, 1))`
  - `ext.conv1d(input, weight, bias=None, stride=1, padding_size='same'|int, dilation=1, groups=...)` — takes 2D `(C, L)` or 3D `(N, C, L)` block + 3D weight `(C_out, C_in//groups, K)`. Worth a quick microbench against manual `extract_slice` UB-loop!
- **32B alignment & UB knobs**:
  - `T=258` is NOT 16-aligned (516 B) and `K=31` is not pow2. Pad load width to `T_PAD=288` (if 3.6.0 accepts non-pow2 `tl.arange`) or `512` with mask, and `K_PAD=32`.
  - `NPUOptions` already defaults `multibuffer=True` and `enable_buffer_insert_optimization=True`. Useful knobs to test: `enable_ubuf_saving=True`, `enable_preload=True`, env `TRITON_ALL_BLOCKS_PARALLEL=1`, and `TRITON_MEMORY_DISPLAY=1` to inspect exact UB allocation.

---

## 5. H2D Check (Don't Trust Unpinned `profile_step_npu.py` 59 ms)

- H3 `PROFILE.md` reported `h2d = 59.2 ms` because `profile_step_npu.py:162` allocates **pageable** CPU tensors (`batch('cpu', 7)` without `.pin_memory()`), whereas real training (`stage.py`) uses `--pin-memory`.
- Measure pinned `non_blocking=True` H2D on 950PR (50.6 MB should be ~4-5 ms over PCIe, not 59 ms). If real Lightning training still stalls on H2D (>5 ms in wall step vs synthetic step), overlap it with a dedicated H2D `torch.npu.Stream()` (`torch_npu.contrib.module.Prefetcher` pattern).
