# CANN 算子审计：我们的 STFT / 复数 / fold 路径到底走的是什么

> 只读审计，不改训练代码。证据来源三类：
> 1. **本机 CANN 安装**（`/usr/local/Ascend/ascend-toolkit/latest`，CANN 9.2.1）的头文件、`opp` 算子配置、
>    `libopapi*.so` 导出符号；
> 2. **gitcode 源码**（`cann/ops-nn`、`cann/ops-math`、`Ascend/pytorch`，浅克隆在 `/workspace/cann-src/`）；
> 3. **实测**：`npu:0` 上逐算子跑一次，用 `torch.profiler` 数 device kernel（`results/cann_op_kernels.json`）。

## 1. 一句话结论

**950PR 上确实有原生 STFT kernel，而且我们一直在用它**（`aten::stft` → `aclStft` → 1 个 `STFT_*` kernel +
1 个 `MirrorPad`）。真正退化的是 **`aten::_fft_r2c`（6 个 kernel，走 DFT 矩阵乘）**、
**`aten::_fft_c2r`（12 个 kernel）** 和 **`aten::istft`（直接不可用）**。
也就是说：我们的 `fast_loss.py` 里那条 `torch.fft.rfft(...)` 的谱计算，比原生 STFT 慢 3 倍 kernel 数；
而 iSTFT 之所以要自己写伴随，是因为 CANN 根本没有 IStft 算子（不是我们想手写）。

## 2. CANN 9.2.1 里到底有哪些 FFT / STFT / 复数 / fold 算子

### 2.1 头文件（`x86_64-linux/include/aclnnop/`，987 个头文件里只有这些相关）

| 头文件 | 导出 API | 说明 |
|---|---|---|
| `acl_stft.h` | `aclStftGetWorkspaceSize` / `aclStft` | **原生 STFT**，`@domain aclnn_fft`，输入含 `window`，支持 `hopLength/nFft/normalized` |
| `acl_rfft1d.h` | `aclRfft1DGetWorkspaceSize` / `aclRfft1D` | **原生实序列 1D FFT** |
| `aclnn_complex.h` | `aclnnComplex` | real+imag → complex |
| `aclnn_real.h` | `aclnnReal` | complex → real |
| `aclnn_im2col.h` / `aclnn_im2col_backward.h` | `aclnnIm2col` / `aclnnIm2colBackward` | im2col / col2im（`F.fold` 走这个） |
| `aclnn_unfold_grad.h` | `aclnnUnfoldGrad` | unfold 的反向 |
| `aclnn_ops_math.h` / `aclnn_ops_cv.h` | — | **没有** `aclnnFftC2C/FftR2C/FftC2R/IStft` 任何条目 |

**没有 `aclannIStft` / `aclnnIStft`，也没有任何 C2C FFT 的 aclnn 入口。** 只有 `aclStft` + `aclRfft1D` 两个。

### 2.2 算子二进制配置（决定 950PR 有没有真 kernel）

`opp/built-in/op_impl/ai_core/tbe/kernel/config/<arch>/` 下按算子分目录，只有 `ascend950/` 里有：

| 算子 | 目录 | 950PR kernel | 支持 dtype（来自 binList / def） |
|---|---|---|---|
| `Stft` | `ops_math/stft.json`（4 个 bin） | **有** | x: float32/**complex64**；y: float32/complex64；window/plan: float32 |
| `Rfft1D` | `ops_math/rfft1_d.json`（1 个 bin） | **有** | x/dft/y: **float32 only** |
| `Complex` | `ops_math/complex_apt.json` | **有** | real/imag fp16/fp32 → complex32/complex64 |
| `Real` | `ops_math/real_apt.json` | **有** | complex32/64 → fp16/fp32 |
| `ComplexAbs` | `ops_math/complex_abs_apt.json` | **有** | complex32/64 → fp16/fp32 |
| `Im2col` | `ops_math/im2col_apt.json` | **有** | fp16（binList）+ bf16/fp32/int8…（def） |
| `UnfoldGrad` | `ops_math/unfold_grad.json` | **有** | bf16/fp16/fp32 |
| `Col2Im` | `ops_cv/col2im.json` | **有** | bf16/fp16/fp32 |

`ascend910b` / `ascend910_93` 的 config 里**一个 FFT/STFT 算子都没有**（所以这套算子是 950 专属）。

### 2.3 源码侧（gitcode，`/workspace/cann-src/`）

| 算子 | 所在库 | `op_host/config/` 架构目录 | def 里声明的 dtype |
|---|---|---|---|
| `Stft` | `ops-math/math/stft` | ascend910_93, ascend910b, **ascend950** | float32 / double / **complex64** |
| `Rfft1D` | `ops-math/math/rfft1_d` | ascend910_93, ascend910b, **ascend950** | **float32 only** |
| `Complex` | `ops-math/math/complex`（含 `op_api/`） | **ascend950** | fp16/fp32 → complex32/64 |
| `ComplexAbs` | `ops-math/math/complex_abs` | ascend350, **ascend950**, ascend960dt | complex32/64 → fp16/fp32 |
| `Im2col` | `ops-math/conversion/im2col`（含 `op_api/`） | ascend350, **ascend950**, 960dt/pr | bf16/fp16/fp32/int* |
| `UnfoldGrad` | `ops-math/conversion/unfold_grad`（含 `op_api/`） | 910_93, 910b, **ascend950**, 960dt/pr | bf16/fp16/fp32/int64 |
| `Col2Im` | 只有 950 的 kernel config，**ops-math 里没有源码目录** | — | bf16/fp16/fp32 |
| `IStft` | **两个库里都不存在** | — | — |
| `FftC2C/R2C/C2R` | 本版本**不在开源库里**（`ops-nn` 无 FFT 目录） | — | — |

`ops-metransform` / `cann-ops-adv` 两个仓库在 gitcode 上 **不存在**（clone 返回 403/项目未找到），
所以 Fold/Unfold 相关的实现只在 ops-math 的 `conversion/` 下。

**Stft 只有 fp32/fp64/complex64，没有 bf16/fp16** → 想在 bf16 autocast 下用原生 STFT，必须显式转 fp32。
`Rfft1D` 更是 **只有 fp32**。

## 3. torch_npu（Ascend/pytorch）里 aten → aclnn 的映射

映射表来自安装包里的生成清单
`torch_npu/csrc/aten/npu_native_functions_by_codegen.yaml`（3294 条），并用 `nm -D` 核对了
`libopapi.so` / `libopapi_math.so` 的导出符号。

| aten 算子 | 有 NPU 实现 | 注册方式 | 背后 aclnn/acl | dtype | 950 是否退化 |
|---|---|---|---|---|---|
| `aten::_fft_r2c` (+`.out`) | 是 | `impl_ns=op_api` | 无 aclnn 直调 → **DFT 矩阵乘分解** | fp32 | **退化（6 kernel）** |
| `aten::_fft_c2c` (+`.out`) | 是 | `op_api` | 同上，MatMulV3/BatchMatMulV3 | complex64 | 退化（4 kernel） |
| `aten::_fft_c2r` (+`.out`) | 是 | `op_api` | 同上 | fp32 | **重退化（12 kernel）** |
| `aten::_fft_r2c_backward` | 是 | `torch.ops.npu.fft_r2c_backward` | 手写 | — | — |
| `aten::_fft_c2c_backward` | **否** | — | — | — | 走 autograd 组合 |
| `aten::stft` (+`.out`) | 是 | `op_api` | **`aclStft`**（`STFT_*` kernel） | fp32/fp64/complex64 | **原生（2 kernel）** |
| `aten::stft_backward` | 是 | `torch.ops.npu.stft_backward` | 手写 | — | — |
| `aten::istft` | **否** | — | — | — | **不可用（RuntimeError）** |
| `aten::unfold` | 视图算子 | `npu_native_functions.yaml` 的 supported 列表 | 无 kernel（0） | 任意 | 不退化（0 kernel） |
| `aten::unfold_backward` | 是 | `op_api` + `aclnnUnfoldGrad` | `aclnnUnfoldGrad` | bf16/fp16/fp32 | 原生 |
| `aten::col2im` / `aten::fold` | 是（注册了） | `acl_op` → `aclnnIm2colBackward` | `aclnnIm2colBackward` | — | **实测报错**（见 §4） |
| `aten::im2col` | 是 | `acl_op` → `aclnnIm2col` | `aclnnIm2col` | fp16/bf16/fp32 | 原生 |
| `aten::view_as_real` / `view_as_complex` | 是 | 视图算子 | 无 kernel（0） | — | 不退化 |
| `aten::_conj` | **否** | — | — | — | 走 view/neg 组合 |
| `aten::complex` | 是 | `acl_op` → `aclnnComplex` | `aclnnComplex` | fp16/fp32→complex | 原生（1 kernel） |
| `aten::real` / `aten::imag` | **否**（未注册） | — | —（`aclnnReal` 存在但没接） | — | 需自己取 `.real`/`.imag` |
| `aten::reflection_pad1d` | 是 | `acl_op` → `MirrorPad` | MirrorPad | 全部 | 原生（1 kernel） |

### torch_npu 暴露的 NPU 专用接口（审计要求第 4 点）

- `torch.ops.npu` 共 **369** 个算子，其中与 FFT/STFT/复数相关的只有三个**反向**算子：
  `torch.ops.npu.fft_c2r_backward`、`torch.ops.npu.fft_r2c_backward`、`torch.ops.npu.stft_backward`。
- `torch_npu.npu` 下只有 `_fft_plan_cache`（`NPUFFTPlanCache` 类），没有 `npu_fft` / `npu_stft` / `npu_fast_fft`。
  这个 plan cache 佐证了正向前向确实走 **plan 式原生 STFT**（CANN `Stft` 算子的第 2 个输入就是 `plan`）。
- **没有** `torch.ops.npu.aclnn_*` 这种 aclnn 直调入口 → 想绕过 aten 直接调 `aclStft`，只能写 C++ 扩展
  （`aclStftGetWorkspaceSize` + `aclStft`，见 `python3.12.16/.../torch_npu/include`）。Python 侧没有路。

## 4. 实测：每个 aten 算子实际下发多少 device kernel

`npu:0`，形状按我们的训练实际用（`bs24`, 8 s, n_fft 2048, hop 512），
每个算子跑 3 次取平均（`results/cann_op_kernels.json`）：

| aten 调用 | device kernel/次 | 判定 | 头部 kernel |
|---|---|---|---|
| `torch.stft(return_complex=True)` | **2.00** | **原生** | `STFT_…`, `MirrorPad_…` |
| `torch.stft(return_complex=False)` | **2.00** | **原生** | 同上 |
| `torch.fft.rfft` (`_fft_r2c`) | 6.00 | 分解（DFT matmul） | `Transpose_float32_int64`, `Slice`, `MatMulV3_FP32` |
| `torch.fft.fft` (`_fft_c2c`) | 4.00 | 分解 | `Transpose`, `MatMulV3`, `BatchMatMulV3` |
| `torch.fft.irfft` (`_fft_c2r`) | **12.00** | 重退化 | `Transpose`, `BatchMatMulV3`, `PadV3` |
| `torch.istft` | **失败** | 不可用 | `RuntimeError: istft(...) expected a length >= ...` |
| `Tensor.unfold` | **0.00** | 纯视图（零 kernel） | — |
| `view_as_real` / `view_as_complex` | **0.00** | 纯视图 | — |
| `complex` 乘法（`complex_mul`） | 1.00 | 原生 | `Mul_…` |
| `reflection_pad1d` | 1.00 | 原生 | `MirrorPad_…` |
| `conv1d`(depthwise k=7) | 3.00 | 原生 | `Conv2DV2`（+normal） |
| `F.fold` / `col2im` | **失败** | 不可用 | `aclnnIm2colBackward failed, error code is 161002` |
| `as_strided`（跨步切片） | **设备故障** | 会打挂 vector core | `aivec error, error code = 95`（MTE invalid GM） |

两条重要负结果：
- **`F.fold`（=col2im）在我们这个形状上直接报 `161002`**，所以 overlap-add 只能用自己写的
  `index_add_`/`scatter`（这也是我们早先必须手写 getattr 的原因）。
- **`as_strided` 手工帧化会打挂 vector core（error 95）**；`unfold` 是 0 kernel 的纯视图，
  用它 + `reshape` 才是安全写法。

## 5. 对我们当前实现路径的判定

| 我们代码里的位置 | 实际走什么 | 评价 |
|---|---|---|
| `train.py::_stft_fc`（数据预处理） | `torch.stft` → **原生 `aclStft`（2 kernel）** | 已经是最优，无需改 |
| `fast_loss.stft_mag_fast`（MR-STFT 谱） | `torch.fft.rfft` → **6 kernel DFT-matmul** | **可换原生 STFT（2 kernel）** |
| 谱的取模 `spec.abs()` | `abs` on complex64 | 1 kernel，无优化空间 |
| `stft_adjoint`（手写伴随） | `irfft` 的**自己实现**（real-plane transpose） | 必须自己写：`_fft_c2r` 12 kernel、`istft` 不可用 |
| iSTFT 重建（`fast_istft`） | 自己写的 fold（`index_add_`） | 必须自己写：`F.fold` 报错 |
| band conv → matmul（`_band_down/up_matmul`） | 手工 4 次移位 matmul | 已经比 conv 慢，默认关闭（实测见 REPORT 第 49 节） |

**结论**：唯一"有原生算子却没吃上"的就是 **STFT 谱本身** —— 我们把 `rfft`（6 kernel，
DFT 矩阵乘）当成了首选，而 950PR 的原生 `aclStft` 只有 2 kernel。这一项是 A 优化（见下）的主线。

## 6. 优化方向（实测数据在 `results/speedup_final3.json`，此处给计划）

| 方向 | 依据 | 预期 |
|---|---|---|
| A1 把 MR-STFT 的谱从 `torch.fft.rfft` 换成 `torch.stft`（原生 `aclStft`） | 6 kernel → 2 kernel；且 STFT 算子内部就是我们手写的 pad+frame+FFT | 损失 fwd+bwd 大幅下降 |
| A2 iSTFT 保留手写伴随 | `_fft_c2r`=12 kernel、`istft` 不可用、`F.fold` 报错 | 无法用原生，只能优化自己的实现 |
| A3 2D rfft 合并多分辨率 | 每个分辨率独立调用 → 每个 6 kernel；合并成一次 2D FFT 可省 (N-1)×5 kernel | 待测 |
| B 反向 elementwise 合并 | 见 `REPORT.md` 第 19–26 节 profile | 待测 |
| C H2D 减半（target 谱预计算存 fp16） | 数据管道 543 MB → ~272 MB | 待测 |
| D 损失进 bf16 autocast | **原生 Stft/Rfft1D 不支持 bf16**（§2.3），必须 fp32 进算子；只能是"外圈 bf16、STFT 内 fp32" | 收益有限，风险高 |
| — | 直接用 `aclStft` 绕过 aten | 需要写 C++ 扩展，Python 无入口（§3 末） |

**不要做的事**：`as_strided` 帧化（打挂设备）、`F.fold` overlap-add（161002）、`aten::istft`（不存在）。
