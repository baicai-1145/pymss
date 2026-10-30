# 国产 GPU 适配说明（Chinese accelerator support）

pymss 的模型全部是纯 PyTorch 算子（`torch.stft` / `torch.fft` / `nn.*`），没有手写 CUDA/HIP
扩展，所以移植的关键不是编译算子，而是**两件事**：

1. **torch 版本下限**：本分支把 `torch>=2.7.1` 降到 **`torch>=2.4`**（代码里唯一需要 ≥2.4 的 API 是
   `torch.amp.autocast(device_type=...)`，2.4 才引入；对 <2.4 还有 `try/except` 回退）。
2. **设备抽象**：各家 SDK 对 `torch` 暴露方式不统一（有的私有 device type，有的直接冒充 CUDA），
   所以设备判断统一走 `pymss_core.device` 的能力档案，不再硬编码 `device.type == "cuda"/"npu"`。

## 支持矩阵

| 厂商 | 厂商软件栈 | 适配 torch | `--device` | torch 来源 | 已知限制 |
|---|---|---|---|---|---|
| 壁仞 Biren | BIRENSUPA / BRPyTorch | 2.4 – 2.8 | `biren`（或 `supa`） | `torch_br` + `torch 2.x+br` | 老版本 SUPA 把设备暴露为 `cuda`，新版本为 `supa`；**实测 BIREN106M + torch-br 1.7.0.20700（torch 2.7.0+cpu，31.75 GiB）**：`stft(return_complex=True)` 报不支持 → 走 `torch.fft` 组合（频谱 2.3e-05 / 波形 1.2e-06）；`hann_window(device=supa)` 崩 → 必须 CPU 建窗再搬；**融合 `_VF.lstm` 数值错误**（0.18–0.47，手写 cell 才对 2e-06）→ 所有 RNN 回退 CPU；**无 fp16**（br10x matmul 只吃 bf16/fp32/s8/u8）→ autocast 用 bf16 |
| 天数智芯 Iluvatar | CoreX SDK 4.4 | **2.7.1** | `iluvatar` | `torch 2.7.1+corex.4.4.0` | 冒充 `cuda`；FFT 走原生路径，若报错用 `PYMSS_FFT_FALLBACK=1` |
| 沐曦 MetaX | MACA 3.x | 2.6 – 2.8 | `metax` | `torch 2.x+metax` | 冒充 `cuda` |
| 燧原 Enflame | TopsRider 3.5 – 3.7 | 2.6 – 2.10 | `enflame`（或 `gcu`） | `torch_gcu` | 私有 device type `gcu`；`transfer_to_gcu` 可让 CUDA 代码直接跑 |
| 海光 Hygon DCU | DTK 25.04.x + DAS | **2.4.1** | `hygon` | `torch 2.4.1+das.opt1.dtk25041` | 冒充 `cuda`（HIP）；**官方适配版最高 2.4.1** |
| 寒武纪 Cambricon MLU | torch_mlu（分支到 `pt2.4.0`） | **2.4** | `mlu` | `torch_mlu` | 私有 device type `mlu`；wheel 不公开分发 |
| 华为昇腾 Ascend | CANN + torch_npu | 2.7.1+ | `npu` | `torch_npu` | 已在基线实现：STFT 走 `torch.fft` 组合、LSTM/GRU 走 CPU 回退 |
| NVIDIA / AMD ROCm | 官方 wheel | 2.4 – 2.10 | `cuda` / `rocm` | PyPI / rocm 索引 | — |
| Apple Silicon | MLX / MPS | 2.4+ | `mps` / `mlx` | PyPI | 用 MLX 全模型后端 |

> 结论：**`torch>=2.4` 是能覆盖全部 7 家的唯一下限**。海光、寒武纪停留在 2.4，其余四家都已在 2.6~2.10 区间。

## 安装

厂商适配版 torch 必须**先装**，且不要被 PyPI 公版覆盖：

```sh
# 举例：天数智芯（corex）
pip install torch==2.7.1+corex.4.4.0 --no-deps        # 用厂商提供的 whl 或官方镜像
pip install --no-deps pymss                          # 只装 pymss，避免依赖解析拉公版 torch

# 举例：海光 DCU
pip install torch==2.4.1+das.opt1.dtk25041 --no-deps
pip install --no-deps pymss
```

在带厂商 SDK 的官方镜像里通常已经预装好 torch，直接：

```sh
pip install --no-deps pymss
```

## 使用

```sh
pymss infer --model <name> --input mix.wav --device iluvatar   # 天数
pymss infer --model <name> --input mix.wav --device biren      # 壁仞
pymss infer --model <name> --input mix.wav --device hygon      # 海光
pymss infer --model <name> --input mix.wav --device gcu        # 燧原
pymss infer --model <name> --input mix.wav --device mlu        # 寒武纪
```

`--device auto` 的探测顺序：CUDA/ROCm → Ascend NPU → GCU → MLU → SUPA → MPS → CPU。
若某个厂商后端在 `torch.cuda.get_device_name()` 里返回的字符串无法识别，可用环境变量强制指定：

```sh
PYMSS_DEVICE_BACKEND=iluvatar pymss infer --device auto ...
```

## 调试开关

| 变量 | 作用 |
|---|---|
| `PYMSS_DEVICE_BACKEND` | 强制厂商识别（`biren` / `iluvatar` / `metax` / `enflame` / `hygon` / `mlu` / `ascend` / `nvidia` / `rocm`） |
| `PYMSS_FFT_FALLBACK=1` | 强制用 `torch.fft` 组合实现 STFT/ISTFT（厂商 complex 支持不完整时用） |
| `PYMSS_RNN_FALLBACK=1` | 强制 LSTM/GRU/RNN 走 CPU 回退 |
| `PYMSS_AMP=0` | 关闭 autocast（厂商 fp16 kernel 不稳时用） |

## 调试任务：脚本化操作容器

开卡只能走网页，但开出来之后的 JupyterLab 有标准 REST + websocket API，
用 `tools/jlab_exec.py` 就能不看屏、不点鼠标地跑命令：

```sh
export JLAB_BASE=https://<host>/notebook_<hash>_task0     # 调试任务的 JupyterLab 地址
python tools/jlab_exec.py --sh 'pip list | grep -Ei "torch|supa"'
python tools/jlab_exec.py -f tools/cn_gpu_probe.py
python tools/jlab_exec.py --new-kernel --sh 'brsmi'        # kernel 被长命令占住时换一个
```

两个必须知道的细节：POST 需要 `_xsrf` cookie（否则建 kernel 返回 403）；
公网 ingress 会偶发重置 TLS，客户端已带重试。

性能测量用 `tools/cn_gpu_bench.py`：它会拒绝在有别的推理进程时开跑（同卡并发
会把数据污染成完全错误的结论），并且只在小形状上跑 CPU 对照组（大分辨率
CPU 卷积的 im2col 会吃掉几十 GB 内存，会直接把宿主 OOM）。

## 在 OpenI 云脑任务里一键采集证据

仓库自带 `tools/cn_gpu_probe.py`，可直接作为**训练任务的启动文件**（无人值守、不限时长），
结果写进 `c2net_context.output_path` 并 `upload_output()` 回传，在任务详情「结果下载」里取：

```sh
# 训练任务配置
#   资源：目标厂商卡型（如 壁仞 1*BIREN106M）
#   镜像：厂商 SDK 镜像
#   访问Internet：是（需要 pip / git clone）
#   代码分支：包含本文件的适配分支
#   启动文件：tools/cn_gpu_probe.py
#   运行参数：--device <alias>            # 例：--device enflame
#             --install-tests             # 首次跑时装 pytest/librosa 等
#   （默认会自己 pip install pymss-core：优先平台内 git，回落 GitHub，需 Internet=是）
```

脚本会依次采集：环境与 torch 构建信息 → 已安装厂商栈（`torch/+supa/+corex/+metax/+gcu/+das/+mlu`）
→ pymss 设备档案 → STFT/ISTFT 原生与回退双路往返误差 → LSTM 原生与 CPU 回退 →
`pytest` 摘要，最后写出 `cn_gpu_report.txt` 与 `cn_gpu_profile.json`。

## 壁仞 BIREN 实测记录（BIREN106M / torch-br 1.7.0.20700）

环境：`torch 2.7.0+cpu` + `torch-br 1.7.0.20700+br1xx` + SUPA SDK `1.9.0.0.rc1`，
驱动 1.10.0，31.75 GiB 显存。模型 `UVR-De-Echo-Aggressive`，10 秒 44.1 kHz 单声道。

### 一、性能：每个新形状都要付一次 220 秒

同一张卡，同一个 3×3 fp32 卷积：

| 形状 | 首次调用 | 稳态中位数 |
|---|---:|---:|
| `(2,144,42,64)` | 3 320 ms | 0.27 ms |
| `(2,192,21,32)` | 979 ms | 0.13 ms |
| `(2,2,1025,860)` | **223 350 ms** | 10.66 ms |
| `(2,32,1025,860)` | **220 838 ms** | 14.88 ms |
| `linear [1024,1024]` | 1 821 ms | 0.15 ms |

稳态是毫秒级，首次调用是分钟级 —— 时间全花在 SUDNN 为新形状挑选 kernel 上。
分离模型的每个 chunk 形状都不同，所以一次 10 秒推理的 388 秒里有 380 秒是这件事，
期间 `brsmi` 显示 **GPU-Util 0%、CPU 100%**（主机侧单线程挑选，卡在等）。

逐个开关实测（每项独立进程，75 秒硬超时，`(2,32,1025,860)` fp32）：

| 开关 | 首次调用 | 结论 |
|---|---:|---|
| 默认 | >75 000 ms | — |
| `BRTB_DISABLE_HEURISTIC_SEARCH=1` | >75 000 ms | **无效**（名字看着像正解，实测不动） |
| `BRTB_ENABLE_BUILD_CACHE=1` | >75 000 ms | 无效 |
| `BRTB_ENABLE_OP_FRONTEND_CACHE=1` | >75 000 ms | 无效 |
| `BRTB_ENABLE_FAST_MAKE_KERNEL_KEY=1` | >75 000 ms | 无效 |
| **`BRTB_ENABLE_FORCE_EAGER_CONV2D=1`** | **1 790 ms** | ✅ 快 125 倍，稳态 21.0 ms（慢 40%，可接受） |

因此 `pymss_core.device._VENDOR_ENV_DEFAULTS` 对 BIREN 默认设
`BRTB_ENABLE_FORCE_EAGER_CONV2D` / `..._CONV2DTRANSPOSE=1`，在 `import torch_br` **之前**注入。
用户自己设了值就不覆盖，`PYMSS_VENDOR_ENV=0` 可整体关闭。

端到端：**388 秒 → 10 秒**（`UVR-De-Echo-Aggressive`，10 秒输入）。

### 二、算子缺口（都是实测，不是猜）

| 操作 | 厂商行为 | pymss 的处理 |
|---|---|---|
| `stft(return_complex=True)` | 报 "does not support return_complex=True now" | 走 `torch.fft` 组合（`npu_fft.stft_npu`） |
| `hann_window(device=supa)` | kernel 崩 | CPU 建窗再搬（`hann_window_npu`） |
| `complex64 * float32` | `pointwise No SUPA Function found` | 在实/虚平面上相乘（`vendor_shims.complex_mul`） |
| `complex64.reshape` | **静默污染设备状态**（`dev -1 is not a device`），报错指向下一个无辜算子 | `Tensor.reshape/view` 走实/虚平面（`vendor_shims.complex_reshape`） |
| `complex64` 的 `torch.cat` | `Concat.cpp:68 ErrorCode: 11`（SUDNN） | 平面上拼接再重组（`vendor_shims.complex_cat`） |
| `complex64.abs()` | `Complex tensor op is not supported now` | 用平面求模（apollo 的 band power） |
| 复数 **D2H**（`complex.cpu()`） | `ErrorCode: 101, invalid device` | 只搬 `view_as_real` 的实数平面（`complex_to_numpy_cpu`） |
| 复数 **H2D**（`.to(device)`） | `Unsupported BR type for ScalarType: 0` | 实数平面上卡、卡上重组（`numpy_complex_to_device`） |
| `F.pad(mode="replicate")` | 警告"回退 CPU"，实际 SUDNN `ReorderWei2Plain() FAILED` + `suError: 101` | `cat`+`expand` 复现 replicate（`replicate_pad_bins`） |
| `F.pad`（band 轴补零） | `Unsupported BR type for ScalarType: 0` | 显式 `cat` 零块 |
| **分组 conv1d**（`1 < groups < in_channels`） | `Conv1d.cpp:120` **801 operation not supported**（`groups=1` 与 depthwise 反而正常） | 按组拆分逐组卷积再 `cat`（`conv1d_grouped_equivalent`，逐位相同） |
| `F.group_norm`（部分通道数） | `GroupNorm.cpp:43` **801** | reshape+mean+var 重建（误差 1e-7） |
| 5 维 `index_add_` | 厂商内部 `vector::reserve` abort | one-hot 矩阵累加（`scatter_add_equivalent`，逐位相同） |
| `scatter_add` / `F.glu` | 无内核 | 数学等价实现 |
| fp16 | "Current SUPA Device does not support float16" | 用 bf16 或 fp32（文档亦写明 SDPA 只支持 BF16/FLOAT32） |
| bf16 的 `F.interpolate(bilinear)` | `no valid backend was found for op: UpsampleBilinear` | 该算子单独用 fp32 算再转回 bf16（`vendor_shims.interpolate`），**AMP 保持开启** |
| 融合 `_VF.lstm` | **静默算错** 0.18–0.47 | RNN 回退 CPU（手写 cell 只差 2e-06） |
| 膨胀卷积（SUDNN 路径） | `terminate()` 杀进程 | 见下节 |
| **`torch.fft`（1024 / 16 点）** | **静默返回全零**，无异常 | 结果级校验：全零即用精确 DFT 矩阵重算 |
| **`torch.fft`（4096 点）** | 首次调用 **600 秒**（plan/autotune），随后报 SUPA 错 | 白名单只放行 2048；其余走 DFT 矩阵（`PYMSS_VENDOR_FFT_SIZES` 可覆盖） |

> 关于 FFT：`torch.fft.*` 在厂商文档里是**明确不支持**的（chapter05 API 限制表）。
> 实测它部分可用，但正确性与性能都依赖形状 —— 所以只在**已证明既正确又快**的
> 尺寸（2048）上用厂商实现，其余一律用精确 DFT 矩阵，并保留结果级校验兜底。

### 二·补：模型家族实测支持矩阵

验收标准是三条同时成立（`tools/vendor_model_verify.py`）：与 CPU 参考的
相对误差 `rel`、卡上两次运行的一致性 `repeat`、输出非零元素数。**只看"跑通"不够** ——
这张卡曾经在 FFT 上静态返回全零，跑通但结果全零的情况真实发生过。

| 家族 | rel | repeat | 结论 |
|---|---:|---:|---|
| `legacy_demucs` | 1.6e-05 | 0 | ✅ |
| `htdemucs` | 4.4e-05 | 8.7e-08 | ✅ |
| `bs_conformer` | 4.7e-05 | 0 | ✅ |
| `mel_band_conformer` | 5.3e-05 | 2.2e-08 | ✅ |
| `legacy_hdemucs` | 6.1e-05 | 1.2e-07 | ✅ |
| `bs_roformer_hyperace` | 6.7e-05 | 5.3e-08 | ✅ |
| `mdx23c` | 7.3e-05 | 8.7e-08 | ✅ |
| `bs_roformer` | 7.5e-05 | 1.2e-08 | ✅ |
| `legacy_tasnet` | 9.9e-05 | 0 | ✅ |
| `mel_band_roformer` | 1.4e-04 | 0 | ✅ |
| `apollo` | — | — | 🔧 复数 cat / 分组卷积已修，待复测 |
| `bandit` / `bandit_v2` | — | — | 🔧 mask 累加在卡上形态异常，定位中 |
| `scnet` / `legacy_v3_demucs` | — | — | 🔧 单算子黑洞，定位中 |

`repeat ≈ 1e-08` 说明卡上运算是**确定性**的 —— 这条必须测，因为早期版本里
退化 FFT 路径会让同输入两次结果不同（0.272 vs 0.334），看起来像精度问题。

### 五、JIT 启动环境问题（原“编译器不可用”结论已撤回）

早期探针以 `PYTHONPATH=/tmp/pc` 覆盖 SDK 环境，导致 JIT 所需的
`br_generator` 模块不可见。以下错误来自该启动环境，不能据此认定编译器损坏：

```
[SUDNN] Failed to init python
[SUDNN] Failed to build the assembly code
[SUDNN] [Kernel] FAILED to compile, since initialize Kernel.
ERROR (SUDNN): SudnnPlanBuilder.h:667 Failed to finalize engine config descriptor
```

调用栈（`/tmp/logs/torch_br-*.log`）：

```
at::supa::do_sudnn_matmul -> at::supa::PtOpFrontend::run -> SUDNNException
```

**历史复现**：以下形状在 SDK 路径被移除时失败。这不是其支持上限；
重现时必须记录并保留原始 `PYTHONPATH`：

```python
import torch, torch_br
torch.supa.set_device(0)
d = torch.device("supa:0")
torch.bmm(torch.randn(1384, 62, 64, device=d), torch.randn(1384, 64, 62, device=d))
```

**历史尝试（SDK 路径修复前）**：以下结果不能用于排除厂商能力，需在正确环境重测。

| 尝试 | 结果 |
|---|---|
| 把 batch 分块（8 / 64 / 173 / 256） | 无效 |
| 操作数 `.contiguous()` | 无效 |
| 4-D 批 matmul 折叠成 3-D bmm | 能改变命中的内核，但不能保证命中预编译库 |
| bf16 autocast | 无效 |
| `BRTB_KERNEL_CACHE_CAPACITY` / `..._DEVICE_MEMORY_MAX_SIZE_MB` | 无效 |
| `BRTB_LOG_DIR` / `BRTB_LOG_LEVEL` / `BRTB_ENABLE_RUN_KERNEL` | 无效 |
| 失败后重试 | 无效（与 scnet 的 D2H 不同） |
| 重启 Jupyter 内核 | 无效 |

**历史观察**：同一容器曾出现某些 bmm 尺寸成功、后来持续失败，而大 GEMM、卷积和
FFT 正常。此前把它归因于容器随时间退化，没有控制启动环境；这一因果结论已撤回。

**已验证修复**：保留 SDK 路径后，完全相同的 3-D bf16 FMHA 最小复现成功，
预热后三次为 0.686 / 0.407 / 0.386 ms。生产规模 bf16 roformer 的显式 FMHA
也成功执行，三轮每轮 24 次厂商调用，零失败、零回退。
启动方式应使用 `PYTHONPATH="/tmp/pc:${PYTHONPATH:-}"`；各工具创建子进程时也必须追加
测试路径而不是覆盖环境。详细数据见文末“环境修复后的验证”。

### 三、CPU 回退的取舍原则

> **只有厂商"崩进程"或"算错结果"才允许回退 CPU；"慢"必须留在卡上解决。**

按这条原则回查：膨胀卷积的 CPU 回退**已解除**。实测（eager 卷积路径，fp32，对比 CPU 参考）：

| padding / dilation | 输出形状 | max err |
|---|---|---:|
| (0,1) / (1,2) | (1,8,62,62) | 5.8e-05 |
| (1,0) / (1,2) | (1,8,64,60) | 6.4e-05 |
| (2,1) / (1,3) | (1,8,66,60) | 6.3e-05 |
| (1,2) / (2,1) | (1,8,62,66) | 5.9e-05 |
| (2,2) / (2,2) 对称 | (1,8,64,64) | 5.5e-05 |

当年的 abort 来自 SUDNN 的 plan builder，eager 内核根本不经过它。
`dilated_conv_blocked()` 因此在 eager 路径启用时直接返回 `False`，
只有用户关掉 eager 时才保留旧的白名单作为安全网。

其余仅存的回退只有 **RNN**（算错，属于必须回退）。

### 四、几个容易踩的坑

- `torch.device("supa:0")` 只有在 `import torch_br` **之后**才存在，否则报
  "Expected one of cpu, cuda, ... device type"。
- 设备串不带 index（`torch.device("supa")`）会解析到"当前设备"，而 PrivateUse1
  后端不会自己设置它 → 各种 `ErrorCode: 101, invalid device`。
  `prepare_device_alias()` 因此会调用厂商的 `set_device(0)`（可用 `PYMSS_DEVICE_INDEX` 覆盖）。
- SUDNN 的报错是**异步**的：算子提交时失败，要到下一个同步点（`.cpu()` / `.numpy()`）
  才抛出，所以 Python 栈常常指向一个无辜的后续算子。定位要逐个算子加
  `torch.supa.synchronize()`。
- 两个推理进程同时占一张 31.75 GiB 的卡（各 17 GiB）会让所有数字彻底失真
  （曾经测出 263 秒/batch 的假象），性能测量必须单进程。
- **厂商错误有两类污染**：一是异步（报错指向后续无辜算子），二是**失败后会把
  "当前设备"重置成 -1**（`complex.reshape` 的 SUDNN 分支就是这样），于是接下去任何
  算子都报 `dev -1 is not a device`。看到这个字符串先怀疑**上一个**算子。
- 有些失败**不抛异常**：`[SUDNN] STATUS_NOT_SUPPORTED ... date type 13`（dtype 13 是
  complex64）只打日志不报错，之后的卷积/FFT 静默返回全零。所以验收必须比对数值，
  不能只看"跑完没报错"。

## 真实模型全类型验证（2026-10-29）

工具 `tools/vendor_realmodel_check.py`：目录内 11 种模型类型各取一个
真实 checkpoint（ModelScope 自动下载），10s 合成立体声推理，逐模型
子进程隔离。**SUPA 可用 6/11**：mdx23c 28.7s、scnet 86.7s、apollo
65.5s、bandit 372s、bandit_v2 112s、vr 17.8s（均输出有限、stem 正确）。
本轮新增 shim 解锁：fold/col2im CPU 回退、conv1d dtype 对齐 +
fp32 重试 + CPU 三级阶梯、`view_as_complex` bf16→fp32、SDPA 非
bf16 回退 math、`F.linear` 全局 dtype 对齐、FFN 门放宽（eval 下
忽略 dropout p）。

未过 5 类（均已定位根因）：bs_roformer（LargeV1）/mel_band ——
遮罩路径的 complex-GEMM 无后端（dim-512 配置触发；dim-384 优化
路径不受影响）；htdemucs —— vendor 缺 Embedding 算子（待加 CPU
回退）；legacy_tasnet —— scatter shim 内 one_hot 矩阵乘 SUBLAS
ErrorCode 15（待重写为 index 形式）；legacy_demucs —— vendor conv
裁剪语义 bug（800vs860）。

## 真实模型全类型验证（2026-10-30，全部通过）

工具 `tools/vendor_realmodel_check.py`：目录内 11 种模型类型各取一个
真实 checkpoint（ModelScope 自动下载），10s 合成立体声推理，逐模型
子进程隔离。**SUPA 全部 11/11 可用**（输出有限、stem 数正确）：
mdx23c 28.7s、scnet 86.7s、apollo 65.5s、bandit 372s、bandit_v2 112s、
vr 17.8s、bs_roformer(LargeV1) 40.1s、mel_band 102s、htdemucs 39.7s、
legacy_demucs 9.8s、legacy_tasnet 23.6s（首轮均为未优化回退路径）。

修复链（详见 `.tmp/BR_OPTIMIZATION_STATE.md`，14 项）：F.linear
dtype 对齐 + 2D 拍平重试、fold/col2im CPU 回退、view_as_complex
升精度、SDPA 非 bf16 回退与跨长度 q/kv、scatter 换 index_add、
embedding 换 one-hot 矩阵乘（aten 钩子设备盲）、matmul 复数分解
守卫、band 分组逐组 2D、FFN 融合算子回退、LayerNorm 类补丁还原、
conv_transpose 阶梯、BLSTM 手写参考循环（含上游分帧丢尾 bug 修复）、
legacy 注意力 CPU 回退、htdemucs 频率嵌入广播加固。

关键平台规律：torch_br 对 nn.LSTM/nn.LayerNorm 类级打补丁且
aten::lstm/mm/embedding 设备盲路由（CPU 张量也进 SUPA IR）——
对这些算子只能参考实现重写；异步内核崩溃会以 SIGABRT 迟到暴露；
深层 stride 下 conv/slice 长度语义漂移需正索引切片+显式补齐。

## 开卡验证清单

在目标卡上按顺序执行（约 10 分钟）：

```sh
# 1. 确认厂商适配版栈（关键：别是 PyPI 公版 torch）
pip list | grep -Ei "torch|corex|metax|gcu|das|br|mlu|npu|dtk"

# 2. 确认后端能被识别
python -c "from pymss.backends import device_profile; print(device_profile('<alias>'))"

# 3. 单元测试（CPU 可跑，验证抽象层没坏）
pytest -q test/ -x -k "not live and not server"

# 4. 真实分离 smoke（10 秒音频）
pymss infer --model <name> --input test.wav --device <alias>
```

## 设计说明

- `pymss_core/device.py` 是**唯一**的设备能力来源：厂商识别、device type、插件导入、能力标志
  （`complex_fft` / `rnn` / `amp` / `channels_last` / `multi_device`）。
- `pymss/backends.py` 是 pymss 侧的门面，CLI 的 `--device` 取值、自动探测、RNN 回退安装都走它。
- 模型代码里的 STFT 分支统一用 `pymss_core.npu_fft.torch_stft_blocked(device)` 判断，
  它内部委托给设备档案，新增厂商只需改档案，不需要再改 6 个模型文件。
- 各家「冒充 CUDA」的后端（天数/沐曦/海光）在 pymss 里仍然按 `cuda` 路径运行，
  只有算子缺失时才通过能力标志切到回退路径。

## 性能：生产规模 roformer 的实测数据

本节先保留环境修复前的历史数据。分块失败、bf16 无收益、MFU 天花板等早期推断均已撤回；
正确环境下的 bf16、SUPAGraph 与显式 FMHA 对照见文末“环境修复后的验证”。

测量口径：`pymss/tools/vendor_perf.py`，`BSRoformer(dim=384, depth=12, stereo=True,
num_stems=1, dim_head=64, heads=8)`，10 秒 44.1 kHz 立体声，fp32，3 次取中位数。
FLOPs 由 `torch.utils.flop_counter` 在 CPU 上对同一输入统计。

### 峰值（8192³ 稠密 GEMM）

| 精度 | TFLOPS |
|---|---|
| bf16 | 284.62 |
| fp32 | 115.74 |

小矩阵衰减严重：2048³ 只有 63.6 TFLOPS（峰值的 22%），4096³ 230.9（81%）。
**这张卡只在矩阵足够大时才跑得满**，所以"合并小算子成大 GEMM"比"优化单个算子"更有效。

### 优化前后的完整链条

| 改动 | 耗时 | 实时倍率 | MFU |
|---|---|---|---|
| 起点（分块 256，交错 rotary） | 8810 ms | 1.14× | 0.50% |
| rotary：复数路径 → 实部/虚部逐元素 | 4934 ms | 2.03× | 0.90% |
| attention 分块 256 → 1024 | 3820 ms | 2.62× | — |
| rotary：交错写入 → `[偶数半|奇数半]` 连续写入 | 2428 ms | 4.12× | 1.83% |
| sdpa：单块不再 `cat` | 2403 ms | 4.16× | 1.85% |

历史同配置累计加速 3.67 倍。14 个家族 `rel` 为 1.45e-05 ~ 1.34e-04；
SCNet 为 5.79e-02，是未解决的精度偏差，不能称为 15 家族全部通过精度验收。

三笔主要优化的依据：

1. **rotary 122 → 35 ms/次**：消融显示它占 5602 ms（63.6%）。复数乘法要经过
   `view_as_complex`/`view_as_real`，而厂商 shim 必须拆成实部/虚部平面，在
   `(62, 862, 8, 64)`（2740 万元素）上反复读写。改成逐元素后降到 35 ms。
   CUDA/MPS 保留原路径，本轮 BR 结果不代表其他设备的性能对照。
2. **attention 分块 256 → 1024**：freq attention 折叠后为 862×8=6896 个 batch，
   每块 256 需 27 块，每块 1024 需 7 块。旧环境下更大 batch 失败，不能作为当前支持上限。
3. **rotary 35 → 9.4 ms/次**：交错布局需要跨步写入。只有 q 和 k 被旋转，且用同一个
   维度置换，点积不变，所以可以改成两段连续写入。算术下限实测 2.5 ms，
   单纯换布局就拿到 3.8 倍。

### 剩余时间去哪了（GPU 事件归因）

| 项目 | 耗时 | 占比 |
|---|---|---|
| attention（含分块 bmm） | 1640 ms | 68% |
| 　其中 sdpa | 915 ms | 38% |
| FeedForward | 275 ms | 11% |
| rotary | 333 ms | 14% |
| mask estimator | 124 ms | 5% |
| stft + istft | 106 ms | 4% |
| RMSNorm | 56 ms | 2% |

这些模块计时存在包含关系：rotary 是 Attention 的一部分，不能把表中所有行相加。
显式 attention 的 time scores 张量大小确为 `496×862×862×4≈1.47 GB`，但据此不能
断言整个模型已经到达带宽下限，也不能推导 MFU 永远低于某个比例。此前计算还混用了
单次与 12 层的流量/算量。新验证发现了 SDK 路径问题、标准 SDPA 的 CPU 回退，以及
可用的厂商 FMHA；应先排除这些实现与环境问题，再进行 roofline 分析。

### 旧实验（不代表厂商能力上限）

| 尝试 | 结果 |
|---|---|
| 全链路 bf16 autocast | **更慢**：3060 ms vs 2409 ms，且 stft 从 38 ms 变 771 ms |
| attention 的 scores/softmax 用 bf16 | 更慢（0.81×）且 `rel=1.13e-02`，精度不可接受 |
| 第二个 bmm 用 bf16 | 更慢（0.88×），`rel=5.33e-03` |
| 提高 attention 分块到 2048 | 触发厂商 JIT 失败 |

### 测量纪律

- **控制并记录启动环境。** 曾观察到同一 STFT 的 39 ms 与 765 ms 差异，但不能仅据此
  判断容器退化。确认了 `br_generator` 缺失来自探针覆盖 SDK Python 路径；应先核对环境，
  再研究内核选择与性能变化。
- **关键数字取中位数。** 厂商栈有偶发非确定性：`bandit_v2` 出现过一次
  `repeat=1.84e-03`（正常 `1.1e-07`），复测三次不再现。
- **分模块归因不要用 `synchronize()`。** 同步会串行化流水线，曾把 RMSNorm 报成
  20.8%（消融实测 0.6%）。用 GPU 事件（`tools/vendor_timeline.py`）或消融法。

### 环境修复后的验证

硬件环境为 BR106M / SUPA 1.9.0.0.rc1、torch 2.7.0+cpu、torch-br
1.7.0.20700+br1xx。模型仍为 9819 万参数的 BS-RoFormer，输入为 10 秒、44.1 kHz
立体声；使用随机权重测性能，不能据此宣称真实音源分离质量合格。

**启动环境修复**：工具原先覆盖 `PYTHONPATH`，清掉了 SDK 的 tensor-engine、sulib/lib
等路径。JIT 最早的错误是 `ModuleNotFoundError: No module named 'br_generator'`。
单变量 A/B 验证：保留路径后，相同 3-D bf16 FMHA 复现成功。8 个工具的子进程均改为追加
测试目录并保留继承路径。不要用覆盖环境的旧启动命令。

```sh
export PYTHONPATH="/tmp/pc:${PYTHONPATH:-}"
python -c "import importlib.util; print(importlib.util.find_spec('br_generator'))"
```

**bf16**：频谱保持 fp32，进入 band_split 时匹配网络权重 dtype；rotary 的逆频率、
位置和三角函数计算保持 fp32，最终表转为激活 dtype，转换模型时清空缓存。
新建模型、相同权重、独立预热后的三轮 AB/BA 对照：

| 路径 | 中位数 | 对照说明 |
|---|---:|---|
| fp32 手工 attention | 2418.1 ms | bf16 修复后的同进程对照 |
| bf16 手工 attention | 2234.3 ms | 加速 1.082×，相对 fp32 的输出 L2 误差 1.64% |

**SUPAGraph**：用新输入验证 replay 确实重新计算，且与 eager 输出一致。
以下图数据测于 SDK 路径修复前的 bf16 手工 attention，不能外推到 FMHA。

| ZERO_WS | eager host / 总耗时 | replay host / 总耗时 |
|---|---:|---:|
| 0 | 2168.9 / 2233.7 ms | 4.3 / 2233.4 ms |
| 1 | 2169.9 / 2234.9 ms | 4.2 / 2225.0 ms |

图捕获有效，显著释放 host；这个配置的单次总延迟收益小。不能说图模式“无对应能力”或
“完全无收益”。安装版 graph 上下文已经自动使用捕获 stream。

环境修复后，bf16 + 显式 FMHA + ZERO_WS=1 的三轮交错测量为：eager host / 总耗时
中位数 1771.4 / 1834.5 ms，replay 为 1.59 / 1824.0 ms。更换输入后 replay 与
同输入 eager 输出逐值一致（相对 L2=0）。此组合主要降低 host 提交成本，单次总耗时
约改善 0.6%；没有据此断言其他配置的上限。

**标准 F.sdpa 与厂商 FMHA 是不同路径**：profiler 显示标准入口选择
`aten::_scaled_dot_product_attention_math`，其 `_safe_softmax` 的 `isneginf` / `where`
触发 CPU 回退。对 isneginf 的进程内试验减少了回退，但没有作为全局补丁安装。
SDK 的 `torch.supa.enable_flash_sdp` 源码明确只设置全局变量，不能控制设备算子。
实际融合入口为 `supa::sudnn_scaled_dot_product_attention_fwd(..., algorithm="FMHA")`，
描述符只接受 2-D/3-D 输入。适配器把 batch/head 折叠成 3-D，再恢复输出形状。

**完整模型 FMHA**：保留 SDK 路径，bf16，同一模型、相同权重和输入，分别预热后交错三轮：

| 路径 | 三轮耗时 | 中位数 |
|---|---|---:|
| 手工 attention | 2214.6 / 2215.8 / 2215.4 ms | 2215.4 ms |
| 显式 suDNN FMHA | 1826.2 / 1827.6 / 1824.3 ms | 1826.2 ms |

加速 **1.213×**，10 秒输入的前向速度约 **5.48× 实时**。计时阶段两种实际形状各调用
厂商算子 36 次，共 72 次，零失败、零回退。输出有限；相对 bf16 手工路径的 L2 误差为
1.26%，尚需真实 checkpoint 验证质量。以上是模型前向，未包含产品级解码、分块拼接及文件 I/O。

新后端显式选择：在 `prepare_device_alias` 安装 shim **之前**设置
`PYMSS_SDPA_BACKEND=sudnn`，并把网络转为 bf16。默认仍为 `manual`；原生算子异常会直接
上报，不静默回退。当前限定为 SUPA 上的无 mask、非 causal、dropout=0、无 GQA 的 bf16
自注意力。其他设备继续原有路径。早期 `PYMSS_SDPA_VENDOR_CHUNK` 的标准 math 实验已移除。

可复用探针：`vendor_bf16_probe.py`、`vendor_graph_probe.py`、
`vendor_native_sdpa_probe.py`、`vendor_safe_softmax_probe.py`、`vendor_fmha_probe.py`、
`vendor_fmha_model_probe.py`，均位于 `tools/`。原始日志保存在任务的 `.tmp/` 目录。

**MFU 口径**：FP32 与 BF16 必须分别使用匹配精度的 GEMM 参考值；还需说明 TF32 等
matmul 设置。不能用不同分母下的百分比直接判断速度，更不能据历史访存估算宣称性能已到上限。

### 原生 RoPE 内核（supa_rope_apply）

**启用方式**：加载 checkpoint 之后、首次前向之前，调用
`pymss_core.vendor_shims.prepare_model_for_vendor_rope(model)`。设备准备
（`prepare_device_alias("biren")`）会经 BIREN profile 自动安装 rope shim；
`PYMSS_ROPE_BACKEND=manual` 可随时退回原路径。未调用 prepare 的模型完全不
受影响（shim 只加速破标记的 rotary，其余设备/模型走原路径，始终安全）。

**为什么需要权重置换**：厂商算子按分半配对旋转（第 i 与第 i+d/2 分量），
而 BS-Roformer 的 q/k 通道是交错对（2i, 2i+1）。prepare 把每个 `to_qkv`
的 q/k 输出行按每头 `[evens|odds]` 重排，让投影本身零成本地发射算子要的
配对；运行时再做 cat 解包反而比原路径更慢（实测 1913 vs 1858 ms）。算子
输出保持 `[even|odd]` 半分布局，与 `apply_rotary_emb_fast` 输出一致，下游
代码零改动。置换后的 checkpoint 保存/重载不适用；重载原始 checkpoint 时
已注册的 `load_state_dict` post-hook 会自动重新置换。

**算子契约（实测）**：x 必须是 bf16 `(batch*heads, seq, head_dim)` 且
dim0 ≤ 1024（超出报 invalid parameter 甚至挂死，rotate 内部按 dim0 分块）；
表必须 fp32（bf16 被 schema 拒绝）`(1, seq, head_dim)`，取 `concat([half, half], -1)`；
seq 实测至 4096 均可；表精度沿 seq 无漂移（单算子相对 fp32 参考误差
约 0.167%，17→1024 平稳）。

**端到端（10 s 立体声，dim=384，depth=12，bf16+FMHA 基线）**：

| 配置 | 中位耗时 | 相对 fp32 锚点 |
|---|---:|---:|
| 原 rotary（未 prepare，回退路径） | 1825.1 ms | 1.5714% |
| prepare + 原生 RoPE | 1728.4 ms | 1.5705% |
| 重载 raw checkpoint + post-hook | 1732.9 ms | 1.5705% |

加速 -96.7 ms（5.3%），精度与原路径等价（两者到 fp32 错距一致，1.24% 为
两条 bf16 舍入路径的正常互差）。定位过程与陷阱见
`tools/vendor_rope_shape_sweep2.py`、`tools/vendor_rope_seq_accuracy.py`、
`tools/vendor_rope_model_probe.py`、`tools/vendor_rope_integration_check.py`；
配套 CPU 测试 `pymss-core/tests/test_vendor_rope.py`（置换代数等价、
幂等、load-hook、env 覆盖、分块上限、mock 算子数值对齐）。

**qkv 单次重排（同 shim 内置）**：rope shim 同时接管已 prepare 模型的
`Attention.forward`（仅 SUPA+bf16+推理态+无 PoPE）：qkv 投影一次
`view(b,n,3,h,d).permute(2,0,3,1,4).contiguous()` 重排为 `(3,B,H,S,D)`，
取代原先三次分散拷贝（q/k 各一次 permute + SDPA shim 里 v 的
`.contiguous()`）；rotary 与 FMHA 适配器此后全部零拷贝视图。与原前向
**逐位一致**（探针 `tools/vendor_attention_chain_probe.py`，rel_to_current
=0.0）。端到端叠加：1823.0 → 1647.1 ms（rope+重排合计 -9.7%），
rel_to_fp32 不变 1.5705%。q/k 合并单调用（`trio[:2]` 免费视图）后再降
至 **1632.5 ms**。其他设备/dtype/训练态/PoPE/未 prepare 模型
一律走原前向。

**融合 FFN（同 shim 内置）**：rope shim 同时接管 `FeedForward.forward`
（仅 SUPA+bf16+推理态+结构匹配：`RMSNorm→Linear→GELU→Dropout(0)→
Linear→Dropout(0)`、双 Linear 带偏置）：两投影+GELU 整体换成单内核
`torch_br.supa_fused_mlp_linear_act_linear_fwd(x, w_up, w_down, b_up,
b_down, act_mode='gelu')`（**`act_mode` 必须小写 'gelu'**，其它拼写会静默
禁用激活导致输出错误；返回 `[output, ...]` 需取首元素；2-D 输入会触发
布局断言，须 3-D 调用）。隔离实测 2.82 ms vs eager 6.43（2.3×），对
bf16 参考误差 0.41%（erf/tanh 不可分辨）。端到端：1823.6 → 1557.4 ms
（叠加 rope+重排后，今日合计 -14.6%），rel_to_fp32 1.5464%（不劣反微升）。
退出开关 `PYMSS_FFN_BACKEND=eager`。

**后续优化线索（已探明未实施）**：freq 侧 (6896,62,64) 因 dim0≤1024
被迫 7 块分块，实测算子在短序列形态效率仅为长序列的 1/4（同元素量
12.4 ms vs 0.45 ms），构当前 rope 剩余成本。自定义 G-mode 内核工具链
已走通（SRTC 运行时编译 + 驱动启动，见
`tools/vendor_rope_srtc_kernel.py`），但 fp32 表指针语义、bf16 向量化、
NUMA 16 区域寻址三题未解。另有 SDK 内置融合算子
`supa_norm_rope_uma_bf16_infer`（RMSNorm+RoPE）未探索。

## 四卡实测终局（2026-10-30，11 类型真实模型全矩阵）

| 卡 | 结果 | 首轮通过 | 关键适配 |
|---|---|---|---|
| 海光 Z100L (DTK/ROCm) | 11/11 | 8/11 | fft_dtype shim（hipFFT 拒 Half→fp32 全路径提升，含 istft/complex32）；BLSTM 分帧丢尾修复推广全设备 |
| 沐曦 N260 (MACA/torch 2.10) | 11/11 | 6/11 | fft_dtype shim 复用（cuFFT NOT_IMPLEMENTED 同症）；环境极简（无 git、ffmpeg 缺库）但卡稳 |
| 天数 MR-V100 (corex/torch 2.7) | 11/11 | 11/11 零修复 | 预防性 fft_dtype shim；兼容性最佳 |
| 天数 BI-V100 | 0/11 | - | 卡级 wedge（首个 GPU 操作死锁），容器重启无效，宿主机故障 |

### 性能要点：JIT 首调税与热态真值（MR-V100 实测）

国产栈按形状现编 kernel，**首次调用含巨额编译税**：scnet 23.9s→热态 3.2s（−87%）、mdx23c 8.2→2.6s、vr 36.8→0.43s（batch16）。单进程 CLI 每次都付；对策：

- `MSSeparator(..., inference_params={"warmup": True})`：加载时静音空跑一遍预编译（`efe781c`）；
- VR 类 launch-bound 模型建议 `batch_size: 16`（热态再 −13%）；
- apollo/bandit 热态无改善，为真算力瓶颈，优化需模型侧手段。

### 半精度 FFT 结论

hipFFT / MACA cuFFT / ixFFT 三家**均不支持** half 变换（16 点小探针会误判通过，真实 stft 才暴露）。`fft_dtype_shim` 保持无条件升 fp32（hygon/metax/iluvatar profile 已启用）；懒探测方案已证伪回滚（`b5dce0a`）。

## ENFLAME T20 判决（2026-10-30，TopsRider 2.5.115 / torch 1.10.0+cu102 / py3.8）

**结论：当前镜像不可用（vendor 运行时级 SIGILL），需换新版 TopsRider 镜像（torch 2.x 线）重测。**

- 硬件与运行时初始化正常（xla:0 可见、8 卡、4GB MC），34/38 常用算子隔离测试通过（含 conv/LSTM/einsum/attention 基础件）。
- 已实证 SIGILL 的原生 kernel：`torch.stft`（设备侧）、`complex mul/reshape/cat`、`unfold`、原生 `as_strided`、autograd `_to_copy` 某些布局。已全部落地垫片（stft→CPU 往返、complex→BIREN 同款实虚分流、unfold/as_strided→gather 物化），但完整模型仍在前向早期 SIGILL，且燧原信号处理器吞掉 faulthandler，无法继续归因。
- Python 层兼容成果（对任何停留在此栈的用户有效，全部已推送）：py3.8 语法/注解（22+63 文件 future import）、with 语句、importlib.resources 双重回退、`torch.load` weights_only/mmap 梯子、`itertools.pairwise`、`F.rms_norm` backport、`_apply(recurse)`、`torch.backends.mps` 守卫、`inference_ctx`（<1.12 用 no_grad）、xla 设备类型→ENFLAME vendor 映射。
- 升级路径：OpenI 建任务时选用较新 TopsRider（≥3.x，torch 2.x）镜像；容器内 pip 升级不可行（wheel 必须与宿主机 /opt/tops runtime 精确匹配）。
