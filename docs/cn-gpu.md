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

### 五、SUDNN 的 JIT 编译器在本镜像里不可用（最重要的一条）

厂商的 `PtOpFrontend` 会为**没有预编译内核**的形状现场生成并编译代码，它依赖一个内嵌的
Python 环境，在本镜像里初始化失败：

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

**判定方法**：同一个算子，用**预编译形状**能跑，用其它形状就报上面这串。最小复现：

```python
import torch, torch_br
torch.supa.set_device(0)
d = torch.device("supa:0")
torch.bmm(torch.randn(1384, 62, 64, device=d), torch.randn(1384, 64, 62, device=d))
```

**已实测无效的尝试**（不要再浪费时间）：

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

**会随时间退化**：同一容器内，`bmm((1384,62,64),(1384,64,62))` 在 20:15 通过（batch
173/346/692/1384/1385/2048/1024 全通过），20:32 起持续失败，且**不加载 pymss 的
任何 shim 也失败**。同期 `8192³` 矩阵乘、`conv2d`、radix-2 FFT 全部正常 —— 说明卡本身
没坏，坏的是需要现场编译的那条路径。**长跑测量必须在干净容器里做，退化后重启调试任务。**

**后果**：生产规模的 roformer（`dim=384, depth=12`，10 秒音频 862 帧）做 attention 时
需要 `bmm(6896, 62, 64)`，命中 JIT 路径，因此**无法稳定运行**；而正确性验收用的
缩小模型（3k–23k 参数、64 个采样点）形状都在预编译库里，所以 15/15 全绿。
**用小模型验收会完全掩盖这条问题。**

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

**累计 3.67 倍**，15 家族数值不变（`rel` 1.45e-05 ~ 1.34e-04）。

三笔主要优化的依据：

1. **rotary 122 → 35 ms/次**：消融显示它占 5602 ms（63.6%）。复数乘法要经过
   `view_as_complex`/`view_as_real`，而厂商 shim 必须拆成实部/虚部平面，在
   `(62, 862, 8, 64)`（2740 万元素）上反复读写。改成逐元素后降到 35 ms。
   CUDA/MPS 上复数路径实测快 4.7 倍，所以按设备选择，不是一刀切。
2. **attention 分块 256 → 1024**：折叠后 53444 行，256 一块要 209 次 bmm，
   1024 一块只要 52 次。1024 是仍能命中预编译内核的最大值（2048 会触发 JIT 失败）。
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

**这个模型在这张卡上是内存带宽受限，不是算力受限。** 5.14 TFLOP 的计算背后，
单是 attention 的 scores 张量就有 `496 × 862 × 862 = 1.47 GB`，每次 attention 至少
4 遍读写（写 scores → softmax 读 → 写 probs → bmm 读）≈ 5.9 GB。按实测带宽
~200 GB/s 算，29 ms 是物理下限，实测纯计算 25 ms、模型内 38 ms —— **已经贴着下限**。

这正是 MFU 只有 1.85% 的原因：算力利用率低不是因为适配没做好，而是因为
`scores` 的搬运量（24 次 attention 合计约 108 GB）远超计算量。

### 被实测否决的优化（不要再试）

| 尝试 | 结果 |
|---|---|
| 全链路 bf16 autocast | **更慢**：3060 ms vs 2409 ms，且 stft 从 38 ms 变 771 ms |
| attention 的 scores/softmax 用 bf16 | 更慢（0.81×）且 `rel=1.13e-02`，精度不可接受 |
| 第二个 bmm 用 bf16 | 更慢（0.88×），`rel=5.33e-03` |
| 提高 attention 分块到 2048 | 触发厂商 JIT 失败 |

### 测量纪律

- **必须在干净的调试任务里测。** 容器退化后同样的 `stft_npu` 从 39 ms 变成 765 ms
  （20 倍），曾据此错误地得出"DFT 矩阵比厂商 FFT 快 12 倍"的结论，方向完全相反。
- **关键数字取中位数。** 厂商栈有偶发非确定性：`bandit_v2` 出现过一次
  `repeat=1.84e-03`（正常 `1.1e-07`），复测三次不再现。
- **分模块归因不要用 `synchronize()`。** 同步会串行化流水线，曾把 RMSNorm 报成
  20.8%（消融实测 0.6%）。用 GPU 事件（`tools/vendor_timeline.py`）或消融法。
