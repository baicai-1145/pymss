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
| `complex64 * float32` | `pointwise No SUPA Function found` | 在实/虚平面上相乘（`masked_spectrum`） |
| 复数 **D2H**（`complex.cpu()`） | `ErrorCode: 101, invalid device` | 只搬 `view_as_real` 的实数平面（`complex_to_numpy_cpu`） |
| 复数 **H2D**（`.to(device)`） | `Unsupported BR type for ScalarType: 0` | 实数平面上卡、卡上重组（`numpy_complex_to_device`） |
| `F.pad(mode="replicate")` | 警告"回退 CPU"，实际 SUDNN `ReorderWei2Plain() FAILED` + `suError: 101` | `cat`+`expand` 复现 replicate（`replicate_pad_bins`） |
| fp16 | "Current SUPA Device does not support float16" | 用 bf16 或 fp32 |
| bf16 autocast | `no valid backend was found for op: UpsampleBilinear` | 分离模型 `amp=False`（档案已注明原因） |
| 融合 `_VF.lstm` | **静默算错** 0.18–0.47 | RNN 回退 CPU（手写 cell 只差 2e-06） |
| 膨胀卷积（SUDNN 路径） | `terminate()` 杀进程 | 见下节 |

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
