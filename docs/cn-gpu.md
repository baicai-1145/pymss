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
| 壁仞 Biren | BIRENSUPA / BRPyTorch | 2.4 – 2.8 | `biren`（或 `supa`） | `torch_br` + `torch 2.x+br` | 老版本 SUPA 把设备暴露为 `cuda`，新版本为 `supa` |
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

## 在 OpenI 云脑任务里一键采集证据

仓库自带 `scripts/cn_gpu_probe.py`，可直接作为**训练任务的启动文件**（无人值守、不限时长），
结果写进 `c2net_context.output_path` 并 `upload_output()` 回传，在任务详情「结果下载」里取：

```sh
# 训练任务配置
#   资源：目标厂商卡型（如 壁仞 1*BIREN106M）
#   镜像：厂商 SDK 镜像
#   访问Internet：是（需要 pip / git clone）
#   代码分支：包含本文件的适配分支
#   启动文件：scripts/cn_gpu_probe.py
#   运行参数：--device <alias>            # 例：--device enflame
#             --install-tests             # 首次跑时装 pytest/librosa 等
#   （默认会自己 pip install pymss-core：优先平台内 git，回落 GitHub，需 Internet=是）
```

脚本会依次采集：环境与 torch 构建信息 → 已安装厂商栈（`torch/+supa/+corex/+metax/+gcu/+das/+mlu`）
→ pymss 设备档案 → STFT/ISTFT 原生与回退双路往返误差 → LSTM 原生与 CPU 回退 →
`pytest` 摘要，最后写出 `cn_gpu_report.txt` 与 `cn_gpu_profile.json`。

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
