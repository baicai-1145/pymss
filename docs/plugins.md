# pymss 插件与操作开发指南

pymss 2.2+ 采用双层扩展模型（见 [plugin-redesign-plan.md](plugin-redesign-plan.md) 的完整背景）：

- **Operation（操作）**：可执行的一项音频处理。声明输入、参数、输出，宿主（CLI / pymss-studio / ComfyUI / 工作流图）从同一份描述生成界面与节点。
- **Provider（服务实现）**：底层服务（当前为重采样）的可替换实现，显式注册与选择。

## 快速开始：写一个操作插件

调用外部原生工具时，可使用 `pymss.plugins.kit.run_native_process(argv, cwd=..., context=...)`。
它使用参数数组启动程序，支持任务取消、超时、有限诊断缓冲及子进程树回收。
省略 `context` 时沿用 `operation_context()` 绑定的上下文，否则创建独立上下文；
独立上下文不具备宿主取消信号。原生程序及其许可证、平台依赖由插件包自行提供。
`output_callback(channel, text)` 可接收运行中的 stdout/stderr 文本片段，适配原生进度；
回调在调用线程执行，输出队列有界，拥塞时只保留近期片段。完整行解析由插件负责，
诊断尾部仍保存在返回值中。`environment` 可显式传入子进程环境。

在 `~/.pymss/plugins/myplugin/`（或 `PYMSS_PLUGINS_DIR` 指定目录）放两个文件。`pyproject.toml` 声明插件身份和它注册的操作（[插件协议 v2](plugin-protocol.md)）：

```toml
[project]
name = "pymss-plugin-myplugin"
version = "0.1.0"

[tool.pymss.plugin]
id = "myplugin"
api_version = 2

[[tool.pymss.plugin.operations]]
id = "myplugin.gain"
title = "Apply Gain"
```

`__init__.py` 中的 `setup()` 注册同一个操作：

```python
def setup(plugin):
    plugin.register_operation(
        id="myplugin.gain",                 # 以插件 id 加点开头
        schema_version=1,
        title="Apply Gain",
        inputs={"audio": {"type": "audio"}},
        params={
            "type": "object",
            "properties": {
                "gain_db": {"type": "number", "default": 0.0,
                             "minimum": -60.0, "maximum": 60.0},
            },
            "additionalProperties": False,
        },
        outputs={"audio": {"type": "audio"}},
        run=run_gain,
    )


def run_gain(ctx, inputs, params):
    audio = inputs["audio"]        # AudioArtifact: [channels, samples] float32 + sample_rate
    gain = 10 ** (params["gain_db"] / 20.0)
    out = audio.audio * gain
    return {"audio": type(audio)(out, audio.sample_rate, channel_layout=audio.channel_layout)}
```

`setup()` 失败，或注册的操作与清单声明不一致时，该插件的所有注册会被回滚，错误进入加载报告。用 `pymss plugins validate <目录>` 可以在不执行代码的情况下检查清单。

## 参数 schema

当前支持 `number` / `integer` / `string` / `boolean` 与 `default` / `minimum` / `maximum` / `enum` / `required` / `additionalProperties`。默认值填充与校验由 runtime 统一执行；宿主只负责展示与传值。

## 使用入口

| 入口 | 命令 / 位置 |
|---|---|
| CLI | `pymss ops list` / `pymss ops describe <id>` / `pymss ops run <id> --input audio=a.wav --param gain_db=-3` |
| 工作流图 | 节点类型 `op.<operation id>`，参数写在 `node.data["params"]`（命名参数，不是 widgets 位置） |
| pymss-studio | 工具页 "插件操作"：自动生成表单，后台任务执行 |
| ComfyUI | 节点 `pymss_op_<id 下划线>`：音频输入端口 + 参数 widget 自动生成 |

## 内置操作

pymss 自带 `pymss.audio.adjust_volume` / `invert_phase` / `normalize_peak` / `trim`，与外部插件走完全相同的注册与执行路径（source 标记为 `pymss.builtin`）。

## Python API

```python
from pymss.operations import create_runtime, default_runtime

runtime = create_runtime(load_plugins=True)   # 隔离实例
runtime = default_runtime()                   # 进程默认（含内置操作）
result = runtime.run("myplugin.gain",
                     inputs={"audio": artifact},
                     params={"gain_db": -3.0})
```

## Provider：替换重采样实现

```python
from pymss.audio_services import default_audio_service

class MyResampler:
    name = "my-resampler"
    def resample(self, audio, orig_sr, target_sr):
        ...

svc = default_audio_service()
svc.register_resampler(MyResampler())
svc.select_resampler("my-resampler")     # 显式选择；selected_resampler() 可查询
svc.resample(audio, 44100, 48000)        # 所有通用路径（图、保存转换、ensemble 对齐）随之改变
```

同名冲突：`register_resampler` 默认拒绝重复注册（传 `replace=True` 覆盖）；能力（capability）注册表同样不再允许内置编码器覆盖插件实现。

## capability / node / CLI

插件在 `setup(plugin)` 中通过 `plugin.register_capability` / `plugin.register_node` / `plugin.register_cli` 注册到全局注册表；`register_cli` 的命令会挂载到 `pymss` CLI（与内置命令同名则跳过并告警）。没有 v2 清单的插件不再加载，迁移方式见 [plugin-protocol.md](plugin-protocol.md#migrating-from-v1)。

## 共享插件管理

插件面向 pymss 内核，Studio 页面是可选扩展。只有 Provider、codec 或 CLI 贡献的插件不需要声明 Operation、节点或 Studio SDK。

`pymss_plugin_manager` 随 pymss 一起分发，提供轻量管理 API；导入它和读取安装记录不会导入模型运行时。`pymss.plugins.service` 提供相同接口的公开别名。管理 API 当前主版本为 `1`，宿主应检测该能力，不能仅根据旧的 `pymss>=2.2` 约束推断已支持。

```python
from pathlib import Path
from pymss_plugin_manager import PluginContext, PluginService

service = PluginService(PluginContext(Path("/data/plugins/cpu")))
service.install("/path/to/plugin")
snapshot = service.list()
service.disable("author.plugin")
service.enable("author.plugin")
service.uninstall("author.plugin")
```

上下文路径是包含 `installed/`、`disabled/` 和 `transactions/` 的运行环境绑定目录。它的父目录是插件存储，兄弟绑定共享安装意图和 revision，文件、依赖 sidecar 和 Python profile 各自准备。不同存储相互隔离；不会自动移动旧目录或虚拟环境。

Studio 的管理界面调用同一服务。CLI 可使用 Studio 返回的 `runtime.bindingDir`，配合该运行环境的解释器显式选择相同绑定：

```sh
python -B -X utf8 -m pymss --plugin-binding /data/plugins/cpu plugins list --json
python -B -X utf8 -m pymss --plugin-binding /data/plugins/cpu install /path/to/plugin
python -B -X utf8 -m pymss --plugin-binding /data/plugins/cpu plugins disable author.plugin
python -B -X utf8 -m pymss --plugin-binding /data/plugins/cpu plugins check-updates --json
python -B -X utf8 -m pymss --plugin-binding /data/plugins/cpu plugins recover --json
```

`--plugin-store` 是同一参数的别名，`PYMSS_PLUGIN_BINDING` 可用于后续 CLI 运行。Studio 检查当前运行环境时，会通过共享 API 登记该解释器的插件绑定、内核位置和可选组件目录；在该环境里直接运行 `pymss install`、`pymss plugins` 或插件命令即可沿用相同绑定，内核更新目录也会随环境检查同步。其他宿主可在目标解释器内调用 `pymss_plugin_manager.environment.register_environment_binding()` 完成同样的关联。

选择顺序为：命令行绑定参数、`PYMSS_PLUGIN_BINDING`、显式旧目录 `PYMSS_PLUGINS_DIR`、当前解释器的登记记录。没有登记的独立 CLI 环境继续使用旧默认目录。记录保存在用户目录 `~/.pymss/environments/`，不会写入 Python 安装目录或 macOS 应用包；`PYMSS_PLUGIN_ENVIRONMENTS_DIR` 可为隔离运行指定记录目录。解释器身份保留虚拟环境链接路径，不把不同 venv 合并为其基础 Python。已存在但损坏的记录会报错；显式指定绑定可以覆盖自动选择。

自动选择内核时保留当前工作目录、解释器的隔离及编码等启动选项，输入文件和本地插件的相对路径含义不变。`python -m pymss` 同样先选择内核，再加载推理依赖；`-I` 或 `-E` 下仍可使用已登记的内核和组件目录。显式绑定参数及上述插件目录环境变量会跳过自动内核选择，适用于独立 CLI 和开发环境。

外部旧目录不会被受管上下文隐式加载。固定管理命令不会先执行插件的 `setup()`；`plugins probe` 才执行独立验证。

持久安装记录、启停意图和进程加载结果分别管理。加载和原生监督进程持有跨进程执行租约，管理操作不能覆盖正在使用的文件；Provider/codec 变更在新进程生效。库调用使用 `PluginPolicy` 传递事件、取消、依赖约束及可选组件事实，不依赖 Studio 模块。

新旧写入器不能因为磁盘格式相同就混用。共享绑定的管理客户端需要支持同一管理 API 与锁规则。缺少公共管理能力的 Studio 只观察旧记录并提示升级。发布包的 artifact 获取和构建发布流水线仍为后续阶段；本接口首先统一现有源码安装与事务管理。
