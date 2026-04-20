# CS Prophet

面向 CS2 demo 的炸弹落点预测项目。核心任务是从一回合的状态序列中预测该回合最终会在 `A` 点还是 `B` 点下包，并提供离线分析、HLTV 下载工具、GSI 实时推理，以及基于进程内存的实时推理实验路径。

## 现在这个仓库的真实状态

这个仓库目前同时维护三条相关但不完全相同的工作流：

| 路线 | 维度 / 采样 | 主要输入 | 主要输出 | 现状 |
| --- | --- | --- | --- | --- |
| 旧版 `v1` | `275` 维, `8 Hz` | `.dem` | `processed/*.parquet` | 兼容保留，离线分析脚本仍使用它 |
| 新版 `v2` | `218` 维, `8 Hz` | `*_full.pkl` | `processed_v2/*.parquet` | 当前离线特征/训练主线 |
| 新版 `v2` 实时对齐 | `218` 维, `2 Hz` | GSI 或内存状态 | `processed_v2_2hz_preplant/*` + `checkpoints/v2_2hz/*` | 当前实时推理默认路径 |

几个最重要的事实：

- 预测目标一直是 `A / B` 二分类，不包含 `other`。
- `realtime_engine` 现在走的是 `v2` 特征，不再是旧版 `275` 维路径。
- 默认实时 checkpoint 是 `checkpoints/v2_2hz/best.pt`。
- `analyze_demo.py` 仍然只适用于旧版 `v1 / 275` 维 parquet。
- `dashboard/app.py` 只是占位；真正使用的是 `realtime_engine` 托管的 `dashboard/index.html`。

## 数据路径规则

大部分脚本不再把 repo 内的 `data/` 当成唯一数据目录，而是通过 `src/utils/paths.py` 解析路径。

优先级如下：

1. 环境变量 `CS_PROPHET_DATA_ROOT`
2. Windows 下默认外部数据根 `H:\CS_Prophet\data`
3. 仓库内 `data/` 作为回退

这意味着 README 里的很多命令都用的是类似 `raw/demos`、`processed`、`processed_v2`、`viz` 这种“数据根相对路径”，而不是硬编码成 `data/...`。

补充说明：

- repo 内 `data/` 现在主要是轻量占位和兼容回退，详见 [data/README.md](data/README.md)
- 仓库上传边界见 [docs/repository-layout.md](docs/repository-layout.md)

## 环境安装

安装完整依赖：

```powershell
pip install -r requirements.txt
```

或者按包安装：

```powershell
pip install -e .
```

如果还需要开发依赖：

```powershell
pip install -e ".[dev]"
```

安装完成后，也可以直接用包入口启动实时服务：

```powershell
cs-prophet --checkpoint checkpoints/v2_2hz/best.pt
```

运行测试：

```powershell
pytest tests -v
```

## 快速开始

### 1. 旧版 `v1`：`.dem -> processed/*.parquet -> 275 维训练/分析`

批量解析本地 demo：

```powershell
python parse_demos.py --demo-dir raw/demos --out-dir processed --resume
```

训练旧版模型：

```powershell
python -m src.model.train --config configs/train_config.yaml
```

烟雾测试：

```powershell
python -m src.model.train --config configs/train_config.smoke.yaml
```

对单个旧版 parquet 做离线回放分析：

```powershell
python analyze_demo.py processed/your_demo.parquet --checkpoint checkpoints/best.pt
```

说明：

- `parse_demos.py` 产出的是旧版 `processed/*.parquet`
- `analyze_demo.py` 只支持这条旧版路径

### 2. 新版 `v2`：提取完整 payload，再转 `processed_v2`

先把一个 demo 提取成 `*_full.pkl`：

```powershell
python tools/demo_full_extract.py raw/demos/your_demo.dem
```

默认输出到活动数据根下的 `viz/your_demo_full.pkl`。

把 `*_full.pkl` 转成 `processed_v2/*.parquet`：

```powershell
python tools/build_processed_v2.py viz --out-dir processed_v2 --resume
```

训练 `v2` 模型：

```powershell
python -m src.model.train --config configs/train_config_v2.yaml
```

烟雾测试：

```powershell
python -m src.model.train --config configs/train_config_v2.smoke.yaml
```

如果只是想快速抽样提取一些 demo 跑通 `v2`：

```powershell
python tools/batch_extract_v2.py --per-map 2 --resume
```

这条批处理默认生成的是 `8 Hz` 的 `processed_v2`，不是 `2 Hz` 实时对齐数据。

### 3. `v2` 实时对齐训练：`2 Hz` / `180` step / `processed_v2_2hz_preplant`

当前实时推理默认对应的是 `2 Hz` 版本。典型流程是：

先按 `2 Hz` 提取：

```powershell
python tools/demo_full_extract.py raw/demos/your_demo.dem --downsample 32 --output-dir viz_2hz
```

再转成 `2 Hz` 训练数据：

```powershell
python tools/build_processed_v2.py viz_2hz --out-dir processed_v2_2hz_preplant --resume
```

训练对应模型：

```powershell
python -m src.model.train --config configs/train_config_v2_2hz.yaml
```

这条配置对应：

- `input_dim = 218`
- `sequence_length = 180`
- `target_tick_rate = 2`
- `save_dir = checkpoints/v2_2hz/`

### 4. ONNX 导出

仓库里保留了导出脚本：

```powershell
python -m src.inference.onnx_export --checkpoint checkpoints/best.pt --output model.onnx
```

但要注意：当前 `onnx_export.py` 仍按旧版 `275` 维 / `720` step 默认值构造导出输入，更适合旧版 `v1` checkpoint，不应直接视为 `v2` 的通用导出入口。

## 实时推理

### 1. GSI 模式

启动服务：

```powershell
python -m src.inference.realtime_engine --checkpoint checkpoints/v2_2hz/best.pt --input gsi --port 3000 --device cpu
```

打开：

```text
http://localhost:3000
```

把下面这个文件放到 CS2 的 Game State Integration 目录：

```text
cfg/gamestate_integration_cs_prophet.cfg
```

这份配置默认推送到 `http://127.0.0.1:3000/gsi`，和 `realtime_engine` 默认端口一致。

### 2. 内存模式

```powershell
python -m src.inference.realtime_engine --checkpoint checkpoints/v2_2hz/best.pt --input memory --port 3000 --device cpu
```

说明：

- 这条路径读取的是运行中的 CS2 进程内存
- 依赖 `src/inference/memory_reader.py` 和 `src/memory_reader/{offsets.json,client_dll.json}`
- 更偏 Windows 本机实验环境，不是跨平台通用方案

## GSI 抓包与调试

如果你不是要直接跑实时预测，而是想先观察 GSI payload 字段，可以用：

```powershell
python tools/gsi_capture.py --port 3001 --out data/gsi_dump.jsonl
```

这时应该使用：

```text
tools/gamestate_integration_csprophet.cfg
```

注意这份工具配置默认打到 `127.0.0.1:3001`，和主服务配置不是同一份。

另外，`tools/gsi_capture.py` 的 `--out` 是普通文件系统路径，不走 `data_root()` 解析；如果不传，默认写 repo 内的 `data/gsi_dump.jsonl`。

## 质检与可视化

生成旧版 `v1` parquet QA 报告：

```powershell
python viz_parquet.py --schema v1 --per-map 2 --open
```

生成新版 `v2` parquet QA 报告：

```powershell
python viz_parquet.py --schema v2 --per-map 2 --open
```

输出位置：

- `v1` -> 活动数据根下的 `viz_report.html`
- `v2` -> 活动数据根下的 `viz_report_v2.html`

`tools/` 目录下还有一批更偏调试/比对的脚本，例如：

- `tools/demo_visualize.py`
- `tools/demo_feature_preview.py`
- `tools/analyze_round_predictions.py`
- `tools/compare_offline_vs_realtime.py`
- `tools/verify_train_infer_parity.py`

这些工具大多是实验/诊断脚本，适合开发期排查，不建议把它们当成唯一入口文档。

## HLTV 下载工具链

只下载 demo：

```powershell
python tools/download_demos.py --config tools/hltv_config.yaml
```

下载后立即解析为旧版 `processed/*.parquet`，不长期保留 `.dem`：

```powershell
python tools/pipeline.py --config tools/hltv_config.yaml
```

这里要注意几件事：

- `tools/pipeline.py` 解析出来的是旧版 `processed/`，不是 `processed_v2`
- `tools/hltv_config.yaml` 目前是“示例加本机配置混合体”，里面的 `proxy`、`allowed_events`、`target_demos` 很可能需要你先改
- 配置里的 `raw/demos`、`raw/manifest.jsonl`、`raw/failed.jsonl` 也会走活动数据根解析
- 解压依赖 `tools/hltv/downloader.py` 中的 `Bandizip` / `bz.exe` 查找逻辑，明显偏 Windows

## 目录说明

```text
CS_Prophet/
├─ configs/                  训练配置
├─ cfg/                      主实时服务使用的 GSI 配置
├─ dashboard/                实时页面静态资源
├─ data/                     轻量回退/占位数据目录
├─ docs/                     仓库说明与设计文档
├─ src/
│  ├─ parser/                旧版 demo -> parquet 解析
│  ├─ features/              v1/v2 特征、数据集、标签逻辑
│  ├─ inference/             GSI / 内存 / checkpoint 推理
│  ├─ memory_reader/         实时内存读取依赖的 offset 元数据
│  ├─ model/                 Transformer 与训练逻辑
│  └─ utils/                 路径与地图工具
├─ tests/                    测试
├─ tools/                    下载、提取、转换、调试脚本
├─ analyze_demo.py           旧版 v1 离线分析
├─ parse_demos.py            旧版 v1 批量解析入口
└─ viz_parquet.py            v1/v2 QA 报告入口
```

## 当前推荐阅读顺序

如果你是第一次接手这个仓库，建议按下面顺序看：

1. [docs/repository-layout.md](docs/repository-layout.md)
2. [src/utils/paths.py](src/utils/paths.py)
3. [configs/train_config_v2_2hz.yaml](configs/train_config_v2_2hz.yaml)
4. [src/features/state_vector_v2.py](src/features/state_vector_v2.py)
5. [src/features/processed_v2.py](src/features/processed_v2.py)
6. [src/inference/realtime_engine.py](src/inference/realtime_engine.py)
7. [src/inference/memory_reader.py](src/inference/memory_reader.py)
8. [WORKLOG.md](WORKLOG.md)

如果你要维护旧版兼容路径，再回头看：

1. [configs/train_config.yaml](configs/train_config.yaml)
2. [src/parser/demo_parser.py](src/parser/demo_parser.py)
3. [src/features/state_vector.py](src/features/state_vector.py)
4. [analyze_demo.py](analyze_demo.py)

## 已知限制

- 标签仍然只覆盖能确认 `A/B` 的回合
- 地图包点框选依赖 `map_utils.py` 中的经验框，不是严格标注基准
- `analyze_demo.py` 还没有 `v2` 对应版本
- `tools/pipeline.py` 仍然是旧版 `v1` 解析链
- `src.inference.onnx_export` 目前仍然默认按旧版 `275` 维输入导出
- 内存读取与 HLTV 解压工具都明显偏 Windows 环境
