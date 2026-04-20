# CS Prophet

一个面向 CS2 demo 的炸弹落点预测项目：从职业比赛 `.dem` 中解析回合状态序列，训练 Transformer 模型预测该回合最终会在 `A` 点还是 `B` 点下包，并支持离线回放分析、ONNX 导出和基于 GSI 的实时推理。

## 项目定位

当前仓库的真实主线不是“回合胜负预测”，而是：

- 输入：一回合内从 `freeze_end` 开始到下包前的状态序列
- 输出：`A / B` 两类下包点概率
- 标签来源：`bomb_planted` 事件
- 数据筛选：只保留能确认是 `A` 或 `B` 的回合，`other` 不进入训练集

如果 `bomb_planted` 事件里带有坐标，代码会优先用 `src/utils/map_utils.py` 中的地图包点框选规则判断 A/B；如果没有坐标，再回退到事件自带的 `site` 字段。

## 目前已经实现的能力

- `.dem -> parquet` 解析流水线
- 275 维状态向量构建
- 基于 T/CT 双流输入的 Transformer + Cross-Attention
- YAML 配置驱动训练
- 最佳 checkpoint 保存与加载
- ONNX 导出与结果校验
- 离线 parquet 回放分析
- 通过 CS2 GSI 做实时推理，并在本地网页展示 A/B 概率
- HLTV demo 下载与“下载后立即解析”的 pipeline

同时也有几块内容仍然是占位或实验性质：

- `dashboard/app.py` 只是 Streamlit 占位页
- 真正可用的实时面板是 `src/inference/realtime_engine.py` 启动后托管的 `dashboard/index.html`
- `src/features/feature_engineering.py` 仍是未实现占位
- `docs/superpowers/` 下的规划文档包含早期方案，和当前代码不完全一致，应以源码与配置为准

## 数据与建模流程

### 1. Demo 解析

`src/parser/demo_parser.py` 的核心逻辑：

- 使用 `demoparser2` 读取 `.dem`
- 从 `round_freeze_end` 开始取有效回合窗口；若没有则回退到 `round_start`
- 原始 64 tick/s 下采样到 8 tick/s
- 最多保留 90 秒，即最多 720 个 step
- 每个 step 展平成一行 parquet
- 每边按玩家名排序后固定到 `5T + 5CT`，不足位置补零

输出目录默认是 `data/processed/`，每个 demo 生成一个 parquet。

### 2. 特征表示

`src/features/state_vector.py` 将每一行 parquet 编码成 275 维向量：

- `0:135`：T 方 5 名玩家，每人 27 维
- `135:270`：CT 方 5 名玩家，每人 27 维
- `270:275`：全局特征

单个玩家的 27 维包含：

- 坐标、血量、护甲、头盔、存活
- 5 维角色 one-hot
- 7 维武器类别 one-hot
- 投掷物持有情况
- 致盲时长、装备价值、开镜、拆包

### 3. 模型

`src/model/transformer.py` 中的 `BombSiteTransformer`：

- T 方输入：`135` 玩家特征 + `5` 全局特征 = `140` 维
- CT 方输入：`135` 维
- 双流线性投影后分别加位置编码
- T 流对 CT 流做 cross-attention
- 再经过 self-attention encoder
- 最后从“最后一个真实 timestep”做 A/B 二分类

### 4. 训练

`src/model/train.py` 使用：

- `FocalLoss`
- AMP 混合精度
- gradient accumulation
- demo 级别切分 train/val/test，避免同一 demo 泄漏到多个集合
- early stopping

默认训练配置在 `configs/train_config.yaml`。

## 环境安装

### 方案一：安装完整依赖

适合要跑解析、训练、测试、HLTV 工具链的情况。

```powershell
pip install -r requirements.txt
```

### 方案二：按包方式安装核心运行依赖

适合只想用核心推理/训练代码。

```powershell
pip install -e .
```

如果还需要开发依赖：

```powershell
pip install -e ".[dev]"
```

## 快速开始

### 1. 批量解析本地 demo

```powershell
python parse_demos.py --demo-dir data/raw/demos --out-dir data/processed --resume
```

也可以直接调用模块版入口：

```powershell
python -m src.parser.demo_parser data/raw/demos data/processed
```

### 2. 训练模型

```powershell
python -m src.model.train --config configs/train_config.yaml
```

快速烟雾测试配置：

```powershell
python -m src.model.train --config configs/train_config.smoke.yaml
```

### 3. 运行测试

```powershell
pytest tests -v
```

### 4. 离线分析单个 parquet

会在固定几个时间点输出 A/B 概率，方便观察模型在回合推进过程中的判断变化。

```powershell
python analyze_demo.py data/processed/your_demo.parquet --checkpoint checkpoints/best.pt
```

### 5. 导出 ONNX

```powershell
python -m src.inference.onnx_export --checkpoint checkpoints/best.pt --output model.onnx
```

### 6. 启动实时推理面板

```powershell
python -m src.inference.realtime_engine --checkpoint checkpoints/best.pt --port 3000 --device cpu
```

启动后打开：

```text
http://localhost:3000
```

要让 CS2 把 GSI 推送给本地服务，需要把下面这个配置文件放到 CS2 的 Game State Integration 目录：

```text
cfg/gamestate_integration_cs_prophet.cfg
```

### 7. 生成 parquet 质检报告

```powershell
python viz_parquet.py --per-map 2
```

结果默认输出到 `data/viz_report.html`。

## HLTV 工具链

仓库里有两条相关脚本线：

### 只下载 demo

```powershell
python tools/download_demos.py --config tools/hltv_config.yaml
```

### 下载后立即解析，不在磁盘长期保留 `.dem`

```powershell
python tools/pipeline.py --config tools/hltv_config.yaml
```

相关说明：

- 配置文件：`tools/hltv_config.yaml`
- 下载记录：`data/raw/manifest.jsonl`
- 失败日志：`data/raw/failed.jsonl`
- 支持代理配置
- 主仓库的解压实现依赖 `Bandizip` 的 `bz.exe`，明显偏 Windows 环境

另外，`checkpoints/smoke/portable_copy/` 里还有一套便携版离线下载/解析脚本，用于把 Python、依赖和解压工具打包到 U 盘场景；这部分是辅助实验产物，不是主仓库运行主线。

## 目录说明

```text
CS_Prophet/
├─ configs/                  训练与推理配置
├─ cfg/                      CS2 GSI 配置文件
├─ dashboard/                实时面板静态页与占位 Streamlit 文件
├─ data/
│  ├─ raw/                   原始 demo、manifest、失败日志
│  ├─ processed/             解析后的 parquet
│  └─ viz_report.html        质检报告示例
├─ docs/superpowers/         历史规划与设计文档
├─ notebooks/                EDA notebook
├─ src/
│  ├─ parser/                demo 解析
│  ├─ features/              标签、向量、数据集
│  ├─ model/                 注意力模块、Transformer、训练
│  ├─ inference/             checkpoint 推理、GSI、ONNX
│  └─ utils/                 地图坐标与包点工具
├─ tests/                    单元测试与工具链测试
├─ tools/                    HLTV 下载与 pipeline
├─ analyze_demo.py           离线回放分析
├─ parse_demos.py            批量解析入口
└─ viz_parquet.py            parquet 质检报告生成
```

## 当前仓库里值得注意的现实情况

- 仓库已经带有大量本地数据、parquet、raw demos 和 checkpoint，不是一个“空模板”
- `checkpoints/best.pt` 与 `checkpoints/smoke/best.pt` 已存在，可直接用于推理或导出
- `WORKLOG.md` 里记录了近期训练诊断与标签质量判断，适合先读一遍
- `CLAUDE.md` 对项目现状总结得比较接近代码，但其中部分测试数量等描述可能已过时

## 已知限制

- 当前训练目标是 `A/B` 二分类，不包含 `other`
- 地图包点框选是经验校准，不是严格标注基准
- 玩家角色虽然在特征里有位置，但默认解析流程通常不会自动补齐真实角色
- `dashboard/app.py` 不是生产可用界面
- 历史计划文档里提到的 `74` 维、`124` 维、`3` 分类等方案，属于旧设计，不代表当前实现

## 建议阅读顺序

如果你是第一次接手这个仓库，推荐按下面顺序看：

1. `WORKLOG.md`
2. `configs/train_config.yaml`
3. `src/parser/demo_parser.py`
4. `src/features/state_vector.py`
5. `src/model/transformer.py`
6. `src/model/train.py`
7. `src/inference/realtime_engine.py`

这样能最快理解项目的数据流和真实运行方式。
