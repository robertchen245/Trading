# Trading — DCA 回测引擎

基于 **vectorbt** 的月度定投回测框架。可插拔权重分配器、多信号系统、多 baseline 对比、Plotly 多图报告。

## 快速开始

```bash
# 1. 创建环境（首次）
conda create -n trade python=3.12 -y
conda activate trade

# 2. 安装项目 + 依赖
pip install -e .
pip install yfinance       # vectorbt 下载行情需要，暂未打入 pyproject.toml

# 3. 跑冒烟测试（首次需下载行情，约 10-30 秒）
pytest tests/ -v

# 4. CLI 一键运行
trading run QQQ,TQQQ --allocator nasdaq_rule
trading experiment --presets
trading report QQQ,TQQQ --allocator smart --vix ^VIX
trading list
trading show smart_signal_fusion
```

> 在中国大陆需设置代理：`export HTTPS_PROXY=http://127.0.0.1:7890`（端口按实际）

## CLI 命令

| 命令 | 用途 | 示例 |
|------|------|------|
| `trading run SYM,SYM` | 单策略回测 + 指标表 | `trading run QQQ,TQQQ --allocator smart --vix ^VIX --start 2021 --end 2025` |
| `trading experiment --presets` | 批量实验 + 排名 CSV | `trading experiment --presets` |
| `trading experiment --spec spec.json` | 自定义 spec 批量实验 | |
| `trading report SYM,SYM` | 生成 HTML 报告 | `trading report QQQ,TQQQ --allocator smart --vix ^VIX --output my_report.html` |
| `trading list` | 列出所有策略和分配器 | `trading list` |
| `trading show NAME` | 查看 preset 详情 | `trading show smart_signal_fusion` |

常用参数：`--start 2021`（年份自动补全为 2021-01-01）、`--budget 5000`、`--weights QQQ=0.7,TQQQ=0.3`、`--signals ^IXIC,^GSPC`

## 项目结构

```
Trading/
├── trading/                 # 核心库
│   ├── data.py              # 行情下载 + Parquet 缓存 + 多信号计算
│   ├── engine.py            # BacktestResult、run_scenarios
│   ├── strategies/dca.py    # DCAParams、SignalSnapshot、WeightAllocator
│   ├── baseline_builders.py # monthly_full_invest、lump_sum、等权
│   ├── scenario_context.py  # ScenarioContext、BaselineBuilder 协议
│   ├── specs.py             # StrategySpec、preset 模板、NL→策略
│   ├── experiment.py        # 批量实验 + 排名 + ALLOCATOR_REGISTRY
│   ├── metrics.py           # CAGR、夏普、回撤、超额序列
│   └── viz.py               # Plotly 图 + HTML/PNG 报告
├── examples/                # 独立运行脚本
├── tests/                   # pytest (10 tests)
├── data/cache/              # Parquet 缓存（gitignore）
└── reports/                 # 输出目录（gitignore）
```

## Agent 报告生成指南

当需要基于回测结果写报告、生成交互式 artifact，或接入 Hermes/Codex 等 agent 输出时，优先使用 agent 中立报告包，不要直接从零猜测应该读取哪些 CSV。

### 报告包入口

使用 `--format package|codex|all` 会在 `--package-dir` 下生成报告包：

```bash
trading report QQQ,SMH --format all --package-dir reports/latest_package
```

每个报告包中，agent 应先读取：

1. `agent_report_index.json` — agent 入口索引，声明推荐读取顺序、文件角色、是否必需、报告用途。
2. `manifest.json` — 回测标题、生成时间、参数、场景列表、数据集注册表。
3. `metrics.csv` — 首要指标表，用来形成摘要、场景排序和收益/风险结论。

### 数据职责

| 文件 | 是否必需 | 报告用途 |
|------|----------|----------|
| `metrics.csv` | 必需 | 摘要、收益/风险排名、指标表 |
| `equity_curve.csv` | 必需 | 组合净值走势、终值路径对比 |
| `drawdown.csv` | 必需 | 回撤风险、压力阶段解释 |
| `monthly_returns.csv` | 必需 | 月度波动、短周期收益分布 |
| `decision_snapshot.csv` | 可选 | 动态策略的信号、权重、预算执行解释 |
| `yearly_weights.csv` | 可选 | 分配器年度目标权重解释 |

CSV 是事实来源；`codex_manifest.json`、`codex_snapshot.json`、Hermes 输出或其他 artifact 文件都应视为面向具体界面的派生结果。若报告包里存在 `agent_report_index.json`，自动 agent 必须优先遵循其中的 `recommended_read_order` 和 `files[].report_use`。

## 核心概念

### DCAParams

月度 DCA 参数对象（frozen dataclass）：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `symbols` | tuple | 必填 | 主策略标的 |
| `start` / `end` | str | 必填 | 回测区间 |
| `monthly_budget` | float | 必填 | 每月投入总额 |
| `default_weights` | dict | 必填 | 默认权重（自动归一化） |
| `signal_symbols` | tuple | `("^IXIC",)` | **多信号标的**，用于年收益/回撤/MA 计算 |
| `vix_symbol` | str\|None | None | VIX 恐惧指数 ticker |
| `drawdown_lookback` | int | 252 | 年内回撤计算窗口（交易日） |
| `ma_window` | int | 200 | 均线偏离计算窗口 |
| `extra_symbols` | tuple | `()` | 仅 baseline 用到的 ticker |
| `max_weight_per_asset` | float\|None | None | 单资产权重上限 |
| `max_gross_exposure` | float\|None | None | 月度预算使用上限 |

向后兼容：`signal_symbol` 属性返回 `signal_symbols[0]`。

### SignalSnapshot（新）

每个定投日自动构建的不可变信号对象。分配器通过它获取所有维度的市场信号：

```python
@dataclass(frozen=True)
class SignalSnapshot:
    invest_date: pd.Timestamp      # 定投日
    invest_year: int               # 定投所在自然年
    annual_returns: pd.DataFrame   # 多指数年收益 (index=year, columns=symbols)
    drawdown: float                # 年内回撤 (-0.20 = 跌 20%)
    ma_deviation: float            # (close - MA) / MA
    vix: float | None              # VIX 当日值
    current_prices: dict           # {symbol: close_price}
```

### WeightAllocator（新协议）

```python
def my_allocator(
    signal: SignalSnapshot,    # 所有信号维度
    default_weights: dict,     # 默认权重
) -> dict[str, float]:         # 返回权重（会被归一化）
```

内置分配器：

| 分配器 | 使用的信号 | 规则 |
|--------|-----------|------|
| `fixed_weight_allocator` | 无 | 始终返回 default_weights |
| `equal_weight_allocator` | 无 | 等权 |
| `nasdaq_rule_allocator` | `annual_returns["^IXIC"]` | 上年涨>20%偏QQQ，跌<0偏TQQQ |
| `smart_allocator` | `drawdown` + `vix` + `annual_returns` | 恐慌+高波动重仓TQQQ，牛年后保守 |

### 向后兼容适配器

旧签名的分配器可用 `adapt_legacy_allocator` 包装：

```python
def old_alloc(invest_year, annual_returns, default_weights) -> dict: ...
wrapped = adapt_legacy_allocator(old_alloc)
```

### BaselineBuilder (Protocol)

```python
def my_baseline(ctx: ScenarioContext) -> BacktestResult: ...
```

内置：
- `monthly_full_invest("QQQ")` — 每月全买单一标的
- `lump_sum_first_day()` — 首日一次性投入
- `equal_weight_monthly_on_strategy_universe()` — 等权 DCA

### 运行模型

```
行情下载 → 对齐 close → 计算多信号(drawdown/MA/VIX)
→ 按月构建 SignalSnapshot → 分配器决策 → order_sizes
→ vectorbt.Portfolio.from_orders → BacktestResult → metrics / viz
```

- `init_cash` = `total_invested`，`cash_sharing=True`
- 定投日 = 每月第一个可用交易日
- 决策快照包含所有信号维度（drawdown、MA 偏离、VIX）

## 新增一个分配器

1. 在 `trading/strategies/dca.py` 实现符合 `WeightAllocator` 签名的函数：

```python
def my_alloc(signal: SignalSnapshot, dw: dict) -> dict:
    if signal.drawdown < -0.3:
        return {"QQQ": 0.4, "TQQQ": 0.6}  # 恐慌抄底
    return dw
```

2. 在 `experiment.py` 的 `ALLOCATOR_REGISTRY` 注册
3. 在 `specs.py` 的 `SUPPORTED_ALLOCATORS` 中添加；如需 preset，也同步添加到 `preset_strategy_specs`

## 环境

- **Conda env**: `trade`（Python 3.12, vectorbt 0.28.5）
- **依赖**: vectorbt, pandas, numpy, plotly, pyarrow, yfinance
- **代理**: yfinance 在中国大陆需要代理；设置 `HTTP_PROXY`/`HTTPS_PROXY`

## 已知问题 & 处理

| 问题 | 处理 |
|------|------|
| vectorbt 1.0 YFData.download 列对齐 bug | data.py 已改为 yfinance 逐一下载 |
| yfinance 快速请求会限流 | `fetch_close_prices` 中每个 ticker 间隔 0.3s |
| `^IXIC` 列名可能被 vectorbt 清洗 | `fetch_annual_returns` 独立用 yfinance 拉取 |
| 首次拉取行情需联网 | 后面走 `data/cache/` Parquet 缓存 |

## 策略扩展想法（未实现）

- **分级加仓分配器** — 标的跌 10%/20%/30% 时动态提升权重
- **75% 上限/再平衡分配器** — 单资产占比超阈值时卖出换现金
- **QLD 跟进日分配器** — 大盘确认反转后用 2x 杠杆追
- **多资产混合** — 股票 ETF + BTC 同框回测
- **周投/双周投** — 将 `get_monthly_invest_dates` 泛化为周期参数
