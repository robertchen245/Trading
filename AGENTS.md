# Trading — DCA 回测引擎

基于 **vectorbt** 的月度定投回测框架。可插拔权重分配器、多 baseline 对比、Plotly 多图报告。

## 快速开始

```bash
# 1. 创建环境（首次）
conda create -n trade python=3.12 -y
conda activate trade

# 2. 安装项目 + 依赖
pip install -e .
pip install yfinance       # vectorbt 下载行情需要，暂未打入 pyproject.toml

# 3. 跑冒烟测试（首次需下载行情，约 10-30 秒）
python3 examples/dynamic_dca_vectorbt.py
pytest tests/ -v

# 4. 批量实验 & 报告
python3 scripts/run_experiments.py
python3 scripts/generate_report.py
```

> 在中国大陆需设置代理：`export HTTPS_PROXY=http://127.0.0.1:7890`（端口按实际）

## 项目结构

```
Trading/
├── trading/                 # 核心库
│   ├── data.py              # 行情下载 + Parquet 缓存
│   ├── engine.py            # BacktestResult、run_scenarios
│   ├── strategies/dca.py    # DCAParams、WeightAllocator、订单构造
│   ├── baseline_builders.py # monthly_full_invest、lump_sum、等权
│   ├── scenario_context.py  # ScenarioContext、BaselineBuilder 协议
│   ├── specs.py             # StrategySpec、preset 模板
│   ├── experiment.py        # 批量实验 + 排名
│   ├── metrics.py           # CAGR、夏普、回撤、超额序列
│   └── viz.py               # Plotly 图 + HTML/PNG 报告
├── examples/                # 独立运行脚本
├── scripts/                 # run_experiments、generate_report
├── tests/                   # pytest
├── data/cache/              # Parquet 缓存（gitignore）
└── reports/                 # 输出目录（gitignore）
```

## 核心概念

### DCAParams

月度 DCA 参数对象（frozen dataclass）：

- `symbols` — 主策略标的元组
- `start` / `end` — 回测区间
- `monthly_budget` — 每月投入总额
- `default_weights` — 默认资产权重（自动归一化）
- `signal_symbol` — 信号标的，用于计算年度涨跌幅（默认 `^IXIC`）
- `extra_symbols` — 仅 baseline 用到的额外 ticker
- `max_weight_per_asset` — 单资产权重上限
- `max_gross_exposure` — 月度预算使用上限

### WeightAllocator (Protocol)

```python
def my_allocator(
    invest_year: int,           # 定投所在自然年
    annual_returns: pd.Series,  # signal_symbol 各年涨跌幅
    default_weights: dict,      # 默认权重
) -> dict[str, float]:          # 返回权重（会被归一化）
```

内置：
- `fixed_weight_allocator` — 始终返回 default_weights
- `nasdaq_rule_allocator` — 上一年涨 >20% 偏 QQQ，跌 <0 偏 TQQQ
- `equal_weight_allocator` — 等权

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
行情下载 → 对齐 close → 按月生成 order_sizes → vectorbt.Portfolio.from_orders
→ BacktestResult → metrics / viz
```

- `init_cash` = `total_invested`，`cash_sharing=True`
- 定投日 = 每月第一个可用交易日
- 分配器按自然年调用一次，同一年各月权重相同

## 新增一个分配器

1. 在 `trading/strategies/dca.py` 实现符合 `WeightAllocator` 签名的函数
2. 传入 `run_dca_portfolio(..., allocator=my_allocator)` 或 `run_scenarios`
3. 若需按自然语言触发，在 `specs.py` 的 `nl_to_strategy_spec` 和 `SUPPORTED_ALLOCATORS` 中注册

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
