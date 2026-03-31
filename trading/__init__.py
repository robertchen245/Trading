"""回测策略引擎：定投、baseline 对比、指标与可视化。"""

from trading.baseline_builders import (
    default_baseline_builders_v1,
    equal_weight_monthly_on_strategy_universe,
    lump_sum_first_day,
    monthly_full_invest,
)
from trading.engine import BacktestResult, portfolio_from_orders, run_dca_portfolio, run_scenarios
from trading.metrics import compare_portfolios, infer_monthly_full_baseline_key, portfolio_metrics_table
from trading.scenario_context import BaselineBuilder, ScenarioContext
from trading.strategies.dca import (
    DCAParams,
    equal_weight_allocator,
    fixed_weight_allocator,
    nasdaq_rule_allocator,
    normalize_weights,
)

__all__ = [
    "BacktestResult",
    "BaselineBuilder",
    "DCAParams",
    "ScenarioContext",
    "compare_portfolios",
    "default_baseline_builders_v1",
    "equal_weight_allocator",
    "equal_weight_monthly_on_strategy_universe",
    "fixed_weight_allocator",
    "infer_monthly_full_baseline_key",
    "lump_sum_first_day",
    "monthly_full_invest",
    "nasdaq_rule_allocator",
    "normalize_weights",
    "portfolio_from_orders",
    "portfolio_metrics_table",
    "run_dca_portfolio",
    "run_scenarios",
]
