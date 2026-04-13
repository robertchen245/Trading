from __future__ import annotations

from trading.engine import BacktestResult, portfolio_from_orders
from trading.scenario_context import BaselineBuilder, ScenarioContext
from trading.strategies.dca import (
    build_order_sizes,
    equal_weight_allocator,
    lump_sum_orders,
    monthly_single_asset_orders,
)


def monthly_full_invest(symbol: str) -> BaselineBuilder:
    """每月全额买入单一标的；`symbol` 须出现在 `ctx.aligned_close` 中。"""

    name = f"monthly_full_{symbol}"

    def build(ctx: ScenarioContext) -> BacktestResult:
        sub = ctx.aligned_close[[symbol]]
        order_sizes = monthly_single_asset_orders(sub, ctx.monthly_budget, symbol)
        portfolio = portfolio_from_orders(
            sub,
            order_sizes,
            ctx.total_invested,
            fee_rate=ctx.params.fee_rate,
            slippage_rate=ctx.params.slippage_rate,
        )
        return BacktestResult(
            name=name,
            portfolio=portfolio,
            order_sizes=order_sizes,
            yearly_weights=None,
            annual_returns=ctx.annual_returns,
            total_invested=ctx.total_invested,
            fee_rate=ctx.params.fee_rate,
            slippage_rate=ctx.params.slippage_rate,
        )

    return build


def lump_sum_first_day() -> BaselineBuilder:
    """首个定投日按 `ctx.first_month_weights` 一次性投入 `total_invested` 并持有。"""

    name = "lump_sum_first_day"

    def build(ctx: ScenarioContext) -> BacktestResult:
        order_sizes = lump_sum_orders(
            ctx.strategy_close,
            ctx.total_invested,
            ctx.first_month_weights,
            ctx.first_invest_date,
        )
        portfolio = portfolio_from_orders(
            ctx.strategy_close,
            order_sizes,
            ctx.total_invested,
            fee_rate=ctx.params.fee_rate,
            slippage_rate=ctx.params.slippage_rate,
        )
        return BacktestResult(
            name=name,
            portfolio=portfolio,
            order_sizes=order_sizes,
            yearly_weights=None,
            annual_returns=ctx.annual_returns,
            total_invested=ctx.total_invested,
            fee_rate=ctx.params.fee_rate,
            slippage_rate=ctx.params.slippage_rate,
        )

    return build


def equal_weight_monthly_on_strategy_universe() -> BaselineBuilder:
    """在主策略标的上每月等权 DCA。"""

    name = "equal_weight_monthly"

    def build(ctx: ScenarioContext) -> BacktestResult:
        n = len(ctx.strategy_close.columns)
        eq_w = {c: 1.0 / n for c in ctx.strategy_close.columns}
        order_sizes, yearly_weights = build_order_sizes(
            asset_prices=ctx.strategy_close,
            monthly_budget=ctx.monthly_budget,
            annual_returns=ctx.annual_returns,
            default_weights=eq_w,
            allocator=equal_weight_allocator,
        )
        portfolio = portfolio_from_orders(
            ctx.strategy_close,
            order_sizes,
            ctx.total_invested,
            fee_rate=ctx.params.fee_rate,
            slippage_rate=ctx.params.slippage_rate,
        )
        return BacktestResult(
            name=name,
            portfolio=portfolio,
            order_sizes=order_sizes,
            yearly_weights=yearly_weights,
            annual_returns=ctx.annual_returns,
            total_invested=ctx.total_invested,
            fee_rate=ctx.params.fee_rate,
            slippage_rate=ctx.params.slippage_rate,
        )

    return build


def default_baseline_builders_v1(benchmark_symbol: str = "QQQ") -> tuple[BaselineBuilder, ...]:
    """与早期版本三条内置 baseline 等价的 builder 元组，需显式传入 `run_scenarios`。"""
    return (
        monthly_full_invest(benchmark_symbol),
        lump_sum_first_day(),
        equal_weight_monthly_on_strategy_universe(),
    )
