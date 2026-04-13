from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import vectorbt as vbt

from trading.data import fetch_annual_returns, fetch_close_prices, get_monthly_invest_dates
from trading.scenario_context import BaselineBuilder, ScenarioContext
from trading.strategies.dca import (
    DCAParams,
    RiskGuardConfig,
    WeightAllocator,
    build_order_plan,
    fixed_weight_allocator,
    normalize_weights,
)


@dataclass(frozen=True)
class BacktestResult:
    """单次回测结果：含组合、订单与各年权重（若有）。"""

    name: str
    portfolio: vbt.Portfolio
    order_sizes: pd.DataFrame
    yearly_weights: pd.DataFrame | None
    annual_returns: pd.Series
    total_invested: float
    fee_rate: float = 0.0
    slippage_rate: float = 0.0
    risk_trigger_count: int = 0
    decision_snapshot: pd.DataFrame | None = None


def _symbols_to_fetch(params: DCAParams) -> list[str]:
    return sorted(set(params.symbols) | set(params.extra_symbols) | {params.signal_symbol})


def _invest_months_and_total(monthly_budget: float, price_index: pd.DatetimeIndex) -> tuple[int, float]:
    invest_dates = get_monthly_invest_dates(price_index)
    n = len(invest_dates)
    return n, monthly_budget * n


def portfolio_from_orders(
    close: pd.DataFrame,
    order_sizes: pd.DataFrame,
    total_invested: float,
    *,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
) -> vbt.Portfolio:
    return vbt.Portfolio.from_orders(
        close=close,
        size=order_sizes,
        init_cash=total_invested,
        fees=fee_rate,
        slippage=slippage_rate,
        cash_sharing=True,
        group_by=True,
        freq="1D",
    )


def _dca_backtest(
    name: str,
    strategy_close: pd.DataFrame,
    monthly_budget: float,
    annual_returns: pd.Series,
    default_weights: dict[str, float],
    allocator: WeightAllocator,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    risk_config: RiskGuardConfig | None = None,
) -> BacktestResult:
    order_sizes, yearly_weights, decision_snapshot, risk_trigger_count = build_order_plan(
        asset_prices=strategy_close,
        monthly_budget=monthly_budget,
        annual_returns=annual_returns,
        default_weights=default_weights,
        allocator=allocator,
        risk_config=risk_config,
    )
    _, total_invested = _invest_months_and_total(monthly_budget, strategy_close.index)
    portfolio = portfolio_from_orders(
        strategy_close,
        order_sizes,
        total_invested,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    return BacktestResult(
        name=name,
        portfolio=portfolio,
        order_sizes=order_sizes,
        yearly_weights=yearly_weights,
        annual_returns=annual_returns,
        total_invested=total_invested,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        risk_trigger_count=risk_trigger_count,
        decision_snapshot=decision_snapshot,
    )


def run_dca_portfolio(
    params: DCAParams,
    allocator: WeightAllocator = fixed_weight_allocator,
    *,
    name: str = "strategy",
) -> BacktestResult:
    """按 DCA 参数与权重分配器运行主策略。"""
    symbols = list(params.symbols)
    all_syms = _symbols_to_fetch(params)
    full_close = fetch_close_prices(all_syms, params.start, params.end, use_cache=params.use_cache)
    strategy_close = full_close[symbols].dropna(how="any")

    annual_returns = fetch_annual_returns(
        params.signal_symbol,
        params.start,
        params.end,
        use_cache=params.use_cache,
    )

    default_w = {s: params.default_weights[s] for s in symbols if s in params.default_weights}
    if not default_w:
        default_w = {s: 1.0 / len(symbols) for s in symbols}
    default_w = normalize_weights(default_w)

    return _dca_backtest(
        name,
        strategy_close,
        params.monthly_budget,
        annual_returns,
        default_w,
        allocator,
        fee_rate=params.fee_rate,
        slippage_rate=params.slippage_rate,
        risk_config=RiskGuardConfig(
            max_weight_per_asset=params.max_weight_per_asset,
            max_gross_exposure=params.max_gross_exposure,
            observe_only=params.risk_observe_only,
        ),
    )


def _first_invest_day_weights(
    strategy_close: pd.DataFrame,
    annual_returns: pd.Series,
    default_weights: dict[str, float],
    allocator: WeightAllocator,
) -> tuple[pd.Timestamp, dict[str, float]]:
    invest_dates = get_monthly_invest_dates(strategy_close.index)
    d0 = invest_dates[0]
    base_w = {
        c: default_weights[c]
        for c in strategy_close.columns
        if c in default_weights and default_weights[c] > 0
    }
    if not base_w:
        base_w = {c: 1.0 / len(strategy_close.columns) for c in strategy_close.columns}
    base_w = normalize_weights(base_w)
    w = allocator(invest_year=d0.year, annual_returns=annual_returns, default_weights=base_w)
    merged = {c: float(w.get(c, 0.0)) for c in strategy_close.columns}
    if sum(merged.values()) <= 0:
        merged = {c: 1.0 / len(strategy_close.columns) for c in strategy_close.columns}
    return d0, normalize_weights(merged)


def run_scenarios(
    params: DCAParams,
    allocator: WeightAllocator = fixed_weight_allocator,
    *,
    baseline_builders: Sequence[BaselineBuilder] = (),
) -> dict[str, BacktestResult]:
    """主策略 + 若干 baseline；行情列为 `symbols` ∪ `extra_symbols`，由调用方保证 builder 所用 ticker 已包含在内。"""
    symbols = list(params.symbols)
    needed_cols = list(dict.fromkeys([*symbols, *params.extra_symbols]))
    all_syms = _symbols_to_fetch(params)
    full_close = fetch_close_prices(all_syms, params.start, params.end, use_cache=params.use_cache)
    aligned_close = full_close[needed_cols].dropna(how="any")
    strategy_close = aligned_close[symbols]

    annual_returns = fetch_annual_returns(
        params.signal_symbol,
        params.start,
        params.end,
        use_cache=params.use_cache,
    )

    default_w = {s: params.default_weights[s] for s in symbols if s in params.default_weights}
    if not default_w:
        default_w = {s: 1.0 / len(symbols) for s in symbols}
    default_w = normalize_weights(default_w)

    strategy = _dca_backtest(
        "strategy",
        strategy_close,
        params.monthly_budget,
        annual_returns,
        default_w,
        allocator,
        fee_rate=params.fee_rate,
        slippage_rate=params.slippage_rate,
        risk_config=RiskGuardConfig(
            max_weight_per_asset=params.max_weight_per_asset,
            max_gross_exposure=params.max_gross_exposure,
            observe_only=params.risk_observe_only,
        ),
    )
    _, total_invested = _invest_months_and_total(params.monthly_budget, strategy_close.index)
    d0, w0 = _first_invest_day_weights(strategy_close, annual_returns, default_w, allocator)

    ctx = ScenarioContext(
        params=params,
        strategy_close=strategy_close,
        aligned_close=aligned_close,
        annual_returns=annual_returns,
        monthly_budget=params.monthly_budget,
        total_invested=total_invested,
        allocator=allocator,
        default_weights=default_w,
        first_invest_date=d0,
        first_month_weights=w0,
    )

    out: dict[str, BacktestResult] = {"strategy": strategy}
    seen: set[str] = {"strategy"}
    for builder in baseline_builders:
        res = builder(ctx)
        if res.name == "strategy":
            raise ValueError("Baseline result name must not be 'strategy'.")
        if res.name in seen:
            raise ValueError(f"Duplicate baseline BacktestResult.name: {res.name!r}")
        seen.add(res.name)
        out[res.name] = res
    return out
