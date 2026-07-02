from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from trading.data import fetch_annual_returns, fetch_close_prices, fetch_vix_data, get_monthly_invest_dates
from trading.rebalance import apply_rebalance_to_plan
from trading.scenario_context import BaselineBuilder, ScenarioContext
from trading.strategies.dca import (
    DCAParams,
    RiskGuardConfig,
    SignalSnapshot,
    WeightAllocator,
    build_order_plan,
    fixed_weight_allocator,
    normalize_weights,
)


class _VectorbtProxy:
    def __init__(self) -> None:
        object.__setattr__(self, "_overrides", {})

    def __getattr__(self, name: str) -> Any:
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        import vectorbt as module

        return getattr(module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__getattribute__(self, "_overrides")[name] = value


vbt = _VectorbtProxy()


class _FallbackPortfolio:
    """Small local portfolio value engine used when vectorbt is unavailable."""

    def __init__(
        self,
        close: pd.DataFrame,
        size: pd.DataFrame,
        init_cash: float,
        fees: float,
        slippage: float,
        reason: Exception,
    ) -> None:
        self._close = close
        self._size = size.reindex_like(close).fillna(0.0)
        self._init_cash = float(init_cash)
        self._fees = float(fees)
        self._slippage = float(slippage)
        self.fallback_reason = reason

    def value(self) -> pd.Series:
        notional = self._size * self._close
        costs = notional.abs().sum(axis=1) * (self._fees + self._slippage)
        cash = self._init_cash - (notional.sum(axis=1) + costs).cumsum()
        positions = self._size.cumsum()
        value = cash + (positions * self._close).sum(axis=1)
        value.name = "value"
        return value

    def stats(self) -> pd.Series:
        return pd.Series(dtype=float)


@dataclass(frozen=True)
class BacktestResult:
    """单次回测结果：含组合、订单与各年权重（若有）。"""

    name: str
    portfolio: Any
    order_sizes: pd.DataFrame
    yearly_weights: pd.DataFrame | None
    annual_returns: pd.DataFrame
    total_invested: float
    fee_rate: float = 0.0
    slippage_rate: float = 0.0
    risk_trigger_count: int = 0
    decision_snapshot: pd.DataFrame | None = None


def _symbols_to_fetch(params: DCAParams) -> list[str]:
    all_syms = set(params.symbols) | set(params.extra_symbols) | set(params.signal_symbols)
    if params.vix_symbol:
        all_syms.add(params.vix_symbol)
    # 排除虚拟现金标的（不是真实 ticker）
    if params.cash_symbol:
        all_syms.discard(params.cash_symbol)
    return sorted(all_syms)


def _data_kwargs(params: DCAParams) -> dict[str, object]:
    return {
        "data_source": params.data_source,
        "local_data_dir": params.local_data_dir,
        "yf_max_retries": params.yf_max_retries,
        "yf_retry_sleep": params.yf_retry_sleep,
        "allow_stale_cache": params.allow_stale_cache,
    }


def _invest_months_and_total(
    monthly_budget: float, price_index: pd.DatetimeIndex
) -> tuple[int, float]:
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
) -> Any:
    try:
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
    except Exception as exc:  # noqa: BLE001
        return _FallbackPortfolio(
            close=close,
            size=order_sizes,
            init_cash=total_invested,
            fees=fee_rate,
            slippage=slippage_rate,
            reason=exc,
        )


def _dca_backtest(
    name: str,
    strategy_close: pd.DataFrame,
    monthly_budget: float,
    annual_returns: pd.DataFrame,
    default_weights: dict[str, float],
    allocator: WeightAllocator,
    fee_rate: float = 0.0,
    slippage_rate: float = 0.0,
    risk_config: RiskGuardConfig | None = None,
    *,
    vix_series: pd.Series | None = None,
    drawdown_lookback: int = 252,
    ma_window: int = 200,
    rebalance_max_weight: float | None = None,
    rebalance_mode: str = "sell",
    cash_symbol: str | None = None,
) -> BacktestResult:
    order_sizes, yearly_weights, decision_snapshot, risk_trigger_count = build_order_plan(
        asset_prices=strategy_close,
        monthly_budget=monthly_budget,
        annual_returns=annual_returns,
        default_weights=default_weights,
        allocator=allocator,
        risk_config=risk_config,
        vix_series=vix_series,
        drawdown_lookback=drawdown_lookback,
        ma_window=ma_window,
    )

    # 组合再平衡：叠加卖出订单
    if rebalance_max_weight is not None and rebalance_max_weight > 0:
        invest_dates = get_monthly_invest_dates(strategy_close.index)
        original_order_sizes = order_sizes.copy()
        order_sizes = apply_rebalance_to_plan(
            order_sizes, strategy_close, invest_dates, rebalance_max_weight,
            mode=rebalance_mode,
        )
        # 更新决策快照
        if rebalance_mode == "tilt":
            triggered = (order_sizes.round(12) != original_order_sizes.round(12)).any(axis=1)
        else:
            triggered = (order_sizes < 0).any(axis=1)
        rebalance_triggered = triggered.sum()
        if decision_snapshot is not None and rebalance_triggered > 0:
            decision_snapshot["rebalance_triggered"] = False
            for dt in invest_dates:
                if dt in decision_snapshot.index and triggered.get(dt, False):
                    decision_snapshot.loc[dt, "rebalance_triggered"] = True
        elif decision_snapshot is not None:
            decision_snapshot["rebalance_triggered"] = False

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
    full_close = fetch_close_prices(
        all_syms, params.start, params.end, use_cache=params.use_cache, **_data_kwargs(params)
    )
    strategy_close = full_close[symbols].dropna(how="any").copy()

    # 注入虚拟现金标的
    if params.cash_symbol:
        cash_col = params.cash_symbol
        if cash_col not in strategy_close.columns:
            strategy_close[cash_col] = 1.0
        symbols_with_cash = list(symbols) + [cash_col]
        strategy_close = strategy_close[symbols_with_cash]
        default_w = {s: params.default_weights[s] for s in symbols if s in params.default_weights}
        if cash_col in params.default_weights:
            default_w[cash_col] = params.default_weights[cash_col]
        else:
            default_w[cash_col] = 0.0
    else:
        default_w = {s: params.default_weights[s] for s in symbols if s in params.default_weights}

    if not default_w:
        default_w = {s: 1.0 / len(strategy_close.columns) for s in strategy_close.columns}
    default_w = normalize_weights(default_w)

    annual_returns = fetch_annual_returns(
        list(params.signal_symbols), params.start, params.end, use_cache=params.use_cache, **_data_kwargs(params)
    )

    vix_series = None
    if params.vix_symbol:
        vix_series = fetch_vix_data(
            params.vix_symbol, params.start, params.end, use_cache=params.use_cache, **_data_kwargs(params)
        )

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
        vix_series=vix_series,
        drawdown_lookback=params.drawdown_lookback,
        ma_window=params.ma_window,
        rebalance_max_weight=params.rebalance_max_weight,
        rebalance_mode=params.rebalance_mode,
        cash_symbol=params.cash_symbol,
    )


def _first_invest_day_weights(
    strategy_close: pd.DataFrame,
    annual_returns: pd.DataFrame,
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

    prices = strategy_close.loc[d0]
    signal = SignalSnapshot(
        invest_date=d0,
        invest_year=d0.year,
        annual_returns=annual_returns,
        drawdown=0.0,
        ma_deviation=0.0,
        vix=None,
        current_prices={c: float(prices[c]) for c in strategy_close.columns},
    )
    w = allocator(signal, default_weights=base_w)
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
    """主策略 + 若干 baseline。"""
    symbols = list(params.symbols)
    cash_col = params.cash_symbol
    needed_cols = [s for s in dict.fromkeys([*symbols, *params.extra_symbols]) if s != cash_col]
    all_syms = _symbols_to_fetch(params)
    full_close = fetch_close_prices(
        all_syms, params.start, params.end, use_cache=params.use_cache, **_data_kwargs(params)
    )
    aligned_close = full_close[needed_cols].dropna(how="any")
    strategy_symbols = [s for s in symbols if s != cash_col]
    strategy_close = aligned_close[strategy_symbols].copy()

    # 注入虚拟现金标的
    if cash_col:
        if cash_col not in strategy_close.columns:
            strategy_close[cash_col] = 1.0

    annual_returns = fetch_annual_returns(
        list(params.signal_symbols), params.start, params.end, use_cache=params.use_cache, **_data_kwargs(params)
    )

    vix_series = None
    if params.vix_symbol:
        vix_series = fetch_vix_data(
            params.vix_symbol, params.start, params.end, use_cache=params.use_cache, **_data_kwargs(params)
        )

    default_w = {s: params.default_weights[s] for s in strategy_close.columns if s in params.default_weights}
    if not default_w:
        default_w = {s: 1.0 / len(strategy_close.columns) for s in strategy_close.columns}
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
        vix_series=vix_series,
        drawdown_lookback=params.drawdown_lookback,
        ma_window=params.ma_window,
        rebalance_max_weight=params.rebalance_max_weight,
        rebalance_mode=params.rebalance_mode,
        cash_symbol=params.cash_symbol,
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
