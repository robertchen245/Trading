from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from trading.data import get_monthly_invest_dates


class WeightAllocator(Protocol):
    def __call__(
        self,
        invest_year: int,
        annual_returns: pd.Series,
        default_weights: dict[str, float],
    ) -> dict[str, float]: ...


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total_weight = float(sum(weights.values()))
    if total_weight <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return {symbol: value / total_weight for symbol, value in weights.items()}


def fixed_weight_allocator(
    invest_year: int,
    annual_returns: pd.Series,
    default_weights: dict[str, float],
) -> dict[str, float]:
    _ = invest_year, annual_returns
    return normalize_weights(default_weights)


def nasdaq_rule_allocator(
    invest_year: int,
    annual_returns: pd.Series,
    default_weights: dict[str, float],
) -> dict[str, float]:
    """根据上一年度信号标的涨跌幅调整 QQQ/TQQQ 配比（计划中的示例规则）；缺标的时退回固定权重。"""
    prev_year = invest_year - 1
    prev_return = annual_returns.get(prev_year, np.nan)
    keys = set(default_weights.keys())
    if keys >= {"QQQ", "TQQQ"}:
        if prev_return > 0.20:
            return normalize_weights({"QQQ": 0.85, "TQQQ": 0.15})
        if prev_return < 0:
            return normalize_weights({"QQQ": 0.60, "TQQQ": 0.40})
    return normalize_weights(default_weights)


def equal_weight_allocator(
    invest_year: int,
    annual_returns: pd.Series,
    default_weights: dict[str, float],
) -> dict[str, float]:
    _ = invest_year, annual_returns
    symbols = tuple(default_weights.keys())
    n = len(symbols)
    if n == 0:
        raise ValueError("default_weights must be non-empty.")
    return {s: 1.0 / n for s in symbols}


@dataclass(frozen=True)
class DCAParams:
    symbols: tuple[str, ...]
    start: str
    end: str
    monthly_budget: float
    default_weights: dict[str, float]
    signal_symbol: str = "^IXIC"
    benchmark_symbol: str = "QQQ"
    """便于示例与 `default_baseline_builders_v1`；`run_scenarios` 不再隐式使用。"""
    extra_symbols: tuple[str, ...] = ()
    """仅用于 baseline 等、不在主策略 `symbols` 中的行情列；须覆盖所有 builder 用到的 ticker。"""
    use_cache: bool = True
    fee_rate: float = 0.0
    slippage_rate: float = 0.0
    max_weight_per_asset: float | None = None
    max_gross_exposure: float | None = None
    risk_observe_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "default_weights", normalize_weights(dict(self.default_weights)))
        if self.fee_rate < 0:
            raise ValueError("fee_rate must be >= 0.")
        if self.slippage_rate < 0:
            raise ValueError("slippage_rate must be >= 0.")
        if self.max_weight_per_asset is not None and not (0 < self.max_weight_per_asset <= 1.0):
            raise ValueError("max_weight_per_asset must be in (0, 1].")
        if self.max_gross_exposure is not None and self.max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be > 0.")


@dataclass(frozen=True)
class RiskGuardConfig:
    max_weight_per_asset: float | None = None
    max_gross_exposure: float | None = None
    observe_only: bool = False


def _apply_weight_cap(
    weights: dict[str, float],
    max_weight_per_asset: float | None,
) -> tuple[dict[str, float], bool]:
    if max_weight_per_asset is None:
        return normalize_weights(weights), False
    capped = normalize_weights(weights)
    changed = False
    for _ in range(len(capped) + 1):
        over = [symbol for symbol, weight in capped.items() if weight > max_weight_per_asset]
        if not over:
            break
        changed = True
        excess = sum(capped[symbol] - max_weight_per_asset for symbol in over)
        for symbol in over:
            capped[symbol] = max_weight_per_asset
        under = [symbol for symbol in capped if symbol not in over]
        if not under:
            break
        under_sum = sum(capped[symbol] for symbol in under)
        if under_sum <= 0:
            equal_add = excess / len(under)
            for symbol in under:
                capped[symbol] += equal_add
        else:
            for symbol in under:
                capped[symbol] += excess * (capped[symbol] / under_sum)
    return normalize_weights(capped), changed


def build_order_sizes(
    asset_prices: pd.DataFrame,
    monthly_budget: float,
    annual_returns: pd.Series,
    default_weights: dict[str, float],
    allocator: WeightAllocator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    order_sizes, yearly_weights, _, _ = build_order_plan(
        asset_prices=asset_prices,
        monthly_budget=monthly_budget,
        annual_returns=annual_returns,
        default_weights=default_weights,
        allocator=allocator,
    )
    return order_sizes, yearly_weights


def build_order_plan(
    asset_prices: pd.DataFrame,
    monthly_budget: float,
    annual_returns: pd.Series,
    default_weights: dict[str, float],
    allocator: WeightAllocator,
    risk_config: RiskGuardConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    invest_dates = get_monthly_invest_dates(asset_prices.index)
    order_sizes = pd.DataFrame(0.0, index=asset_prices.index, columns=asset_prices.columns)
    cfg = risk_config or RiskGuardConfig()

    yearly_weight_rows: list[dict[str, float]] = []
    seen_years: set[int] = set()
    decision_rows: list[dict[str, float | int | bool | pd.Timestamp]] = []
    risk_trigger_count = 0

    base_w = {
        c: default_weights[c] for c in asset_prices.columns if c in default_weights and default_weights[c] > 0
    }
    if not base_w:
        base_w = {c: 1.0 / len(asset_prices.columns) for c in asset_prices.columns}
    base_w = normalize_weights(base_w)

    for invest_date in invest_dates:
        weights_raw = allocator(
            invest_year=invest_date.year,
            annual_returns=annual_returns,
            default_weights=base_w,
        )
        merged = {c: float(weights_raw.get(c, 0.0)) for c in asset_prices.columns}
        if sum(merged.values()) <= 0:
            merged = {c: 1.0 / len(asset_prices.columns) for c in asset_prices.columns}
        alloc_weights = normalize_weights(merged)
        guarded_weights, weight_cap_triggered = _apply_weight_cap(
            alloc_weights,
            cfg.max_weight_per_asset,
        )
        exposure_multiplier = min(cfg.max_gross_exposure, 1.0) if cfg.max_gross_exposure is not None else 1.0
        exposure_triggered = exposure_multiplier < 1.0
        risk_triggered = weight_cap_triggered or exposure_triggered
        if risk_triggered:
            risk_trigger_count += 1

        applied_weights = alloc_weights if cfg.observe_only else guarded_weights
        applied_budget = monthly_budget if cfg.observe_only else monthly_budget * exposure_multiplier

        prices_on_day = asset_prices.loc[invest_date]
        for symbol, weight in applied_weights.items():
            if symbol not in order_sizes.columns:
                continue
            cash_to_invest = applied_budget * weight
            order_sizes.loc[invest_date, symbol] = cash_to_invest / prices_on_day[symbol]

        if invest_date.year not in seen_years:
            yearly_weight_rows.append(
                {
                    "year": invest_date.year,
                    **{f"{symbol}_weight": applied_weights[symbol] for symbol in asset_prices.columns},
                }
            )
            seen_years.add(invest_date.year)
        decision_row: dict[str, float | int | bool | pd.Timestamp] = {
            "invest_date": invest_date,
            "invest_year": invest_date.year,
            "signal_return_prev_year": float(annual_returns.get(invest_date.year - 1, np.nan)),
            "planned_budget": monthly_budget,
            "applied_budget": applied_budget,
            "budget_utilization": applied_budget / monthly_budget if monthly_budget > 0 else np.nan,
            "weight_cap_triggered": weight_cap_triggered,
            "gross_exposure_triggered": exposure_triggered,
            "risk_triggered": risk_triggered,
            "risk_observe_only": cfg.observe_only,
        }
        for symbol in asset_prices.columns:
            decision_row[f"{symbol}_w_pre"] = alloc_weights[symbol]
            decision_row[f"{symbol}_w_post"] = guarded_weights[symbol]
            decision_row[f"{symbol}_w_applied"] = applied_weights[symbol]
        decision_rows.append(decision_row)

    yearly_weights = pd.DataFrame(yearly_weight_rows)
    if not yearly_weights.empty:
        yearly_weights = yearly_weights.set_index("year")
    decision_snapshot = pd.DataFrame(decision_rows)
    if not decision_snapshot.empty:
        decision_snapshot = decision_snapshot.set_index("invest_date")
    return order_sizes, yearly_weights, decision_snapshot, risk_trigger_count


def monthly_single_asset_orders(
    asset_prices: pd.DataFrame,
    monthly_budget: float,
    symbol: str,
) -> pd.DataFrame:
    """每月全额买入单一标的（baseline）。"""
    invest_dates = get_monthly_invest_dates(asset_prices.index)
    order_sizes = pd.DataFrame(0.0, index=asset_prices.index, columns=asset_prices.columns)
    if symbol not in order_sizes.columns:
        raise ValueError(f"Symbol {symbol} not in price columns.")
    for invest_date in invest_dates:
        price = asset_prices.loc[invest_date, symbol]
        order_sizes.loc[invest_date, symbol] = monthly_budget / price
    return order_sizes


def lump_sum_orders(
    asset_prices: pd.DataFrame,
    total_cash: float,
    weights: dict[str, float],
    invest_date: pd.Timestamp,
) -> pd.DataFrame:
    """在指定日一次性按权重买入并持有。"""
    order_sizes = pd.DataFrame(0.0, index=asset_prices.index, columns=asset_prices.columns)
    w = normalize_weights({c: weights.get(c, 0.0) for c in asset_prices.columns})
    prices_on_day = asset_prices.loc[invest_date]
    for symbol, weight in w.items():
        cash = total_cash * weight
        order_sizes.loc[invest_date, symbol] = cash / prices_on_day[symbol]
    return order_sizes
