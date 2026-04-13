from __future__ import annotations

import numpy as np
import pandas as pd

from trading.engine import BacktestResult


def equity_curve(result: BacktestResult) -> pd.Series:
    s = result.portfolio.value()
    s.name = result.name
    return s


def _years_between(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 0.0
    delta = index[-1] - index[0]
    return max(delta.days / 365.25, 1e-9)


def portfolio_metrics_row(
    result: BacktestResult,
    *,
    risk_free_annual: float = 0.02,
    trading_days_per_year: int = 252,
) -> pd.Series:
    """单行量化指标（与 total_invested 对齐的终值与收益率）。"""
    v = equity_curve(result)
    idx = pd.DatetimeIndex(v.index)
    years = _years_between(idx)
    final_value = float(v.iloc[-1])
    invested = float(result.total_invested)
    total_return = final_value / invested - 1.0 if invested > 0 else float("nan")
    cagr = (final_value / invested) ** (1.0 / years) - 1.0 if invested > 0 and years > 0 else float("nan")

    daily = v.pct_change().dropna()
    vol_annual = float(daily.std() * np.sqrt(trading_days_per_year)) if len(daily) > 1 else float("nan")
    rf_daily = risk_free_annual / trading_days_per_year
    excess = daily - rf_daily
    sharpe = (
        float(excess.mean() / daily.std() * np.sqrt(trading_days_per_year))
        if len(daily) > 1 and daily.std() > 0
        else float("nan")
    )

    downside = daily[daily < 0]
    down_std = downside.std()
    sortino = (
        float((daily.mean() - rf_daily) / down_std * np.sqrt(trading_days_per_year))
        if down_std and down_std > 0
        else float("nan")
    )

    running_max = v.cummax()
    dd = v / running_max - 1.0
    max_dd = float(dd.min())
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else float("nan")

    try:
        stats = result.portfolio.stats()
        stat_dict = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
    except Exception:  # noqa: BLE001
        stat_dict = {}

    row = {
        "scenario": result.name,
        "final_value": final_value,
        "total_invested": invested,
        "total_return": total_return,
        "CAGR": cagr,
        "max_drawdown": max_dd,
        "volatility_annual": vol_annual,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "start": idx[0],
        "end": idx[-1],
        "years": years,
        "fee_rate": float(result.fee_rate),
        "slippage_rate": float(result.slippage_rate),
        "risk_trigger_count": int(result.risk_trigger_count),
    }
    if result.decision_snapshot is not None and not result.decision_snapshot.empty:
        budget_util = pd.to_numeric(result.decision_snapshot.get("budget_utilization"), errors="coerce")
        if budget_util is not None:
            row["avg_budget_utilization"] = float(budget_util.mean())
    for k in ("Win Rate [%]", "Profit Factor", "Expectancy", "Omega Ratio"):
        if k in stat_dict:
            row[k] = stat_dict[k]

    return pd.Series(row)


def portfolio_metrics_table(results: dict[str, BacktestResult], **kwargs) -> pd.DataFrame:
    rows = [portfolio_metrics_row(r, **kwargs) for r in results.values()]
    return pd.DataFrame(rows).set_index("scenario")


def infer_monthly_full_baseline_key(results: dict[str, BacktestResult]) -> str | None:
    """从场景名中选取 `monthly_full_*` 字典序第一条，供默认超额对比。"""
    keys = [k for k in results if k != "strategy" and k.startswith("monthly_full_")]
    return sorted(keys)[0] if keys else None


def compare_portfolios(
    results: dict[str, BacktestResult],
    *,
    baseline_key: str | None = None,
    **kwargs,
) -> pd.DataFrame:
    """场景指标表；`baseline_key` 为 None 时尝试 `infer_monthly_full_baseline_key` 计算 `vs_baseline_final_ratio`。"""
    table = portfolio_metrics_table(results, **kwargs)
    key = baseline_key if baseline_key is not None else infer_monthly_full_baseline_key(results)
    if key is not None and key in table.index:
        base_final = float(table.loc[key, "final_value"])
        if base_final > 0:
            table["vs_baseline_final_ratio"] = table["final_value"] / base_final - 1.0
    return table


def excess_equity_vs_baseline(
    strategy: BacktestResult,
    baseline: BacktestResult,
) -> pd.Series:
    """策略净值相对 baseline 净值减 1（按日期内连接）。"""
    vs = strategy.portfolio.value()
    vb = baseline.portfolio.value()
    joined = pd.concat([vs, vb], axis=1, join="inner")
    joined.columns = ["strategy", "baseline"]
    out = joined["strategy"] / joined["baseline"] - 1.0
    out.name = "excess_vs_baseline"
    return out
