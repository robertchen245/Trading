from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt


# 从项目根目录运行 / Run from the project root:
# python examples/dynamic_dca_vectorbt.py
START_DATE = "2016-01-01"
END_DATE = "2026-01-01"
MONTHLY_BUDGET = 5000.0
ASSET_SYMBOLS = ["QQQ", "TQQQ"]
NASDAQ_SYMBOL = "^IXIC"
DEFAULT_WEIGHTS = {
    "QQQ": 0.7,
    "TQQQ": 0.3,
}


@dataclass(frozen=True)
class BacktestResult:
    # 回测结果容器 / Container for backtest outputs
    portfolio: vbt.Portfolio
    order_sizes: pd.DataFrame
    yearly_weights: pd.DataFrame
    annual_returns: pd.Series
    total_invested: float


def _remove_timezone(index: pd.Index) -> pd.Index:
    # 统一移除时区，避免后续按年/月分组时出现索引兼容问题
    # Remove timezone info to avoid grouping/index alignment issues later
    if getattr(index, "tz", None) is not None:
        return index.tz_localize(None)
    return index


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    # 归一化权重，确保比例和为 1 / Normalize weights so they sum to 1
    total_weight = float(sum(weights.values()))
    if total_weight <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return {symbol: value / total_weight for symbol, value in weights.items()}


def fetch_close_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    # 下载收盘价并整理成统一格式 / Download close prices in a consistent format
    data = vbt.YFData.download(symbols, start=start, end=end)
    close = data.get("Close")
    if isinstance(close, pd.Series):
        close = close.to_frame()

    # 某些指数代码在数据源返回时会被清洗，例如 "^IXIC" 可能变成 "IXIC"。
    # Some index tickers may be normalized by the data source, for example "^IXIC" -> "IXIC".
    if len(symbols) == 1 and close.shape[1] == 1 and symbols[0] not in close.columns:
        close.columns = symbols

    close.index = _remove_timezone(close.index)
    close = close[symbols].dropna(how="any")
    return close.sort_index()


def fetch_nasdaq_annual_returns(
    symbol: str = NASDAQ_SYMBOL,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.Series:
    # 计算纳指每年末到下一年末的涨跌幅 / Calculate year-over-year Nasdaq returns
    close = fetch_close_prices([symbol], start=start, end=end)[symbol]
    annual_close = close.groupby(close.index.year).last()
    annual_returns = annual_close.pct_change().dropna()
    annual_returns.index.name = "year"
    annual_returns.name = f"{symbol}_annual_return"
    return annual_returns


def get_monthly_invest_dates(price_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # 每个月取第一个可交易日作为定投日 / Use the first trading day of each month
    calendar = pd.Series(price_index, index=price_index)
    invest_dates = calendar.groupby(calendar.index.to_period("M")).first()
    return pd.DatetimeIndex(invest_dates.to_list())


def allocation_for_year(
    invest_year: int,
    annual_returns: pd.Series,
    default_weights: dict[str, float],
) -> dict[str, float]:
    prev_year = invest_year - 1
    prev_return = annual_returns.get(prev_year, np.nan)

    # 这里是下一年配比逻辑的预留位置，目前先返回固定比例。
    # This is the placeholder for next-year allocation logic; for now we use fixed weights.
    #
    # 示例 / Example:
    # if prev_return > 0.20:
    #     return {"QQQ": 0.85, "TQQQ": 0.15}
    # if prev_return < 0:
    #     return {"QQQ": 0.60, "TQQQ": 0.40}
    _ = prev_return

    return normalize_weights(default_weights)


def build_order_sizes(
    asset_prices: pd.DataFrame,
    monthly_budget: float,
    annual_returns: pd.Series,
    default_weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # 把“每月投入金额”转换成“每个资产买多少股”
    # Convert monthly cash allocation into share order sizes per asset
    invest_dates = get_monthly_invest_dates(asset_prices.index)
    order_sizes = pd.DataFrame(0.0, index=asset_prices.index, columns=asset_prices.columns)

    yearly_weight_rows: list[dict[str, float]] = []
    seen_years: set[int] = set()

    for invest_date in invest_dates:
        weights = allocation_for_year(
            invest_year=invest_date.year,
            annual_returns=annual_returns,
            default_weights=default_weights,
        )
        prices_on_day = asset_prices.loc[invest_date]

        for symbol, weight in weights.items():
            cash_to_invest = monthly_budget * weight
            order_sizes.loc[invest_date, symbol] = cash_to_invest / prices_on_day[symbol]

        if invest_date.year not in seen_years:
            # 只记录每年的一次目标配比，便于输出查看
            # Record yearly target weights once for easier inspection
            yearly_weight_rows.append(
                {
                    "year": invest_date.year,
                    **{f"{symbol}_weight": weights[symbol] for symbol in asset_prices.columns},
                }
            )
            seen_years.add(invest_date.year)

    yearly_weights = pd.DataFrame(yearly_weight_rows).set_index("year")
    return order_sizes, yearly_weights


def run_backtest(
    start: str = START_DATE,
    end: str = END_DATE,
    monthly_budget: float = MONTHLY_BUDGET,
    default_weights: dict[str, float] | None = None,
) -> BacktestResult:
    # 当前示例：先抓纳指年度涨幅，再按规则生成每月订单
    # Current example: fetch annual Nasdaq returns first, then build monthly orders
    weights = normalize_weights(default_weights or DEFAULT_WEIGHTS)
    annual_returns = fetch_nasdaq_annual_returns(start=start, end=end)
    asset_prices = fetch_close_prices(ASSET_SYMBOLS, start=start, end=end)
    order_sizes, yearly_weights = build_order_sizes(
        asset_prices=asset_prices,
        monthly_budget=monthly_budget,
        annual_returns=annual_returns,
        default_weights=weights,
    )

    invest_months = int((order_sizes > 0).any(axis=1).sum())
    total_invested = monthly_budget * invest_months

    # 这里用一次性初始现金来近似“每月追加投入”，未使用现金会留到账户中等待下次定投。
    # We approximate periodic contributions with upfront cash; unused cash remains idle until the next DCA date.
    portfolio = vbt.Portfolio.from_orders(
        close=asset_prices,
        size=order_sizes,
        init_cash=total_invested,
        cash_sharing=True,
        group_by=True,
        freq="1D",
    )

    return BacktestResult(
        portfolio=portfolio,
        order_sizes=order_sizes,
        yearly_weights=yearly_weights,
        annual_returns=annual_returns,
        total_invested=total_invested,
    )


def print_summary(result: BacktestResult) -> None:
    portfolio_value = result.portfolio.value()
    final_value = float(portfolio_value.iloc[-1])
    total_return_pct = (final_value / result.total_invested - 1.0) * 100.0

    print("=== Dynamic DCA Backtest / 动态定投回测（当前为占位配比规则） ===")
    print(f"Total invested / 累计投入: {result.total_invested:,.2f}")
    print(f"Final portfolio value / 期末组合价值: {final_value:,.2f}")
    print(f"Total return / 总收益率: {total_return_pct:.2f}%")
    print()

    print("=== Yearly allocation used / 各年使用的配比 ===")
    print(result.yearly_weights.to_string(float_format=lambda x: f"{x:.2%}"))
    print()

    print("=== Nasdaq annual returns / 纳指年度涨幅 ===")
    print(result.annual_returns.to_string(float_format=lambda x: f"{x:.2%}"))
    print()

    print("=== vectorbt stats / 回测统计 ===")
    print(result.portfolio.stats().to_string())


def main() -> None:
    result = run_backtest()
    print_summary(result)

if __name__ == "__main__":
    main()
