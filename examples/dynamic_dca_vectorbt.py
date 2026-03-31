"""从项目根目录运行: python examples/dynamic_dca_vectorbt.py"""

from __future__ import annotations

from trading.engine import BacktestResult, run_dca_portfolio
from trading.metrics import portfolio_metrics_row
from trading.strategies.dca import DCAParams, fixed_weight_allocator


def print_summary(result: BacktestResult) -> None:
    portfolio_value = result.portfolio.value()
    final_value = float(portfolio_value.iloc[-1])
    total_return_pct = (final_value / result.total_invested - 1.0) * 100.0

    print("=== Dynamic DCA Backtest / 动态定投回测 ===")
    print(f"Total invested / 累计投入: {result.total_invested:,.2f}")
    print(f"Final portfolio value / 期末组合价值: {final_value:,.2f}")
    print(f"Total return / 总收益率: {total_return_pct:.2f}%")
    print()

    if result.yearly_weights is not None:
        print("=== Yearly allocation used / 各年使用的配比 ===")
        print(result.yearly_weights.to_string(float_format=lambda x: f"{x:.2%}"))
        print()

    print("=== Nasdaq annual returns / 纳指年度涨幅 ===")
    print(result.annual_returns.to_string(float_format=lambda x: f"{x:.2%}"))
    print()

    print("=== Key metrics / 核心指标 ===")
    print(portfolio_metrics_row(result).to_string())
    print()

    print("=== vectorbt stats / 回测统计 ===")
    print(result.portfolio.stats().to_string())


def main() -> None:
    params = DCAParams(
        symbols=("QQQ", "TQQQ"),
        start="2016-01-01",
        end="2026-01-01",
        monthly_budget=5000.0,
        default_weights={"QQQ": 0.7, "TQQQ": 0.3},
        signal_symbol="^IXIC",
        benchmark_symbol="QQQ",
        use_cache=True,
    )
    result = run_dca_portfolio(params, allocator=fixed_weight_allocator)
    print_summary(result)


if __name__ == "__main__":
    main()
