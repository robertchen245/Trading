#!/usr/bin/env python3
"""从项目根目录运行: python scripts/generate_report.py [--png]"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from trading.baseline_builders import default_baseline_builders_v1
from trading.engine import run_scenarios
from trading.metrics import compare_portfolios
from trading.strategies.dca import DCAParams, fixed_weight_allocator, nasdaq_rule_allocator
from trading.viz import (
    fig_drawdown,
    fig_equity_comparison,
    fig_monthly_returns_heatmap,
    fig_rolling_sharpe,
    fig_summary_dashboard,
    fig_yearly_weights_stacked,
    write_figure_image,
    write_report_html,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def _default_params() -> DCAParams:
    return DCAParams(
        symbols=("QQQ", "TQQQ"),
        start="2016-01-01",
        end="2026-01-01",
        monthly_budget=5000.0,
        default_weights={"QQQ": 0.7, "TQQQ": 0.3},
        signal_symbol="^IXIC",
        benchmark_symbol="QQQ",
        use_cache=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="生成回测 HTML 报告（可选 PNG）")
    parser.add_argument("--png", action="store_true", help="导出 PNG（需安装 kaleido）")
    parser.add_argument(
        "--allocator",
        choices=("fixed", "nasdaq_rule"),
        default="fixed",
        help="主策略权重规则",
    )
    args = parser.parse_args()

    params = _default_params()
    allocator = nasdaq_rule_allocator if args.allocator == "nasdaq_rule" else fixed_weight_allocator
    results = run_scenarios(
        params,
        allocator=allocator,
        baseline_builders=default_baseline_builders_v1(params.benchmark_symbol),
    )

    table = compare_portfolios(results)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = REPORTS_DIR / f"report_{ts}"

    strategy = results["strategy"]
    figures: list[tuple[str, object]] = [
        ("场景指标", _table_figure(table)),
        ("总览（净值/回撤/超额）", fig_summary_dashboard(results)),
        ("净值对比", fig_equity_comparison(results)),
        ("回撤", fig_drawdown(results)),
        ("策略月度收益热力图", fig_monthly_returns_heatmap(strategy.portfolio.value())),
        ("策略滚动夏普", fig_rolling_sharpe(strategy)),
    ]
    if strategy.yearly_weights is not None and not strategy.yearly_weights.empty:
        figures.append(("年度目标权重", fig_yearly_weights_stacked(strategy.yearly_weights)))

    table.to_csv(prefix.with_suffix(".metrics.csv"))
    write_report_html(figures, prefix.with_suffix(".html"))

    if args.png:
        for title, fig in figures:
            safe = "".join(c if c.isalnum() else "_" for c in title)[:40]
            try:
                write_figure_image(fig, prefix.parent / f"{prefix.name}_{safe}.png")
            except Exception as e:  # noqa: BLE001
                print(f"跳过 PNG「{title}」: {e}")

    print(f"已写入: {prefix.with_suffix('.html')}")
    print(f"指标 CSV: {prefix.with_suffix('.metrics.csv')}")


def _table_figure(table: pd.DataFrame):
    import plotly.graph_objects as go

    # 格式化数值列便于阅读
    display = table.reset_index()
    return go.Figure(
        data=[
            go.Table(
                header=dict(values=list(display.columns), fill_color="paleturquoise", align="left"),
                cells=dict(
                    values=[display[c].astype(str) for c in display.columns],
                    align="left",
                ),
            ),
        ],
    ).update_layout(title="场景指标对比")


if __name__ == "__main__":
    main()
