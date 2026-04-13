#!/usr/bin/env python3
"""从项目根目录运行: python scripts/generate_report.py [--png]"""
from __future__ import annotations

import argparse
from dataclasses import replace
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


def _params_from_args(args: argparse.Namespace) -> DCAParams:
    base = _default_params()
    return replace(
        base,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        max_weight_per_asset=args.max_weight_per_asset,
        max_gross_exposure=args.max_gross_exposure,
        risk_observe_only=args.risk_observe_only,
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
    parser.add_argument("--fee-rate", type=float, default=0.0, help="单笔成交手续费率（如 0.001=0.1%）")
    parser.add_argument("--slippage-rate", type=float, default=0.0, help="单笔成交滑点率（如 0.0005=0.05%）")
    parser.add_argument(
        "--max-weight-per-asset",
        type=float,
        default=None,
        help="单资产权重上限（0~1），为空表示不限制",
    )
    parser.add_argument(
        "--max-gross-exposure",
        type=float,
        default=None,
        help="每次定投预算使用上限（如 0.8 表示最多投入月预算 80%）",
    )
    parser.add_argument(
        "--risk-observe-only",
        action="store_true",
        help="仅统计风控触发，不实际截断下单",
    )
    args = parser.parse_args()

    params = _params_from_args(args)
    allocator = nasdaq_rule_allocator if args.allocator == "nasdaq_rule" else fixed_weight_allocator
    results = run_scenarios(
        params,
        allocator=allocator,
        baseline_builders=default_baseline_builders_v1(params.benchmark_symbol),
    )

    table = compare_portfolios(results)
    cost_impact = _cost_impact_table(params, allocator, results)
    risk_diag = _risk_diagnostics_table(results["strategy"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = REPORTS_DIR / f"report_{ts}"

    strategy = results["strategy"]
    figures: list[tuple[str, object]] = [
        ("场景指标", _table_figure(table, title="场景指标对比")),
        ("成本前后对比", _table_figure(cost_impact, title="成本前后指标变化")),
        ("风控诊断", _table_figure(risk_diag, title="风控触发与预算执行")),
        ("总览（净值/回撤/超额）", fig_summary_dashboard(results)),
        ("净值对比", fig_equity_comparison(results)),
        ("回撤", fig_drawdown(results)),
        ("策略月度收益热力图", fig_monthly_returns_heatmap(strategy.portfolio.value())),
        ("策略滚动夏普", fig_rolling_sharpe(strategy)),
    ]
    if strategy.yearly_weights is not None and not strategy.yearly_weights.empty:
        figures.append(("年度目标权重", fig_yearly_weights_stacked(strategy.yearly_weights)))

    table.to_csv(prefix.with_suffix(".metrics.csv"))
    cost_impact.to_csv(prefix.with_name(f"{prefix.name}.cost_impact.csv"))
    if strategy.decision_snapshot is not None and not strategy.decision_snapshot.empty:
        strategy.decision_snapshot.to_csv(prefix.with_name(f"{prefix.name}.decision_snapshot.csv"))
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
    print(f"成本对比 CSV: {prefix.with_name(f'{prefix.name}.cost_impact.csv')}")
    if strategy.decision_snapshot is not None and not strategy.decision_snapshot.empty:
        print(f"决策快照 CSV: {prefix.with_name(f'{prefix.name}.decision_snapshot.csv')}")


def _cost_impact_table(
    params: DCAParams,
    allocator,
    results_with_cost,
) -> pd.DataFrame:
    with_cost_table = compare_portfolios(results_with_cost)
    if params.fee_rate == 0.0 and params.slippage_rate == 0.0:
        no_cost_table = with_cost_table.copy()
    else:
        no_cost_params = replace(params, fee_rate=0.0, slippage_rate=0.0)
        no_cost_results = run_scenarios(
            no_cost_params,
            allocator=allocator,
            baseline_builders=default_baseline_builders_v1(params.benchmark_symbol),
        )
        no_cost_table = compare_portfolios(no_cost_results)

    rows: list[dict[str, float | str]] = []
    for scenario in with_cost_table.index:
        if scenario not in no_cost_table.index:
            continue
        rows.append(
            {
                "scenario": scenario,
                "final_value_no_cost": float(no_cost_table.loc[scenario, "final_value"]),
                "final_value_with_cost": float(with_cost_table.loc[scenario, "final_value"]),
                "final_value_delta": float(with_cost_table.loc[scenario, "final_value"])
                - float(no_cost_table.loc[scenario, "final_value"]),
                "CAGR_no_cost": float(no_cost_table.loc[scenario, "CAGR"]),
                "CAGR_with_cost": float(with_cost_table.loc[scenario, "CAGR"]),
                "CAGR_delta": float(with_cost_table.loc[scenario, "CAGR"])
                - float(no_cost_table.loc[scenario, "CAGR"]),
                "max_drawdown_no_cost": float(no_cost_table.loc[scenario, "max_drawdown"]),
                "max_drawdown_with_cost": float(with_cost_table.loc[scenario, "max_drawdown"]),
                "max_drawdown_delta": float(with_cost_table.loc[scenario, "max_drawdown"])
                - float(no_cost_table.loc[scenario, "max_drawdown"]),
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def _risk_diagnostics_table(strategy_result) -> pd.DataFrame:
    if strategy_result.decision_snapshot is None or strategy_result.decision_snapshot.empty:
        return pd.DataFrame(
            [
                {
                    "metric": "risk_trigger_count",
                    "value": strategy_result.risk_trigger_count,
                }
            ]
        ).set_index("metric")
    snap = strategy_result.decision_snapshot
    metrics = {
        "risk_trigger_count": int(strategy_result.risk_trigger_count),
        "weight_cap_trigger_count": int(pd.to_numeric(snap["weight_cap_triggered"]).sum()),
        "gross_exposure_trigger_count": int(pd.to_numeric(snap["gross_exposure_triggered"]).sum()),
        "avg_budget_utilization": float(pd.to_numeric(snap["budget_utilization"]).mean()),
        "min_budget_utilization": float(pd.to_numeric(snap["budget_utilization"]).min()),
        "max_budget_utilization": float(pd.to_numeric(snap["budget_utilization"]).max()),
    }
    return pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    ).set_index("metric")


def _table_figure(table: pd.DataFrame, *, title: str):
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
    ).update_layout(title=title)


if __name__ == "__main__":
    main()
