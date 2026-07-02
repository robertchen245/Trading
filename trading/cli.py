#!/usr/bin/env python3
"""Trading CLI — 统一的命令行入口。

用法:
  trading run QQQ,TQQQ --start 2021 --end 2026 --budget 5000 \\
      --weights QQQ=0.7,TQQQ=0.3 --allocator nasdaq_rule

  trading experiment --presets                   # 跑 preset 批量实验
  trading experiment --spec spec.json            # 跑自定义 spec

  trading report QQQ,TQQQ --allocator smart \\
      --vix ^VIX --output my_report.html

  trading list                                   # 列出所有策略/分配器
  trading show balanced_fixed                    # 查看 preset 详情
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd


def _parse_weights(raw: str) -> dict[str, float]:
    """解析 'QQQ=0.7,TQQQ=0.3' → {'QQQ': 0.7, 'TQQQ': 0.3}"""
    out = {}
    for pair in raw.split(","):
        k, v = pair.split("=")
        out[k.strip()] = float(v)
    return out


def _parse_symbols(raw: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in raw.split(","))


def _normalize_date(raw: str) -> str:
    """'2021' → '2021-01-01', '2025' → '2025-01-01', 完整日期直接返回。"""
    raw = raw.strip()
    if len(raw) == 4 and raw.isdigit():
        return f"{raw}-01-01"
    return raw


def _add_common_dca_args(parser: argparse.ArgumentParser, require_symbols: bool = True) -> None:
    if require_symbols:
        parser.add_argument("symbols", help="标的列表 (逗号分隔)，如 QQQ,TQQQ")
    parser.add_argument("--start", default="2016-01-01", help="起始日期 (默认 2016-01-01)")
    parser.add_argument("--end", default="2026-01-01", help="结束日期 (默认 2026-01-01)")
    parser.add_argument("--budget", type=float, default=5000.0, help="月度定投金额 (默认 5000)")
    parser.add_argument("--weights", default="QQQ=0.7,TQQQ=0.3", help="权重 (如 QQQ=0.7,TQQQ=0.3)")
    parser.add_argument(
        "--allocator",
        default="fixed",
        choices=["fixed", "nasdaq_rule", "equal_weight", "smart", "trend_follow", "momentum_rotation"],
        help="分配器 (默认 fixed)",
    )
    parser.add_argument("--benchmark", default="QQQ", help="基准标的 (默认 QQQ)")
    parser.add_argument("--signals", default="^IXIC", help="信号标的，逗号分隔 (默认 ^IXIC)")
    parser.add_argument("--vix", default=None, help="VIX 标的 (如 ^VIX)")
    parser.add_argument("--fee", "--fee-rate", dest="fee", type=float, default=0.0, help="手续费率")
    parser.add_argument("--slippage", "--slippage-rate", dest="slippage", type=float, default=0.0, help="滑点率")
    parser.add_argument("--max-weight", "--max-weight-per-asset", dest="max_weight", type=float, default=None, help="单资产权重上限")
    parser.add_argument("--max-exposure", "--max-gross-exposure", dest="max_exposure", type=float, default=None, help="预算使用上限")
    parser.add_argument("--risk-observe-only", action="store_true", help="仅统计风控触发，不实际截断下单")
    parser.add_argument("--rebalance-max", type=float, default=None, help="再平衡阈值 (如 0.75)")
    parser.add_argument("--rebalance-mode", default="sell", choices=["sell", "tilt"],
                        help="再平衡模式: sell(卖出超阈值) 或 tilt(倾斜新资金)")
    parser.add_argument("--cash", default=None, help="虚拟现金标的名称 (如 CASH)")
    parser.add_argument("--no-cache", action="store_true", help="禁用行情缓存")
    parser.add_argument("--data-source", default="auto", choices=["auto", "auto_ibkr", "local", "ibkr", "yfinance", "stooq"],
                        help="行情数据源: auto/auto_ibkr/local/ibkr/yfinance/stooq")
    parser.add_argument("--local-data-dir", default=None, help="本地行情 CSV/parquet 目录")
    parser.add_argument("--yf-retries", type=int, default=3, help="yfinance 单标的最大重试次数")
    parser.add_argument("--yf-retry-sleep", type=float, default=1.0, help="yfinance 重试等待秒数")
    parser.add_argument("--no-stale-cache", action="store_true", help="所有在线源失败时不使用旧缓存兜底")


def cmd_run(args: argparse.Namespace) -> None:
    """运行单策略回测，打印指标表。"""
    from trading import (
        DCAParams,
        compare_portfolios,
        default_baseline_builders_v1,
        run_scenarios,
    )
    from trading.experiment import resolve_allocator

    symbols = _parse_symbols(args.symbols)
    weights = _parse_weights(args.weights)
    signal_symbols = _parse_symbols(args.signals)
    allocator = resolve_allocator(args.allocator)
    start = _normalize_date(args.start)
    end = _normalize_date(args.end)

    params = DCAParams(
        symbols=symbols,
        start=start,
        end=end,
        monthly_budget=args.budget,
        default_weights=weights,
        signal_symbols=signal_symbols,
        vix_symbol=args.vix,
        benchmark_symbol=args.benchmark,
        use_cache=not args.no_cache,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
        max_weight_per_asset=args.max_weight,
        max_gross_exposure=args.max_exposure,
        risk_observe_only=args.risk_observe_only,
        rebalance_max_weight=args.rebalance_max,
        rebalance_mode=args.rebalance_mode,
        cash_symbol=args.cash,
        data_source=args.data_source,
        local_data_dir=args.local_data_dir,
        yf_max_retries=args.yf_retries,
        yf_retry_sleep=args.yf_retry_sleep,
        allow_stale_cache=not args.no_stale_cache,
    )

    results = run_scenarios(
        params,
        allocator=allocator,
        baseline_builders=default_baseline_builders_v1(args.benchmark),
    )

    table = compare_portfolios(results)

    # 美化输出
    pd.set_option("display.float_format", "{:,.2f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)

    cols = ["final_value", "total_invested", "CAGR", "max_drawdown", "sharpe", "vs_baseline_final_ratio"]
    available = [c for c in cols if c in table.columns]
    print(table[available].to_string())


def cmd_experiment(args: argparse.Namespace) -> None:
    """批量实验。"""
    from datetime import datetime

    from trading.experiment import run_experiments
    from trading.specs import StrategySpec, preset_strategy_specs

    Path("reports").mkdir(exist_ok=True)

    if args.presets:
        specs = preset_strategy_specs()
    elif args.spec:
        with open(args.spec) as f:
            payloads = json.load(f)
        if isinstance(payloads, dict):
            payloads = [payloads]
        specs = [StrategySpec.from_dict(p) for p in payloads]
    elif getattr(args, "symbols", None):
        # 用命令行参数构造单个 spec
        symbols = _parse_symbols(args.symbols)
        weights = _parse_weights(args.weights)
        specs = [
            StrategySpec(
                name=args.name or "cli_experiment",
                symbols=symbols,
                start=_normalize_date(args.start),
                end=_normalize_date(args.end),
                monthly_budget=args.budget,
                default_weights=weights,
                allocator=args.allocator,
                signal_symbols=_parse_symbols(args.signals),
                vix_symbol=args.vix,
                benchmark_symbol=args.benchmark,
                use_cache=not args.no_cache,
                data_source=args.data_source,
                local_data_dir=args.local_data_dir,
                yf_max_retries=args.yf_retries,
                yf_retry_sleep=args.yf_retry_sleep,
                allow_stale_cache=not args.no_stale_cache,
                fee_rate=args.fee,
                slippage_rate=args.slippage,
                max_weight_per_asset=args.max_weight,
                max_gross_exposure=args.max_exposure,
                risk_observe_only=args.risk_observe_only,
                rebalance_max_weight=args.rebalance_max,
                rebalance_mode=args.rebalance_mode,
                cash_symbol=args.cash,
            )
        ]
    else:
        print("错误: 需要 --presets、--spec 或 symbols 参数", file=sys.stderr)
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path(f"reports/experiments_{ts}.summary.csv")
    ranking_path = Path(f"reports/experiments_{ts}.ranking.csv")

    summary, ranking = run_experiments(specs, output_dir=Path("reports"), ts=ts)
    print(f"策略数: {len(specs)}")
    print(f"汇总: {summary_path}")
    print(f"排名: {ranking_path}")

    if not ranking.empty:
        print()
        pd.set_option("display.float_format", "{:,.2f}".format)
        pd.set_option("display.width", 200)
        cols = ["strategy_name", "final_value", "CAGR", "max_drawdown", "sharpe", "rank"]
        available = [c for c in cols if c in ranking.columns]
        print(ranking[available].to_string(index=False))


def cmd_report(args: argparse.Namespace) -> None:
    """生成可视化报告与 agent 可读报告包。"""
    from trading import (
        DCAParams,
        compare_portfolios,
        default_baseline_builders_v1,
        run_scenarios,
    )
    from trading.experiment import resolve_allocator
    from trading.reporting import build_report_package, write_codex_artifact, write_report_package
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

    symbols = _parse_symbols(args.symbols)
    weights = _parse_weights(args.weights)
    signal_symbols = _parse_symbols(args.signals)
    allocator = resolve_allocator(args.allocator)
    start = _normalize_date(args.start)
    end = _normalize_date(args.end)

    params = DCAParams(
        symbols=symbols,
        start=start,
        end=end,
        monthly_budget=args.budget,
        default_weights=weights,
        signal_symbols=signal_symbols,
        vix_symbol=args.vix,
        benchmark_symbol=args.benchmark,
        use_cache=not args.no_cache,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
        max_weight_per_asset=args.max_weight,
        max_gross_exposure=args.max_exposure,
        risk_observe_only=args.risk_observe_only,
        rebalance_max_weight=args.rebalance_max,
        rebalance_mode=args.rebalance_mode,
        cash_symbol=args.cash,
        data_source=args.data_source,
        local_data_dir=args.local_data_dir,
        yf_max_retries=args.yf_retries,
        yf_retry_sleep=args.yf_retry_sleep,
        allow_stale_cache=not args.no_stale_cache,
    )

    results = run_scenarios(
        params,
        allocator=allocator,
        baseline_builders=default_baseline_builders_v1(args.benchmark),
    )

    package = build_report_package(
        results,
        title=f"{','.join(symbols)} 回测报告",
        params=params,
        allocator_name=args.allocator,
    )
    output_format = args.format
    table = compare_portfolios(results)
    cost_impact = _cost_impact_table(params, allocator, results)
    risk_diag = _risk_diagnostics_table(results["strategy"])
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

    if output_format in {"html", "all"}:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        prefix = out.with_suffix("")
        table.to_csv(prefix.with_suffix(".metrics.csv"))
        cost_impact.to_csv(prefix.with_name(f"{prefix.name}.cost_impact.csv"))
        if strategy.decision_snapshot is not None and not strategy.decision_snapshot.empty:
            strategy.decision_snapshot.to_csv(prefix.with_name(f"{prefix.name}.decision_snapshot.csv"))
        write_report_html(figures, out)
        print(f"HTML 报告已生成: {out}")
        print(f"指标 CSV: {prefix.with_suffix('.metrics.csv')}")
        print(f"成本对比 CSV: {prefix.with_name(f'{prefix.name}.cost_impact.csv')}")
        if strategy.decision_snapshot is not None and not strategy.decision_snapshot.empty:
            print(f"决策快照 CSV: {prefix.with_name(f'{prefix.name}.decision_snapshot.csv')}")

    if output_format in {"package", "codex", "all"}:
        package_dir = Path(args.package_dir)
        write_report_package(package, package_dir)
        print(f"通用报告包已生成: {package_dir}")

    if output_format in {"codex", "all"}:
        manifest_path, snapshot_path = write_codex_artifact(package, args.package_dir)
        print(f"Codex artifact manifest: {manifest_path}")
        print(f"Codex artifact snapshot: {snapshot_path}")

    if args.png and output_format in {"html", "all"}:
        out = Path(args.output)
        prefix = out.with_suffix("")
        for title, fig in figures:
            safe = "".join(c if c.isalnum() else "_" for c in title)[:40]
            try:
                image_path = prefix.parent / f"{prefix.name}_{safe}.png"
                write_figure_image(fig, image_path)
            except Exception as exc:  # noqa: BLE001
                print(f"跳过 PNG「{title}」: {exc}")


def _cost_impact_table(params, allocator, results_with_cost) -> pd.DataFrame:
    from trading import compare_portfolios, default_baseline_builders_v1, run_scenarios

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
    weight_cap_triggered = (
        pd.to_numeric(snap["weight_cap_triggered"]).sum()
        if "weight_cap_triggered" in snap
        else 0
    )
    gross_exposure_triggered = (
        pd.to_numeric(snap["gross_exposure_triggered"]).sum()
        if "gross_exposure_triggered" in snap
        else 0
    )
    metrics = {
        "risk_trigger_count": int(strategy_result.risk_trigger_count),
        "weight_cap_trigger_count": int(weight_cap_triggered),
        "gross_exposure_trigger_count": int(gross_exposure_triggered),
        "avg_budget_utilization": float(pd.to_numeric(snap["budget_utilization"]).mean()),
        "min_budget_utilization": float(pd.to_numeric(snap["budget_utilization"]).min()),
        "max_budget_utilization": float(pd.to_numeric(snap["budget_utilization"]).max()),
    }
    return pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    ).set_index("metric")


def _table_figure(table: pd.DataFrame, *, title: str):
    import plotly.graph_objects as go

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


def cmd_list(args: argparse.Namespace) -> None:
    """列出所有可用策略和分配器。"""
    from trading.experiment import ALLOCATOR_REGISTRY
    from trading.specs import SUPPORTED_ALLOCATORS, preset_strategy_specs

    print("=== 分配器 ===")
    for name in SUPPORTED_ALLOCATORS:
        fn = ALLOCATOR_REGISTRY.get(name)
        doc = getattr(fn, "__doc__", "") or ""
        first_line = doc.strip().split("\n")[0] if doc else "(无描述)"
        print(f"  {name:<20} {first_line[:60]}")

    print()
    print("=== 预设策略 ===")
    for spec in preset_strategy_specs():
        print(f"  {spec.name:<25} {spec.allocator:<15} "
              f"{','.join(spec.symbols):<20} "
              f"budget={spec.monthly_budget:,.0f} "
              f"start={spec.start}")


def cmd_show(args: argparse.Namespace) -> None:
    """显示预设策略详情。"""
    from trading.specs import preset_strategy_specs

    for spec in preset_strategy_specs():
        if spec.name == args.name:
            print(f"名称: {spec.name}")
            print(f"标的: {spec.symbols}")
            print(f"区间: {spec.start} → {spec.end}")
            print(f"月预算: {spec.monthly_budget:,.0f}")
            print(f"权重: {spec.default_weights}")
            print(f"分配器: {spec.allocator}")
            print(f"信号: {spec.signal_symbols}")
            if spec.vix_symbol:
                print(f"VIX: {spec.vix_symbol}")
            if spec.max_weight_per_asset:
                print(f"权重上限: {spec.max_weight_per_asset}")
            if spec.max_gross_exposure:
                print(f"预算上限: {spec.max_gross_exposure}")
            return

    print(f"未找到策略: {args.name}")
    print("可用策略:")
    for spec in preset_strategy_specs():
        print(f"  {spec.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="trading",
        description="DCA 回测引擎 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  trading run QQQ,TQQQ --allocator nasdaq_rule
  trading run QQQ,TQQQ --allocator smart --vix ^VIX
  trading experiment --presets
  trading report QQQ,TQQQ --allocator smart --vix ^VIX
  trading list
  trading show smart_signal_fusion
        """.strip(),
    )

    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="运行单策略回测")
    _add_common_dca_args(p_run)
    p_run.set_defaults(func=cmd_run)

    # experiment
    p_exp = sub.add_parser("experiment", help="批量实验")
    p_exp.add_argument("--presets", action="store_true", help="使用内置 preset 策略")
    p_exp.add_argument("--spec", help="JSON spec 文件路径")
    p_exp.add_argument("--name", help="策略名称 (非 preset 模式)")
    _add_common_dca_args(p_exp, require_symbols=False)
    p_exp.set_defaults(func=cmd_experiment)

    # report
    p_rep = sub.add_parser("report", help="生成 HTML 报告")
    _add_common_dca_args(p_rep)
    p_rep.add_argument("--output", default="reports/report.html", help="输出路径 (默认 reports/report.html)")
    p_rep.add_argument("--png", action="store_true", help="导出 PNG（需安装 kaleido）")
    p_rep.add_argument(
        "--format",
        default="html",
        choices=["html", "package", "codex", "all"],
        help="输出格式: html/package/codex/all",
    )
    p_rep.add_argument(
        "--package-dir",
        default="reports/latest_package",
        help="通用报告包与 Codex artifact JSON 输出目录",
    )
    p_rep.set_defaults(func=cmd_report)

    # list
    p_list = sub.add_parser("list", help="列出所有策略和分配器")
    p_list.set_defaults(func=cmd_list)

    # show
    p_show = sub.add_parser("show", help="查看预设策略详情")
    p_show.add_argument("name", help="策略名称")
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
