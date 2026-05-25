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
    parser.add_argument("--allocator", default="fixed", choices=["fixed", "nasdaq_rule", "equal_weight", "smart"],
                        help="分配器 (默认 fixed)")
    parser.add_argument("--benchmark", default="QQQ", help="基准标的 (默认 QQQ)")
    parser.add_argument("--signals", default="^IXIC", help="信号标的，逗号分隔 (默认 ^IXIC)")
    parser.add_argument("--vix", default=None, help="VIX 标的 (如 ^VIX)")
    parser.add_argument("--fee", type=float, default=0.0, help="手续费率")
    parser.add_argument("--slippage", type=float, default=0.0, help="滑点率")
    parser.add_argument("--max-weight", type=float, default=None, help="单资产权重上限")
    parser.add_argument("--max-exposure", type=float, default=None, help="预算使用上限")
    parser.add_argument("--rebalance-max", type=float, default=None, help="再平衡阈值 (如 0.75 = QQQ超75%时卖出)")
    parser.add_argument("--no-cache", action="store_true", help="禁用行情缓存")


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
        rebalance_max_weight=args.rebalance_max,
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
    """生成 HTML 可视化报告。"""
    from trading import (
        DCAParams,
        default_baseline_builders_v1,
        run_scenarios,
    )
    from trading.experiment import resolve_allocator
    from trading.viz import fig_equity_comparison, fig_summary_dashboard, write_report_html

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
        rebalance_max_weight=args.rebalance_max,
    )

    results = run_scenarios(
        params,
        allocator=allocator,
        baseline_builders=default_baseline_builders_v1(args.benchmark),
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    write_report_html(
        [
            ("总览", fig_summary_dashboard(results)),
            ("净值对比", fig_equity_comparison(results)),
        ],
        out,
    )
    print(f"报告已生成: {out}")


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
