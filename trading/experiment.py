from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from trading.baseline_builders import default_baseline_builders_v1
from trading.engine import BacktestResult, run_scenarios
from trading.metrics import compare_portfolios
from trading.specs import StrategySpec
from trading.strategies.dca import (
    equal_weight_allocator,
    fixed_weight_allocator,
    momentum_rotation_allocator,
    nasdaq_rule_allocator,
    smart_allocator,
    trend_follow_allocator,
)

AllocatorFn = Callable[..., dict[str, float]]

ALLOCATOR_REGISTRY: dict[str, AllocatorFn] = {
    "fixed": fixed_weight_allocator,
    "nasdaq_rule": nasdaq_rule_allocator,
    "equal_weight": equal_weight_allocator,
    "smart": smart_allocator,
    "trend_follow": trend_follow_allocator,
    "momentum_rotation": momentum_rotation_allocator,
}


@dataclass(frozen=True)
class ExperimentResult:
    spec: StrategySpec
    results: dict[str, BacktestResult]
    metrics: pd.DataFrame


def resolve_allocator(name: str) -> AllocatorFn:
    if name not in ALLOCATOR_REGISTRY:
        raise ValueError(f"Unsupported allocator name: {name!r}")
    return ALLOCATOR_REGISTRY[name]


def run_experiment(spec: StrategySpec) -> ExperimentResult:
    params = spec.to_params()
    allocator = resolve_allocator(spec.allocator)
    results = run_scenarios(
        params,
        allocator=allocator,
        baseline_builders=default_baseline_builders_v1(params.benchmark_symbol),
    )
    metrics = compare_portfolios(results)
    return ExperimentResult(spec=spec, results=results, metrics=metrics)


def run_experiments(
    specs: list[StrategySpec],
    *,
    output_dir: Path | None = None,
    ts: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """批量运行实验并返回 (summary, ranking) 两个 DataFrame。

    若提供 output_dir 和 ts，自动写出 CSV。
    """
    from datetime import datetime as dt_lib

    runs = [run_experiment(spec) for spec in specs]
    summary_rows: list[pd.DataFrame] = []
    for run in runs:
        table = run.metrics.reset_index().rename(columns={"index": "scenario"})
        table.insert(0, "strategy_name", run.spec.name)
        summary_rows.append(table)
    summary = pd.concat(summary_rows, axis=0, ignore_index=True) if summary_rows else pd.DataFrame()
    ranking = rank_experiments(summary)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _ts = ts or dt_lib.now().strftime("%Y%m%d_%H%M%S")
        summary.to_csv(output_dir / f"experiments_{_ts}.summary.csv", index=False)
        ranking.to_csv(output_dir / f"experiments_{_ts}.ranking.csv", index=False)

    return summary, ranking


def rank_experiments(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    strategy_rows = summary[summary["scenario"] == "strategy"].copy()
    strategy_rows = strategy_rows.sort_values(by=["CAGR", "sharpe"], ascending=[False, False]).reset_index(drop=True)
    strategy_rows.insert(0, "rank", range(1, len(strategy_rows) + 1))
    return strategy_rows
