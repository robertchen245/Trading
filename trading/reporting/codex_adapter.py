from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading.reporting.package import package_frame_records


_COLUMN_LABELS = {
    "scenario": "场景 ID",
    "scenario_label": "场景",
    "date": "日期",
    "value": "组合净值",
    "drawdown": "回撤",
    "monthly_return": "月度收益率",
    "year": "年份",
    "month": "月份",
    "final_value": "期末市值",
    "total_invested": "累计投入",
    "total_return": "总收益率",
    "CAGR": "年化收益率",
    "max_drawdown": "最大回撤",
    "volatility_annual": "年化波动率",
    "sharpe": "夏普比率",
    "sortino": "索提诺比率",
    "calmar": "Calmar 比率",
    "start": "开始日期",
    "end": "结束日期",
    "years": "回测年数",
    "fee_rate": "手续费率",
    "slippage_rate": "滑点率",
    "risk_trigger_count": "风控触发次数",
    "avg_budget_utilization": "平均预算使用率",
    "vs_baseline_final_ratio": "相对基准终值差",
    "invest_date": "定投日期",
    "invest_year": "定投年份",
    "signal_return_prev_year": "上一年信号收益",
    "signal_drawdown": "信号回撤",
    "signal_ma_deviation": "均线偏离",
    "signal_vix": "VIX",
    "planned_budget": "计划投入",
    "applied_budget": "实际投入",
    "budget_utilization": "预算使用率",
    "weight_cap_triggered": "权重上限触发",
    "gross_exposure_triggered": "总暴露上限触发",
    "risk_triggered": "风控触发",
    "risk_observe_only": "仅观测风控",
}

_PREFERRED_TABLE_COLUMNS = [
    "scenario_label",
    "final_value",
    "total_invested",
    "total_return",
    "CAGR",
    "max_drawdown",
    "volatility_annual",
    "sharpe",
    "sortino",
    "calmar",
    "vs_baseline_final_ratio",
    "start",
    "end",
    "years",
    "fee_rate",
    "slippage_rate",
    "risk_trigger_count",
    "avg_budget_utilization",
    "scenario",
]

_PERCENT_COLUMNS = {
    "total_return",
    "CAGR",
    "max_drawdown",
    "volatility_annual",
    "vs_baseline_final_ratio",
    "monthly_return",
    "fee_rate",
    "slippage_rate",
    "avg_budget_utilization",
    "signal_return_prev_year",
    "signal_drawdown",
    "signal_ma_deviation",
    "budget_utilization",
}

_METRICS_DISPLAY_DECIMALS = {
    "final_value": 0,
    "total_invested": 0,
    "total_return": 4,
    "CAGR": 4,
    "max_drawdown": 4,
    "volatility_annual": 4,
    "sharpe": 2,
    "sortino": 2,
    "calmar": 2,
    "years": 2,
    "fee_rate": 4,
    "slippage_rate": 4,
    "avg_budget_utilization": 4,
    "vs_baseline_final_ratio": 4,
}

_SCENARIO_LABELS = {
    "strategy": "策略",
    "baseline": "基准",
    "monthly_full_QQQ": "QQQ 定投",
    "monthly_full_SMH": "SMH 定投",
    "lump_sum_first_day": "首日一次性买入",
    "equal_weight_monthly": "等权月度定投",
}


def _sample_records(records: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    if len(records) <= max_rows:
        return records
    if max_rows <= 0:
        return []
    idx = np.linspace(0, len(records) - 1, num=max_rows, dtype=int)
    seen: set[int] = set()
    sampled = []
    for i in idx:
        if int(i) in seen:
            continue
        seen.add(int(i))
        sampled.append(records[int(i)])
    return sampled


def _numeric_columns(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return []
    out = []
    for key in records[0]:
        if any(isinstance(row.get(key), int | float) and not isinstance(row.get(key), bool) for row in records):
            out.append(key)
    return out


def _ordered_record_keys(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return []
    keys = list(records[0].keys())
    preferred = [key for key in _PREFERRED_TABLE_COLUMNS if key in keys]
    return preferred + [key for key in keys if key not in preferred]


def _table_columns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    numeric = set(_numeric_columns(records))
    columns = []
    for key in _ordered_record_keys(records):
        col_type = "number" if key in numeric else "text"
        if key in {"date", "start", "end", "invest_date"}:
            col_type = "date"
        if "return" in key or key in {"CAGR", "max_drawdown", "sharpe", "sortino", "calmar"}:
            col_type = "number"
        column = {"field": key, "label": _column_label(key), "type": col_type}
        if (
            key in _PERCENT_COLUMNS
            or key.endswith("_weight")
            or key.endswith("_w_pre")
            or key.endswith("_w_post")
            or key.endswith("_w_applied")
        ):
            column["format"] = "percent"
        columns.append(column)
    return columns


def _column_label(key: str) -> str:
    if key in _COLUMN_LABELS:
        return _COLUMN_LABELS[key]
    if key.endswith("_weight"):
        return f"{key.removesuffix('_weight')} 权重"
    if key.endswith("_w_pre"):
        return f"{key.removesuffix('_w_pre')} 原始权重"
    if key.endswith("_w_post"):
        return f"{key.removesuffix('_w_post')} 风控后权重"
    if key.endswith("_w_applied"):
        return f"{key.removesuffix('_w_applied')} 实际权重"
    return key


def _scenario_label(name: Any, overrides: dict[str, str] | None = None) -> str:
    text = str(name)
    if overrides and text in overrides:
        return overrides[text]
    if text in _SCENARIO_LABELS:
        return _SCENARIO_LABELS[text]
    if text.startswith("monthly_full_"):
        return f"{text.removeprefix('monthly_full_')} 定投"
    return text


def _scenario_label_records(
    records: list[dict[str, Any]],
    overrides: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    out = []
    for row in records:
        new_row = dict(row)
        if "scenario" in new_row:
            new_row["scenario_label"] = _scenario_label(new_row["scenario"], overrides)
        out.append(new_row)
    return out


def _scenario_label_overrides(pkg_manifest: dict[str, Any]) -> dict[str, str]:
    params = pkg_manifest.get("params")
    if not isinstance(params, dict):
        return {}
    weights = params.get("default_weights")
    if not isinstance(weights, dict):
        return {}
    positive = [
        (str(symbol), float(weight))
        for symbol, weight in weights.items()
        if isinstance(weight, int | float) and float(weight) > 0
    ]
    if len(positive) == 1:
        return {"strategy": f"{positive[0][0]} 定投"}
    return {}


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and np.isfinite(value)


def _row_label(row: dict[str, Any]) -> str:
    return str(row.get("scenario_label") or _scenario_label(row.get("scenario")))


def _money(value: Any) -> str:
    return f"{float(value):,.0f}"


def _pct(value: Any) -> str:
    return f"{float(value):.2%}"


def _metric_line(metrics: list[dict[str, Any]]) -> str:
    strategy = next((row for row in metrics if row.get("scenario") == "strategy"), metrics[0] if metrics else None)
    if not strategy:
        return "未生成策略指标。"
    final_value = strategy.get("final_value")
    cagr = strategy.get("CAGR")
    max_drawdown = strategy.get("max_drawdown")
    sharpe = strategy.get("sharpe")
    parts = []
    if isinstance(final_value, int | float):
        parts.append(f"期末市值 {final_value:,.0f}")
    if isinstance(cagr, int | float):
        parts.append(f"年化收益率 {cagr:.2%}")
    if isinstance(max_drawdown, int | float):
        parts.append(f"最大回撤 {max_drawdown:.2%}")
    if isinstance(sharpe, int | float):
        parts.append(f"夏普比率 {sharpe:.2f}")
    return f"{_row_label(strategy)}概览：" + "，".join(parts) + "。" if parts else "策略指标见下方表格。"


def _best_by(metrics: list[dict[str, Any]], field: str, *, highest: bool = True) -> dict[str, Any] | None:
    rows = [row for row in metrics if _is_number(row.get(field))]
    if not rows:
        return None
    return max(rows, key=lambda row: float(row[field])) if highest else min(rows, key=lambda row: float(row[field]))


def _sort_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: float(row["final_value"]) if _is_number(row.get("final_value")) else float("-inf"),
        reverse=True,
    )


def _round_records(records: list[dict[str, Any]], decimals_by_field: dict[str, int]) -> list[dict[str, Any]]:
    rounded = []
    for row in records:
        out = dict(row)
        for field, decimals in decimals_by_field.items():
            value = out.get(field)
            if not _is_number(value):
                continue
            rounded_value = round(float(value), decimals)
            out[field] = int(rounded_value) if decimals == 0 else rounded_value
        rounded.append(out)
    return rounded


def _summary_lines(
    metrics: list[dict[str, Any]],
    *,
    has_decision_snapshot: bool,
    has_yearly_weights: bool,
) -> list[str]:
    if not metrics:
        return ["- 未生成策略指标。"]

    lines: list[str] = []
    base = next((row for row in metrics if row.get("scenario") == "strategy"), metrics[0])
    best_final = _best_by(metrics, "final_value", highest=True)
    if best_final and base and best_final is not base and _is_number(base.get("final_value")):
        lift = float(best_final["final_value"]) / float(base["final_value"]) - 1.0
        lines.append(
            f"- 终值最高的是 {_row_label(best_final)}，期末市值 {_money(best_final['final_value'])}，"
            f"较 {_row_label(base)} 高 {_pct(lift)}。"
        )
    else:
        lines.append(f"- {_metric_line(metrics)}")

    deepest = _best_by(metrics, "max_drawdown", highest=False)
    shallowest = _best_by(metrics, "max_drawdown", highest=True)
    if deepest and shallowest and deepest is not shallowest:
        lines.append(
            f"- 回撤压力最大的是 {_row_label(deepest)}（{_pct(deepest['max_drawdown'])}），"
            f"回撤较浅的是 {_row_label(shallowest)}（{_pct(shallowest['max_drawdown'])}）。"
        )

    if base and _is_number(base.get("final_value")):
        details = [f"期末市值 {_money(base['final_value'])}"]
        if _is_number(base.get("CAGR")):
            details.append(f"年化收益率 {_pct(base['CAGR'])}")
        if _is_number(base.get("max_drawdown")):
            details.append(f"最大回撤 {_pct(base['max_drawdown'])}")
        if _is_number(base.get("sharpe")):
            details.append(f"夏普比率 {float(base['sharpe']):.2f}")
        lines.append(f"- {_row_label(base)}：" + "，".join(details) + "。")

    sections = ["场景指标", "净值曲线", "回撤", "月度收益"]
    if has_decision_snapshot:
        sections.append("定投决策快照")
    if has_yearly_weights:
        sections.append("年度目标权重")
    lines.append(f"- 下方展示：{'、'.join(sections)}。")
    lines.append("- 口径提示：组合净值包含尚未投入的现金，累计投入为全期间计划投入总额。")
    return lines


def _executive_summary(
    metrics: list[dict[str, Any]],
    generated_at: Any,
    *,
    has_decision_snapshot: bool,
    has_yearly_weights: bool,
) -> str:
    lines = [
        "## 摘要",
        "",
        *_summary_lines(
            metrics,
            has_decision_snapshot=has_decision_snapshot,
            has_yearly_weights=has_yearly_weights,
        ),
    ]
    if generated_at:
        lines.append(f"- 生成时间：{generated_at}。")
    return "\n".join(lines)


def _source(
    dataset: str,
    columns: list[str],
    description: str,
    *,
    metric_definitions: list[str] | None = None,
) -> dict[str, Any]:
    cols = ", ".join(columns) if columns else "*"
    query: dict[str, Any] = {
        "language": "sql",
        "engine": "report_package",
        "description": description,
        "sql": f"SELECT {cols} FROM {dataset}",
        "tables_used": [dataset],
    }
    if metric_definitions:
        query["metric_definitions"] = metric_definitions
    return {
        "label": "Trading 报告包",
        "query": query,
    }


def build_codex_artifact(
    package: dict[str, Any],
    *,
    max_curve_rows: int = 2000,
    max_table_rows: int = 500,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert a neutral report package to Codex Data Analytics artifact payloads."""
    pkg_manifest = package["manifest"]
    title = str(pkg_manifest.get("title") or "Trading Backtest Report")
    generated_at = pkg_manifest.get("generated_at")
    scenario_overrides = _scenario_label_overrides(pkg_manifest)

    metrics = _round_records(
        _sort_metrics(_scenario_label_records(package_frame_records(package, "metrics"), scenario_overrides)),
        _METRICS_DISPLAY_DECIMALS,
    )
    equity_curve = _scenario_label_records(
        _sample_records(package_frame_records(package, "equity_curve"), max_curve_rows),
        scenario_overrides,
    )
    drawdown = _scenario_label_records(
        _sample_records(package_frame_records(package, "drawdown"), max_curve_rows),
        scenario_overrides,
    )
    monthly_returns = _scenario_label_records(
        _sample_records(package_frame_records(package, "monthly_returns"), max_table_rows),
        scenario_overrides,
    )
    decision_snapshot = _sample_records(package_frame_records(package, "decision_snapshot"), max_table_rows)
    yearly_weights = package_frame_records(package, "yearly_weights")

    datasets = {
        "metrics": metrics,
        "equity_curve": equity_curve,
        "drawdown": drawdown,
        "monthly_returns": monthly_returns,
        "decision_snapshot": decision_snapshot,
        "yearly_weights": yearly_weights,
    }

    charts: list[dict[str, Any]] = [
        {
            "id": "equity_curve",
            "title": "净值曲线",
            "dataset": "equity_curve",
            "source": _source(
                "equity_curve",
                ["date", "scenario", "scenario_label", "value"],
                "读取由 BacktestResult.portfolio.value() 生成的各场景组合净值。",
            ),
            "type": "line",
            "encodings": {
                "x": {"field": "date", "type": "temporal"},
                "y": {"field": "value", "type": "quantitative"},
                "color": {"field": "scenario_label", "type": "nominal"},
            },
        },
        {
            "id": "drawdown",
            "title": "回撤",
            "dataset": "drawdown",
            "source": _source(
                "drawdown",
                ["date", "scenario", "scenario_label", "drawdown"],
                "读取由组合净值计算得到的各场景回撤序列。",
                metric_definitions=["回撤 = 组合净值 / 历史最高组合净值 - 1"],
            ),
            "type": "line",
            "encodings": {
                "x": {"field": "date", "type": "temporal"},
                "y": {"field": "drawdown", "type": "quantitative"},
                "color": {"field": "scenario_label", "type": "nominal"},
            },
        },
        {
            "id": "monthly_returns",
            "title": "月度收益",
            "dataset": "monthly_returns",
            "source": _source(
                "monthly_returns",
                ["date", "scenario", "scenario_label", "monthly_return", "year", "month"],
                "读取由日度组合净值聚合得到的月度收益。",
                metric_definitions=["月度收益率 = 当月每日 (1 + 日收益率) 连乘 - 1"],
            ),
            "type": "bar",
            "encodings": {
                "x": {"field": "date", "type": "temporal"},
                "y": {"field": "monthly_return", "type": "quantitative"},
                "color": {"field": "scenario_label", "type": "nominal"},
            },
        },
    ]
    if yearly_weights:
        charts.append(
            {
                "id": "yearly_weights",
                "title": "年度目标权重",
                "dataset": "yearly_weights",
                "source": _source(
                    "yearly_weights",
                    list(yearly_weights[0].keys()) if yearly_weights else [],
                    "读取策略分配器记录的年度目标权重。",
                ),
                "type": "bar",
                "encodings": {
                    "x": {"field": "year", "type": "nominal"},
                    "y": {"fields": [c for c in yearly_weights[0] if c.endswith("_weight")]},
                },
            }
        )

    tables: list[dict[str, Any]] = [
        {
            "id": "metrics",
            "title": "场景指标",
            "dataset": "metrics",
            "source": _source(
                "metrics",
                list(metrics[0].keys()) if metrics else [],
                "读取由 trading.metrics.compare_portfolios 计算得到的场景指标。",
                metric_definitions=[
                    "期末市值 = 回测最后一个交易日的组合净值",
                    "年化收益率 = 基于观测回测年数折算的年化收益",
                    "最大回撤 = min(组合净值 / 历史最高组合净值 - 1)",
                    "平均预算使用率 = 策略定投日实际投入金额 / 计划投入金额 的均值",
                ],
            ),
            "columns": _table_columns(metrics),
            "defaultSort": {"field": "final_value", "direction": "desc"},
        }
    ]
    if decision_snapshot:
        tables.append(
            {
                "id": "decision_snapshot",
                "title": "定投决策快照",
                "dataset": "decision_snapshot",
                "source": _source(
                    "decision_snapshot",
                    list(decision_snapshot[0].keys()) if decision_snapshot else [],
                    "读取策略在每个定投日记录的信号、权重和预算使用情况。",
                ),
                "columns": _table_columns(decision_snapshot),
                "defaultSort": {"field": "invest_date", "direction": "asc"},
            }
        )

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "executive_summary",
            "type": "markdown",
            "body": _executive_summary(
                metrics,
                generated_at,
                has_decision_snapshot=bool(decision_snapshot),
                has_yearly_weights=bool(yearly_weights),
            ),
        },
        {
            "id": "equity_curve_context",
            "type": "markdown",
            "body": (
                "## 组合净值走势\n\n"
                "净值曲线用于比较策略与各基准在同一交易日历下的累计表现。"
                "当前组合净值包含尚未投入的现金，因此定投场景的起点通常等于全期间计划投入总额；"
                "如果要观察纯持仓市值，可以在后续报告包里增加 invested-only 曲线。"
            ),
        },
        {"id": "equity_curve", "type": "chart", "chartId": "equity_curve"},
        {
            "id": "drawdown_context",
            "type": "markdown",
            "body": "## 回撤风险\n\n回撤图展示每个场景相对自身历史高点的下跌幅度，用来观察持有过程中的压力。",
        },
        {"id": "drawdown", "type": "chart", "chartId": "drawdown"},
        {
            "id": "metrics_context",
            "type": "markdown",
            "body": (
                "## 场景指标\n\n"
                "指标表保留收益、风险、成本和预算使用情况的精确数值。"
                "默认按期末市值排序，适合先看结果排名，再回到上方曲线判断路径是否可接受。"
            ),
        },
        {"id": "metrics", "type": "table", "tableId": "metrics"},
        {
            "id": "monthly_returns_context",
            "type": "markdown",
            "body": "## 月度收益\n\n月度收益用于观察短周期波动，以及不同场景在单月维度上的差异。",
        },
        {"id": "monthly_returns", "type": "chart", "chartId": "monthly_returns"},
    ]
    if decision_snapshot:
        blocks.append(
            {
                "id": "decision_snapshot_context",
                "type": "markdown",
                "body": "## 定投决策快照\n\n决策快照展示策略在每个定投日记录的信号、权重和预算执行情况。",
            }
        )
        blocks.append({"id": "decision_snapshot", "type": "table", "tableId": "decision_snapshot"})
    if yearly_weights:
        blocks.append(
            {
                "id": "yearly_weights_context",
                "type": "markdown",
                "body": "## 年度目标权重\n\n年度目标权重展示分配器每年计划采用的资产配置比例。",
            }
        )
        blocks.append({"id": "yearly_weights", "type": "chart", "chartId": "yearly_weights"})

    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "由通用报告包生成的交互式 Trading 回测报告。",
        "generatedAt": generated_at,
        "sources": [
            {
                "id": "trading_report_package",
                "label": "Trading 报告包",
                "description": "由 trading.reporting.package 基于 BacktestResult 对象生成。",
            }
        ],
        "blocks": blocks,
        "charts": charts,
        "tables": tables,
    }
    snapshot = {
        "version": 1,
        "status": "ready",
        "generatedAt": generated_at,
        "datasets": datasets,
    }
    return manifest, snapshot


def write_codex_artifact(
    package: dict[str, Any],
    output_dir: str | Path,
    *,
    max_curve_rows: int = 2000,
    max_table_rows: int = 500,
) -> tuple[Path, Path]:
    manifest, snapshot = build_codex_artifact(
        package,
        max_curve_rows=max_curve_rows,
        max_table_rows=max_table_rows,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "codex_manifest.json"
    snapshot_path = out / "codex_snapshot.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path, snapshot_path
