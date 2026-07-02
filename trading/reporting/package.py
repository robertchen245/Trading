from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from trading.engine import BacktestResult
from trading.metrics import compare_portfolios, equity_curve


_DATASET_GUIDE: dict[str, dict[str, Any]] = {
    "metrics": {
        "role": "primary_metrics",
        "required": True,
        "description": "场景级收益、风险、成本和预算使用指标。写报告时先读这张表。",
        "report_use": ["executive_summary", "scenario_comparison", "metrics_table"],
    },
    "equity_curve": {
        "role": "primary_timeseries",
        "required": True,
        "description": "各场景组合净值时间序列，用于净值走势和终值对比。",
        "report_use": ["equity_chart", "path_comparison"],
    },
    "drawdown": {
        "role": "risk_timeseries",
        "required": True,
        "description": "各场景相对自身历史高点的回撤时间序列。",
        "report_use": ["drawdown_chart", "risk_commentary"],
    },
    "monthly_returns": {
        "role": "returns_timeseries",
        "required": True,
        "description": "由日度净值聚合的月度收益，用于观察短周期波动。",
        "report_use": ["monthly_return_chart", "volatility_context"],
    },
    "decision_snapshot": {
        "role": "strategy_decisions",
        "required": False,
        "description": "策略每个定投日的信号、权重和预算执行记录；仅在动态策略有决策日志时存在。",
        "report_use": ["decision_audit", "allocator_explanation"],
    },
    "yearly_weights": {
        "role": "allocator_weights",
        "required": False,
        "description": "策略分配器每年的目标权重；仅在分配器输出年度权重时存在。",
        "report_use": ["allocation_chart", "allocator_explanation"],
    },
}


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out = frame.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    out = out.astype(object).where(pd.notna(out), None)
    return out.to_dict(orient="records")


def _series_to_long_frame(series_by_name: dict[str, pd.Series], value_name: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for name, series in series_by_name.items():
        frame = series.rename(value_name).reset_index()
        frame.columns = ["date", value_name]
        frame.insert(1, "scenario", name)
        rows.append(frame)
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame(columns=["date", "scenario", value_name])


def _equity_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
    return _series_to_long_frame({name: equity_curve(result) for name, result in results.items()}, "value")


def _drawdown_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
    series_by_name = {}
    for name, result in results.items():
        value = equity_curve(result)
        series_by_name[name] = value / value.cummax() - 1.0
    return _series_to_long_frame(series_by_name, "drawdown")


def _monthly_returns_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for name, result in results.items():
        daily = equity_curve(result).pct_change().dropna()
        monthly = (1.0 + daily).resample("ME").prod() - 1.0
        if monthly.empty:
            continue
        frame = monthly.rename("monthly_return").reset_index()
        frame.columns = ["date", "monthly_return"]
        frame.insert(1, "scenario", name)
        frame["year"] = pd.to_datetime(frame["date"]).dt.year
        frame["month"] = pd.to_datetime(frame["date"]).dt.month
        rows.append(frame)
    return (
        pd.concat(rows, axis=0, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["date", "scenario", "monthly_return", "year", "month"])
    )


def _decision_snapshot_frame(result: BacktestResult | None) -> pd.DataFrame:
    if result is None or result.decision_snapshot is None or result.decision_snapshot.empty:
        return pd.DataFrame()
    frame = result.decision_snapshot.reset_index()
    if "invest_date" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "invest_date"})
    return frame


def _yearly_weights_frame(result: BacktestResult | None) -> pd.DataFrame:
    if result is None or result.yearly_weights is None or result.yearly_weights.empty:
        return pd.DataFrame()
    frame = result.yearly_weights.reset_index()
    return frame.rename(columns={frame.columns[0]: "year"})


def _report_file_entries(datasets: dict[str, str | None]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = [
        {
            "path": "manifest.json",
            "role": "package_manifest",
            "required": True,
            "description": "报告包元数据、参数、场景列表和数据集注册表。",
            "report_use": ["source_inventory", "parameter_context", "dataset_registry"],
        }
    ]
    for name in (
        "metrics",
        "equity_curve",
        "drawdown",
        "monthly_returns",
        "decision_snapshot",
        "yearly_weights",
    ):
        filename = datasets.get(name)
        if not filename:
            continue
        guide = _DATASET_GUIDE[name]
        entries.append(
            {
                "path": filename,
                "dataset": name,
                "role": guide["role"],
                "required": guide["required"],
                "description": guide["description"],
                "report_use": guide["report_use"],
            }
        )
    return entries


def build_agent_report_index(manifest: dict[str, Any]) -> dict[str, Any]:
    """Build a small handoff index so agents know which report files to read."""
    files = manifest.get("report_files")
    if not isinstance(files, list):
        datasets = manifest.get("datasets", {})
        files = _report_file_entries(datasets if isinstance(datasets, dict) else {})
    return {
        "version": 1,
        "title": manifest.get("title"),
        "generated_at": manifest.get("generated_at"),
        "entrypoint": "manifest.json",
        "recommended_read_order": [item["path"] for item in files],
        "files": files,
        "agent_instructions": [
            "先读取 manifest.json 理解参数、场景和数据集注册表。",
            "用 metrics.csv 形成摘要和场景排序；再用 equity_curve.csv、drawdown.csv、monthly_returns.csv 支撑图表和风险解释。",
            "只有当 optional 文件存在时，才使用 decision_snapshot.csv 和 yearly_weights.csv 解释动态分配器。",
            "CSV 是事实来源；Codex/Hermes 等 artifact 文件应视为面向具体界面的派生输出。",
        ],
    }


def build_report_package(
    results: dict[str, BacktestResult],
    *,
    title: str = "Trading Backtest Report",
    params: Any | None = None,
    allocator_name: str | None = None,
    baseline_key: str | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Build an agent-neutral report package from backtest results."""
    generated = generated_at or datetime.now()
    metrics = compare_portfolios(results, baseline_key=baseline_key).reset_index()
    equity = _equity_frame(results)
    drawdown = _drawdown_frame(results)
    monthly_returns = _monthly_returns_frame(results)
    strategy = results.get("strategy")
    decision_snapshot = _decision_snapshot_frame(strategy)
    yearly_weights = _yearly_weights_frame(strategy)
    datasets = {
        "metrics": "metrics.csv",
        "equity_curve": "equity_curve.csv",
        "drawdown": "drawdown.csv",
        "monthly_returns": "monthly_returns.csv",
        "decision_snapshot": "decision_snapshot.csv" if not decision_snapshot.empty else None,
        "yearly_weights": "yearly_weights.csv" if not yearly_weights.empty else None,
    }

    manifest = {
        "title": title,
        "generated_at": generated.isoformat(timespec="seconds"),
        "allocator": allocator_name,
        "params": asdict(params) if is_dataclass(params) else (params or {}),
        "scenarios": list(results.keys()),
        "datasets": datasets,
        "report_files": _report_file_entries(datasets),
    }

    return {
        "manifest": manifest,
        "frames": {
            "metrics": metrics,
            "equity_curve": equity,
            "drawdown": drawdown,
            "monthly_returns": monthly_returns,
            "decision_snapshot": decision_snapshot,
            "yearly_weights": yearly_weights,
        },
    }


def write_report_package(package: dict[str, Any], output_dir: str | Path) -> Path:
    """Write report manifest and datasets to a directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = dict(package["manifest"])
    frames: dict[str, pd.DataFrame] = package["frames"]
    for name, frame in frames.items():
        if frame.empty and name in {"decision_snapshot", "yearly_weights"}:
            continue
        frame.to_csv(out / f"{name}.csv", index=False)

    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (out / "agent_report_index.json").write_text(
        json.dumps(build_agent_report_index(manifest), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return out


def package_frame_records(package: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """Return JSON-safe records for a package dataset."""
    frame = package["frames"].get(name)
    if frame is None or frame.empty:
        return []
    return _frame_to_records(frame)
