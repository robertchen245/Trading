from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trading.engine import BacktestResult
from trading.reporting import build_codex_artifact, build_report_package


class _Portfolio:
    def __init__(self, values: pd.Series) -> None:
        self._values = values

    def value(self) -> pd.Series:
        return self._values

    def stats(self) -> pd.Series:
        return pd.Series(dtype=float)


@dataclass(frozen=True)
class _PreviewCase:
    name: str
    results: dict[str, BacktestResult]
    params: dict[str, Any]
    expect_decision_snapshot: bool = False
    expect_yearly_weights: bool = False
    min_curve_rows_per_scenario: int = 20


def _curve(index: pd.DatetimeIndex, *, start: float, final: float, wave: float = 0.03) -> list[float]:
    x = np.linspace(0.0, 1.0, len(index))
    trend = start * (final / start) ** x
    cycle = 1.0 + wave * np.sin(2 * np.pi * x) + wave / 2 * np.sin(7 * np.pi * x)
    values = trend * cycle
    values[0] = start
    values[-1] = final
    return values.tolist()


def _decision_snapshot(
    index: pd.DatetimeIndex,
    *,
    symbols: tuple[str, ...],
    with_vix: bool = False,
    risk_every: int | None = None,
    rebalance_every: int | None = None,
) -> pd.DataFrame:
    dates = index[::21][:18]
    rows: list[dict[str, Any]] = []
    for i, date in enumerate(dates):
        risk = risk_every is not None and i % risk_every == 0
        row: dict[str, Any] = {
            "invest_date": date,
            "invest_year": int(date.year),
            "signal_return_prev_year": -0.08 + i * 0.012,
            "signal_drawdown": -0.05 - (i % 5) * 0.035,
            "signal_ma_deviation": -0.06 + (i % 7) * 0.025,
            "signal_vix": 18.0 + (i % 6) * 4 if with_vix else np.nan,
            "planned_budget": 5000.0,
            "applied_budget": 2500.0 if risk else 5000.0,
            "budget_utilization": 0.5 if risk else 1.0,
            "weight_cap_triggered": risk,
            "gross_exposure_triggered": risk,
            "risk_triggered": risk,
            "risk_observe_only": False,
        }
        if rebalance_every is not None:
            row["rebalance_triggered"] = i % rebalance_every == 0
        raw = np.linspace(1.0, len(symbols), len(symbols))
        raw = np.roll(raw, i % len(symbols))
        weights = raw / raw.sum()
        for symbol, weight in zip(symbols, weights, strict=True):
            row[f"{symbol}_w_pre"] = float(weight)
            row[f"{symbol}_w_post"] = float(min(weight, 0.75))
            row[f"{symbol}_w_applied"] = float(min(weight, 0.75))
        rows.append(row)
    return pd.DataFrame(rows).set_index("invest_date")


def _yearly_weights(symbols: tuple[str, ...]) -> pd.DataFrame:
    years = [2020, 2021, 2022, 2023]
    rows = []
    for i, _year in enumerate(years):
        raw = np.linspace(1.0, len(symbols), len(symbols))
        raw = np.roll(raw, i % len(symbols))
        weights = raw / raw.sum()
        rows.append({f"{symbol}_weight": float(weight) for symbol, weight in zip(symbols, weights, strict=True)})
    return pd.DataFrame(rows, index=years)


def _result(
    name: str,
    values: list[float],
    index: pd.DatetimeIndex,
    *,
    symbols: tuple[str, ...] = ("QQQ",),
    total_invested: float = 630000.0,
    decision_snapshot: pd.DataFrame | None = None,
    yearly_weights: pd.DataFrame | None = None,
    risk_trigger_count: int = 0,
) -> BacktestResult:
    orders = pd.DataFrame(0.0, index=index, columns=list(symbols))
    orders.iloc[::21, 0] = 1.0
    annual_returns = pd.DataFrame({symbol: [0.08, -0.12, 0.22] for symbol in symbols}, index=[2020, 2021, 2022])
    return BacktestResult(
        name=name,
        portfolio=_Portfolio(pd.Series(values, index=index, name=name)),
        order_sizes=orders,
        yearly_weights=yearly_weights,
        annual_returns=annual_returns,
        total_invested=total_invested,
        risk_trigger_count=risk_trigger_count,
        decision_snapshot=decision_snapshot,
    )


def _case(
    name: str,
    finals: dict[str, float],
    *,
    symbols: tuple[str, ...],
    default_weights: dict[str, float],
    strategy_decision: pd.DataFrame | None = None,
    strategy_yearly_weights: pd.DataFrame | None = None,
    risk_trigger_count: int = 0,
    wave: float = 0.035,
) -> _PreviewCase:
    index = pd.bdate_range("2020-01-02", periods=260)
    results = {}
    for scenario, final in finals.items():
        results[scenario] = _result(
            scenario,
            _curve(index, start=630000.0, final=final, wave=wave),
            index,
            symbols=symbols,
            decision_snapshot=strategy_decision if scenario == "strategy" else None,
            yearly_weights=strategy_yearly_weights if scenario == "strategy" else None,
            risk_trigger_count=risk_trigger_count if scenario == "strategy" else 0,
        )
    return _PreviewCase(
        name=name,
        results=results,
        params={"symbols": symbols, "default_weights": default_weights},
        expect_decision_snapshot=strategy_decision is not None,
        expect_yearly_weights=strategy_yearly_weights is not None,
    )


def _preview_cases() -> list[_PreviewCase]:
    index = pd.bdate_range("2020-01-02", periods=260)
    return [
        _case(
            "single_asset_dca_compare",
            {"strategy": 850000, "monthly_full_QQQ": 920000, "monthly_full_SMH": 1180000, "monthly_full_SPY": 760000},
            symbols=("QQQ", "SMH", "SPY"),
            default_weights={"QQQ": 1.0},
        ),
        _case(
            "leveraged_etf_compare",
            {"strategy": 980000, "monthly_full_QQQ": 850000, "monthly_full_QLD": 1230000, "monthly_full_TQQQ": 1540000},
            symbols=("QQQ", "QLD", "TQQQ"),
            default_weights={"QQQ": 1.0},
            wave=0.08,
        ),
        _case(
            "lump_sum_vs_monthly_dca",
            {"strategy": 910000, "monthly_full_QQQ": 880000, "lump_sum_first_day": 1160000},
            symbols=("QQQ",),
            default_weights={"QQQ": 1.0},
        ),
        _case(
            "fixed_vs_dynamic_weights",
            {"strategy": 1240000, "equal_weight_monthly": 990000, "monthly_full_QQQ": 920000},
            symbols=("QQQ", "SMH"),
            default_weights={"QQQ": 0.5, "SMH": 0.5},
            strategy_decision=_decision_snapshot(index, symbols=("QQQ", "SMH")),
            strategy_yearly_weights=_yearly_weights(("QQQ", "SMH")),
        ),
        _case(
            "trend_signal_cash_filter",
            {"strategy": 1010000, "monthly_full_QQQ": 1120000, "lump_sum_first_day": 980000},
            symbols=("QQQ", "CASH"),
            default_weights={"QQQ": 0.8, "CASH": 0.2},
            strategy_decision=_decision_snapshot(index, symbols=("QQQ", "CASH"), risk_every=4),
            strategy_yearly_weights=_yearly_weights(("QQQ", "CASH")),
            risk_trigger_count=5,
        ),
        _case(
            "drawdown_tier_buy",
            {"strategy": 1360000, "monthly_full_QQQ": 980000, "monthly_full_TQQQ": 1420000},
            symbols=("QQQ", "TQQQ"),
            default_weights={"QQQ": 0.7, "TQQQ": 0.3},
            strategy_decision=_decision_snapshot(index, symbols=("QQQ", "TQQQ"), risk_every=3),
            strategy_yearly_weights=_yearly_weights(("QQQ", "TQQQ")),
            risk_trigger_count=6,
            wave=0.07,
        ),
        _case(
            "vix_risk_control",
            {"strategy": 1050000, "monthly_full_QQQ": 980000, "monthly_full_TQQQ": 1510000},
            symbols=("QQQ", "TQQQ"),
            default_weights={"QQQ": 0.6, "TQQQ": 0.4},
            strategy_decision=_decision_snapshot(index, symbols=("QQQ", "TQQQ"), with_vix=True, risk_every=2),
            strategy_yearly_weights=_yearly_weights(("QQQ", "TQQQ")),
            risk_trigger_count=9,
            wave=0.08,
        ),
        _case(
            "asset_cap_rebalance",
            {"strategy": 1100000, "monthly_full_SMH": 1320000, "equal_weight_monthly": 1010000},
            symbols=("QQQ", "SMH"),
            default_weights={"QQQ": 0.25, "SMH": 0.75},
            strategy_decision=_decision_snapshot(index, symbols=("QQQ", "SMH"), risk_every=3, rebalance_every=4),
            strategy_yearly_weights=_yearly_weights(("QQQ", "SMH")),
            risk_trigger_count=6,
        ),
        _case(
            "multi_asset_mix",
            {"strategy": 1190000, "monthly_full_QQQ": 930000, "monthly_full_SMH": 1480000, "equal_weight_monthly": 1030000},
            symbols=("QQQ", "SMH", "GLD", "TLT", "BTC"),
            default_weights={"QQQ": 0.25, "SMH": 0.25, "GLD": 0.2, "TLT": 0.2, "BTC": 0.1},
            strategy_decision=_decision_snapshot(index, symbols=("QQQ", "SMH", "GLD", "TLT", "BTC")),
            strategy_yearly_weights=_yearly_weights(("QQQ", "SMH", "GLD", "TLT", "BTC")),
        ),
        _case(
            "signal_source_compare",
            {"strategy": 1090000, "signal_ixic": 990000, "signal_spy": 1060000, "signal_smh": 1270000, "signal_vix": 890000},
            symbols=("QQQ", "SMH"),
            default_weights={"QQQ": 0.5, "SMH": 0.5},
            strategy_decision=_decision_snapshot(index, symbols=("QQQ", "SMH"), with_vix=True),
            strategy_yearly_weights=_yearly_weights(("QQQ", "SMH")),
        ),
    ]


class ReportPreviewVariantTests(unittest.TestCase):
    def test_codex_preview_handles_ten_strategy_and_signal_variants(self) -> None:
        percent_fields = {"total_return", "CAGR", "max_drawdown", "volatility_annual", "vs_baseline_final_ratio"}
        for case in _preview_cases():
            with self.subTest(case=case.name):
                package = build_report_package(
                    case.results,
                    title=case.name,
                    params=case.params,
                    allocator_name=case.name,
                )
                manifest, snapshot = build_codex_artifact(package, max_curve_rows=240, max_table_rows=80)

                self.assertEqual(manifest["surface"], "report")
                self.assertEqual(snapshot["status"], "ready")
                self.assertEqual(manifest["blocks"][0]["body"], f"# {case.name}")
                self.assertIn("口径提示", manifest["blocks"][1]["body"])

                metric_rows = snapshot["datasets"]["metrics"]
                final_values = [row["final_value"] for row in metric_rows]
                self.assertEqual(final_values, sorted(final_values, reverse=True))
                winner = metric_rows[0]["scenario_label"]
                self.assertIn(winner, manifest["blocks"][1]["body"])

                metrics_table = next(table for table in manifest["tables"] if table["id"] == "metrics")
                metric_columns = {column["field"]: column for column in metrics_table["columns"]}
                self.assertEqual(metrics_table["defaultSort"], {"field": "final_value", "direction": "desc"})
                self.assertEqual(metrics_table["columns"][0]["field"], "scenario_label")
                for field in percent_fields:
                    if field in metric_columns:
                        self.assertEqual(metric_columns[field]["format"], "percent")

                curve_rows = snapshot["datasets"]["equity_curve"]
                drawdown_rows = snapshot["datasets"]["drawdown"]
                self.assertGreaterEqual(len(curve_rows), case.min_curve_rows_per_scenario * len(case.results))
                self.assertGreaterEqual(len(drawdown_rows), case.min_curve_rows_per_scenario * len(case.results))
                self.assertTrue(all("scenario_label" in row for row in curve_rows))
                self.assertTrue(all("scenario_label" in row for row in drawdown_rows))
                curve_chart = next(chart for chart in manifest["charts"] if chart["id"] == "equity_curve")
                self.assertEqual(curve_chart["encodings"]["color"]["field"], "scenario_label")

                table_ids = {table["id"] for table in manifest["tables"]}
                chart_ids = {chart["id"] for chart in manifest["charts"]}
                if case.expect_decision_snapshot:
                    self.assertIn("decision_snapshot", table_ids)
                    self.assertGreater(len(snapshot["datasets"]["decision_snapshot"]), 0)
                    decision_table = next(table for table in manifest["tables"] if table["id"] == "decision_snapshot")
                    decision_columns = {column["field"]: column for column in decision_table["columns"]}
                    self.assertEqual(decision_columns["budget_utilization"]["format"], "percent")
                else:
                    self.assertNotIn("decision_snapshot", table_ids)

                if case.expect_yearly_weights:
                    self.assertIn("yearly_weights", chart_ids)
                    self.assertGreater(len(snapshot["datasets"]["yearly_weights"]), 0)
                    weight_chart = next(chart for chart in manifest["charts"] if chart["id"] == "yearly_weights")
                    self.assertTrue(weight_chart["encodings"]["y"]["fields"])
                else:
                    self.assertNotIn("yearly_weights", chart_ids)

                report_files = package["manifest"]["report_files"]
                report_datasets = {item.get("dataset") for item in report_files}
                self.assertIn("metrics", report_datasets)
                self.assertIn("equity_curve", report_datasets)
                self.assertIn("drawdown", report_datasets)


if __name__ == "__main__":
    unittest.main()
