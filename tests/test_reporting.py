from __future__ import annotations

import json
import tempfile
import unittest

import pandas as pd

from trading.engine import BacktestResult
from trading.reporting import build_codex_artifact, build_report_package, write_codex_artifact, write_report_package


class _Portfolio:
    def __init__(self, values: pd.Series) -> None:
        self._values = values

    def value(self) -> pd.Series:
        return self._values

    def stats(self) -> pd.Series:
        return pd.Series(dtype=float)


def _result(name: str, values: list[float]) -> BacktestResult:
    index = pd.to_datetime(["2020-01-02", "2020-02-03", "2020-03-02"])
    orders = pd.DataFrame({"AAA": [1.0, 1.0, 1.0]}, index=index)
    decision = pd.DataFrame(
        {
            "invest_date": index,
            "budget_utilization": [1.0, 1.0, 1.0],
            "risk_triggered": [False, False, True],
        }
    ).set_index("invest_date")
    return BacktestResult(
        name=name,
        portfolio=_Portfolio(pd.Series(values, index=index, name=name)),
        order_sizes=orders,
        yearly_weights=pd.DataFrame({"AAA_weight": [1.0]}, index=[2020]) if name == "strategy" else None,
        annual_returns=pd.DataFrame({"AAA": [0.1]}, index=[2019]),
        total_invested=300.0,
        decision_snapshot=decision if name == "strategy" else None,
    )


class ReportingTests(unittest.TestCase):
    def test_report_package_writes_neutral_files(self) -> None:
        package = build_report_package(
            {
                "strategy": _result("strategy", [300.0, 330.0, 360.0]),
                "baseline": _result("baseline", [300.0, 315.0, 330.0]),
            },
            title="Test Report",
            allocator_name="fixed",
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = write_report_package(package, tmp)
            self.assertTrue((out / "manifest.json").exists())
            self.assertTrue((out / "agent_report_index.json").exists())
            self.assertTrue((out / "metrics.csv").exists())
            self.assertTrue((out / "equity_curve.csv").exists())
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            index = json.loads((out / "agent_report_index.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["title"], "Test Report")
        self.assertIn("strategy", manifest["scenarios"])
        self.assertIn("report_files", manifest)
        self.assertEqual(index["entrypoint"], "manifest.json")
        self.assertIn("metrics.csv", index["recommended_read_order"])

    def test_codex_artifact_contains_manifest_and_snapshot(self) -> None:
        package = build_report_package(
            {
                "strategy": _result("strategy", [300.0, 330.0, 360.0]),
                "baseline": _result("baseline", [300.0, 315.0, 330.0]),
            },
            title="Test Report",
            params={"default_weights": {"AAA": 1.0}},
            allocator_name="fixed",
        )
        manifest, snapshot = build_codex_artifact(package)

        self.assertEqual(manifest["title"], "Test Report")
        self.assertEqual(snapshot["status"], "ready")
        self.assertIn("equity_curve", {chart["id"] for chart in manifest["charts"]})
        self.assertIn("metrics", {table["id"] for table in manifest["tables"]})
        self.assertIn("## 摘要", manifest["blocks"][1]["body"])
        self.assertIn("口径提示", manifest["blocks"][1]["body"])
        metrics_table = next(table for table in manifest["tables"] if table["id"] == "metrics")
        self.assertIn("期末市值", {column["label"] for column in metrics_table["columns"]})
        self.assertEqual(metrics_table["columns"][0]["field"], "scenario_label")
        cagr_column = next(column for column in metrics_table["columns"] if column["field"] == "CAGR")
        self.assertEqual(cagr_column["format"], "percent")
        self.assertIn("scenario_label", snapshot["datasets"]["equity_curve"][0])
        self.assertEqual(snapshot["datasets"]["metrics"][0]["scenario_label"], "AAA 定投")
        self.assertIsInstance(snapshot["datasets"]["metrics"][0]["final_value"], int)
        self.assertIn("equity_curve", snapshot["datasets"])
        self.assertGreater(len(snapshot["datasets"]["metrics"]), 0)

    def test_codex_metrics_table_matches_summary_winner(self) -> None:
        package = build_report_package(
            {
                "strategy": _result("strategy", [300.0, 315.0, 330.0]),
                "baseline": _result("baseline", [300.0, 330.0, 390.0]),
            },
            title="Test Report",
            params={"default_weights": {"AAA": 1.0}},
            allocator_name="fixed",
        )
        manifest, snapshot = build_codex_artifact(package)

        summary = manifest["blocks"][1]["body"]
        self.assertIn("终值最高的是 基准", summary)
        self.assertEqual(snapshot["datasets"]["metrics"][0]["scenario_label"], "基准")
        metrics_table = next(table for table in manifest["tables"] if table["id"] == "metrics")
        total_return_column = next(column for column in metrics_table["columns"] if column["field"] == "total_return")
        self.assertEqual(total_return_column["format"], "percent")

    def test_write_codex_artifact_outputs_json_files(self) -> None:
        package = build_report_package(
            {"strategy": _result("strategy", [300.0, 330.0, 360.0])},
            title="Test Report",
        )
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path, snapshot_path = write_codex_artifact(package, tmp)
            self.assertTrue(manifest_path.exists())
            self.assertTrue(snapshot_path.exists())
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))

        self.assertIn("datasets", payload)


if __name__ == "__main__":
    unittest.main()
