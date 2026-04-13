from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd

if "vectorbt" not in sys.modules:
    fake_vectorbt = types.ModuleType("vectorbt")

    class _FakePortfolioFactory:
        @staticmethod
        def from_orders(*_args, **_kwargs):
            return object()

    fake_vectorbt.Portfolio = _FakePortfolioFactory
    sys.modules["vectorbt"] = fake_vectorbt

from trading.experiment import rank_experiments, run_experiments
from trading.specs import StrategySpec


class ExperimentModuleTests(unittest.TestCase):
    @patch("trading.experiment.compare_portfolios")
    @patch("trading.experiment.run_scenarios")
    def test_run_experiments_builds_summary(self, mock_run_scenarios, mock_compare_portfolios) -> None:
        mock_run_scenarios.return_value = {"strategy": object()}
        mock_compare_portfolios.return_value = pd.DataFrame(
            [{"scenario": "strategy", "CAGR": 0.12, "sharpe": 1.1, "final_value": 12345.0}]
        ).set_index("scenario")

        specs = [
            StrategySpec(
                name="s1",
                symbols=("QQQ", "TQQQ"),
                start="2020-01-01",
                end="2021-01-01",
                monthly_budget=1000.0,
                default_weights={"QQQ": 0.7, "TQQQ": 0.3},
            )
        ]
        runs, summary = run_experiments(specs)
        self.assertEqual(len(runs), 1)
        self.assertIn("strategy_name", summary.columns)
        self.assertEqual(summary.iloc[0]["strategy_name"], "s1")

    def test_rank_experiments_orders_by_cagr_then_sharpe(self) -> None:
        summary = pd.DataFrame(
            [
                {"strategy_name": "A", "scenario": "strategy", "CAGR": 0.1, "sharpe": 1.2},
                {"strategy_name": "B", "scenario": "strategy", "CAGR": 0.15, "sharpe": 1.0},
                {"strategy_name": "C", "scenario": "monthly_full_QQQ", "CAGR": 0.2, "sharpe": 0.9},
            ]
        )
        ranked = rank_experiments(summary)
        self.assertEqual(list(ranked["strategy_name"]), ["B", "A"])
        self.assertEqual(list(ranked["rank"]), [1, 2])


if __name__ == "__main__":
    unittest.main()
