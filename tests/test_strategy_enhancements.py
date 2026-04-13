from __future__ import annotations

import sys
import types
import unittest
from dataclasses import replace
from unittest.mock import patch

import pandas as pd

fake_vectorbt = types.ModuleType("vectorbt")

class _FakePortfolio:
    def __init__(self, close: pd.DataFrame, size: pd.DataFrame, init_cash: float, fees: float, slippage: float):
        self._close = close
        self._size = size
        self._init_cash = init_cash
        self._fees = fees
        self._slippage = slippage

    def value(self) -> pd.Series:
        spent = (self._size * self._close).sum(axis=1)
        costs = spent * (self._fees + self._slippage)
        cash = self._init_cash - (spent + costs).cumsum()
        position_value = (self._size.cumsum() * self._close).sum(axis=1)
        return position_value + cash

    def stats(self) -> pd.Series:
        return pd.Series(dtype=float)

class _FakePortfolioFactory:
    @staticmethod
    def from_orders(close, size, init_cash, fees=0.0, slippage=0.0, **_kwargs):
        return _FakePortfolio(close=close, size=size, init_cash=init_cash, fees=fees, slippage=slippage)

fake_vectorbt.Portfolio = _FakePortfolioFactory
sys.modules["vectorbt"] = fake_vectorbt

import trading.engine as engine
from trading.engine import run_dca_portfolio
from trading.strategies.dca import DCAParams, RiskGuardConfig, build_order_plan, fixed_weight_allocator

engine.vbt.Portfolio = _FakePortfolioFactory


class StrategyEnhancementsTests(unittest.TestCase):
    def test_weight_cap_generates_risk_snapshot(self) -> None:
        index = pd.to_datetime(["2021-01-04", "2021-02-01"])
        prices = pd.DataFrame({"AAA": [100.0, 110.0], "BBB": [50.0, 55.0]}, index=index)
        annual_returns = pd.Series({2020: 0.1})

        def all_in_aaa(**_kwargs) -> dict[str, float]:
            return {"AAA": 1.0, "BBB": 0.0}

        _, _, decision_snapshot, risk_trigger_count = build_order_plan(
            asset_prices=prices,
            monthly_budget=1000.0,
            annual_returns=annual_returns,
            default_weights={"AAA": 1.0, "BBB": 0.0},
            allocator=all_in_aaa,
            risk_config=RiskGuardConfig(max_weight_per_asset=0.6),
        )

        self.assertEqual(risk_trigger_count, 2)
        self.assertTrue(bool(decision_snapshot.iloc[0]["risk_triggered"]))
        self.assertAlmostEqual(float(decision_snapshot.iloc[0]["AAA_w_post"]), 0.6, places=6)
        self.assertAlmostEqual(float(decision_snapshot.iloc[0]["BBB_w_post"]), 0.4, places=6)

    def test_gross_exposure_cap_reduces_allocated_budget(self) -> None:
        index = pd.to_datetime(["2021-01-04", "2021-02-01"])
        prices = pd.DataFrame({"AAA": [100.0, 110.0], "BBB": [100.0, 100.0]}, index=index)
        annual_returns = pd.Series({2020: 0.1})

        order_sizes, _, decision_snapshot, risk_trigger_count = build_order_plan(
            asset_prices=prices,
            monthly_budget=1000.0,
            annual_returns=annual_returns,
            default_weights={"AAA": 0.5, "BBB": 0.5},
            allocator=fixed_weight_allocator,
            risk_config=RiskGuardConfig(max_gross_exposure=0.5),
        )

        invested_notional = float((order_sizes * prices).sum().sum())
        self.assertAlmostEqual(invested_notional, 1000.0, places=6)
        self.assertEqual(risk_trigger_count, 2)
        self.assertAlmostEqual(float(decision_snapshot.iloc[0]["budget_utilization"]), 0.5, places=6)

    @patch("trading.engine.fetch_annual_returns")
    @patch("trading.engine.fetch_close_prices")
    def test_costs_reduce_final_value(self, mock_fetch_close_prices, mock_fetch_annual_returns) -> None:
        index = pd.to_datetime(
            ["2020-01-02", "2020-01-31", "2020-02-03", "2020-02-28", "2020-03-02", "2020-03-31"]
        )
        close = pd.DataFrame(
            {
                "QQQ": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "TQQQ": [50.0, 51.0, 52.0, 53.0, 54.0, 55.0],
                "^IXIC": [10000.0, 10050.0, 10100.0, 10120.0, 10200.0, 10250.0],
            },
            index=index,
        )
        mock_fetch_close_prices.return_value = close
        mock_fetch_annual_returns.return_value = pd.Series({2019: 0.12, 2020: 0.08})

        base_params = DCAParams(
            symbols=("QQQ", "TQQQ"),
            start="2020-01-01",
            end="2020-04-01",
            monthly_budget=1000.0,
            default_weights={"QQQ": 0.7, "TQQQ": 0.3},
            signal_symbol="^IXIC",
            use_cache=False,
        )
        result_no_cost = run_dca_portfolio(base_params)
        result_with_cost = run_dca_portfolio(replace(base_params, fee_rate=0.01, slippage_rate=0.005))

        final_no_cost = float(result_no_cost.portfolio.value().iloc[-1])
        final_with_cost = float(result_with_cost.portfolio.value().iloc[-1])
        self.assertLess(final_with_cost, final_no_cost)


if __name__ == "__main__":
    unittest.main()
