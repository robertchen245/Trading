from __future__ import annotations

import unittest

from trading.specs import StrategySpec, nl_to_strategy_spec


class StrategySpecTests(unittest.TestCase):
    def test_from_dict_and_to_params(self) -> None:
        spec = StrategySpec.from_dict(
            {
                "name": "my_spec",
                "symbols": ["QQQ", "TQQQ"],
                "start": "2020-01-01",
                "end": "2021-01-01",
                "monthly_budget": 2000,
                "default_weights": {"QQQ": 0.7, "TQQQ": 0.3},
                "allocator": "fixed",
                "fee_rate": 0.001,
                "slippage_rate": 0.0005,
            }
        )
        params = spec.to_params()
        self.assertEqual(params.symbols, ("QQQ", "TQQQ"))
        self.assertAlmostEqual(params.fee_rate, 0.001, places=8)
        self.assertAlmostEqual(params.slippage_rate, 0.0005, places=8)

    def test_nl_to_strategy_spec_detects_aggressive_and_allocator(self) -> None:
        spec = nl_to_strategy_spec("做一个激进的纳指规则策略，主要QQQ和TQQQ", name="nl_test")
        self.assertEqual(spec.name, "nl_test")
        self.assertEqual(spec.allocator, "nasdaq_rule")
        self.assertEqual(spec.symbols[:2], ("QQQ", "TQQQ"))
        self.assertGreater(spec.default_weights["TQQQ"], 0.3)


if __name__ == "__main__":
    unittest.main()
