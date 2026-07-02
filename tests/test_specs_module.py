from __future__ import annotations

import unittest

from trading.specs import StrategySpec


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
                "data_source": "auto_ibkr",
                "yf_max_retries": 5,
                "fee_rate": 0.001,
                "slippage_rate": 0.0005,
            }
        )
        params = spec.to_params()
        self.assertEqual(params.symbols, ("QQQ", "TQQQ"))
        self.assertEqual(params.data_source, "auto_ibkr")
        self.assertEqual(params.yf_max_retries, 5)
        self.assertAlmostEqual(params.fee_rate, 0.001, places=8)
        self.assertAlmostEqual(params.slippage_rate, 0.0005, places=8)


if __name__ == "__main__":
    unittest.main()
