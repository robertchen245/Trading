"""再平衡模块的单元测试。"""

from __future__ import annotations

import unittest

import pandas as pd

from trading.rebalance import apply_rebalance_to_plan, compute_rebalance_orders


class RebalanceComputeTests(unittest.TestCase):
    def test_sell_when_over_threshold(self) -> None:
        """QQQ 占比 90%，阈值 75%，应卖出至 75%。"""
        orders = compute_rebalance_orders(
            current_shares={"QQQ": 100.0, "TQQQ": 10.0},
            current_prices={"QQQ": 100.0, "TQQQ": 50.0},
            total_value=10500.0,  # QQQ 10000 + TQQQ 500
            max_weight=0.75,
        )
        # QQQ current: 10000/10500 = 95.2% > 75%
        # target QQQ value: 10500 * 0.75 = 7875
        # excess: 10000 - 7875 = 2125
        # sell shares: floor(2125/100) = 21
        self.assertIn("QQQ", orders)
        self.assertLess(orders["QQQ"], 0)
        self.assertEqual(orders["QQQ"], -21)
        # TQQQ: 500/10500 = 4.8% < 75%, no sell
        self.assertNotIn("TQQQ", orders)

    def test_no_sell_when_within_limit(self) -> None:
        """所有资产都在阈值内，不卖出。"""
        orders = compute_rebalance_orders(
            current_shares={"QQQ": 50.0, "TQQQ": 10.0},
            current_prices={"QQQ": 100.0, "TQQQ": 50.0},
            total_value=5500.0,  # QQQ 5000/5500 = 90.9% → 实际上超过
            max_weight=0.95,  # 高阈值
        )
        self.assertEqual(orders, {})  # 都低于 95%

    def test_zero_total_value(self) -> None:
        """组合市值为 0，不卖。"""
        orders = compute_rebalance_orders(
            current_shares={},
            current_prices={},
            total_value=0.0,
            max_weight=0.75,
        )
        self.assertEqual(orders, {})


class RebalancePlanTests(unittest.TestCase):
    def test_apply_rebalance_injects_sells(self) -> None:
        """再平衡应在 order_sizes 中添加卖出订单。"""
        index = pd.to_datetime(["2020-01-02", "2020-02-03", "2020-03-02"])
        prices = pd.DataFrame(
            {"QQQ": [100.0, 105.0, 110.0], "TQQQ": [50.0, 51.0, 55.0]},
            index=index,
        )

        # 模拟：第1个月全部买 QQQ，第2个月全部买 QQQ
        # DCA orders: 月预算 1000
        order_sizes = pd.DataFrame(0.0, index=index, columns=["QQQ", "TQQQ"])
        order_sizes.loc[index[0], "QQQ"] = 10.0  # 买 10 股 QQQ
        order_sizes.loc[index[1], "QQQ"] = 9.523  # 买 ~9.5 股 QQQ ($1000/$105)
        order_sizes.loc[index[2], "TQQQ"] = 18.18  # 买 TQQQ ($1000/$55)

        invest_dates = pd.DatetimeIndex(index)
        adjusted = apply_rebalance_to_plan(order_sizes, prices, invest_dates, max_weight=0.75)

        # 第1个月：初始持仓为 0，触发 → 买入正常
        self.assertGreater(adjusted.loc[index[0], "QQQ"], 0)

        # 第2个月：已持 10 股 QQQ @$105 = $1050 (100%)，触发再平衡卖出。
        # 超额 $262.5 → 卖出 2 股。DCA 买入 9.523 股 → 净买入 7.523。
        # 净买入应小于原始 DCA 买入（说明有卖出发生）
        orig_buy = order_sizes.loc[index[1], "QQQ"]
        self.assertAlmostEqual(adjusted.loc[index[1], "QQQ"], 7.523, places=2)
        self.assertLess(adjusted.loc[index[1], "QQQ"], orig_buy)  # 卖出减少了净买入

    def test_no_rebalance_when_disabled(self) -> None:
        """rebalance_max_weight 无效时不触发。"""
        index = pd.to_datetime(["2020-01-02", "2020-02-03"])
        prices = pd.DataFrame({"QQQ": [100.0, 105.0]}, index=index)
        order_sizes = pd.DataFrame(0.0, index=index, columns=["QQQ"])
        order_sizes.loc[index[0], "QQQ"] = 10.0

        adjusted = apply_rebalance_to_plan(
            order_sizes, prices, pd.DatetimeIndex(index), max_weight=0.0
        )
        # 应该和原始一样
        self.assertTrue((adjusted == order_sizes).all().all())


if __name__ == "__main__":
    unittest.main()
