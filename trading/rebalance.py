"""组合再平衡模块：在定投日检查各资产市值占比，超出上限自动卖出。

纯函数设计 —— 输入持仓状态 + 价格，返回卖出订单。不依赖 vectorbt，
方便单元测试。
"""

from __future__ import annotations

import pandas as pd


def compute_rebalance_orders(
    current_shares: dict[str, float],
    current_prices: dict[str, float],
    total_value: float,
    max_weight: float,
) -> dict[str, int]:
    """计算再平衡所需的卖出订单（股数，整数向下取整）。

    参数:
        current_shares: 当前各标的持有股数
        current_prices: 当前各标的价格
        total_value: 组合当前总市值
        max_weight: 单资产最大权重阈值 (e.g. 0.75 → 卖出超过 75% 的部分)

    返回:
        {symbol: sell_shares} — 负值表示卖出，0 表示不动。
        不包含买入（由 DCA 订单另行处理）。
    """
    if max_weight <= 0 or max_weight >= 1.0:
        return {}

    orders: dict[str, int] = {}
    for symbol in current_shares:
        shares = current_shares.get(symbol, 0.0)
        price = current_prices.get(symbol, 0.0)
        if shares <= 0 or price <= 0 or total_value <= 0:
            continue

        current_weight = (shares * price) / total_value
        if current_weight <= max_weight:
            continue

        # 需要降到 max_weight
        target_value = total_value * max_weight
        excess_value = (shares * price) - target_value
        sell_shares = int(excess_value / price)

        if sell_shares > 0:
            orders[symbol] = -sell_shares

    return orders


def apply_rebalance_to_plan(
    order_sizes: pd.DataFrame,
    asset_prices: pd.DataFrame,
    invest_dates: pd.DatetimeIndex,
    max_weight: float,
) -> pd.DataFrame:
    """在现有的 DCA 订单表上叠加再平衡卖出操作。

    按时间顺序遍历每个定投日，模拟持仓累积，在每次定投前
    检查是否需要卖出超阈值资产。

    参数:
        order_sizes: build_order_plan 输出的原始买入订单 (DataFrame)
        asset_prices: 价格矩阵
        invest_dates: 定投日列表
        max_weight: 再平衡阈值

    返回:
        调整后的 order_sizes（含负值卖出订单）
    """
    if max_weight <= 0 or max_weight >= 1.0:
        return order_sizes

    adjusted = order_sizes.copy()
    cumulative: dict[str, float] = {c: 0.0 for c in order_sizes.columns}

    for invest_date in invest_dates:
        prices = asset_prices.loc[invest_date]

        # 计算当前总市值
        total_value = sum(
            cumulative[s] * float(prices[s]) for s in order_sizes.columns if s in prices.index
        )

        # 在买入前先做再平衡卖出
        if total_value > 0:
            current_prices = {s: float(prices[s]) for s in order_sizes.columns if s in prices.index}
            sell_orders = compute_rebalance_orders(
                current_shares=cumulative,
                current_prices=current_prices,
                total_value=total_value,
                max_weight=max_weight,
            )
            for symbol, shares in sell_orders.items():
                adjusted.loc[invest_date, symbol] += shares
                cumulative[symbol] += shares  # 卖出减少持仓

        # 应用当天的 DCA 买入订单
        for symbol in order_sizes.columns:
            buy_shares = order_sizes.loc[invest_date, symbol]
            if buy_shares > 0:
                cumulative[symbol] += buy_shares

    return adjusted
