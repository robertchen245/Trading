"""组合再平衡模块：支持两种模式。

- "sell": 卖出超阈值资产，买回 CASH（或释放现金到池）
- "tilt": 不卖，而是把当月 DCA 预算全部倾斜给占比最低的资产
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

    "sell" 模式使用：返回 {symbol: negative_shares}，卖出的现金由 vectorbt 管理。
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

        target_value = total_value * max_weight
        excess_value = (shares * price) - target_value
        sell_shares = int(excess_value / price)

        if sell_shares > 0:
            orders[symbol] = -sell_shares

    return orders


def compute_tilt_weights(
    current_shares: dict[str, float],
    current_prices: dict[str, float],
    total_value: float,
    target_weights: dict[str, float],
    default_weights: dict[str, float],
    max_weight: float,
) -> dict[str, float]:
    """"tilt" 模式：不卖超阈值资产，而是将 DCA 预算倾斜给最瘦的资产。

    当前占比 > max_weight 的资产 → 分配权重降为 0（这个月不买它）
    被释放的预算重新分配给占比最低的资产。

    返回调整后的 DCA 权重（归一化后）。
    """
    if max_weight <= 0 or max_weight >= 1.0 or total_value <= 0:
        return dict(default_weights)

    tilted = dict(default_weights)

    # 找出超阈值的资产，把它们从 DCA 预算中移除
    overweight: list[str] = []
    for symbol, weight in default_weights.items():
        shares = current_shares.get(symbol, 0.0)
        price = current_prices.get(symbol, 0.0)
        if shares > 0 and price > 0:
            current_w = (shares * price) / total_value
            if current_w > max_weight:
                overweight.append(symbol)

    if not overweight:
        return dict(default_weights)

    # 释放超阈值资产的预算
    freed = 0.0
    for symbol in overweight:
        freed += tilted.get(symbol, 0.0)
        tilted[symbol] = 0.0

    if freed <= 0:
        return dict(default_weights)

    # 重新分配给占比最低的资产
    underweight = [s for s in tilted if tilted[s] > 0]
    if not underweight:
        return dict(default_weights)

    # 按当前占比的倒数加权：占比越低的获得越多
    inv_weights = {}
    for symbol in underweight:
        shares = current_shares.get(symbol, 0.0)
        price = current_prices.get(symbol, 0.0)
        if current_shares.get(symbol, 0.0) == 0:
            # 从未持有过的资产，给高优先级
            inv_weights[symbol] = 10.0
        elif shares > 0 and price > 0:
            current_w = (shares * price) / total_value
            inv_weights[symbol] = 1.0 / max(current_w, 0.01)
        else:
            inv_weights[symbol] = 1.0

    total_inv = sum(inv_weights.values())
    if total_inv <= 0:
        # 等分
        each = freed / len(underweight)
        for symbol in underweight:
            tilted[symbol] += each
    else:
        for symbol in underweight:
            tilted[symbol] += freed * (inv_weights[symbol] / total_inv)

    # 归一化
    total = sum(tilted.values())
    if total > 0:
        tilted = {k: v / total for k, v in tilted.items()}

    return tilted


def apply_rebalance_to_plan(
    order_sizes: pd.DataFrame,
    asset_prices: pd.DataFrame,
    invest_dates: pd.DatetimeIndex,
    max_weight: float,
    *,
    mode: str = "sell",
) -> pd.DataFrame:
    """在 DCA 订单上叠加再平衡操作。

    mode="sell": 卖出超阈值资产（订单含负值）。
    mode="tilt": 不修改 order_sizes（倾斜由分配器层面处理）。

    返回调整后的 order_sizes。
    """
    if max_weight <= 0 or max_weight >= 1.0:
        return order_sizes

    if mode == "tilt":
        adjusted = order_sizes.copy()
        cumulative: dict[str, float] = {c: 0.0 for c in order_sizes.columns}

        for invest_date in invest_dates:
            prices = asset_prices.loc[invest_date]
            current_prices = {
                s: float(prices[s]) for s in order_sizes.columns if s in prices.index
            }
            total_value = sum(
                cumulative.get(s, 0.0) * current_prices.get(s, 0.0)
                for s in order_sizes.columns
            )

            planned_value = {
                symbol: max(0.0, float(order_sizes.loc[invest_date, symbol]) * current_prices.get(symbol, 0.0))
                for symbol in order_sizes.columns
            }
            planned_budget = sum(planned_value.values())
            if total_value > 0 and planned_budget > 0:
                planned_weights = {
                    symbol: value / planned_budget
                    for symbol, value in planned_value.items()
                }
                tilted_weights = compute_tilt_weights(
                    current_shares=cumulative,
                    current_prices=current_prices,
                    total_value=total_value,
                    target_weights=planned_weights,
                    default_weights=planned_weights,
                    max_weight=max_weight,
                )
                if tilted_weights != planned_weights:
                    for symbol in order_sizes.columns:
                        price = current_prices.get(symbol, 0.0)
                        adjusted.loc[invest_date, symbol] = (
                            planned_budget * tilted_weights.get(symbol, 0.0) / price
                            if price > 0
                            else 0.0
                        )

            for symbol in order_sizes.columns:
                buy_shares = adjusted.loc[invest_date, symbol]
                if buy_shares > 0:
                    cumulative[symbol] = cumulative.get(symbol, 0.0) + buy_shares

        return adjusted

    # sell 模式
    adjusted = order_sizes.copy()
    cumulative: dict[str, float] = {c: 0.0 for c in order_sizes.columns}

    for invest_date in invest_dates:
        prices = asset_prices.loc[invest_date]

        total_value = sum(
            cumulative.get(s, 0.0) * float(prices.get(s, 0.0))
            for s in order_sizes.columns
        )

        if total_value > 0:
            current_prices = {
                s: float(prices[s]) for s in order_sizes.columns if s in prices.index
            }
            sell_orders = compute_rebalance_orders(
                current_shares=cumulative,
                current_prices=current_prices,
                total_value=total_value,
                max_weight=max_weight,
            )
            for symbol, shares in sell_orders.items():
                adjusted.loc[invest_date, symbol] += shares
                cumulative[symbol] = cumulative.get(symbol, 0.0) + shares

        # 应用当天的 DCA 买入订单
        for symbol in order_sizes.columns:
            buy_shares = order_sizes.loc[invest_date, symbol]
            if buy_shares > 0:
                cumulative[symbol] = cumulative.get(symbol, 0.0) + buy_shares

    return adjusted
