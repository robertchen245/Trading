from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd

from trading.strategies.dca import DCAParams, WeightAllocator


@dataclass(frozen=True)
class ScenarioContext:
    """单次 `run_scenarios` 的只读上下文，供 `BaselineBuilder` 使用。"""

    params: DCAParams
    strategy_close: pd.DataFrame
    aligned_close: pd.DataFrame
    annual_returns: pd.Series
    monthly_budget: float
    total_invested: float
    allocator: WeightAllocator
    default_weights: dict[str, float]
    first_invest_date: pd.Timestamp
    first_month_weights: dict[str, float]


class BaselineBuilder(Protocol):
    def __call__(self, ctx: ScenarioContext) -> Any: ...
