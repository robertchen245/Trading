from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from trading.strategies.dca import DCAParams

SUPPORTED_ALLOCATORS = ("fixed", "nasdaq_rule", "equal_weight", "smart")


@dataclass(frozen=True)
class StrategySpec:
    """策略规格：用于批量实验与后续 LLM 生成。"""

    name: str
    symbols: tuple[str, ...]
    start: str
    end: str
    monthly_budget: float
    default_weights: dict[str, float]
    allocator: str = "fixed"
    signal_symbols: tuple[str, ...] = ("^IXIC",)
    vix_symbol: str | None = None
    drawdown_lookback: int = 252
    ma_window: int = 200
    benchmark_symbol: str = "QQQ"
    extra_symbols: tuple[str, ...] = ()
    use_cache: bool = True
    fee_rate: float = 0.0
    slippage_rate: float = 0.0
    max_weight_per_asset: float | None = None
    max_gross_exposure: float | None = None
    risk_observe_only: bool = False
    rebalance_max_weight: float | None = None
    rebalance_mode: str = "sell"
    cash_symbol: str | None = None

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("StrategySpec.name must be non-empty.")
        if len(self.symbols) == 0:
            raise ValueError("StrategySpec.symbols must be non-empty.")
        if self.monthly_budget <= 0:
            raise ValueError("StrategySpec.monthly_budget must be > 0.")
        if self.allocator not in SUPPORTED_ALLOCATORS:
            raise ValueError(f"Unsupported allocator: {self.allocator!r}")
        missing = [symbol for symbol in self.symbols if symbol not in self.default_weights]
        if missing:
            raise ValueError(f"default_weights missing symbols: {missing}")

    def to_params(self) -> DCAParams:
        self.validate()
        return DCAParams(
            symbols=self.symbols,
            start=self.start,
            end=self.end,
            monthly_budget=self.monthly_budget,
            default_weights=self.default_weights,
            signal_symbols=self.signal_symbols,
            vix_symbol=self.vix_symbol,
            drawdown_lookback=self.drawdown_lookback,
            ma_window=self.ma_window,
            benchmark_symbol=self.benchmark_symbol,
            extra_symbols=self.extra_symbols,
            use_cache=self.use_cache,
            fee_rate=self.fee_rate,
            slippage_rate=self.slippage_rate,
            max_weight_per_asset=self.max_weight_per_asset,
            max_gross_exposure=self.max_gross_exposure,
            risk_observe_only=self.risk_observe_only,
            rebalance_max_weight=self.rebalance_max_weight,
            rebalance_mode=self.rebalance_mode,
            cash_symbol=self.cash_symbol,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrategySpec":
        return cls(
            name=str(payload["name"]),
            symbols=tuple(payload["symbols"]),
            start=str(payload["start"]),
            end=str(payload["end"]),
            monthly_budget=float(payload["monthly_budget"]),
            default_weights={str(k): float(v) for k, v in dict(payload["default_weights"]).items()},
            allocator=str(payload.get("allocator", "fixed")),
            signal_symbols=tuple(payload.get("signal_symbols", ("^IXIC",))),
            vix_symbol=_maybe_str(payload.get("vix_symbol")),
            drawdown_lookback=int(payload.get("drawdown_lookback", 252)),
            ma_window=int(payload.get("ma_window", 200)),
            benchmark_symbol=str(payload.get("benchmark_symbol", "QQQ")),
            extra_symbols=tuple(payload.get("extra_symbols", ())),
            use_cache=bool(payload.get("use_cache", True)),
            fee_rate=float(payload.get("fee_rate", 0.0)),
            slippage_rate=float(payload.get("slippage_rate", 0.0)),
            max_weight_per_asset=_maybe_float(payload.get("max_weight_per_asset")),
            max_gross_exposure=_maybe_float(payload.get("max_gross_exposure")),
            risk_observe_only=bool(payload.get("risk_observe_only", False)),
            rebalance_max_weight=_maybe_float(payload.get("rebalance_max_weight")),
            rebalance_mode=str(payload.get("rebalance_mode", "sell")),
            cash_symbol=_maybe_str(payload.get("cash_symbol")),
        )


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def preset_strategy_specs() -> list[StrategySpec]:
    """内置策略模板。"""
    return [
        StrategySpec(
            name="balanced_fixed",
            symbols=("QQQ", "TQQQ"),
            start="2016-01-01",
            end="2026-01-01",
            monthly_budget=5000.0,
            default_weights={"QQQ": 0.75, "TQQQ": 0.25},
            allocator="fixed",
        ),
        StrategySpec(
            name="aggressive_nasdaq_rule",
            symbols=("QQQ", "TQQQ"),
            start="2016-01-01",
            end="2026-01-01",
            monthly_budget=5000.0,
            default_weights={"QQQ": 0.6, "TQQQ": 0.4},
            allocator="nasdaq_rule",
            max_weight_per_asset=0.8,
        ),
        StrategySpec(
            name="defensive_cap_exposure",
            symbols=("QQQ", "TQQQ"),
            start="2016-01-01",
            end="2026-01-01",
            monthly_budget=5000.0,
            default_weights={"QQQ": 0.85, "TQQQ": 0.15},
            allocator="fixed",
            max_weight_per_asset=0.9,
            max_gross_exposure=0.85,
        ),
        StrategySpec(
            name="smart_signal_fusion",
            symbols=("QQQ", "TQQQ"),
            start="2016-01-01",
            end="2026-01-01",
            monthly_budget=5000.0,
            default_weights={"QQQ": 0.7, "TQQQ": 0.3},
            allocator="smart",
            signal_symbols=("^IXIC",),
            vix_symbol="^VIX",
            drawdown_lookback=252,
            ma_window=200,
        ),
        StrategySpec(
            name="balanced_rebalance_75",
            symbols=("QQQ", "TQQQ"),
            start="2016-01-01",
            end="2026-01-01",
            monthly_budget=5000.0,
            default_weights={"QQQ": 0.7, "TQQQ": 0.3},
            allocator="smart",
            signal_symbols=("^IXIC",),
            vix_symbol="^VIX",
            rebalance_max_weight=0.75,
        ),
    ]


def nl_to_strategy_spec(prompt: str, *, name: str = "nl_generated_strategy") -> StrategySpec:
    """MVP: 将自然语言转为策略草案。"""
    text = prompt.lower()
    symbols = _extract_symbols(prompt) or ("QQQ", "TQQQ")

    # 分配器检测
    allocator = "fixed"
    if "smart" in text or "多信号" in text or "vix" in text:
        allocator = "smart"
    elif "nasdaq_rule" in text or "纳指规则" in text or "动量" in text:
        allocator = "nasdaq_rule"
    elif "等权" in text or "equal weight" in text:
        allocator = "equal_weight"

    # 激进/保守
    if "激进" in text or "aggressive" in text:
        qqq_weight, tqqq_weight = 0.6, 0.4
    elif "保守" in text or "defensive" in text:
        qqq_weight, tqqq_weight = 0.85, 0.15
    else:
        qqq_weight, tqqq_weight = 0.75, 0.25

    # VIX 检测
    vix_sym = "^VIX" if ("vix" in text or "恐惧" in text) else None

    default_weights = _guess_weights(symbols, qqq_weight, tqqq_weight)
    return StrategySpec(
        name=name,
        symbols=symbols,
        start="2016-01-01",
        end="2026-01-01",
        monthly_budget=5000.0,
        default_weights=default_weights,
        allocator=allocator,
        vix_symbol=vix_sym,
    )


def _extract_symbols(prompt: str) -> tuple[str, ...]:
    candidates = re.findall(r"\b[A-Z^][A-Z0-9^]{1,7}\b", prompt)
    unique = []
    seen: set[str] = set()
    for symbol in candidates:
        if symbol in seen:
            continue
        seen.add(symbol)
        unique.append(symbol)
    return tuple(unique[:8])


def _guess_weights(
    symbols: tuple[str, ...], qqq_weight: float, tqqq_weight: float
) -> dict[str, float]:
    if len(symbols) == 1:
        return {symbols[0]: 1.0}
    if "QQQ" in symbols and "TQQQ" in symbols:
        rest = [s for s in symbols if s not in {"QQQ", "TQQQ"}]
        leftover = max(0.0, 1.0 - qqq_weight - tqqq_weight)
        out = {"QQQ": qqq_weight, "TQQQ": tqqq_weight}
        if rest:
            each = leftover / len(rest) if leftover > 0 else 0.0
            for symbol in rest:
                out[symbol] = each
        return out
    equal = 1.0 / len(symbols)
    return {symbol: equal for symbol in symbols}
