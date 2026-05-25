from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import vectorbt as vbt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"


def _remove_timezone(index: pd.Index) -> pd.Index:
    if getattr(index, "tz", None) is not None:
        return index.tz_localize(None)
    return index


def _cache_key(symbols: list[str], start: str, end: str) -> str:
    payload = json.dumps({"symbols": sorted(symbols), "start": start, "end": end}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def fetch_close_prices(
    symbols: list[str],
    start: str,
    end: str,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """下载收盘价；可选写入 ``data/cache`` 下的 parquet。"""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"close_{_cache_key(symbols, start, end)}.parquet"

    if use_cache and cache_path.exists():
        close = pd.read_parquet(cache_path)
        close.index = _remove_timezone(close.index)
        return close.sort_index()

    # 逐一下载避免 vectorbt YFData 的列对齐 bug
    import time as _time
    import yfinance as yf
    series_list = []
    for sym in symbols:
        ticker = yf.Ticker(sym)
        hist = ticker.history(start=start, end=end, auto_adjust=False)
        if hist.empty:
            raise ValueError(f"No data for symbol: {sym}")
        s = hist["Close"].rename(sym)
        s.index = _remove_timezone(s.index)
        series_list.append(s)
        _time.sleep(0.3)  # 避免 yfinance 限流

    close = pd.concat(series_list, axis=1).dropna(how="any").sort_index()

    if use_cache:
        close.to_parquet(cache_path)

    return close


def fetch_annual_returns(
    symbol: str,
    start: str,
    end: str,
    *,
    use_cache: bool = True,
) -> pd.Series:
    """标的按自然年收盘计算的年度收益率序列（index=year）。"""
    close = fetch_close_prices([symbol], start=start, end=end, use_cache=use_cache)[symbol]
    annual_close = close.groupby(close.index.year).last()
    annual_returns = annual_close.pct_change().dropna()
    annual_returns.index.name = "year"
    annual_returns.name = f"{symbol}_annual_return"
    return annual_returns


def get_monthly_invest_dates(price_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    calendar = pd.Series(price_index, index=price_index)
    invest_dates = calendar.groupby(calendar.index.to_period("M")).first()
    return pd.DatetimeIndex(invest_dates.to_list())
