from __future__ import annotations

import hashlib
import json
import time as _time
from pathlib import Path

import pandas as pd

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
    symbols: list[str],
    start: str,
    end: str,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """多标的年度收益率 DataFrame。

    index = year (int), columns = symbols.
    每个标的按自然年最后一个收盘价计算 pct_change。
    """
    close = fetch_close_prices(symbols, start=start, end=end, use_cache=use_cache)

    result = {}
    for sym in symbols:
        if sym not in close.columns:
            continue
        annual_close = close[sym].groupby(close.index.year).last()
        ann_ret = annual_close.pct_change().dropna()
        ann_ret.name = sym
        result[sym] = ann_ret

    if not result:
        return pd.DataFrame()

    df = pd.DataFrame(result)
    df.index.name = "year"
    return df


def fetch_vix_data(
    vix_symbol: str,
    start: str,
    end: str,
    *,
    use_cache: bool = True,
) -> pd.Series | None:
    """获取 VIX 收盘价序列（用于恐惧信号）。"""
    try:
        close = fetch_close_prices([vix_symbol], start=start, end=end, use_cache=use_cache)
        if vix_symbol in close.columns:
            return close[vix_symbol]
        return close.iloc[:, 0]
    except Exception:
        return None


def get_monthly_invest_dates(price_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    calendar = pd.Series(price_index, index=price_index)
    invest_dates = calendar.groupby(calendar.index.to_period("M")).first()
    return pd.DatetimeIndex(invest_dates.to_list())
