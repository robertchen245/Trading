from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"
_LOCAL_DATA_DIR = _PROJECT_ROOT / "data" / "local"

_DATA_SOURCES = {"auto", "auto_ibkr", "local", "ibkr", "yfinance", "stooq"}


class DataFetchError(RuntimeError):
    """Raised when no configured data source can produce close prices."""


@dataclass(frozen=True)
class DataFetchConfig:
    """Runtime knobs for resilient market-data loading.

    The defaults prefer deterministic local inputs first, then yfinance, then
    Stooq for simple US tickers. Environment variables can override the same
    settings without changing strategy specs:

    - TRADING_DATA_SOURCE=auto|auto_ibkr|local|ibkr|yfinance|stooq
    - TRADING_DATA_LOCAL_DIR=/path/to/csvs
    - TRADING_YF_MAX_RETRIES=3
    - TRADING_YF_RETRY_SLEEP=1.0
    - TRADING_YF_SYMBOL_SLEEP=0.3
    - TRADING_ALLOW_STALE_CACHE=1
    - TRADING_IBKR_HOST=127.0.0.1
    - TRADING_IBKR_PORT=7497
    - TRADING_IBKR_CLIENT_ID=19
    - TRADING_IBKR_REQUEST_SLEEP=11.0
    """

    source: str = "auto"
    local_data_dir: Path = _LOCAL_DATA_DIR
    yf_max_retries: int = 3
    yf_retry_sleep: float = 1.0
    yf_symbol_sleep: float = 0.3
    allow_stale_cache: bool = True
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 7497
    ibkr_client_id: int = 19
    ibkr_exchange: str = "SMART"
    ibkr_currency: str = "USD"
    ibkr_timeout: float = 10.0
    ibkr_request_sleep: float = 11.0

    @classmethod
    def from_env(
        cls,
        *,
        source: str | None = None,
        local_data_dir: str | Path | None = None,
        yf_max_retries: int | None = None,
        yf_retry_sleep: float | None = None,
        allow_stale_cache: bool | None = None,
    ) -> "DataFetchConfig":
        selected_source = source or os.getenv("TRADING_DATA_SOURCE", "auto")
        local_dir = Path(local_data_dir or os.getenv("TRADING_DATA_LOCAL_DIR", _LOCAL_DATA_DIR))
        max_retries = _env_int("TRADING_YF_MAX_RETRIES", 3) if yf_max_retries is None else yf_max_retries
        retry_sleep = _env_float("TRADING_YF_RETRY_SLEEP", 1.0) if yf_retry_sleep is None else yf_retry_sleep
        symbol_sleep = _env_float("TRADING_YF_SYMBOL_SLEEP", 0.3)
        stale = _env_bool("TRADING_ALLOW_STALE_CACHE", True) if allow_stale_cache is None else allow_stale_cache
        cfg = cls(
            source=selected_source,
            local_data_dir=local_dir,
            yf_max_retries=max(1, int(max_retries)),
            yf_retry_sleep=max(0.0, float(retry_sleep)),
            yf_symbol_sleep=max(0.0, float(symbol_sleep)),
            allow_stale_cache=bool(stale),
            ibkr_host=os.getenv("TRADING_IBKR_HOST", "127.0.0.1"),
            ibkr_port=_env_int("TRADING_IBKR_PORT", 7497),
            ibkr_client_id=_env_int("TRADING_IBKR_CLIENT_ID", 19),
            ibkr_exchange=os.getenv("TRADING_IBKR_EXCHANGE", "SMART"),
            ibkr_currency=os.getenv("TRADING_IBKR_CURRENCY", "USD"),
            ibkr_timeout=_env_float("TRADING_IBKR_TIMEOUT", 10.0),
            ibkr_request_sleep=_env_float("TRADING_IBKR_REQUEST_SLEEP", 11.0),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.source not in _DATA_SOURCES:
            raise ValueError(f"Unsupported data source: {self.source!r}")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _warn(message: str) -> None:
    print(f"[trading.data] {message}", file=sys.stderr)


def _remove_timezone(index: pd.Index) -> pd.Index:
    if getattr(index, "tz", None) is not None:
        return index.tz_localize(None)
    return index


def _cache_key(symbols: list[str], start: str, end: str) -> str:
    payload = json.dumps({"symbols": sorted(symbols), "start": start, "end": end}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _safe_symbol(symbol: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", symbol).strip("_")
    return safe or "symbol"


def _symbol_cache_key(symbol: str, start: str, end: str) -> str:
    payload = json.dumps({"symbol": symbol, "start": start, "end": end}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _symbol_cache_path(symbol: str, start: str, end: str) -> Path:
    return _CACHE_DIR / f"close_symbol_{_safe_symbol(symbol)}_{_symbol_cache_key(symbol, start, end)}.parquet"


def _normalize_close_series(series: pd.Series, symbol: str, start: str, end: str) -> pd.Series:
    s = series.copy()
    s.name = symbol
    s.index = pd.to_datetime(s.index)
    s.index = _remove_timezone(s.index)
    s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return s[(s.index >= start_ts) & (s.index < end_ts)]


def _read_symbol_cache(symbol: str, start: str, end: str) -> pd.Series | None:
    path = _symbol_cache_path(symbol, start, end)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if symbol in frame.columns:
        series = frame[symbol]
    else:
        series = frame.iloc[:, 0].rename(symbol)
    return _normalize_close_series(series, symbol, start, end)


def _write_symbol_cache(symbol: str, start: str, end: str, series: pd.Series) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    series.to_frame(symbol).to_parquet(_symbol_cache_path(symbol, start, end))


def _read_stale_symbol_cache(symbol: str, start: str, end: str) -> pd.Series | None:
    pattern = f"close_symbol_{_safe_symbol(symbol)}_*.parquet"
    candidates = sorted(_CACHE_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        frame = pd.read_parquet(path)
        series = frame[symbol] if symbol in frame.columns else frame.iloc[:, 0].rename(symbol)
        series = _normalize_close_series(series, symbol, start, end)
        if not series.empty:
            _warn(
                f"using stale cache for {symbol}: {series.index.min().date()} "
                f"to {series.index.max().date()}"
            )
            return series
    return None


def _local_data_candidates(symbol: str, local_data_dir: Path) -> list[Path]:
    safe = _safe_symbol(symbol)
    names = [symbol, safe]
    out: list[Path] = []
    for name in names:
        out.extend(
            [
                local_data_dir / f"{name}.csv",
                local_data_dir / f"{name}.parquet",
            ]
        )
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in out:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def _close_column(frame: pd.DataFrame) -> str:
    normalized = {str(c).strip().lower(): c for c in frame.columns}
    for name in ("close", "adj close", "adj_close", "price"):
        if name in normalized:
            return normalized[name]
    return frame.columns[-1]


def _date_column(frame: pd.DataFrame) -> str:
    normalized = {str(c).strip().lower(): c for c in frame.columns}
    for name in ("date", "datetime", "time", "timestamp"):
        if name in normalized:
            return normalized[name]
    return frame.columns[0]


def _fetch_symbol_from_local_csv(
    symbol: str,
    start: str,
    end: str,
    local_data_dir: Path,
) -> pd.Series:
    for path in _local_data_candidates(symbol, local_data_dir):
        if not path.exists():
            continue
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        if frame.empty:
            continue
        date_col = _date_column(frame)
        close_col = symbol if symbol in frame.columns else _close_column(frame)
        series = pd.Series(frame[close_col].values, index=pd.to_datetime(frame[date_col]), name=symbol)
        series = _normalize_close_series(series, symbol, start, end)
        if not series.empty:
            return series
    raise DataFetchError(f"local data not found for {symbol!r} in {local_data_dir}")


def _fetch_symbol_from_yfinance(
    symbol: str,
    start: str,
    end: str,
    cfg: DataFetchConfig,
) -> pd.Series:
    import yfinance as yf

    last_error: Exception | None = None
    for attempt in range(1, cfg.yf_max_retries + 1):
        try:
            hist = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False)
            if not hist.empty and "Close" in hist.columns:
                series = _normalize_close_series(hist["Close"], symbol, start, end)
                if not series.empty:
                    return series
            last_error = DataFetchError(f"yfinance returned no close data for {symbol}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        if attempt < cfg.yf_max_retries:
            _time.sleep(cfg.yf_retry_sleep * attempt)
    raise DataFetchError(f"yfinance failed for {symbol}: {last_error}")


def _ibkr_contract(symbol: str, cfg: DataFetchConfig):
    try:
        from ib_insync import Index, Stock
    except ImportError as exc:
        raise DataFetchError(
            "ibkr data source requires ib_insync. Install with `pip install -e .[ibkr]` "
            "or `pip install ib_insync`."
        ) from exc

    index_map = {
        "^VIX": ("VIX", "CBOE"),
        "VIX": ("VIX", "CBOE"),
        "^GSPC": ("SPX", "CBOE"),
        "SPX": ("SPX", "CBOE"),
    }
    if symbol in index_map:
        ib_symbol, exchange = index_map[symbol]
        return Index(ib_symbol, exchange, cfg.ibkr_currency)
    if symbol.startswith("^"):
        raise DataFetchError(
            f"IBKR index mapping for {symbol!r} is not built in. Use a local CSV "
            "or a tradeable ETF proxy in signal_symbols."
        )
    return Stock(symbol, cfg.ibkr_exchange, cfg.ibkr_currency)


def _ibkr_duration_for_range(start: pd.Timestamp, end: pd.Timestamp) -> str:
    days = max(1, int((end - start).days) + 1)
    return f"{min(days, 365)} D"


def _fetch_symbol_from_ibkr(
    symbol: str,
    start: str,
    end: str,
    cfg: DataFetchConfig,
) -> pd.Series:
    try:
        from ib_insync import IB, util
    except ImportError as exc:
        raise DataFetchError(
            "ibkr data source requires ib_insync. Install with `pip install -e .[ibkr]` "
            "or `pip install ib_insync`."
        ) from exc

    ib = IB()
    start_ts = pd.Timestamp(start).normalize()
    cursor = pd.Timestamp(end).normalize()
    frames: list[pd.DataFrame] = []
    try:
        ib.connect(cfg.ibkr_host, cfg.ibkr_port, clientId=cfg.ibkr_client_id, timeout=cfg.ibkr_timeout)
        contract = _ibkr_contract(symbol, cfg)
        qualified = ib.qualifyContracts(contract)
        contract = qualified[0] if qualified else contract
        while cursor > start_ts:
            chunk_start = max(start_ts, cursor - pd.Timedelta(days=365))
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=cursor.strftime("%Y%m%d 23:59:59"),
                durationStr=_ibkr_duration_for_range(chunk_start, cursor),
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                keepUpToDate=False,
            )
            if bars:
                frame = util.df(bars)
                if frame is not None and not frame.empty and "close" in frame.columns:
                    frames.append(frame[["date", "close"]])
            cursor = chunk_start
            if cursor > start_ts and cfg.ibkr_request_sleep > 0:
                ib.sleep(cfg.ibkr_request_sleep)
    except Exception as exc:  # noqa: BLE001
        raise DataFetchError(f"IBKR failed for {symbol}: {exc}") from exc
    finally:
        if ib.isConnected():
            ib.disconnect()

    if not frames:
        raise DataFetchError(f"IBKR returned no close data for {symbol}")

    frame = pd.concat(frames, axis=0, ignore_index=True)
    series = pd.Series(frame["close"].values, index=pd.to_datetime(frame["date"]), name=symbol)
    series = _normalize_close_series(series, symbol, start, end)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if series.empty:
        raise DataFetchError(f"IBKR close data for {symbol} did not overlap requested range")
    return series


def _stooq_symbol(symbol: str) -> str | None:
    if symbol.startswith("^"):
        return None
    s = symbol.lower().replace("-", ".")
    if "." not in s:
        s = f"{s}.us"
    return s


def _fetch_symbol_from_stooq(symbol: str, start: str, end: str) -> pd.Series:
    stooq_sym = _stooq_symbol(symbol)
    if stooq_sym is None:
        raise DataFetchError(f"stooq does not support symbol shape {symbol!r}")
    start_s = pd.Timestamp(start).strftime("%Y%m%d")
    end_s = pd.Timestamp(end).strftime("%Y%m%d")
    query = urllib.parse.urlencode({"s": stooq_sym, "i": "d", "d1": start_s, "d2": end_s})
    url = f"https://stooq.com/q/d/l/?{query}"
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            frame = pd.read_csv(response)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DataFetchError(f"stooq failed for {symbol}: {exc}") from exc
    if frame.empty or "Close" not in frame.columns:
        raise DataFetchError(f"stooq returned no close data for {symbol}")
    return _normalize_close_series(pd.Series(frame["Close"].values, index=frame["Date"]), symbol, start, end)


def _provider_order(source: str) -> tuple[str, ...]:
    if source == "auto":
        return ("local", "yfinance", "stooq")
    if source == "auto_ibkr":
        return ("local", "ibkr", "yfinance", "stooq")
    return (source,)


def _fetch_symbol_close(
    symbol: str,
    start: str,
    end: str,
    *,
    use_cache: bool,
    cfg: DataFetchConfig,
) -> pd.Series:
    if use_cache:
        cached = _read_symbol_cache(symbol, start, end)
        if cached is not None and not cached.empty:
            return cached

    errors: list[str] = []
    for provider in _provider_order(cfg.source):
        try:
            if provider == "local":
                series = _fetch_symbol_from_local_csv(symbol, start, end, cfg.local_data_dir)
            elif provider == "ibkr":
                series = _fetch_symbol_from_ibkr(symbol, start, end, cfg)
            elif provider == "yfinance":
                series = _fetch_symbol_from_yfinance(symbol, start, end, cfg)
            elif provider == "stooq":
                series = _fetch_symbol_from_stooq(symbol, start, end)
            else:
                raise DataFetchError(f"unknown provider {provider!r}")
            if not series.empty:
                if use_cache:
                    _write_symbol_cache(symbol, start, end, series)
                if provider != "local" and cfg.yf_symbol_sleep > 0:
                    _time.sleep(cfg.yf_symbol_sleep)
                return series
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{provider}: {exc}")

    if use_cache and cfg.allow_stale_cache:
        stale = _read_stale_symbol_cache(symbol, start, end)
        if stale is not None and not stale.empty:
            return stale

    detail = "; ".join(errors) if errors else "no provider attempted"
    raise DataFetchError(f"No data for symbol {symbol!r}. Tried {cfg.source}: {detail}")


def fetch_close_prices(
    symbols: list[str],
    start: str,
    end: str,
    *,
    use_cache: bool = True,
    data_source: str | None = None,
    local_data_dir: str | Path | None = None,
    yf_max_retries: int | None = None,
    yf_retry_sleep: float | None = None,
    allow_stale_cache: bool | None = None,
) -> pd.DataFrame:
    """Load close prices with cache and provider fallback.

    The public return shape stays the same as the original implementation:
    a date-indexed DataFrame with one close column per requested symbol. The
    fetch path is now more defensive against yfinance throttling:

    1. exact multi-symbol cache
    2. exact per-symbol cache
    3. local CSV/parquet files
    4. IBKR, when explicitly selected or source=auto_ibkr
    5. yfinance with retries/backoff
    6. Stooq for simple US tickers
    7. stale per-symbol cache, if enabled
    """
    unique_symbols = list(dict.fromkeys(symbols))
    if not unique_symbols:
        raise ValueError("symbols must be non-empty.")

    cfg = DataFetchConfig.from_env(
        source=data_source,
        local_data_dir=local_data_dir,
        yf_max_retries=yf_max_retries,
        yf_retry_sleep=yf_retry_sleep,
        allow_stale_cache=allow_stale_cache,
    )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"close_{_cache_key(unique_symbols, start, end)}.parquet"

    if use_cache and cache_path.exists():
        close = pd.read_parquet(cache_path)
        close.index = _remove_timezone(close.index)
        return close.sort_index()[unique_symbols]

    series_list = [
        _fetch_symbol_close(symbol, start, end, use_cache=use_cache, cfg=cfg)
        for symbol in unique_symbols
    ]

    close = pd.concat(series_list, axis=1).dropna(how="any").sort_index()
    close = close[unique_symbols]
    if close.empty:
        raise DataFetchError(f"No overlapping close-price dates for symbols: {unique_symbols}")

    if use_cache:
        close.to_parquet(cache_path)

    return close


def fetch_annual_returns(
    symbols: list[str],
    start: str,
    end: str,
    *,
    use_cache: bool = True,
    data_source: str | None = None,
    local_data_dir: str | Path | None = None,
    yf_max_retries: int | None = None,
    yf_retry_sleep: float | None = None,
    allow_stale_cache: bool | None = None,
) -> pd.DataFrame:
    """Return annual pct-change table for one or more signal symbols."""
    close = fetch_close_prices(
        symbols,
        start=start,
        end=end,
        use_cache=use_cache,
        data_source=data_source,
        local_data_dir=local_data_dir,
        yf_max_retries=yf_max_retries,
        yf_retry_sleep=yf_retry_sleep,
        allow_stale_cache=allow_stale_cache,
    )

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
    data_source: str | None = None,
    local_data_dir: str | Path | None = None,
    yf_max_retries: int | None = None,
    yf_retry_sleep: float | None = None,
    allow_stale_cache: bool | None = None,
) -> pd.Series | None:
    """Fetch VIX close data; return None if all providers fail."""
    try:
        close = fetch_close_prices(
            [vix_symbol],
            start=start,
            end=end,
            use_cache=use_cache,
            data_source=data_source,
            local_data_dir=local_data_dir,
            yf_max_retries=yf_max_retries,
            yf_retry_sleep=yf_retry_sleep,
            allow_stale_cache=allow_stale_cache,
        )
        if vix_symbol in close.columns:
            return close[vix_symbol]
        return close.iloc[:, 0]
    except Exception as exc:  # noqa: BLE001
        _warn(f"VIX data unavailable for {vix_symbol}: {exc}")
        return None


def get_monthly_invest_dates(price_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    calendar = pd.Series(price_index, index=price_index)
    invest_dates = calendar.groupby(calendar.index.to_period("M")).first()
    return pd.DatetimeIndex(invest_dates.to_list())
