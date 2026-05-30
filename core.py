# Core logic for Monkey Tester.
# Everything lives here until a section gets unwieldy, then split into core/ and data/.

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import pandas as pd
import akshare as ak
from pydantic import BaseModel, field_validator

CACHE_DIR = Path(__file__).parent / "cache"

# ---------------------------------------------------------------------------
# Signal spec schema
# ---------------------------------------------------------------------------

VALID_OPERATORS = {">", ">=", "<", "<=", "==", "crosses_above", "crosses_below"}
VALID_MODES = {"screen", "backtest"}
VALID_MARKETS = {"CN", "US"}
VALID_LOGIC = {"AND", "OR"}


class Condition(BaseModel):
    metric: str
    operator: str
    value: Union[
        float, str
    ]  # threshold (float) or metric name for cross comparisons (str)
    window: Union[int, None] = None  # lookback period in days where relevant
    timeframe: str = "1d"

    @field_validator("operator")
    @classmethod
    def operator_must_be_valid(cls, v: str) -> str:
        if v not in VALID_OPERATORS:
            raise ValueError(f"operator must be one of {VALID_OPERATORS}, got {v!r}")
        return v


class SignalSpec(BaseModel):
    mode: str  # "screen" or "backtest"
    market: str  # "CN" or "US"
    universe: Union[str, list[str]]  # "all_a_shares", "csi300", or explicit ticker list
    conditions: list[
        Condition
    ]  # used for screen mode; also as fallback if entry absent
    logic: str = "AND"  # how conditions combine: "AND" or "OR"

    # backtest-only fields
    entry: Union[list[Condition], None] = None
    exit: Union[list[Condition], None] = None
    holding_period: Union[int, None] = None  # max bars to hold
    date_range: Union[tuple[str, str], None] = None

    @field_validator("mode")
    @classmethod
    def mode_must_be_valid(cls, v: str) -> str:
        if v not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {v!r}")
        return v

    @field_validator("market")
    @classmethod
    def market_must_be_valid(cls, v: str) -> str:
        if v not in VALID_MARKETS:
            raise ValueError(f"market must be one of {VALID_MARKETS}, got {v!r}")
        return v

    @field_validator("logic")
    @classmethod
    def logic_must_be_valid(cls, v: str) -> str:
        if v not in VALID_LOGIC:
            raise ValueError(f"logic must be one of {VALID_LOGIC}, got {v!r}")
        return v


# ---------------------------------------------------------------------------
# Spec → engine translator
# ---------------------------------------------------------------------------


def _evaluate_condition(cond: Condition, df: pd.DataFrame) -> pd.Series:
    """Evaluate one Condition against a DataFrame, returning a boolean Series.

    One value per bar — True means that bar satisfies the condition.
    Metrics that don't exist in df return all-False with a printed warning.
    New metrics are supported by computing them onto df before calling this
    (see indicators section below); no changes needed here.
    """
    if cond.metric not in df.columns:
        print(
            f"[translator] metric {cond.metric!r} not in DataFrame — condition always False"
        )
        return pd.Series(False, index=df.index)

    lhs = df[cond.metric]

    # Right-hand side: numeric threshold or another column name
    if isinstance(cond.value, str):
        if cond.value not in df.columns:
            print(
                f"[translator] value column {cond.value!r} not in DataFrame — condition always False"
            )
            return pd.Series(False, index=df.index)
        rhs = df[cond.value]
    else:
        rhs = cond.value

    op = cond.operator
    if op == ">":
        mask = lhs > rhs
    elif op == ">=":
        mask = lhs >= rhs  # noqa: E701
    elif op == "<":
        mask = lhs < rhs  # noqa: E701
    elif op == "<=":
        mask = lhs <= rhs  # noqa: E701
    elif op == "==":
        mask = lhs == rhs  # noqa: E701
    elif op in ("crosses_above", "crosses_below"):
        prev_lhs = lhs.shift(1)
        prev_rhs = rhs.shift(1) if isinstance(rhs, pd.Series) else rhs
        if op == "crosses_above":
            mask = (lhs > rhs) & (prev_lhs <= prev_rhs)
        else:
            mask = (lhs < rhs) & (prev_lhs >= prev_rhs)
    else:
        print(f"[translator] unknown operator {op!r} — condition always False")
        return pd.Series(False, index=df.index)

    return mask.fillna(False)


def _combine_masks(masks: list[pd.Series], logic: str) -> pd.Series:
    """Combine boolean Series with AND or OR logic."""
    result = masks[0]
    for m in masks[1:]:
        result = (result & m) if logic == "AND" else (result | m)
    return result.fillna(False)


def translate(spec: SignalSpec, df: pd.DataFrame) -> pd.Series:
    """Translate a SignalSpec into a boolean Series aligned to df's index.

    Screen mode:  evaluates spec.conditions — caller takes .iloc[-1] for
                  the latest bar (Step 3 / screening).
    Backtest mode: evaluates spec.entry conditions — caller uses this inside
                  Strategy.next() to decide when to enter (Step 4 / backtesting).

    df must already contain all metric columns referenced in the spec.
    Metrics not yet in df will return all-False and print a warning —
    compute them via the indicators functions before calling translate().
    """
    conditions = (
        spec.entry if spec.mode == "backtest" and spec.entry else spec.conditions
    )

    if not conditions:
        return pd.Series(True, index=df.index)

    masks = [_evaluate_condition(cond, df) for cond in conditions]
    return _combine_masks(masks, spec.logic)


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------


class MarketDataProvider(ABC):
    """Abstract base — every market provider must implement these three methods
    and return DataFrames with the normalized English column set."""

    @abstractmethod
    def get_daily_history(
        self,
        symbol: str,
        start: str,
        end: str,
        adjust: str = "hfq",
        cache_only: bool = False,
    ) -> pd.DataFrame | None:
        """Return daily OHLCV + pct_change + turnover_rate for one symbol.

        Columns: date (str YYYY-MM-DD), open, high, close, low, volume,
                 pct_change (%), turnover_rate (%).
        cache_only=True: return cached data only, never trigger a live fetch.
        Returns None on failure — never raises to the caller.
        """

    @abstractmethod
    def list_symbols(self, market: str) -> list[str]:
        """Return all tradeable symbols for the given market."""

    @abstractmethod
    def get_snapshot(self, symbols: list[str]) -> pd.DataFrame | None:
        """Return the most recent completed bar for each symbol (for screening)."""


# ---------------------------------------------------------------------------
# AKShare China A-share provider
# ---------------------------------------------------------------------------

# AKShare column names → normalized English names
_CN_COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "涨跌幅": "pct_change",
    "换手率": "turnover_rate",
}

_NORMALIZED_COLS = [
    "date",
    "open",
    "high",
    "close",
    "low",
    "volume",
    "pct_change",
    "turnover_rate",
]


class AKShareCNProvider(MarketDataProvider):
    """China A-share daily data via AKShare (free, no API key required).

    Two caches per symbol:
      cn_<symbol>_raw.parquet   — raw pct_change + turnover_rate (always matches broker)
      cn_<symbol>_hfq.parquet   — hfq-adjusted OHLCV (for backtesting engine)

    get_daily_history() always returns raw pct_change/turnover_rate regardless of
    the adjust parameter. When adjust="hfq" the OHLCV columns are back-adjusted;
    when adjust="" they are raw. Signal metrics are never silently swapped.
    """

    def get_daily_history(
        self,
        symbol: str,
        start: str,
        end: str,
        adjust: str = "hfq",
        cache_only: bool = False,
    ) -> pd.DataFrame | None:
        today = pd.Timestamp.today().strftime("%Y-%m-%d")

        # Raw data is always the source of truth for pct_change/turnover_rate
        raw = self._get_or_fetch(
            symbol, start, end, adjust="", today=today, cache_only=cache_only
        )
        if raw is None or raw.empty:
            return None

        if not adjust:
            return raw

        # For adjusted prices, fetch hfq OHLCV and swap in raw signal metrics
        hfq = self._get_or_fetch(
            symbol, start, end, adjust=adjust, today=today, cache_only=cache_only
        )
        if hfq is None or hfq.empty:
            return raw

        ohlcv = [
            c
            for c in ["date", "open", "high", "close", "low", "volume"]
            if c in hfq.columns
        ]
        signals = [
            c for c in ["date", "pct_change", "turnover_rate"] if c in raw.columns
        ]
        result = hfq[ohlcv].merge(raw[signals], on="date", how="left")
        return result.sort_values("date").reset_index(drop=True)

    def list_symbols(self, market: str = "CN") -> list[str]:
        if market != "CN":
            print(f"[AKShareCNProvider] only CN market is supported, got {market!r}")
            return []
        try:
            df = ak.stock_info_a_code_name()
            return df["code"].tolist()
        except Exception as e:
            print(f"[AKShare] could not list symbols: {e}")
            return []

    def get_snapshot(self, symbols: list[str]) -> pd.DataFrame | None:
        """Return the most recent completed bar for each symbol."""
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        start = (pd.Timestamp.today() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        rows = []
        for sym in symbols:
            df = self.get_daily_history(sym, start=start, end=end)
            if df is not None and not df.empty:
                row = df.iloc[-1].to_dict()
                row["symbol"] = sym
                rows.append(row)
        if not rows:
            return None
        return pd.DataFrame(rows).reset_index(drop=True)

    # --- private helpers ---

    def _get_or_fetch(
        self,
        symbol: str,
        start: str,
        end: str,
        adjust: str,
        today: str,
        cache_only: bool = False,
    ) -> pd.DataFrame | None:
        """Load from cache, fetching missing ranges from AKShare unless cache_only=True.

        cache_only=True is used by screening — screens must never trigger live API
        calls over a large universe (cache-first rule).
        """
        cached = self._load_cache(symbol, adjust)

        if cached is not None:
            # Drop today's bar — may have been cached before final close
            if cached["date"].max() == today:
                cached = cached[cached["date"] < today]
            if cached.empty:
                cached = None

        if cached is not None:
            cached_start = cached["date"].min()
            cached_end = cached["date"].max()
            needs_before = start < cached_start
            needs_after = cached_end < end

            if not needs_before and not needs_after:
                return self._filter(cached, start, end)

            if cache_only:
                return self._filter(cached, start, end)

            pieces = [cached]
            if needs_before:
                old_end = (pd.Timestamp(cached_start) - pd.Timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                )
                old = self._fetch(symbol, start, old_end, adjust)
                if old is not None and not old.empty:
                    pieces.insert(0, old)
            if needs_after:
                new_start = (pd.Timestamp(cached_end) + pd.Timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                )
                new = self._fetch(symbol, new_start, end, adjust)
                if new is not None and not new.empty:
                    pieces.append(new)

            combined = (
                pd.concat(pieces, ignore_index=True)
                .drop_duplicates("date")
                .sort_values("date")
                .reset_index(drop=True)
            )
            self._save_cache(combined, symbol, adjust)
            return self._filter(combined, start, end)

        if cache_only:
            return None

        # No cache — fetch full range
        df = self._fetch(symbol, start, end, adjust)
        if df is None or df.empty:
            return None
        self._save_cache(df, symbol, adjust)
        return df

    def _fetch(
        self, symbol: str, start: str, end: str, adjust: str
    ) -> pd.DataFrame | None:
        """Try EastMoney first; fall back to Tencent if that fails."""
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust=adjust,
                timeout=10,
            )
            if df is not None and not df.empty:
                df = df.rename(columns=_CN_COLUMN_MAP)
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                keep = [c for c in _NORMALIZED_COLS if c in df.columns]
                return df[keep].sort_values("date").reset_index(drop=True)
        except Exception as e:
            print(
                f"[AKShare/EM] EastMoney unavailable for {symbol}: {e} — trying Tencent fallback"
            )

        return self._fetch_tx(symbol, start, end, adjust)

    def _fetch_tx(
        self, symbol: str, start: str, end: str, adjust: str
    ) -> pd.DataFrame | None:
        """Tencent Finance fallback (proxy.finance.qq.com).

        Returns OHLC + computed pct_change. Volume and turnover_rate will be
        NaN because the Tencent API does not expose share-count volume.
        Signals that depend on volume_vs_avg20 or turnover_rate will not fire
        until the cache is rebuilt via a working EastMoney connection.
        """
        prefix = "sh" if symbol.startswith("6") else "sz"
        try:
            df = ak.stock_zh_a_hist_tx(
                symbol=f"{prefix}{symbol}",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust=adjust,
            )
        except Exception as e:
            print(f"[AKShare/TX] Tencent fallback also failed for {symbol}: {e}")
            return None

        if df is None or df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["pct_change"] = df["close"].pct_change() * 100
        df["volume"] = float("nan")
        df["turnover_rate"] = float("nan")

        print(
            f"[AKShare/TX] fetched {symbol} via Tencent — volume/turnover_rate unavailable"
        )
        keep = [c for c in _NORMALIZED_COLS if c in df.columns]
        return df[keep].sort_values("date").reset_index(drop=True)

    def _cache_path(self, symbol: str, adjust: str) -> Path:
        label = adjust if adjust else "raw"
        return CACHE_DIR / f"cn_{symbol}_{label}.parquet"

    def _load_cache(self, symbol: str, adjust: str) -> pd.DataFrame | None:
        path = self._cache_path(symbol, adjust)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                return None
        return None

    def _save_cache(self, df: pd.DataFrame, symbol: str, adjust: str) -> None:
        CACHE_DIR.mkdir(exist_ok=True)
        df.to_parquet(self._cache_path(symbol, adjust), index=False)

    def _filter(self, df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cache warming
# ---------------------------------------------------------------------------


def warm_cache(
    symbols: list[str],
    start: str = "2022-01-01",
    adjust: str = "hfq",
) -> None:
    """Download and cache history for a list of symbols.

    Run this once before screening or backtesting a large universe.
    Only fetches dates not already on disk — safe to re-run.
    """
    provider = AKShareCNProvider()
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {sym} ...", end=" ", flush=True)
        result = provider.get_daily_history(sym, start=start, end=end, adjust=adjust)
        print("ok" if result is not None else "unavailable")
    print("Cache warm complete.")


# ---------------------------------------------------------------------------
# Screening  (Phase 2 Step 3)
# ---------------------------------------------------------------------------


def _resolve_universe(
    universe: Union[str, list[str]], provider: MarketDataProvider
) -> list[str]:
    """Convert a universe spec to a flat list of symbol strings."""
    if isinstance(universe, list):
        return universe
    if universe == "csi300":
        try:
            df = ak.index_stock_cons_weight_csindex(symbol="000300")
            return df["成分券代码"].tolist()
        except Exception as e:
            print(f"[screen] could not fetch CSI 300 constituents: {e}")
            return []
    if universe == "all_a_shares":
        return provider.list_symbols("CN")
    print(
        f"[screen] unknown universe {universe!r} — pass a list of symbols or 'csi300' / 'all_a_shares'"
    )
    return []


def run_screen(spec: SignalSpec, provider: MarketDataProvider) -> pd.DataFrame:
    """Run a screen spec against cached data, returning a DataFrame of matches.

    Evaluates spec.conditions against the most recent bar for each symbol.
    Only reads from cache — never triggers a live AKShare call (cache-first rule).
    Symbols not yet in cache are silently skipped; run warm_cache() first.

    Returns a DataFrame with columns [symbol, date, + all cached metric columns],
    sorted by pct_change descending. Empty DataFrame if nothing matches.
    """
    assert spec.mode == "screen", "run_screen only accepts mode='screen'"

    symbols = _resolve_universe(spec.universe, provider)
    if not symbols:
        print("[screen] no symbols resolved from universe")
        return pd.DataFrame()

    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    # 60 days covers ~42 trading days — enough warmup for all standard 20-bar indicators
    start = (pd.Timestamp.today() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")

    matches = []
    for sym in symbols:
        df = provider.get_daily_history(sym, start=start, end=end, cache_only=True)
        if df is None or df.empty:
            continue

        from indicators import add_indicators  # lazy import avoids circular dependency

        df = add_indicators(df, sym)

        mask = translate(spec, df)
        if not mask.iloc[-1]:  # only the latest bar counts for a screen
            continue

        row = df.iloc[-1].to_dict()
        row["symbol"] = sym
        matches.append(row)

    if not matches:
        return pd.DataFrame()

    result = pd.DataFrame(matches)
    front = [c for c in ["symbol", "date"] if c in result.columns]
    rest = [c for c in result.columns if c not in front]
    result = result[front + rest].reset_index(drop=True)

    if "pct_change" in result.columns:
        result = result.sort_values("pct_change", ascending=False).reset_index(
            drop=True
        )

    return result


# ---------------------------------------------------------------------------
# Board-aware limit-up / limit-down thresholds  (Phase 2 Step 5)
# ---------------------------------------------------------------------------
#
# A-share daily price limits by board:
#   STAR Market  688xxx           ±20%
#   ChiNext      300xxx / 301xxx  ±20%
#   Main board   everything else  ±10%
#   ST / *ST     any board        ±5%  (requires caller to pass is_st=True)
#
# Thresholds use 0.1 pp of headroom (9.9 / 19.9 / 4.9) so a stock that closes
# exactly at the limit is caught robustly even with minor float rounding.


def classify_board(symbol: str) -> str:
    """Return the exchange board for a CN A-share symbol.

    Returns one of: "star", "chinext", "main".
    Symbol must be a 6-digit string (leading zeros preserved).
    """
    if symbol.startswith("688"):
        return "star"
    if symbol.startswith("300") or symbol.startswith("301"):
        return "chinext"
    return "main"


def limit_up_pct(symbol: str, is_st: bool = False) -> float:
    """Return the effective limit-up threshold (%) for a symbol.

    Use >= this value against pct_change to detect a limit-up hit.
    Pass is_st=True for ST / *ST stocks (any board).
    """
    if is_st:
        return 4.9
    return 19.9 if classify_board(symbol) in ("star", "chinext") else 9.9


def limit_down_pct(symbol: str, is_st: bool = False) -> float:
    """Return the effective limit-down threshold (%) for a symbol.

    Use <= this value against pct_change to detect a limit-down hit.
    """
    if is_st:
        return -4.9
    return -19.9 if classify_board(symbol) in ("star", "chinext") else -9.9


# ---------------------------------------------------------------------------
# Backtesting  (Phase 2 Step 4)
# ---------------------------------------------------------------------------

# Default A-share transaction costs:
#   0.1% covers broker commission + exchange fee on both sides.
#   Stamp duty (0.05%, sell-only) is absorbed into the per-side rate.
#   Round-trip cost ≈ 0.2%, which matches realistic retail A-share trading.
_DEFAULT_COMMISSION = 0.001
_DEFAULT_CASH = 1_000_000  # CNY; keeps position sizing reasonable on A-share prices


def run_backtest(
    spec: SignalSpec,
    symbol: str,
    provider: MarketDataProvider,
    cash: float = _DEFAULT_CASH,
    commission: float = _DEFAULT_COMMISSION,
) -> "pd.Series | None":
    """Run a single-symbol backtest for the given SignalSpec.

    Returns the backtesting.py stats Series on success, or None if there is
    insufficient data. Always uses hfq-adjusted prices so splits/dividends
    don't create false signals. Transaction costs are applied on every trade.

    Entry logic:  spec.entry conditions (falls back to spec.conditions if absent).
    Exit logic:   spec.exit conditions OR spec.holding_period — whichever fires first.
    """
    from backtesting import (
        Backtest,
        Strategy,
    )  # imported here to keep top-level imports light

    assert spec.mode == "backtest", "run_backtest only accepts mode='backtest'"

    # Resolve date range
    if spec.date_range:
        start, end = spec.date_range
    else:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        start = (pd.Timestamp.today() - pd.Timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    df = provider.get_daily_history(symbol, start=start, end=end, adjust="hfq")
    if df is None or df.empty:
        print(f"[backtest] no data for {symbol}")
        return None
    if len(df) < 20:
        print(f"[backtest] {symbol} has only {len(df)} bars — too few to backtest")
        return None

    from indicators import add_indicators  # lazy import avoids circular dependency

    df = add_indicators(df, symbol)

    # --- Precompute entry and exit boolean arrays via the translator ---
    # We evaluate conditions over the full DataFrame once here, then pass the
    # resulting arrays into the Strategy so next() never re-scans history.

    entry_conditions = spec.entry if spec.entry else spec.conditions
    _entry_spec = SignalSpec(
        mode="backtest",
        market=spec.market,
        universe=[symbol],
        conditions=entry_conditions or [],
        logic=spec.logic,
    )
    entry_arr = translate(_entry_spec, df).values

    exit_arr = None
    if spec.exit:
        _exit_spec = SignalSpec(
            mode="backtest",
            market=spec.market,
            universe=[symbol],
            conditions=spec.exit,
            logic=spec.logic,
        )
        exit_arr = translate(_exit_spec, df).values

    holding = spec.holding_period

    # --- Build the Strategy class dynamically ---
    # Captures entry_arr, exit_arr, holding via closure — each call to
    # run_backtest gets its own Strategy class, which is fine since it's
    # used immediately and discarded.

    class SpecStrategy(Strategy):
        def init(self):
            self._entry_bar = None

        def next(self):
            i = len(self.data) - 1  # 0-based index of the current bar

            if not self.position:
                if i < len(entry_arr) and entry_arr[i]:
                    self.buy()
                    self._entry_bar = i
            else:
                should_exit = bool(
                    exit_arr is not None and i < len(exit_arr) and exit_arr[i]
                )
                if holding and self._entry_bar is not None:
                    if (i - self._entry_bar) >= holding:
                        should_exit = True
                if should_exit:
                    self.position.close()
                    self._entry_bar = None

    # --- Prepare DataFrame for backtesting.py ---
    # Needs a DatetimeIndex and capitalised OHLCV column names.
    bt_df = (
        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        .set_index(pd.to_datetime(df["date"]))
        .drop(columns=["date"])
    )

    bt = Backtest(
        bt_df,
        SpecStrategy,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        finalize_trades=True,
    )
    return bt.run()


# ---------------------------------------------------------------------------
# Phase 1 proof-of-concept screen (kept for reference, superseded by run_screen)
# ---------------------------------------------------------------------------


def hardcoded_limit_up_screen(
    symbols: list[str],
    min_turnover: float = 10.0,
) -> pd.DataFrame:
    """Screen for stocks that hit limit-up on the most recent trading day
    with turnover rate above min_turnover %.

    Uses >= 9.9 % change to catch limit-ups robustly (main-board ±10% limit).
    This is a throwaway proof-of-concept to verify the data path end to end.
    """
    provider = AKShareCNProvider()
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    matches = []
    for sym in symbols:
        df = provider.get_daily_history(sym, start=start, end=end)
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        if (
            last["pct_change"] >= limit_up_pct(sym)
            and last["turnover_rate"] > min_turnover
        ):
            matches.append(
                {
                    "symbol": sym,
                    "date": last["date"],
                    "pct_change": last["pct_change"],
                    "turnover_rate": last["turnover_rate"],
                    "close": last["close"],
                }
            )

    return (
        pd.DataFrame(matches)
        if matches
        else pd.DataFrame(
            columns=["symbol", "date", "pct_change", "turnover_rate", "close"]
        )
    )
