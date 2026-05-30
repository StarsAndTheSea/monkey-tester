"""
Framework metrics for Monkey signal research.

Each function accepts a normalized OHLCV DataFrame (as returned by
AKShareCNProvider.get_daily_history) and returns a pd.Series aligned to
df's index. All functions are lookahead-free — see tests/test_lookahead.py.

add_indicators(df, symbol) is the main entry point. It computes all
standard metrics and returns an augmented copy of df whose new columns are
immediately usable in a SignalSpec via the translator.

Column names added by add_indicators:
    consecutive_limit_ups   — bars of consecutive limit-up ending at t
    turnover_rate_5d        — 5-bar rolling mean of turnover_rate
    turnover_rate_10d       — 10-bar rolling mean of turnover_rate
    volume_vs_avg20         — volume / 20-bar rolling mean of volume
    range_compression20     — (H-L)/C divided by its 20-bar rolling mean
    gap_open_pct            — (open - prev_close) / prev_close × 100
    days_since_limit_up     — trading bars since last limit-up (0 on LU day,
                              NaN if no prior limit-up in the window)
    above_ma5               — 1 if close > 5-day MA, else 0
    above_ma10              — 1 if close > 10-day MA, else 0
    above_ma20              — 1 if close > 20-day MA, else 0
    above_ma40              — 1 if close > 40-day MA (~2-month), else 0
    above_ma120             — 1 if close > 120-day MA (~6-month), else 0
    macd_golden             — 1 if DIF > DEA (golden cross state), else 0
    kdj_golden              — 1 if K > D (golden cross state), else 0
    cci_golden              — 1 if CCI > 0 (bullish state), else 0
    rsi_golden              — 1 if RSI(14) > 50 (bullish state), else 0
"""

import numpy as np
import pandas as pd

from core import limit_up_pct


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def consecutive_limit_ups(
    df: pd.DataFrame,
    symbol: str,
    is_st: bool = False,
) -> pd.Series:
    """Number of consecutive limit-up days ending at each bar.

    0 on any bar that did not hit limit-up; 1 on a single limit-up, 2 on
    the second consecutive limit-up, and so on.
    """
    thresh = limit_up_pct(symbol, is_st)
    is_lu = (df["pct_change"] >= thresh).astype(int)
    # Each 0 bar starts a new group; cumsum within each True-run gives 1,2,3…
    groups = (is_lu == 0).cumsum()
    return is_lu.groupby(groups).cumsum().astype(int)


def turnover_rate_nd(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """N-bar rolling mean of turnover_rate (%)."""
    return df["turnover_rate"].rolling(n).mean()


def volume_vs_avg(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Current volume divided by the prior N-bar rolling mean of volume.

    Compares today against the preceding N bars (today excluded from the
    average), so a genuine 2× spike yields a ratio of exactly 2.0.
    >1 means above-average; <1 means below-average.
    NaN during the warm-up period (first n+1 bars).
    """
    avg = df["volume"].rolling(n).mean().shift(1)   # prior N bars only
    return (df["volume"] / avg).replace([np.inf, -np.inf], np.nan)


def range_compression(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """(High - Low) / Close divided by the prior N-bar rolling mean of that ratio.

    Compares today's range against the preceding N bars (today excluded),
    so a bar with exactly half the average range yields a ratio of 0.5.
    <1 = compressed; >1 = expanded.
    NaN during the warm-up period (first n+1 bars).
    """
    daily_range = (df["high"] - df["low"]) / df["close"]
    avg_range = daily_range.rolling(n).mean().shift(1)   # prior N bars only
    return (daily_range / avg_range).replace([np.inf, -np.inf], np.nan)


def close_above_ma(df: pd.DataFrame, n: int) -> pd.Series:
    """1 if close is above the N-bar simple moving average, 0 otherwise.

    The MA is computed over the current bar and the preceding n-1 bars
    (standard definition, no shift). NaN during the first n-1 bars.
    """
    ma = df["close"].rolling(n).mean()
    return (df["close"] > ma).astype(float)


def gap_open_pct(df: pd.DataFrame) -> pd.Series:
    """Overnight gap as a percentage of the previous bar's close.

    (open_t - close_{t-1}) / close_{t-1} × 100.
    NaN on the first bar (no previous close).
    Positive = gap up; negative = gap down.
    """
    prev_close = df["close"].shift(1)
    return (df["open"] - prev_close) / prev_close * 100


def days_since_limit_up(
    df: pd.DataFrame,
    symbol: str,
    is_st: bool = False,
) -> pd.Series:
    """Trading bars elapsed since the most recent limit-up.

    0 on a limit-up day itself.
    1, 2, 3… on subsequent bars.
    NaN if no limit-up has occurred yet in the DataFrame.

    Use with a threshold such as `days_since_limit_up < 10` to find stocks
    in the aftermath of a limit-up event.
    """
    thresh = limit_up_pct(symbol, is_st)
    is_lu = df["pct_change"] >= thresh
    pos = pd.Series(np.arange(len(df), dtype=float), index=df.index)
    last_lu_pos = pos.where(is_lu).ffill()   # NaN until first limit-up
    return pos - last_lu_pos                  # NaN → no prior LU; 0 → LU day


# ---------------------------------------------------------------------------
# MACD / KDJ / CCI / RSI helpers
# ---------------------------------------------------------------------------

def _macd_lines(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """Return (DIF, DEA) Series using standard EMA smoothing."""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea


def _kdj_lines(
    df: pd.DataFrame,
    n: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """Return (K, D) Series using Wilder's 1/3 smoothing (Chinese convention).

    RSV warm-up NaNs are filled with 50 before smoothing so K and D start
    near the neutral midpoint rather than the first valid RSV value.
    """
    low_n  = df["low"].rolling(n).min()
    high_n = df["high"].rolling(n).max()
    denom  = (high_n - low_n).replace(0, np.nan)
    rsv    = ((df["close"] - low_n) / denom * 100).fillna(50)
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    return k, d


def _cci(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Commodity Channel Index over n bars."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def _rsi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing (alpha = 1/n)."""
    delta    = df["close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=n - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=n - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd_golden(df: pd.DataFrame) -> pd.Series:
    """1 if DIF > DEA (golden cross state), 0 if DIF < DEA (dead cross)."""
    dif, dea = _macd_lines(df)
    return (dif > dea).astype(float)


def kdj_golden(df: pd.DataFrame) -> pd.Series:
    """1 if K > D (golden cross state), 0 if K < D (dead cross)."""
    k, d = _kdj_lines(df)
    return (k > d).astype(float)


def cci_golden(df: pd.DataFrame) -> pd.Series:
    """1 if CCI(14) > 0 (bullish state), 0 otherwise."""
    return (_cci(df) > 0).astype(float)


def rsi_golden(df: pd.DataFrame) -> pd.Series:
    """1 if RSI(14) > 50 (bullish state), 0 otherwise."""
    return (_rsi(df) > 50).astype(float)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def add_indicators(
    df: pd.DataFrame,
    symbol: str,
    is_st: bool = False,
) -> pd.DataFrame:
    """Return a copy of df with all standard indicator columns appended.

    Call this before translate() whenever a SignalSpec references any of
    the framework metric column names listed in the module docstring.
    """
    df = df.copy()
    df["consecutive_limit_ups"] = consecutive_limit_ups(df, symbol, is_st)
    df["turnover_rate_5d"]      = turnover_rate_nd(df, n=5)
    df["turnover_rate_10d"]     = turnover_rate_nd(df, n=10)
    df["volume_vs_avg20"]       = volume_vs_avg(df, n=20)
    df["range_compression20"]   = range_compression(df, n=20)
    df["gap_open_pct"]          = gap_open_pct(df)
    df["days_since_limit_up"]   = days_since_limit_up(df, symbol, is_st)
    df["above_ma5"]             = close_above_ma(df, 5)
    df["above_ma10"]            = close_above_ma(df, 10)
    df["above_ma20"]            = close_above_ma(df, 20)
    df["above_ma40"]            = close_above_ma(df, 40)
    df["above_ma120"]           = close_above_ma(df, 120)
    df["macd_golden"]           = macd_golden(df)
    df["kdj_golden"]            = kdj_golden(df)
    df["cci_golden"]            = cci_golden(df)
    df["rsi_golden"]            = rsi_golden(df)
    return df
