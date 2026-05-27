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
    return df
