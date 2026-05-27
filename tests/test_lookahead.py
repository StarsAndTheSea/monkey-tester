"""
Lookahead bias tests.

Rule: a signal at bar t must only depend on data from bars 0..t.
Method: for N random bars t, replace all rows after t with extreme random
values, recompute the signal, and assert the value at bar t is unchanged.
A changed value means bar t used data from the future.

One test deliberately uses a future-leaking indicator to confirm the
detection machinery itself works.
"""

import numpy as np
import pandas as pd
import pytest

from core import Condition, SignalSpec, translate
from indicators import (
    consecutive_limit_ups, turnover_rate_nd, volume_vs_avg,
    range_compression, gap_open_pct, days_since_limit_up,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df():
    """60 bars of synthetic daily data with all standard columns."""
    n = 60
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.015, n))
    return pd.DataFrame({
        "date":         dates.strftime("%Y-%m-%d"),
        "open":         close * 0.99,
        "high":         close * 1.01,
        "low":          close * 0.98,
        "close":        close,
        "volume":       rng.integers(100_000, 500_000, n).astype(float),
        "pct_change":   rng.normal(0.1, 1.5, n),
        "turnover_rate":rng.uniform(1, 10, n),
    })


# ---------------------------------------------------------------------------
# Lookahead detection helper
# ---------------------------------------------------------------------------

_NUMERIC_COLS = ["open", "high", "close", "low", "volume", "pct_change", "turnover_rate"]


def _has_lookahead(signal_fn, df: pd.DataFrame, n_checks: int = 20, seed: int = 0) -> bool:
    """Return True if signal_fn leaks future data into any checked bar.

    For each sampled bar t, replaces rows t+1..end with extreme random values
    and checks whether the signal at bar t changes. A changed value at t means
    the computation used data from after t — lookahead bias.
    """
    baseline = signal_fn(df)

    rng = np.random.default_rng(seed)
    # exclude the last bar (nothing after it to perturb) and the first few
    # bars where rolling windows would be NaN anyway
    eligible = np.arange(5, len(df) - 1)
    check_bars = rng.choice(eligible, size=min(n_checks, len(eligible)), replace=False)

    numeric_positions = [df.columns.get_loc(c) for c in _NUMERIC_COLS if c in df.columns]

    for t in check_bars:
        perturbed = df.copy()
        n_future = len(df) - t - 1
        perturbed.iloc[t + 1:, numeric_positions] = rng.normal(
            0, 1e6, size=(n_future, len(numeric_positions))
        )
        perturbed_signal = signal_fn(perturbed)

        # NaN == NaN for this comparison — treat them as equal (both unknown)
        v_base = baseline.iloc[t]
        v_new  = perturbed_signal.iloc[t]
        both_nan = pd.isna(v_base) and pd.isna(v_new)
        if not both_nan and v_base != v_new:
            return True

    return False


# ---------------------------------------------------------------------------
# Infrastructure sanity check — must run first
# ---------------------------------------------------------------------------

def test_leaky_indicator_is_detected(df):
    """A shift(-1) signal uses tomorrow's close — test must catch this."""
    def leaky(d):
        return (d["close"].shift(-1) > d["close"]).fillna(False)

    assert _has_lookahead(leaky, df), (
        "Expected lookahead to be detected but wasn't — "
        "_has_lookahead() is broken and all passing tests below are meaningless."
    )


# ---------------------------------------------------------------------------
# Clean signal tests — these must all pass (no lookahead)
# ---------------------------------------------------------------------------

def test_pct_change_threshold(df):
    """`pct_change > 1.0` only reads the current bar's column."""
    spec = SignalSpec(
        mode="screen", market="CN", universe=["FAKE"],
        conditions=[Condition(metric="pct_change", operator=">", value=1.0)],
    )
    assert not _has_lookahead(lambda d: translate(spec, d), df)


def test_multi_condition_and(df):
    """AND of two threshold conditions — both must be clean."""
    spec = SignalSpec(
        mode="screen", market="CN", universe=["FAKE"],
        conditions=[
            Condition(metric="pct_change",   operator=">",  value=1.0),
            Condition(metric="turnover_rate", operator=">",  value=5.0),
        ],
        logic="AND",
    )
    assert not _has_lookahead(lambda d: translate(spec, d), df)


def test_multi_condition_or(df):
    """OR of two threshold conditions."""
    spec = SignalSpec(
        mode="screen", market="CN", universe=["FAKE"],
        conditions=[
            Condition(metric="pct_change",   operator=">",  value=2.0),
            Condition(metric="turnover_rate", operator=">",  value=8.0),
        ],
        logic="OR",
    )
    assert not _has_lookahead(lambda d: translate(spec, d), df)


def test_rolling_mean_crossover(df):
    """`close crosses_above ma5` — rolling window uses only past bars."""
    def signal_fn(d):
        d = d.copy()
        d["ma5"] = d["close"].rolling(5).mean()
        spec = SignalSpec(
            mode="screen", market="CN", universe=["FAKE"],
            conditions=[Condition(metric="close", operator="crosses_above", value="ma5")],
        )
        return translate(spec, d)

    assert not _has_lookahead(signal_fn, df)


def test_rolling_mean_comparison(df):
    """`close > ma20` — 20-bar lookback, no future data."""
    def signal_fn(d):
        d = d.copy()
        d["ma20"] = d["close"].rolling(20).mean()
        spec = SignalSpec(
            mode="screen", market="CN", universe=["FAKE"],
            conditions=[Condition(metric="close", operator=">", value="ma20")],
        )
        return translate(spec, d)

    assert not _has_lookahead(signal_fn, df)


def test_volume_vs_avg(df):
    """`volume > 5-bar rolling mean of volume` — common volume surge pattern."""
    def signal_fn(d):
        d = d.copy()
        d["vol_avg5"] = d["volume"].rolling(5).mean()
        spec = SignalSpec(
            mode="screen", market="CN", universe=["FAKE"],
            conditions=[Condition(metric="volume", operator=">", value="vol_avg5")],
        )
        return translate(spec, d)

    assert not _has_lookahead(signal_fn, df)


# ---------------------------------------------------------------------------
# Indicator lookahead tests (Phase 2 Step 8)
# ---------------------------------------------------------------------------

@pytest.fixture
def df_with_lu(df):
    """df fixture with a few forced limit-up bars for LU-based indicators."""
    d = df.copy()
    d.loc[d.index[10], "pct_change"] = 10.0   # bar 10: limit-up
    d.loc[d.index[11], "pct_change"] = 10.0   # bar 11: second consecutive LU
    d.loc[d.index[30], "pct_change"] = 10.0   # bar 30: isolated LU
    return d


def test_consecutive_limit_ups_no_lookahead(df_with_lu):
    assert not _has_lookahead(
        lambda d: consecutive_limit_ups(d, "600519"), df_with_lu
    )


def test_turnover_rate_nd_no_lookahead(df):
    assert not _has_lookahead(lambda d: turnover_rate_nd(d, n=5), df)


def test_volume_vs_avg_no_lookahead(df):
    assert not _has_lookahead(lambda d: volume_vs_avg(d, n=20), df)


def test_range_compression_no_lookahead(df):
    assert not _has_lookahead(lambda d: range_compression(d, n=20), df)


def test_gap_open_pct_no_lookahead(df):
    assert not _has_lookahead(gap_open_pct, df)


def test_days_since_limit_up_no_lookahead(df_with_lu):
    assert not _has_lookahead(
        lambda d: days_since_limit_up(d, "600519"), df_with_lu
    )
