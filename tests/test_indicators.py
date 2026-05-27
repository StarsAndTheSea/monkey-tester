"""
Correctness tests for indicators.py (Phase 2 Step 9).

Each test constructs a DataFrame with known values and asserts exact
or near-exact output. This is the hand-verification step — a silent
calculation bug here corrupts every downstream backtest and screen.
"""

import numpy as np
import pandas as pd
import pytest

from indicators import (
    add_indicators,
    consecutive_limit_ups,
    days_since_limit_up,
    gap_open_pct,
    range_compression,
    turnover_rate_nd,
    volume_vs_avg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(pct_changes, **extra_cols):
    """Build a minimal DataFrame from a pct_change list plus optional columns."""
    n = len(pct_changes)
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    df = pd.DataFrame({
        "date":         dates.strftime("%Y-%m-%d"),
        "open":         extra_cols.get("open",  np.ones(n) * 100.0),
        "high":         extra_cols.get("high",  np.ones(n) * 101.0),
        "low":          extra_cols.get("low",   np.ones(n) * 99.0),
        "close":        extra_cols.get("close", np.ones(n) * 100.0),
        "volume":       extra_cols.get("volume",np.ones(n) * 1_000.0),
        "pct_change":   np.array(pct_changes, dtype=float),
        "turnover_rate":extra_cols.get("turnover_rate", np.ones(n) * 5.0),
    })
    return df


# ---------------------------------------------------------------------------
# consecutive_limit_ups
# ---------------------------------------------------------------------------

def test_clu_no_limit_ups():
    df = _make_df([0.5, 1.0, -0.3, 2.0, 9.8])   # none hit 9.9
    result = consecutive_limit_ups(df, "600519")
    assert list(result) == [0, 0, 0, 0, 0]


def test_clu_single_limit_up():
    df = _make_df([0.0, 0.0, 10.0, 0.0, 0.0])
    result = consecutive_limit_ups(df, "600519")
    assert list(result) == [0, 0, 1, 0, 0]


def test_clu_consecutive_run():
    df = _make_df([0.0, 10.0, 10.0, 10.0, 0.0, 10.0])
    result = consecutive_limit_ups(df, "600519")
    assert list(result) == [0, 1, 2, 3, 0, 1]


def test_clu_star_market_threshold():
    # STAR symbol 688001 needs >= 19.9 for limit-up, not 9.9
    df = _make_df([10.0, 10.0, 20.0, 20.0])
    result = consecutive_limit_ups(df, "688001")
    # 10.0 is below STAR's 19.9 threshold
    assert list(result) == [0, 0, 1, 2]


def test_clu_st_threshold():
    # ST stocks: threshold is 4.9; 5.0 should count
    df = _make_df([5.0, 5.0, 0.0, 5.0])
    result = consecutive_limit_ups(df, "600001", is_st=True)
    assert list(result) == [1, 2, 0, 1]


# ---------------------------------------------------------------------------
# turnover_rate_nd
# ---------------------------------------------------------------------------

def test_turnover_rate_nd_basic():
    tr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    df = _make_df([0.0] * 6, turnover_rate=np.array(tr))
    result = turnover_rate_nd(df, n=3)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)   # (1+2+3)/3
    assert result.iloc[3] == pytest.approx(3.0)   # (2+3+4)/3
    assert result.iloc[5] == pytest.approx(5.0)   # (4+5+6)/3


# ---------------------------------------------------------------------------
# volume_vs_avg
# ---------------------------------------------------------------------------

def test_volume_vs_avg_constant():
    # Constant volume → ratio = 1.0 everywhere after warm-up
    # warm-up is n+1 bars because the average is shifted by 1
    df = _make_df([0.0] * 25, volume=np.ones(25) * 500.0)
    result = volume_vs_avg(df, n=5)
    assert pd.isna(result.iloc[0])          # warm-up
    assert pd.isna(result.iloc[4])          # still warm-up (n+1 bars needed)
    assert result.iloc[5]  == pytest.approx(1.0)
    assert result.iloc[24] == pytest.approx(1.0)


def test_volume_vs_avg_spike():
    # 20 bars of vol=100, then one bar of vol=200
    vols = np.ones(21) * 100.0
    vols[-1] = 200.0
    df = _make_df([0.0] * 21, volume=vols)
    result = volume_vs_avg(df, n=20)
    assert result.iloc[-1] == pytest.approx(200.0 / 100.0)


# ---------------------------------------------------------------------------
# range_compression
# ---------------------------------------------------------------------------

def test_range_compression_constant():
    # Constant H-L-C → compression ratio = 1.0 after warm-up
    # warm-up is n+1 bars because the average is shifted by 1
    df = _make_df(
        [0.0] * 25,
        high=np.ones(25) * 102.0,
        low=np.ones(25)  * 98.0,
        close=np.ones(25) * 100.0,
    )
    result = range_compression(df, n=5)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[4])          # still warm-up (n+1 bars needed)
    assert result.iloc[5]  == pytest.approx(1.0)
    assert result.iloc[24] == pytest.approx(1.0)


def test_range_compression_tight_bar():
    # 20 bars with range 4/100 = 0.04, then one bar with range 2/100 = 0.02
    highs  = np.ones(21) * 102.0
    lows   = np.ones(21) * 98.0
    closes = np.ones(21) * 100.0
    highs[-1]  = 101.0
    lows[-1]   = 99.0
    df = _make_df([0.0] * 21, high=highs, low=lows, close=closes)
    result = range_compression(df, n=20)
    assert result.iloc[-1] == pytest.approx(0.5)   # half the average range


# ---------------------------------------------------------------------------
# gap_open_pct
# ---------------------------------------------------------------------------

def test_gap_open_pct_no_gap():
    # open == prev close every bar → gap = 0
    closes = np.array([100.0, 102.0, 101.0, 103.0])
    opens  = np.array([100.0, 100.0, 102.0, 101.0])   # open_t == close_{t-1}
    df = _make_df([0.0] * 4, open=opens, close=closes)
    result = gap_open_pct(df)
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(0.0)
    assert result.iloc[2] == pytest.approx(0.0)
    assert result.iloc[3] == pytest.approx(0.0)


def test_gap_open_pct_gap_up():
    # Bar 1 opens 10% above bar 0's close
    closes = np.array([100.0, 110.0])
    opens  = np.array([100.0, 110.0])   # open_1 = 110, close_0 = 100 → +10%
    df = _make_df([0.0] * 2, open=opens, close=closes)
    result = gap_open_pct(df)
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(10.0)


def test_gap_open_pct_gap_down():
    closes = np.array([100.0, 90.0])
    opens  = np.array([100.0, 90.0])   # open_1 = 90, close_0 = 100 → -10%
    df = _make_df([0.0] * 2, open=opens, close=closes)
    result = gap_open_pct(df)
    assert result.iloc[1] == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# days_since_limit_up
# ---------------------------------------------------------------------------

def test_dslu_no_limit_ups():
    df = _make_df([0.0, 1.0, 2.0, 9.8])
    result = days_since_limit_up(df, "600519")
    assert all(pd.isna(result))


def test_dslu_basic_sequence():
    # bar 2 = LU; days after: 0, 1, 2, 3
    df = _make_df([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    result = days_since_limit_up(df, "600519")
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == pytest.approx(0.0)
    assert result.iloc[3] == pytest.approx(1.0)
    assert result.iloc[5] == pytest.approx(3.0)


def test_dslu_resets_on_second_lu():
    # bar 1 = LU, bar 3 = LU → resets counter
    df = _make_df([0.0, 10.0, 0.0, 10.0, 0.0])
    result = days_since_limit_up(df, "600519")
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pytest.approx(0.0)
    assert result.iloc[2] == pytest.approx(1.0)
    assert result.iloc[3] == pytest.approx(0.0)
    assert result.iloc[4] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# add_indicators integration
# ---------------------------------------------------------------------------

def test_add_indicators_columns_present():
    df = _make_df([0.0] * 30)
    out = add_indicators(df, "600519")
    expected_cols = {
        "consecutive_limit_ups", "turnover_rate_5d", "turnover_rate_10d",
        "volume_vs_avg20", "range_compression20", "gap_open_pct",
        "days_since_limit_up",
    }
    assert expected_cols.issubset(set(out.columns))


def test_add_indicators_does_not_mutate_input():
    df = _make_df([0.0] * 30)
    original_cols = list(df.columns)
    add_indicators(df, "600519")
    assert list(df.columns) == original_cols
