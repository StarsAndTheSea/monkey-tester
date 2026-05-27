"""
Tests for factors.py — factor library, simulation engine, and lookahead check.
No network required (uses synthetic data).
"""

import numpy as np
import pandas as pd
import pytest

from core import Condition
from factors import (
    Factor,
    FactorGroup,
    FactorLibrary,
    add_factor,
    add_group,
    check_lookahead,
    load_library,
    run_factor_backtest,
    save_library,
    _factor_mask,
    _group_entry_mask,
    _group_exit_mask,
    _simulate_group,
    _compute_stats,
    _equity_curve,
)
from indicators import add_indicators


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df():
    """80 bars of synthetic daily data with all indicators computed."""
    n = 80
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.015, n))
    base = pd.DataFrame({
        "date":          dates.strftime("%Y-%m-%d"),
        "open":          close * 0.99,
        "high":          close * 1.01,
        "low":           close * 0.98,
        "close":         close,
        "volume":        rng.integers(100_000, 500_000, n).astype(float),
        "pct_change":    rng.normal(0.1, 1.5, n),
        "turnover_rate": rng.uniform(1, 10, n),
    })
    base.loc[base.index[20], "pct_change"] = 10.0
    base.loc[base.index[21], "pct_change"] = 10.0
    return add_indicators(base, "600519")


@pytest.fixture
def simple_factor():
    return Factor(
        id="f_vol",
        name="High volume",
        conditions=[Condition(metric="volume_vs_avg20", operator=">=", value=0.5)],
    )


@pytest.fixture
def entry_factor():
    return Factor(
        id="f_entry",
        name="Entry signal",
        conditions=[Condition(metric="pct_change", operator=">", value=0.0)],
    )


@pytest.fixture
def exit_factor():
    return Factor(
        id="f_exit",
        name="Exit signal",
        conditions=[Condition(metric="pct_change", operator="<", value=0.0)],
    )


@pytest.fixture
def simple_group(entry_factor, exit_factor):
    return FactorGroup(
        id="g_simple",
        name="Simple group",
        entry_factor_ids=["f_entry"],
        exit_factor_ids=["f_exit"],
        capital=50_000.0,
        holding_period=10,
    )


@pytest.fixture
def library(entry_factor, exit_factor):
    lib = FactorLibrary()
    lib = add_factor(lib, entry_factor)
    lib = add_factor(lib, exit_factor)
    return lib


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_factor_requires_conditions():
    with pytest.raises(Exception):
        Factor(id="bad", name="bad", conditions=[])


def test_group_requires_entry_factors():
    with pytest.raises(Exception):
        FactorGroup(id="bad", name="bad", entry_factor_ids=[], capital=10_000)


def test_group_requires_positive_capital():
    with pytest.raises(Exception):
        FactorGroup(id="bad", name="bad", entry_factor_ids=["f"], capital=-1000)


# ---------------------------------------------------------------------------
# Library persistence
# ---------------------------------------------------------------------------

def test_add_factor_creates_new_library(simple_factor):
    lib = FactorLibrary()
    lib2 = add_factor(lib, simple_factor)
    assert len(lib.factors) == 0    # original unchanged
    assert len(lib2.factors) == 1


def test_add_factor_duplicate_raises(simple_factor):
    lib = add_factor(FactorLibrary(), simple_factor)
    with pytest.raises(ValueError, match="already exists"):
        add_factor(lib, simple_factor, overwrite=False)


def test_add_factor_overwrite_replaces(simple_factor):
    lib = add_factor(FactorLibrary(), simple_factor)
    updated = Factor(id="f_vol", name="Updated", conditions=simple_factor.conditions)
    lib2 = add_factor(lib, updated, overwrite=True)
    assert lib2.factor_by_id("f_vol").name == "Updated"


def test_add_group_validates_references(simple_group):
    lib = FactorLibrary()   # no factors yet
    with pytest.raises(ValueError, match="unknown factor"):
        add_group(lib, simple_group)


def test_save_and_load_roundtrip(tmp_path, monkeypatch, entry_factor, exit_factor, simple_group):
    monkeypatch.setattr("factors._LIBRARY_PATH", tmp_path / "lib.json")
    lib = FactorLibrary()
    lib = add_factor(lib, entry_factor)
    lib = add_factor(lib, exit_factor)
    lib = add_group(lib, simple_group)
    save_library(lib)
    lib2 = load_library()
    assert len(lib2.factors) == 2
    assert len(lib2.groups) == 1
    assert lib2.group_by_id("g_simple").capital == 50_000.0


def test_load_missing_library_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("factors._LIBRARY_PATH", tmp_path / "nonexistent.json")
    lib = load_library()
    assert lib.factors == []
    assert lib.groups == []


# ---------------------------------------------------------------------------
# Factor evaluation
# ---------------------------------------------------------------------------

def test_factor_mask_returns_boolean_series(df, simple_factor):
    mask = _factor_mask(simple_factor, df)
    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool or set(mask.unique()).issubset({True, False})
    assert len(mask) == len(df)


def test_group_entry_mask_is_and_of_factors(df, entry_factor, simple_factor):
    factors = {"f_entry": entry_factor, "f_vol": simple_factor}
    group = FactorGroup(
        id="g_and", name="AND group",
        entry_factor_ids=["f_entry", "f_vol"],
        capital=10_000,
    )
    mask = _group_entry_mask(group, factors, df)
    f1 = _factor_mask(entry_factor, df)
    f2 = _factor_mask(simple_factor, df)
    expected = f1 & f2
    pd.testing.assert_series_equal(mask.reset_index(drop=True), expected.reset_index(drop=True))


def test_group_exit_mask_is_or_of_factors(df, entry_factor, exit_factor):
    factors = {"f_entry": entry_factor, "f_exit": exit_factor}
    group = FactorGroup(
        id="g_or", name="OR exit",
        entry_factor_ids=["f_entry"],
        exit_factor_ids=["f_entry", "f_exit"],
        capital=10_000,
    )
    mask = _group_exit_mask(group, factors, df)
    f1 = _factor_mask(entry_factor, df)
    f2 = _factor_mask(exit_factor, df)
    expected = f1 | f2
    pd.testing.assert_series_equal(mask.reset_index(drop=True), expected.reset_index(drop=True))


def test_group_exit_mask_none_when_no_exit_factors(df, entry_factor):
    factors = {"f_entry": entry_factor}
    group = FactorGroup(id="g", name="g", entry_factor_ids=["f_entry"], capital=10_000)
    assert _group_exit_mask(group, factors, df) is None


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------

def test_simulate_group_returns_list(df, entry_factor, exit_factor):
    em = _factor_mask(entry_factor, df)
    xm = _factor_mask(exit_factor, df)
    trades = _simulate_group(df, em, xm, capital=50_000, holding_period=10)
    assert isinstance(trades, list)


def test_simulate_group_trade_fields(df, entry_factor, exit_factor):
    em = _factor_mask(entry_factor, df)
    xm = _factor_mask(exit_factor, df)
    trades = _simulate_group(df, em, xm, capital=50_000, holding_period=10)
    if trades:
        t = trades[0]
        assert "entry_date" in t and "exit_date" in t
        assert "pnl" in t and "return_pct" in t
        assert "capital" in t and "shares" in t
        assert t["capital"] == 50_000


def test_simulate_holding_period_respected(df, entry_factor):
    # Force entry on every bar, no exit conditions
    em = pd.Series(True, index=df.index)
    trades = _simulate_group(df, em, None, capital=10_000, holding_period=5)
    for t in trades[:-1]:   # last trade may be force-closed
        entry = df.index[df["date"] == t["entry_date"]][0]
        exit_ = df.index[df["date"] == t["exit_date"]][0]
        assert (exit_ - entry) <= 5


def test_simulate_no_double_entry(df, entry_factor):
    # If entry fires every bar, we should only enter after the previous trade closes
    em = pd.Series(True, index=df.index)
    trades = _simulate_group(df, em, None, capital=10_000, holding_period=3)
    # Trades should not overlap
    for i in range(1, len(trades)):
        assert trades[i]["entry_date"] >= trades[i - 1]["exit_date"]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_compute_stats_empty_trades():
    stats = _compute_stats([], pd.DataFrame({"pnl_today": [], "equity": []}))
    assert stats["num_trades"] == 0
    assert stats["total_pnl"] == 0.0


def test_compute_stats_with_trades(df, entry_factor, exit_factor):
    em = _factor_mask(entry_factor, df)
    xm = _factor_mask(exit_factor, df)
    trades = _simulate_group(df, em, xm, capital=50_000, holding_period=10)
    eq_df = _equity_curve(trades, df["date"])
    stats = _compute_stats(trades, eq_df)
    assert stats["num_trades"] == len(trades)
    assert 0 <= stats["win_rate_pct"] <= 100
    assert isinstance(stats["total_pnl"], float)
    assert isinstance(stats["sharpe_ratio"], float)


# ---------------------------------------------------------------------------
# run_factor_backtest integration
# ---------------------------------------------------------------------------

class _FakeProvider:
    """Returns the synthetic df regardless of symbol/dates."""
    def __init__(self, df):
        self._df = df.copy()

    def get_daily_history(self, symbol, start=None, end=None, adjust="hfq", cache_only=False):
        return self._df.copy()


def test_run_factor_backtest_returns_all_groups(df, entry_factor, exit_factor):
    lib = FactorLibrary()
    lib = add_factor(lib, entry_factor)
    lib = add_factor(lib, exit_factor)

    g1 = FactorGroup(id="g1", name="G1", entry_factor_ids=["f_entry"],
                     exit_factor_ids=["f_exit"], capital=50_000)
    g2 = FactorGroup(id="g2", name="G2", entry_factor_ids=["f_entry"],
                     capital=30_000, holding_period=5)
    lib = add_group(lib, g1)
    lib = add_group(lib, g2)

    result = run_factor_backtest([g1, g2], "600519", _FakeProvider(df), library=lib)

    assert "groups" in result
    assert "g1" in result["groups"]
    assert "g2" in result["groups"]
    assert "combined" in result


def test_run_factor_backtest_combined_equity_shape(df, entry_factor, exit_factor):
    lib = FactorLibrary()
    lib = add_factor(lib, entry_factor)
    lib = add_factor(lib, exit_factor)
    g = FactorGroup(id="g1", name="G1", entry_factor_ids=["f_entry"],
                    exit_factor_ids=["f_exit"], capital=50_000)
    lib = add_group(lib, g)

    result = run_factor_backtest([g], "600519", _FakeProvider(df), library=lib)
    eq = result["combined"]["equity_curve"]
    assert isinstance(eq, pd.DataFrame)
    assert "equity" in eq.columns
    assert len(eq) == len(df)


def test_run_factor_backtest_unknown_factor_returns_error(df, entry_factor):
    lib = FactorLibrary()
    lib = add_factor(lib, entry_factor)
    g = FactorGroup(id="g1", name="G1", entry_factor_ids=["nonexistent"], capital=10_000)
    result = run_factor_backtest([g], "600519", _FakeProvider(df), library=lib)
    assert "error" in result


def test_run_factor_backtest_two_groups_independent(df, entry_factor, exit_factor):
    """Two groups can have trades at overlapping times — positions are independent."""
    lib = FactorLibrary()
    lib = add_factor(lib, entry_factor)
    lib = add_factor(lib, exit_factor)

    g1 = FactorGroup(id="g1", name="G1", entry_factor_ids=["f_entry"],
                     exit_factor_ids=["f_exit"], capital=50_000)
    g2 = FactorGroup(id="g2", name="G2", entry_factor_ids=["f_entry"],
                     exit_factor_ids=["f_exit"], capital=30_000)
    lib = add_group(lib, g1)
    lib = add_group(lib, g2)

    result = run_factor_backtest([g1, g2], "600519", _FakeProvider(df), library=lib)
    t1 = result["groups"]["g1"]["trades"]
    t2 = result["groups"]["g2"]["trades"]
    # Both groups can have trades on the same date
    assert isinstance(t1, list) and isinstance(t2, list)
    # Capital is separate per group
    if t1:
        assert t1[0]["capital"] == 50_000
    if t2:
        assert t2[0]["capital"] == 30_000


# ---------------------------------------------------------------------------
# Lookahead bias check
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_df():
    """Raw OHLCV data WITHOUT add_indicators applied — input for check_lookahead."""
    n = 80
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.015, n))
    return pd.DataFrame({
        "date":          dates.strftime("%Y-%m-%d"),
        "open":          close * 0.99,
        "high":          close * 1.01,
        "low":           close * 0.98,
        "close":         close,
        "volume":        rng.integers(100_000, 500_000, n).astype(float),
        "pct_change":    rng.normal(0.1, 1.5, n),
        "turnover_rate": rng.uniform(1, 10, n),
    })


def test_check_lookahead_clean_factor_passes(raw_df, simple_factor):
    """Standard metrics computed by add_indicators should have no lookahead."""
    assert check_lookahead(simple_factor, raw_df) is True


def test_check_lookahead_leaky_factor_fails(raw_df, monkeypatch):
    """An indicator computed with shift(-1) should be caught as leaky."""
    import factors as _factors_mod
    from indicators import add_indicators as _real_add_indicators

    def _leaky_add_indicators(df, symbol):
        result = _real_add_indicators(df, symbol)
        # Peeks one bar ahead — leaky by definition
        result["leaky_metric"] = result["pct_change"].shift(-1).fillna(0)
        return result

    monkeypatch.setattr(_factors_mod, "add_indicators", _leaky_add_indicators)

    leaky = Factor(
        id="f_leaky",
        name="Leaky",
        conditions=[Condition(metric="leaky_metric", operator=">", value=0)],
    )
    assert check_lookahead(leaky, raw_df) is False
