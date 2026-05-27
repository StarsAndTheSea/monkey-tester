"""
Tests for the Phase 2a framework template library.

Verifies:
- All five templates load and parse into validated SignalSpec instances
- Every template carries status="unvalidated"
- Every template's conditions run through the translator without error
- The translator returns a boolean Series of the expected length
"""

import numpy as np
import pandas as pd
import pytest

from core import translate
from indicators import add_indicators
from templates import get_template, get_templates, UNVALIDATED_NOTE

EXPECTED_IDS = [
    "t1_momentum_entry",
    "t1_exit_test",
    "t2_macro_thesis",
    "t3_stealth_accumulation",
    "t3_exit_test",
]


# ---------------------------------------------------------------------------
# Synthetic DataFrame with all indicator columns
# ---------------------------------------------------------------------------

@pytest.fixture
def df():
    """60 bars of synthetic data with all indicator columns pre-computed."""
    n = 60
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    rng = np.random.default_rng(7)
    close = 100 * np.cumprod(1 + rng.normal(0.001, 0.015, n))
    base = pd.DataFrame({
        "date":         dates.strftime("%Y-%m-%d"),
        "open":         close * 0.99,
        "high":         close * 1.01,
        "low":          close * 0.98,
        "close":        close,
        "volume":       rng.integers(100_000, 500_000, n).astype(float),
        "pct_change":   rng.normal(0.1, 1.5, n),
        "turnover_rate":rng.uniform(1, 10, n),
    })
    # Force a few limit-up bars so LU-based indicators have something to compute
    base.loc[base.index[20], "pct_change"] = 10.0
    base.loc[base.index[21], "pct_change"] = 10.0
    return add_indicators(base, "600519")


# ---------------------------------------------------------------------------
# Loading and structure
# ---------------------------------------------------------------------------

def test_all_five_templates_load():
    templates = get_templates()
    assert len(templates) == 5


def test_template_ids_match_expected():
    ids = [t["id"] for t in get_templates()]
    assert ids == EXPECTED_IDS


def test_every_template_has_required_keys():
    for t in get_templates():
        assert {"id", "name", "description", "rationale", "status", "spec"}.issubset(t.keys()), \
            f"{t['id']} is missing required keys"


def test_every_template_is_unvalidated():
    for t in get_templates():
        assert t["status"] == "unvalidated", \
            f"{t['id']} has status={t['status']!r} — must be 'unvalidated'"


def test_get_template_by_id():
    for expected_id in EXPECTED_IDS:
        entry = get_template(expected_id)
        assert entry is not None, f"get_template({expected_id!r}) returned None"
        assert entry["id"] == expected_id


def test_get_template_unknown_id_returns_none():
    assert get_template("does_not_exist") is None


# ---------------------------------------------------------------------------
# SignalSpec validity
# ---------------------------------------------------------------------------

def test_all_specs_are_valid_signal_specs():
    from core import SignalSpec
    for t in get_templates():
        assert isinstance(t["spec"], SignalSpec), \
            f"{t['id']} spec is not a SignalSpec instance"


def test_screen_templates_use_screen_mode():
    screen_ids = {"t3_stealth_accumulation"}
    for t in get_templates():
        if t["id"] in screen_ids:
            assert t["spec"].mode == "screen", f"{t['id']} should be mode='screen'"


def test_backtest_templates_use_backtest_mode():
    backtest_ids = {"t1_momentum_entry", "t1_exit_test", "t2_macro_thesis", "t3_exit_test"}
    for t in get_templates():
        if t["id"] in backtest_ids:
            assert t["spec"].mode == "backtest", f"{t['id']} should be mode='backtest'"


def test_backtest_templates_have_entry_conditions():
    backtest_ids = {"t1_momentum_entry", "t1_exit_test", "t2_macro_thesis", "t3_exit_test"}
    for t in get_templates():
        if t["id"] in backtest_ids:
            assert t["spec"].entry, f"{t['id']} has no entry conditions"


def test_backtest_templates_have_exit_or_holding_period():
    backtest_ids = {"t1_momentum_entry", "t1_exit_test", "t2_macro_thesis", "t3_exit_test"}
    for t in get_templates():
        if t["id"] in backtest_ids:
            has_exit = bool(t["spec"].exit) or (t["spec"].holding_period is not None)
            assert has_exit, f"{t['id']} has neither exit conditions nor holding_period"


# ---------------------------------------------------------------------------
# Translator integration — all specs must run without error
# ---------------------------------------------------------------------------

def test_all_templates_translate_without_error(df):
    for t in get_templates():
        spec = t["spec"]
        result = translate(spec, df)
        assert isinstance(result, pd.Series), \
            f"{t['id']} translate() did not return a Series"
        assert len(result) == len(df), \
            f"{t['id']} result length {len(result)} != df length {len(df)}"
        assert result.dtype == bool or result.dtype == object, \
            f"{t['id']} result dtype unexpected: {result.dtype}"


def test_exit_conditions_translate_without_error(df):
    """Exit conditions must also translate cleanly (run_backtest evaluates them separately)."""
    from core import Condition, SignalSpec
    for t in get_templates():
        spec = t["spec"]
        if not spec.exit:
            continue
        exit_spec = SignalSpec(
            mode="backtest", market=spec.market, universe=spec.universe,
            conditions=spec.exit, logic=spec.logic,
        )
        result = translate(exit_spec, df)
        assert isinstance(result, pd.Series), \
            f"{t['id']} exit translate() did not return a Series"


# ---------------------------------------------------------------------------
# UNVALIDATED_NOTE constant
# ---------------------------------------------------------------------------

def test_unvalidated_note_is_non_empty():
    assert isinstance(UNVALIDATED_NOTE, str) and len(UNVALIDATED_NOTE) > 0
