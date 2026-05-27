"""
Factor library and multi-group backtesting engine.

Concepts
--------
Factor
    A named, reusable set of Conditions evaluated with AND logic.
    e.g. "momentum" = [pct_change >= 5, volume_vs_avg20 >= 2]

FactorGroup
    A named collection of Factor IDs with a CNY capital allocation.
    Entry fires when ALL entry factors are active on the same bar.
    Exit fires when ANY exit factor is active (OR logic), or when the
    holding period expires — whichever comes first.
    Two groups run independently: both can be in position at the same time.

FactorLibrary
    The full catalogue of factors and groups, persisted to factor_library.json.

Usage
-----
    from factors import load_library, save_library, run_factor_backtest
    from core import AKShareCNProvider

    lib = load_library()
    results = run_factor_backtest(lib.groups, "600519", AKShareCNProvider())
    for gid, r in results["groups"].items():
        print(gid, r["stats"])
"""

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator

from core import Condition, SignalSpec, translate
from indicators import add_indicators

_LIBRARY_PATH = Path(__file__).parent / "factor_library.json"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class Factor(BaseModel):
    """A named, reusable set of conditions (all must be true — AND logic)."""
    id:          str
    name:        str
    description: str = ""
    conditions:  list[Condition]

    @field_validator("conditions")
    @classmethod
    def at_least_one_condition(cls, v: list) -> list:
        if not v:
            raise ValueError("A factor must have at least one condition.")
        return v


class FactorGroup(BaseModel):
    """A tradeable strategy: factor IDs + capital allocation."""
    id:               str
    name:             str
    description:      str = ""
    entry_factor_ids: list[str]            # ALL must fire to enter
    exit_factor_ids:  list[str] = []       # ANY fires to exit
    capital:          float                # CNY per trade
    holding_period:   Union[int, None] = None   # max bars; None = no hard limit
    market:           str = "CN"

    @field_validator("capital")
    @classmethod
    def capital_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("capital must be > 0")
        return v

    @field_validator("entry_factor_ids")
    @classmethod
    def at_least_one_entry(cls, v: list) -> list:
        if not v:
            raise ValueError("entry_factor_ids must contain at least one factor.")
        return v


class FactorLibrary(BaseModel):
    """Persisted catalogue of all factors and groups."""
    factors: list[Factor]       = []
    groups:  list[FactorGroup]  = []

    def factor_by_id(self, fid: str) -> "Factor | None":
        return next((f for f in self.factors if f.id == fid), None)

    def group_by_id(self, gid: str) -> "FactorGroup | None":
        return next((g for g in self.groups if g.id == gid), None)


# ---------------------------------------------------------------------------
# Library persistence
# ---------------------------------------------------------------------------

def load_library() -> FactorLibrary:
    """Load the factor library from disk. Returns an empty library if not found."""
    if not _LIBRARY_PATH.exists():
        return FactorLibrary()
    try:
        raw = json.loads(_LIBRARY_PATH.read_text(encoding="utf-8"))
        return FactorLibrary.model_validate(raw)
    except Exception as e:
        print(f"[factors] could not load library: {e} — returning empty library")
        return FactorLibrary()


def save_library(lib: FactorLibrary) -> None:
    """Persist the factor library to disk."""
    _LIBRARY_PATH.write_text(
        lib.model_dump_json(indent=2),
        encoding="utf-8",
    )


def add_factor(lib: FactorLibrary, factor: Factor, overwrite: bool = False) -> FactorLibrary:
    """Return a new library with the factor added (or replaced if overwrite=True)."""
    existing = [f for f in lib.factors if f.id != factor.id]
    if not overwrite and any(f.id == factor.id for f in lib.factors):
        raise ValueError(f"Factor id {factor.id!r} already exists. Pass overwrite=True to replace.")
    return FactorLibrary(factors=existing + [factor], groups=lib.groups)


def add_group(lib: FactorLibrary, group: FactorGroup, overwrite: bool = False) -> FactorLibrary:
    """Return a new library with the group added (or replaced if overwrite=True)."""
    existing = [g for g in lib.groups if g.id != group.id]
    if not overwrite and any(g.id == group.id for g in lib.groups):
        raise ValueError(f"Group id {group.id!r} already exists. Pass overwrite=True to replace.")
    _validate_group_references(group, lib)
    return FactorLibrary(factors=lib.factors, groups=existing + [group])


def _validate_group_references(group: FactorGroup, lib: FactorLibrary) -> None:
    """Raise if a group references a factor ID that doesn't exist in the library."""
    known = {f.id for f in lib.factors}
    for fid in group.entry_factor_ids + group.exit_factor_ids:
        if fid not in known:
            raise ValueError(
                f"Group {group.id!r} references unknown factor {fid!r}. "
                "Add the factor to the library first."
            )


# ---------------------------------------------------------------------------
# Factor evaluation
# ---------------------------------------------------------------------------

def _factor_mask(factor: Factor, df: pd.DataFrame, market: str = "CN") -> pd.Series:
    """Return a boolean Series — True where all factor conditions are met."""
    spec = SignalSpec(
        mode="screen",
        market=market,
        universe=[],
        conditions=factor.conditions,
        logic="AND",
    )
    return translate(spec, df)


def _group_entry_mask(group: FactorGroup, factors: dict[str, Factor], df: pd.DataFrame) -> pd.Series:
    """Return True on bars where ALL entry factors fire simultaneously."""
    masks = [_factor_mask(factors[fid], df, group.market) for fid in group.entry_factor_ids]
    result = masks[0]
    for m in masks[1:]:
        result = result & m
    return result.fillna(False)


def _group_exit_mask(group: FactorGroup, factors: dict[str, Factor], df: pd.DataFrame) -> "pd.Series | None":
    """Return True on bars where ANY exit factor fires. Returns None if no exit factors defined."""
    if not group.exit_factor_ids:
        return None
    masks = [_factor_mask(factors[fid], df, group.market) for fid in group.exit_factor_ids]
    result = masks[0]
    for m in masks[1:]:
        result = result | m
    return result.fillna(False)


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------

def _simulate_group(
    df:               pd.DataFrame,
    entry_mask:       pd.Series,
    exit_mask:        "pd.Series | None",
    capital:          float,
    holding_period:   "int | None",
) -> list[dict]:
    """Simulate trades for one group on a prepared DataFrame.

    Entry/exit at the CLOSE of the signal bar.
    One position at a time per group — new entry signals while in position
    are skipped until the position is closed.

    Returns a list of trade dicts:
        entry_date, exit_date, entry_price, exit_price,
        capital, shares, pnl, return_pct
    """
    entry_arr = entry_mask.values
    exit_arr  = exit_mask.values if exit_mask is not None else None
    closes    = df["close"].values
    dates     = df["date"].values
    n         = len(df)

    trades       = []
    in_position  = False
    entry_bar    = None
    entry_price  = None

    for i in range(n):
        if not in_position:
            if entry_arr[i]:
                in_position  = True
                entry_bar    = i
                entry_price  = float(closes[i])
        else:
            should_exit = False
            if exit_arr is not None and exit_arr[i]:
                should_exit = True
            if holding_period is not None and (i - entry_bar) >= holding_period:
                should_exit = True
            if i == n - 1:  # force-close on last bar
                should_exit = True

            if should_exit:
                exit_price  = float(closes[i])
                shares      = capital / entry_price
                pnl         = (exit_price - entry_price) * shares
                return_pct  = (exit_price - entry_price) / entry_price * 100
                trades.append({
                    "entry_date":   str(dates[entry_bar]),
                    "exit_date":    str(dates[i]),
                    "entry_price":  entry_price,
                    "exit_price":   exit_price,
                    "capital":      capital,
                    "shares":       round(shares, 4),
                    "pnl":          round(pnl, 2),
                    "return_pct":   round(return_pct, 4),
                })
                in_position = False
                entry_bar   = None
                entry_price = None

    return trades


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _equity_curve(trades: list[dict], dates: pd.Series) -> pd.DataFrame:
    """Build a daily equity (cumulative PnL) DataFrame from a trade list.

    Rows = every trading day in dates.
    Columns: pnl_today (PnL realised on this day), equity (cumulative).
    """
    pnl_by_date: dict[str, float] = {}
    for t in trades:
        d = t["exit_date"]
        pnl_by_date[d] = pnl_by_date.get(d, 0.0) + t["pnl"]

    daily_pnl = pd.Series(0.0, index=dates)
    for d, p in pnl_by_date.items():
        if d in daily_pnl.index:
            daily_pnl[d] = p

    eq = daily_pnl.cumsum()
    return pd.DataFrame({"pnl_today": daily_pnl, "equity": eq}, index=dates)


def _compute_stats(trades: list[dict], equity_df: pd.DataFrame) -> dict:
    """Summarise a list of trades and the resulting equity curve."""
    if not trades:
        return {
            "num_trades":     0,
            "total_pnl":      0.0,
            "total_return_pct": 0.0,
            "win_rate_pct":   0.0,
            "avg_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio":   0.0,
        }

    total_pnl      = sum(t["pnl"] for t in trades)
    total_capital  = sum(t["capital"] for t in trades)
    returns        = [t["return_pct"] for t in trades]
    wins           = [r for r in returns if r > 0]
    win_rate       = len(wins) / len(trades) * 100

    # Max drawdown from equity curve
    eq             = equity_df["equity"]
    roll_max       = eq.cummax()
    drawdown       = eq - roll_max
    max_dd         = float(drawdown.min())
    # Express as % of peak capital deployed (avoid div-by-zero)
    peak_capital   = total_capital / max(len(trades), 1)
    max_dd_pct     = (max_dd / peak_capital * 100) if peak_capital else 0.0

    # Sharpe — annualised, from daily equity changes
    daily_ret      = equity_df["pnl_today"]
    active_days    = daily_ret[daily_ret != 0]
    if len(active_days) > 1:
        ann_factor = np.sqrt(252)
        sharpe     = float(active_days.mean() / active_days.std() * ann_factor)
    else:
        sharpe = 0.0

    return {
        "num_trades":       len(trades),
        "total_pnl":        round(total_pnl, 2),
        "total_return_pct": round(total_pnl / total_capital * 100, 4) if total_capital else 0.0,
        "win_rate_pct":     round(win_rate, 2),
        "avg_return_pct":   round(float(np.mean(returns)), 4),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "sharpe_ratio":     round(sharpe, 4),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_factor_backtest(
    groups:     list[FactorGroup],
    symbol:     str,
    provider,
    library:    "FactorLibrary | None" = None,
    date_range: "tuple[str, str] | None" = None,
) -> dict:
    """Run all groups on one symbol and return per-group + combined results.

    Args:
        groups:     FactorGroup instances to backtest (must exist in library).
        symbol:     6-digit A-share code.
        provider:   MarketDataProvider instance.
        library:    FactorLibrary to look up Factor definitions. If None,
                    loads from disk.
        date_range: (start, end) strings YYYY-MM-DD. Defaults to last 3 years.

    Returns a dict:
        symbol      — the symbol tested
        groups      — dict[group_id → group_result]
        combined    — combined equity curve + total stats across all groups

    Each group_result:
        group_id, trades (list[dict]), equity_curve (pd.DataFrame), stats (dict)
    """
    if library is None:
        library = load_library()

    factors_dict: dict[str, Factor] = {f.id: f for f in library.factors}

    # Validate all referenced factors exist
    for g in groups:
        for fid in g.entry_factor_ids + g.exit_factor_ids:
            if fid not in factors_dict:
                return {
                    "symbol": symbol,
                    "error":  f"Group {g.id!r} references unknown factor {fid!r}.",
                }

    # Resolve date range
    if date_range:
        start, end = date_range
    else:
        end   = pd.Timestamp.today().strftime("%Y-%m-%d")
        start = (pd.Timestamp.today() - pd.Timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    # Fetch data once — shared across all groups
    df = provider.get_daily_history(symbol, start=start, end=end, adjust="hfq")
    if df is None or df.empty:
        return {"symbol": symbol, "error": f"No data in cache for {symbol}."}
    if len(df) < 20:
        return {"symbol": symbol, "error": f"{symbol} has only {len(df)} bars — too few to backtest."}

    df = add_indicators(df, symbol)
    df = df.reset_index(drop=True)
    dates = df["date"]

    # --- Simulate each group ---
    group_results: dict[str, dict] = {}

    for group in groups:
        try:
            entry_mask = _group_entry_mask(group, factors_dict, df)
            exit_mask  = _group_exit_mask(group, factors_dict, df)
            trades     = _simulate_group(df, entry_mask, exit_mask, group.capital, group.holding_period)
            eq_df      = _equity_curve(trades, dates)
            stats      = _compute_stats(trades, eq_df)
            group_results[group.id] = {
                "group_id":     group.id,
                "name":         group.name,
                "trades":       trades,
                "equity_curve": eq_df,
                "stats":        stats,
            }
        except Exception as e:
            group_results[group.id] = {
                "group_id": group.id,
                "name":     group.name,
                "error":    str(e),
            }

    # --- Combined equity across all groups ---
    combined_pnl = pd.Series(0.0, index=dates)
    for gr in group_results.values():
        if "equity_curve" in gr:
            combined_pnl = combined_pnl.add(gr["equity_curve"]["pnl_today"], fill_value=0.0)
    combined_eq = combined_pnl.cumsum()
    combined_eq_df = pd.DataFrame({"pnl_today": combined_pnl, "equity": combined_eq}, index=dates)

    all_trades = [t for gr in group_results.values() for t in gr.get("trades", [])]
    combined_stats = _compute_stats(all_trades, combined_eq_df)
    combined_stats["note"] = "Aggregated across all groups — capital is additive."

    return {
        "symbol":   symbol,
        "groups":   group_results,
        "combined": {
            "equity_curve": combined_eq_df,
            "stats":        combined_stats,
        },
    }


# ---------------------------------------------------------------------------
# Lookahead bias check
# ---------------------------------------------------------------------------

def check_lookahead(
    factor: Factor,
    raw_df: pd.DataFrame,
    symbol: str = "600519",
    market: str = "CN",
) -> bool:
    """Return True if the factor is free of lookahead bias, False otherwise.

    Accepts a raw OHLCV DataFrame (before indicators). Re-applies
    add_indicators at each sub-length so that any indicator which uses
    future data produces a different value on the truncated DataFrame —
    a mismatch at bar t flags lookahead.

    Args:
        factor:  Factor to test.
        raw_df:  DataFrame with columns date/open/high/low/close/volume/
                 pct_change/turnover_rate — NOT yet processed by add_indicators.
        symbol:  Symbol passed to add_indicators (affects limit-up thresholds).
        market:  Market code passed to condition evaluation.
    """
    if len(raw_df) < 10:
        return True  # too short to test reliably

    full_df   = add_indicators(raw_df.copy(), symbol)
    full_mask = _factor_mask(factor, full_df, market)

    for t in range(5, min(20, len(raw_df))):
        sub_raw = raw_df.iloc[:t + 1].copy().reset_index(drop=True)
        sub_df  = add_indicators(sub_raw, symbol)
        sub_m   = _factor_mask(factor, sub_df, market)
        if bool(full_mask.iloc[t]) != bool(sub_m.iloc[-1]):
            print(
                f"[lookahead] factor {factor.id!r} is leaky at bar {t}: "
                f"full={bool(full_mask.iloc[t])} sub={bool(sub_m.iloc[-1])}"
            )
            return False

    return True
