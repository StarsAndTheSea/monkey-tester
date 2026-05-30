import copy
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import AKShareCNProvider, Condition, SignalSpec, run_backtest, run_screen
from nl_parser import parse_nl
from templates import get_templates, UNVALIDATED_NOTE

load_dotenv()
_NL_AVAILABLE = bool(os.getenv("DEEPSEEK_API_KEY", ""))

_CACHE_DIR = Path(__file__).parent / "cache"
_SPECS_DIR = Path(__file__).parent / "specs"

st.set_page_config(page_title="Monkey Tester", layout="wide")

# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------

_templates      = get_templates()
_template_by_id = {t["id"]: t for t in _templates}

# All metrics the condition builder exposes
_METRICS: dict[str, str] = {
    "pct_change":            "Daily change %",
    "volume_vs_avg20":       "Volume ÷ 20-day avg",
    "turnover_rate":         "Turnover rate % (today)",
    "turnover_rate_5d":      "Turnover rate % (5-day avg)",
    "turnover_rate_10d":     "Turnover rate % (10-day avg)",
    "consecutive_limit_ups": "Consecutive limit-up days",
    "range_compression20":   "Range ÷ 20-day avg range",
    "gap_open_pct":          "Gap open % vs prev close",
    "days_since_limit_up":   "Days since last limit-up",
    "close":                 "Close price",
    "volume":                "Volume (shares)",
}

_OPERATORS = [">=", ">", "<=", "<", "=="]

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "stage":              "input",   # "input" | "confirm" | "results"
    "market":             "CN",
    "mode":               "Screen",
    "logic":              "AND",
    "universe_input":     "600519",
    "conditions":         [],        # screen conditions / backtest fallback
    "entry_conditions":   [],        # backtest entry
    "exit_conditions":    [],        # backtest exit (optional)
    "use_holding_period": False,
    "holding_bars":       5,
    "bt_date_start":      (pd.Timestamp.today() - pd.Timedelta(days=3 * 365)).date(),
    "bt_date_end":        pd.Timestamp.today().date(),
    "bt_strict_mode":     False,
    "loaded_template_id": None,
    "spec_json":          "",
    "run_results":        None,
    "nl_result":          None,
    # Factor Groups tab
    "fg_groups_data":     None,   # list of group dicts, loaded from disk
    "fg_results":         None,   # results from run_factor_backtest
    "fg_symbol":          "600519",
    "fg_strict_mode":     False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        # Copy mutable defaults so session state mutations don't affect _DEFAULTS
        st.session_state[_k] = copy.copy(_v) if isinstance(_v, list) else _v

# ---------------------------------------------------------------------------
# Condition-builder callbacks
# ---------------------------------------------------------------------------

def _do_add_condition(section: str) -> None:
    metric = st.session_state.get(f"_cb_metric_{section}")
    op     = st.session_state.get(f"_cb_op_{section}")
    val    = st.session_state.get(f"_cb_val_{section}", 0.0)
    if metric and op:
        st.session_state[section].append(
            {"metric": metric, "operator": op, "value": float(val)}
        )


def _do_remove_condition(section: str, idx: int) -> None:
    lst = st.session_state.get(section, [])
    if 0 <= idx < len(lst):
        lst.pop(idx)


def _apply_nl_spec(spec) -> None:
    """Populate the condition builder from a parsed SignalSpec."""
    st.session_state["mode"]  = spec.mode.capitalize()
    st.session_state["logic"] = spec.logic
    st.session_state["universe_input"] = (
        ", ".join(spec.universe) if isinstance(spec.universe, list) else spec.universe
    )
    if spec.mode == "screen":
        st.session_state["conditions"] = [
            {"metric": c.metric, "operator": c.operator, "value": c.value}
            for c in (spec.conditions or [])
        ]
    else:
        st.session_state["entry_conditions"] = [
            {"metric": c.metric, "operator": c.operator, "value": c.value}
            for c in (spec.entry or [])
        ]
        st.session_state["exit_conditions"] = [
            {"metric": c.metric, "operator": c.operator, "value": c.value}
            for c in (spec.exit or [])
        ]
        if spec.holding_period:
            st.session_state["use_holding_period"] = True
            st.session_state["holding_bars"]       = spec.holding_period


def _render_condition_builder(section: str, label: str) -> None:
    """Render a self-contained condition builder for one condition list."""
    st.markdown(f"**{label}**")

    # Input row: metric | operator | value | Add button
    c1, c2, c3, c4 = st.columns([3, 1.2, 1.5, 0.8])
    with c1:
        st.selectbox(
            "Metric", options=list(_METRICS.keys()),
            format_func=lambda m: f"{m}  —  {_METRICS[m]}",
            key=f"_cb_metric_{section}", label_visibility="collapsed",
        )
    with c2:
        st.selectbox("Op", options=_OPERATORS, key=f"_cb_op_{section}",
                     label_visibility="collapsed")
    with c3:
        st.number_input("Value", value=0.0, step=0.1, format="%.2f",
                        key=f"_cb_val_{section}", label_visibility="collapsed")
    with c4:
        st.button("Add", key=f"_add_{section}",
                  on_click=_do_add_condition, args=(section,),
                  use_container_width=True)

    # Current conditions list
    _conds = st.session_state.get(section, [])
    if _conds:
        for _i, _c in enumerate(_conds):
            _r1, _r2 = st.columns([11, 1])
            with _r1:
                st.markdown(f"- `{_c['metric']}` **{_c['operator']}** `{_c['value']}`")
            with _r2:
                st.button("×", key=f"_rm_{section}_{_i}",
                          on_click=_do_remove_condition, args=(section, _i))
    else:
        st.caption("_No conditions yet — add one above._")


# ---------------------------------------------------------------------------
# Spec builder — assembles a SignalSpec from current session state
# ---------------------------------------------------------------------------

def _build_spec_from_builder() -> SignalSpec | None:
    mode   = st.session_state["mode"].lower()
    market = st.session_state["market"]
    logic  = st.session_state["logic"]

    raw_u = st.session_state.get("universe_input", "").strip()
    if raw_u in ("csi300", "all_a_shares"):
        universe: str | list[str] = raw_u
    else:
        universe = [s.strip() for s in raw_u.split(",") if s.strip()]
        if not universe:
            return None

    _strict = st.session_state.get("bt_strict_mode", False)
    holding = (
        None
        if _strict
        else (
            int(st.session_state.get("holding_bars", 5))
            if st.session_state.get("use_holding_period")
            else None
        )
    )

    _ds = st.session_state.get("bt_date_start")
    _de = st.session_state.get("bt_date_end")
    _date_range = (str(_ds), str(_de)) if (_ds and _de and mode == "backtest") else None

    if mode == "screen":
        conds = [Condition(**c) for c in st.session_state.get("conditions", [])]
        if not conds:
            return None
        return SignalSpec(mode="screen", market=market, universe=universe,
                          conditions=conds, logic=logic)
    else:
        entry = [Condition(**c) for c in st.session_state.get("entry_conditions", [])]
        exit_ = [Condition(**c) for c in st.session_state.get("exit_conditions", [])]
        if not entry:
            return None
        return SignalSpec(
            mode="backtest", market=market, universe=universe,
            conditions=[], logic=logic,
            entry=entry,
            exit=exit_ if exit_ else None,
            holding_period=holding,
            date_range=_date_range,
        )


# ---------------------------------------------------------------------------
# Helpers — spec display (used in confirm stage)
# ---------------------------------------------------------------------------

def _parse_spec_json(json_str: str) -> tuple[SignalSpec | None, str | None]:
    try:
        data = json.loads(json_str)
        spec = SignalSpec.model_validate(data)
        return spec, None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    except Exception as e:
        return None, str(e)


def _condition_line(c: Condition) -> str:
    return f"`{c.metric}` {c.operator} `{c.value}`"


def _render_spec_summary(spec: SignalSpec) -> None:
    m1, m2, m3, m4 = st.columns(4)
    universe_str = (
        spec.universe
        if isinstance(spec.universe, str)
        else (spec.universe[0] if len(spec.universe) == 1 else ", ".join(spec.universe))
    )
    m1.metric("Mode",     spec.mode.capitalize())
    m2.metric("Market",   spec.market)
    m3.metric("Universe", universe_str)
    m4.metric("Logic",    spec.logic)
    st.write("")
    if spec.mode == "screen" or (spec.mode == "backtest" and not spec.entry):
        if spec.conditions:
            st.markdown("**Conditions**")
            for c in spec.conditions:
                st.markdown(f"- {_condition_line(c)}")
    else:
        if spec.entry:
            st.markdown("**Entry conditions**")
            for c in spec.entry:
                st.markdown(f"- {_condition_line(c)}")
        if spec.exit:
            st.markdown("**Exit conditions**")
            for c in spec.exit:
                st.markdown(f"- {_condition_line(c)}")
    extras = []
    if spec.holding_period:
        extras.append(f"Max hold: **{spec.holding_period} bars**")
    if spec.date_range:
        extras.append(f"Date range: **{spec.date_range[0]}** → **{spec.date_range[1]}**")
    if extras:
        st.caption("  ·  ".join(extras))


def _stat(stats: pd.Series, key: str, default: float = 0) -> float:
    try:
        v = stats[key]
        return float(v) if pd.notna(v) else default
    except (KeyError, TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Helpers — charts
# ---------------------------------------------------------------------------

def _plot_equity_vs_bh(equity_curve: pd.DataFrame, bh_df: "pd.DataFrame | None") -> None:
    eq = equity_curve["Equity"]
    eq_norm  = eq / eq.iloc[0] * 100
    eq_dates = eq_norm.index.strftime("%Y-%m-%d")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq_dates, y=eq_norm.values,
                             name="Strategy", line=dict(color="#2196F3", width=2),
                             hovertemplate="%{x}  %{y:.1f}<extra></extra>"))
    if bh_df is not None and not bh_df.empty:
        fig.add_trace(go.Scatter(x=bh_df["date"], y=bh_df["bh"].values,
                                 name="Buy & Hold",
                                 line=dict(color="#9E9E9E", width=1.5, dash="dot"),
                                 hovertemplate="%{x}  %{y:.1f}<extra></extra>"))
    fig.add_hline(y=100, line_dash="dash", line_color="#888", line_width=1, opacity=0.4)
    fig.update_layout(
        title="Equity curve (indexed to 100 at entry)", height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Index", hovermode="x unified",
        xaxis=dict(type="category", nticks=8),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_drawdown(equity_curve: pd.DataFrame) -> None:
    if "DrawdownPct" not in equity_curve.columns:
        return
    dd       = equity_curve["DrawdownPct"] * 100
    dd_dates = dd.index.strftime("%Y-%m-%d")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd_dates, y=dd.values, fill="tozeroy",
                             fillcolor="rgba(244,67,54,0.15)",
                             line=dict(color="rgba(244,67,54,0.7)", width=1),
                             hovertemplate="%{x}  %{y:.1f}%<extra></extra>"))
    fig.update_layout(title="Drawdown (%)", height=190,
                      margin=dict(l=0, r=0, t=40, b=24),
                      yaxis_title="%", showlegend=False, hovermode="x unified",
                      xaxis=dict(type="category", nticks=8))
    st.plotly_chart(fig, use_container_width=True)


def _plot_trade_distribution(trades_df: pd.DataFrame) -> None:
    if trades_df.empty or "ReturnPct" not in trades_df.columns:
        return
    returns = trades_df["ReturnPct"] * 100
    wins    = returns[returns >= 0]
    losses  = returns[returns <  0]
    fig = go.Figure()
    if not losses.empty:
        fig.add_trace(go.Histogram(x=losses, nbinsx=12, name="Loss",
                                   marker_color="rgba(244,67,54,0.7)"))
    if not wins.empty:
        fig.add_trace(go.Histogram(x=wins, nbinsx=12, name="Win",
                                   marker_color="rgba(76,175,80,0.7)"))
    fig.add_vline(x=0, line_dash="dash", line_color="#888", line_width=1, opacity=0.6)
    fig.update_layout(
        barmode="overlay",
        title=f"Trade return distribution  (n = {len(returns)})", height=220,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_title="Return per trade (%)", yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_monthly_heatmap(equity_curve: pd.DataFrame) -> None:
    """Monthly returns heatmap (years × months) from an equity curve."""
    if "Equity" not in equity_curve.columns or len(equity_curve) < 2:
        return

    # Resample to month-end equity, compute monthly % return
    monthly_eq = equity_curve["Equity"].resample("ME").last().dropna()
    if len(monthly_eq) < 2:
        return
    monthly_ret = monthly_eq.pct_change().dropna() * 100

    years  = sorted(monthly_ret.index.year.unique())
    months = list(range(1, 13))
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    # Build grid: rows = years, cols = months
    grid = []
    text = []
    for yr in years:
        row, row_txt = [], []
        for mo in months:
            mask = (monthly_ret.index.year == yr) & (monthly_ret.index.month == mo)
            if mask.any():
                v = float(monthly_ret[mask].iloc[0])
                row.append(v)
                row_txt.append(f"{v:+.1f}%")
            else:
                row.append(None)
                row_txt.append("")
        grid.append(row)
        text.append(row_txt)

    # Symmetric color scale centred at 0
    _vals = [v for row in grid for v in row if v is not None]
    _abs_max = max(abs(min(_vals)), abs(max(_vals)), 1) if _vals else 5

    fig = go.Figure(go.Heatmap(
        z=grid,
        x=month_labels,
        y=[str(y) for y in years],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorscale="RdYlGn",
        zmid=0,
        zmin=-_abs_max,
        zmax=_abs_max,
        colorbar=dict(title="%", thickness=12, len=0.8),
        hoverongaps=False,
        hovertemplate="%{y}-%{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Monthly returns (%)",
        height=max(180, 80 + len(years) * 40),
        margin=dict(l=0, r=0, t=40, b=40),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_price_with_trades(bh_df: "pd.DataFrame | None", trades_df: "pd.DataFrame | None") -> None:
    """Close price over time with entry (▲) and exit (▼) markers."""
    if bh_df is None or bh_df.empty or "close" not in bh_df.columns:
        return

    # Use string dates so Plotly treats x as categories — no weekend/holiday gaps
    dates  = bh_df["date"]
    prices = bh_df["close"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=prices,
        name="Close (hfq)",
        line=dict(color="#9E9E9E", width=1.2),
        hovertemplate="%{x}  %{y:.2f}<extra></extra>",
    ))

    if trades_df is not None and not trades_df.empty:
        if "EntryTime" in trades_df.columns and "EntryPrice" in trades_df.columns:
            entry_dates = pd.to_datetime(trades_df["EntryTime"]).dt.strftime("%Y-%m-%d")
            fig.add_trace(go.Scatter(
                x=entry_dates, y=trades_df["EntryPrice"],
                mode="markers", name="Entry",
                marker=dict(symbol="triangle-up", size=12, color="#4CAF50",
                            line=dict(color="white", width=1)),
                hovertemplate="Entry  %{x}  %{y:.2f}<extra></extra>",
            ))
        if "ExitTime" in trades_df.columns and "ExitPrice" in trades_df.columns:
            exit_dates = pd.to_datetime(trades_df["ExitTime"]).dt.strftime("%Y-%m-%d")
            fig.add_trace(go.Scatter(
                x=exit_dates, y=trades_df["ExitPrice"],
                mode="markers", name="Exit",
                marker=dict(symbol="triangle-down", size=12, color="#F44336",
                            line=dict(color="white", width=1)),
                hovertemplate="Exit  %{x}  %{y:.2f}<extra></extra>",
            ))

    fig.update_layout(
        title="Price with entry / exit markers (hfq-adjusted)",
        height=320,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Price (CNY)",
        hovermode="x unified",
        xaxis=dict(type="category", nticks=8),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Factor Groups helpers + charts
# ---------------------------------------------------------------------------

_FGG_FILE = Path(__file__).parent / "factor_groups.json"


def _fgg_load() -> list[dict]:
    """Load groups from disk and hydrate per-group condition keys in session state."""
    if not _FGG_FILE.exists():
        return []
    try:
        items = json.loads(_FGG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    groups = []
    for g in items:
        gid = g["id"]
        if f"fg_entry_{gid}" not in st.session_state:
            st.session_state[f"fg_entry_{gid}"] = g.get("entry_conds", [])
        if f"fg_exit_{gid}" not in st.session_state:
            st.session_state[f"fg_exit_{gid}"] = g.get("exit_conds", [])
        groups.append({k: v for k, v in g.items() if k not in ("entry_conds", "exit_conds")})
    return groups


def _fgg_save(groups: list[dict]) -> None:
    """Write groups to disk, picking up latest widget values and conditions."""
    full = []
    for g in groups:
        gid = g["id"]
        full.append({
            "id":             gid,
            "name":           st.session_state.get(f"fg_name_{gid}", g.get("name", "Group")),
            "capital":        st.session_state.get(f"fg_cap_{gid}",  g.get("capital", 50_000.0)),
            "holding_period": int(st.session_state.get(f"fg_hp_{gid}", 0)) or None,
            "entry_conds":    st.session_state.get(f"fg_entry_{gid}", []),
            "exit_conds":     st.session_state.get(f"fg_exit_{gid}",  []),
        })
    _FGG_FILE.write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")


def _get_stock_name(symbol: str) -> str:
    """Return Chinese company name for a symbol; falls back to empty string."""
    if "stock_names" not in st.session_state:
        try:
            import akshare as ak
            df = ak.stock_info_a_code_name()
            st.session_state["stock_names"] = dict(zip(df["code"], df["name"]))
        except Exception:
            st.session_state["stock_names"] = {}
    return st.session_state["stock_names"].get(symbol, "")


def _sym_label(symbol: str) -> str:
    """Return 'symbol  中文名' if name is available, else just symbol."""
    name = _get_stock_name(symbol)
    return f"{symbol}  {name}" if name else symbol


def _render_one_backtest_result(sym: str, stats, bh_df) -> None:
    """Render metric cards + full chart set for one completed backtest result."""
    if stats is None:
        st.warning(
            f"No data in cache for **{sym}**. "
            "Add it via the sidebar Warm Cache button."
        )
        return

    _ret = _stat(stats, "Return [%]")
    _bh  = _stat(stats, "Buy & Hold Return [%]")

    _c1, _c2, _c3, _c4 = st.columns(4)
    _c1.metric("Return",       f"{_ret:.1f}%", delta=f"{_ret - _bh:.1f}% vs B&H")
    _c2.metric("Buy & Hold",   f"{_bh:.1f}%")
    _c3.metric("Sharpe Ratio", f"{_stat(stats, 'Sharpe Ratio'):.2f}")
    _c4.metric("Max Drawdown", f"{_stat(stats, 'Max. Drawdown [%]'):.1f}%")

    _c5, _c6, _c7, _c8 = st.columns(4)
    _c5.metric("# Trades",  int(_stat(stats, "# Trades")))
    _c6.metric("Win Rate",  f"{_stat(stats, 'Win Rate [%]'):.1f}%")
    _c7.metric("Avg Trade", f"{_stat(stats, 'Avg. Trade [%]'):.2f}%")
    _c8.metric("SQN",       f"{_stat(stats, 'SQN'):.2f}")

    st.write("")

    _equity_curve = stats.get("_equity_curve")
    _trades_df    = stats.get("_trades")

    try:
        if _equity_curve is not None and not _equity_curve.empty:
            if "Equity" in _equity_curve.columns:
                _plot_equity_vs_bh(_equity_curve, bh_df)
            if "DrawdownPct" in _equity_curve.columns:
                _plot_drawdown(_equity_curve)
            _plot_monthly_heatmap(_equity_curve)
        _plot_price_with_trades(bh_df, _trades_df)
        if _trades_df is not None and not _trades_df.empty:
            _plot_trade_distribution(_trades_df)
    except Exception as _chart_err:
        import traceback as _tb
        st.warning(f"Chart rendering error: {_chart_err}")
        with st.expander("Chart traceback"):
            st.code(_tb.format_exc())

    with st.expander("All stats"):
        _scalar = {k: str(v) for k, v in stats.items() if not str(k).startswith("_")}
        st.dataframe(
            pd.DataFrame.from_dict(_scalar, orient="index", columns=["Value"]),
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def _save_run(spec_json: str, label: str = "") -> Path:
    _SPECS_DIR.mkdir(exist_ok=True)
    ts   = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    path = _SPECS_DIR / f"{ts}.json"
    path.write_text(
        json.dumps({"label": label, "spec": json.loads(spec_json)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _load_saved_specs() -> list[dict]:
    if not _SPECS_DIR.exists():
        return []
    out = []
    for f in sorted(_SPECS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            out.append({"path": f, "label": data.get("label") or f.stem, "spec": data.get("spec")})
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Template callback — populates condition builder from a template
# ---------------------------------------------------------------------------

def _on_template_change() -> None:
    tid = st.session_state.get("loaded_template_id")
    if not tid or tid not in _template_by_id:
        return
    spec = _template_by_id[tid]["spec"]

    st.session_state["mode"]  = spec.mode.capitalize()
    st.session_state["logic"] = spec.logic

    if isinstance(spec.universe, list):
        st.session_state["universe_input"] = ", ".join(spec.universe)
    else:
        st.session_state["universe_input"] = spec.universe

    if spec.mode == "screen":
        st.session_state["conditions"] = [
            {"metric": c.metric, "operator": c.operator, "value": c.value}
            for c in (spec.conditions or [])
        ]
    else:
        st.session_state["entry_conditions"] = [
            {"metric": c.metric, "operator": c.operator, "value": c.value}
            for c in (spec.entry or [])
        ]
        st.session_state["exit_conditions"] = [
            {"metric": c.metric, "operator": c.operator, "value": c.value}
            for c in (spec.exit or [])
        ]
        if spec.holding_period:
            st.session_state["use_holding_period"] = True
            st.session_state["holding_bars"]       = spec.holding_period


# ---------------------------------------------------------------------------
# Persistent header
# ---------------------------------------------------------------------------

st.title("Monkey Tester")
st.caption("Monkeys are smart")
st.divider()

# ---------------------------------------------------------------------------
# Sidebar — cache management + saved runs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Monkey Tester")

    # Cache management
    st.subheader("Cache")
    _cached_syms: list[str] = []
    if _CACHE_DIR.exists():
        _stems = {f.stem for f in _CACHE_DIR.glob("cn_*.parquet")}
        _cached_syms = sorted({
            s.replace("cn_", "").replace("_raw", "").replace("_hfq", "")
            for s in _stems
        })
    if _cached_syms:
        st.caption(f"{len(_cached_syms)} symbol(s) cached:")
        st.code("  ".join(_cached_syms))
    else:
        st.caption("No symbols cached yet.")

    _sb_new_sym = st.text_input(
        "Add symbol(s)", placeholder="e.g. 600519, 000001",
        key="_sb_new_sym",
        label_visibility="collapsed",
    )
    if st.button("Warm cache", key="_sb_warm", use_container_width=True):
        import io
        import sys as _sys
        _syms_to_warm = [s.strip() for s in _sb_new_sym.split(",") if s.strip()]
        if _syms_to_warm:
            _sb_prog  = st.progress(0, text="Starting...")
            _sb_prov  = AKShareCNProvider()
            _sb_end   = pd.Timestamp.today().strftime("%Y-%m-%d")
            _sb_start = (pd.Timestamp.today() - pd.Timedelta(days=5 * 365)).strftime("%Y-%m-%d")
            _sb_ok, _sb_fail, _sb_errors = [], [], {}
            for _sbi, _sbsym in enumerate(_syms_to_warm):
                _sb_prog.progress((_sbi + 1) / len(_syms_to_warm), text=f"Fetching {_sbsym}…")
                # Redirect stdout so AKShare's internal error prints are captured
                _buf = io.StringIO()
                _old_stdout, _sys.stdout = _sys.stdout, _buf
                try:
                    _sb_res = _sb_prov.get_daily_history(
                        _sbsym, start=_sb_start, end=_sb_end, adjust="hfq"
                    )
                except Exception as _sbe:
                    _sb_res = None
                    _sb_errors[_sbsym] = str(_sbe)
                finally:
                    _sys.stdout = _old_stdout
                _captured = _buf.getvalue().strip()
                if _sb_res is not None and not _sb_res.empty:
                    _sb_ok.append(_sbsym)
                else:
                    _sb_fail.append(_sbsym)
                    if _sbsym not in _sb_errors and _captured:
                        _sb_errors[_sbsym] = _captured
            _sb_prog.empty()
            st.session_state["_sb_warm_ok"]     = _sb_ok
            st.session_state["_sb_warm_fail"]   = _sb_fail
            st.session_state["_sb_warm_errors"] = _sb_errors
            st.rerun()
        else:
            st.warning("Enter at least one symbol.")

    _warm_ok     = st.session_state.pop("_sb_warm_ok",     None)
    _warm_fail   = st.session_state.pop("_sb_warm_fail",   None)
    _warm_errors = st.session_state.pop("_sb_warm_errors", {})
    if _warm_ok is not None or _warm_fail is not None:
        if _warm_ok:
            st.success(f"Successfully warmed cache for: {', '.join(_warm_ok)}")
        if _warm_fail:
            for _fsym in _warm_fail:
                _detail = _warm_errors.get(_fsym, "")
                _msg = f"Failed to fetch historical data for **{_fsym}**."
                if _detail:
                    _msg += f"\n\n`{_detail}`"
                else:
                    _msg += "\n\nAKShare returned no data. Check: VPN connected to China, valid 6-digit symbol."
                st.error(_msg)

    st.divider()

    # Saved runs
    st.subheader("Saved runs")
    _sb_saved = _load_saved_specs()
    if not _sb_saved:
        st.caption("No saved runs yet.")
    else:
        for _sbr in _sb_saved[:15]:
            _sbr_name  = _sbr["path"].stem   # timestamp slug
            _sbr_label = _sbr.get("label") or _sbr_name
            _l1, _l2 = st.columns([5, 1])
            with _l1:
                if st.button(_sbr_label, key=f"_sb_load_{_sbr_name}", use_container_width=True,
                             help="Load this run into the confirm stage"):
                    st.session_state["spec_json"] = json.dumps(_sbr["spec"], indent=2)
                    st.session_state["stage"]     = "confirm"
                    st.rerun()
            with _l2:
                if st.button("✕", key=f"_sb_del_{_sbr_name}",
                             help="Delete this saved run"):
                    try:
                        _sbr["path"].unlink()
                    except Exception:
                        pass
                    st.rerun()


def _render_backtest_tab():
    # ---------------------------------------------------------------------------
    # Stage: INPUT
    # ---------------------------------------------------------------------------

    if st.session_state["stage"] == "input":

        # Apply any pending NL spec BEFORE widgets render (avoids bound-key error)
        _pending = st.session_state.pop("_nl_pending_spec", None)
        if _pending is not None:
            _apply_nl_spec(_pending)

        # Controls row
        _ctrl1, _ctrl2, _ctrl3, _ = st.columns([1.5, 1.5, 1.5, 4])
        with _ctrl1:
            st.radio("Market", options=["CN", "US"], key="market", horizontal=True,
                     help="CN = China A-shares via AKShare.")
        with _ctrl2:
            st.radio("Mode", options=["Screen", "Backtest"], key="mode", horizontal=True,
                     help="Screen: today's data only.  Backtest: test over history.")
        with _ctrl3:
            st.radio("Logic", options=["AND", "OR"], key="logic", horizontal=True,
                     help="How conditions combine: AND = all must be true, OR = any must be true.")

        st.write("")

        # Universe
        _mode_now = st.session_state["mode"]
        st.text_input(
            "Universe",
            key="universe_input",
            placeholder="e.g.  600519   or   600519, 000001   or   csi300",
            help=(
                "One or more 6-digit A-share codes (comma-separated), or 'csi300' for the CSI 300 index. "
                "Backtest runs one symbol at a time."
            ),
        )

        st.write("")

        # Condition builder(s)
        if _mode_now == "Screen":
            _render_condition_builder("conditions", "Conditions")
        else:
            _render_condition_builder("entry_conditions", "Entry conditions")
            st.write("")
            _render_condition_builder(
                "exit_conditions",
                "Exit conditions  *(optional — any firing exits the position)*",
            )
            st.write("")

            # Strict mode
            st.checkbox(
                "Strict mode — only exit when exit condition fires (disable time-limit close)",
                key="bt_strict_mode",
                help="When on, the position is held until an exit condition fires. "
                     "The max holding period is ignored. Sell quantity equals buy quantity.",
            )

            # Max hold (hidden in strict mode)
            if not st.session_state.get("bt_strict_mode"):
                _hp1, _hp2, _ = st.columns([2.5, 1.5, 6])
                with _hp1:
                    st.checkbox("Use max holding period (bars)", key="use_holding_period")
                if st.session_state.get("use_holding_period"):
                    with _hp2:
                        st.number_input("Bars", min_value=1, max_value=500,
                                        key="holding_bars", label_visibility="collapsed",
                                        help="Exit after this many bars even if exit conditions haven't fired.")

            st.write("")

            # Date range
            _dr1, _dr2, _ = st.columns([2, 2, 6])
            with _dr1:
                st.date_input("Start date", key="bt_date_start",
                              help="Backtest start date. Data must be in cache.")
            with _dr2:
                st.date_input("End date", key="bt_date_end",
                              help="Backtest end date.")

        st.write("")

        # Optional NL parser
        with st.expander(
            "Describe your strategy in plain English — AI will fill the builder (optional)",
            expanded=bool(st.session_state.get("nl_result")),
        ):
            if not _NL_AVAILABLE:
                st.warning("Add `DEEPSEEK_API_KEY` to your `.env` file to use this feature.")
            else:
                st.text_area(
                    "Strategy description",
                    key="_nl_text",
                    height=80,
                    placeholder=(
                        "e.g. Enter when volume is 3× the 20-day average and pct change > 5%. "
                        "Exit after 5 bars or when turnover rate exceeds 20%."
                    ),
                    label_visibility="collapsed",
                )
                if st.button("Parse with AI", key="_nl_parse_btn", type="secondary"):
                    _nl_text = st.session_state.get("_nl_text", "").strip()
                    if _nl_text:
                        with st.spinner("Parsing with DeepSeek…"):
                            _nl_res = parse_nl(_nl_text)
                        st.session_state["nl_result"] = _nl_res
                        if _nl_res.get("spec"):
                            st.session_state["_nl_pending_spec"] = _nl_res["spec"]
                        st.rerun()

                _nl_res = st.session_state.get("nl_result")
                if _nl_res:
                    if _nl_res.get("spec"):
                        st.success("Conditions filled — review and edit below.")
                        if _nl_res.get("notes"):
                            st.caption(f"AI notes: {_nl_res['notes']}")
                    else:
                        st.error(f"Could not parse: {_nl_res.get('error', 'unknown error')}")

        st.write("")

        # Optional template loader
        with st.expander("Load a framework template as a starting point (optional)"):
            st.selectbox(
                "Template",
                options=[t["id"] for t in _templates],
                format_func=lambda tid: _template_by_id[tid]["name"],
                index=None,
                placeholder="— select a template to fill the builder above —",
                key="loaded_template_id",
                on_change=_on_template_change,
            )
            _loaded_t = _template_by_id.get(st.session_state.get("loaded_template_id"))
            if _loaded_t:
                st.caption(_loaded_t["rationale"])
                st.warning(UNVALIDATED_NOTE)

        st.write("")

        # Run button
        _mode_lower = st.session_state["mode"].lower()
        _has_conditions = (
            len(st.session_state.get("conditions", [])) > 0
            if _mode_lower == "screen"
            else len(st.session_state.get("entry_conditions", [])) > 0
        )
        _has_universe = bool(st.session_state.get("universe_input", "").strip())

        _run_col, _ = st.columns([1, 5])
        with _run_col:
            _run = st.button(
                "Run", type="primary", use_container_width=True,
                disabled=not (_has_conditions and _has_universe),
            )

        if not _has_conditions:
            st.caption(
                "Add at least one "
                + ("condition" if _mode_lower == "screen" else "entry condition")
                + " to enable Run."
            )

        if _run:
            _spec = _build_spec_from_builder()
            if _spec is None:
                st.error("Could not build a valid spec — check the universe and conditions.")
            else:
                st.session_state["spec_json"] = _spec.model_dump_json(indent=2)
                st.session_state["stage"]     = "confirm"
                st.rerun()

    # ---------------------------------------------------------------------------
    # Stage: CONFIRM
    # ---------------------------------------------------------------------------

    elif st.session_state["stage"] == "confirm":

        st.subheader("Review the spec — adjust anything before running")
        st.caption(
            "This is exactly what will run. Edit the JSON directly if you need to change a value, "
            "add a condition, or adjust the date range."
        )

        st.write("")

        _spec_now, _err = _parse_spec_json(st.session_state.get("spec_json", ""))

        if _spec_now:
            with st.container(border=True):
                _render_spec_summary(_spec_now)

        if _err:
            st.error(f"Invalid spec — fix the JSON below before running.  \n{_err}")

        st.write("")

        st.text_area(
            "Spec (JSON)",
            key="spec_json",
            height=340,
            help="Edit any field and the summary above updates live. Invalid JSON disables the run button.",
        )

        st.write("")

        _btn_col1, _btn_col2, _ = st.columns([1, 1.5, 5])
        with _btn_col1:
            if st.button("Back", use_container_width=True):
                st.session_state["stage"] = "input"
                st.rerun()
        with _btn_col2:
            _confirm = st.button(
                "Confirm & Run",
                type="primary",
                use_container_width=True,
                disabled=(_err is not None),
            )

        if _confirm and _spec_now:
            try:
                _provider = AKShareCNProvider()

                if _spec_now.mode == "screen":
                    with st.spinner("Screening universe — reading from cache..."):
                        _screen_df = run_screen(_spec_now, _provider)
                    st.session_state["run_results"] = {"mode": "screen", "df": _screen_df}

                else:
                    _symbols = (
                        _spec_now.universe
                        if isinstance(_spec_now.universe, list)
                        else [_spec_now.universe]
                    )
                    if _spec_now.date_range:
                        _date_start, _date_end = _spec_now.date_range
                    else:
                        _date_end   = pd.Timestamp.today().strftime("%Y-%m-%d")
                        _date_start = (pd.Timestamp.today() - pd.Timedelta(days=3 * 365)).strftime("%Y-%m-%d")

                    _bt_results: dict = {}
                    for _sym in _symbols:
                        with st.spinner(f"Backtesting {_sym}..."):
                            _stats = run_backtest(_spec_now, _sym, _provider)

                        _entry: dict = {"stats": _stats, "bh": None}

                        if _stats is not None:
                            _px = _provider.get_daily_history(
                                _sym, start=_date_start, end=_date_end,
                                adjust="hfq", cache_only=True,
                            )
                            if _px is not None and not _px.empty and len(_px) > 1:
                                _px_sub = _px[["date", "close"]].copy()
                                _px_sub["bh"] = _px_sub["close"] / _px_sub["close"].iloc[0] * 100
                                _entry["bh"] = _px_sub.reset_index(drop=True)

                        _bt_results[_sym] = _entry

                    st.session_state["run_results"] = {"mode": "backtest", "results": _bt_results}

                st.session_state["stage"] = "results"
                st.rerun()

            except Exception as _run_err:
                import traceback as _tb
                st.error(f"Run failed: {_run_err}")
                with st.expander("Full traceback"):
                    st.code(_tb.format_exc())

    # ---------------------------------------------------------------------------
    # Stage: RESULTS
    # ---------------------------------------------------------------------------

    elif st.session_state["stage"] == "results":

        _rr   = st.session_state.get("run_results") or {}
        _mode = _rr.get("mode")

        _nav1, _nav2, _nav3, _ = st.columns([1, 1, 1.5, 3])
        with _nav1:
            if st.button("Edit spec", use_container_width=True):
                st.session_state["stage"] = "confirm"
                st.rerun()
        with _nav2:
            if st.button("Start over", use_container_width=True):
                st.session_state["stage"]       = "input"
                st.session_state["run_results"] = None
                st.rerun()
        with _nav3:
            _save_label = st.text_input(
                "Label (optional)", placeholder="e.g. GCL momentum test",
                key="_save_label", label_visibility="collapsed",
            )
            if st.button("Save this run", use_container_width=True, key="_btn_save_run"):
                _spec_to_save = st.session_state.get("spec_json", "")
                if _spec_to_save:
                    _saved_path = _save_run(_spec_to_save, label=_save_label.strip())
                    st.success(f"Saved → {_saved_path.name}")

        st.write("")

        # --- Screen results ---
        if _mode == "screen":
            _df = _rr.get("df", pd.DataFrame())
            if _df.empty:
                st.info(
                    "No stocks matched all conditions on the latest trading day in cache.  \n"
                    "If you expected results, make sure the cache is populated — "
                    "run `warm_cache()` from a Python console or notebook first."
                )
            else:
                _date_str = _df["date"].iloc[0] if "date" in _df.columns else "the latest bar"
                st.success(f"**{len(_df)} stock(s)** matched on {_date_str}.")
                _show_cols = [c for c in [
                    "symbol", "date", "close", "pct_change", "turnover_rate",
                    "volume_vs_avg20", "range_compression20", "turnover_rate_5d",
                    "consecutive_limit_ups", "gap_open_pct",
                ] if c in _df.columns]
                st.dataframe(_df[_show_cols], use_container_width=True)

                if len(_df) >= 2:
                    st.write("")

                    # Sorted bar chart — daily change % per symbol
                    if "pct_change" in _df.columns:
                        _sorted = _df[["symbol", "pct_change"]].sort_values(
                            "pct_change", ascending=False
                        )
                        _fig_pc = go.Figure(go.Bar(
                            x=_sorted["symbol"], y=_sorted["pct_change"],
                            marker_color=[
                                "#4CAF50" if v >= 0 else "#F44336"
                                for v in _sorted["pct_change"]
                            ],
                            hovertemplate="%{x}  %{y:.2f}%<extra></extra>",
                        ))
                        _fig_pc.update_layout(
                            title="Daily change % — matched stocks",
                            height=240,
                            margin=dict(l=0, r=0, t=40, b=0),
                            yaxis_title="%",
                            showlegend=False,
                        )
                        st.plotly_chart(_fig_pc, use_container_width=True)

                    # Metric distributions (histograms) for numeric columns
                    _dist_metrics = [c for c in [
                        "turnover_rate", "volume_vs_avg20", "turnover_rate_5d",
                    ] if c in _df.columns]
                    if _dist_metrics:
                        _dcols = st.columns(len(_dist_metrics))
                        for _dc, _dm in zip(_dcols, _dist_metrics):
                            with _dc:
                                _fig_d = go.Figure(go.Histogram(
                                    x=_df[_dm], nbinsx=10,
                                    marker_color="#2196F3",
                                    hovertemplate="%{x:.2f}  count %{y}<extra></extra>",
                                ))
                                _fig_d.update_layout(
                                    title=_dm, height=200,
                                    margin=dict(l=0, r=0, t=30, b=0),
                                    showlegend=False,
                                )
                                st.plotly_chart(_fig_d, use_container_width=True)

                    # Board breakdown
                    try:
                        from core import classify_board
                        _board_counts = _df["symbol"].apply(classify_board).value_counts()
                        if len(_board_counts) > 1:
                            _fig_b = go.Figure(go.Bar(
                                x=[b.upper() for b in _board_counts.index],
                                y=_board_counts.values,
                                marker_color="#78909C",
                                text=_board_counts.values, textposition="outside",
                                hovertemplate="%{x}  %{y}<extra></extra>",
                            ))
                            _fig_b.update_layout(
                                title="Board breakdown",
                                height=220,
                                margin=dict(l=0, r=0, t=40, b=0),
                                showlegend=False,
                                yaxis=dict(visible=False),
                            )
                            st.plotly_chart(_fig_b, use_container_width=True)
                    except Exception:
                        pass

        # --- Backtest results ---
        elif _mode == "backtest":
            _results = _rr.get("results", {})
            if not _results:
                st.info("No backtest results.")
            else:
                for _sym, _entry in _results.items():
                    _stats = _entry.get("stats")
                    _bh_df = _entry.get("bh")
                    st.subheader(_sym_label(_sym))
                    _render_one_backtest_result(_sym, _stats, _bh_df)
                    st.divider()

        else:
            st.warning("No results available — return to input and run a spec.")



def _render_factor_groups_tab():
    """Factor Groups tab — groups with inline entry/exit conditions."""
    import uuid as _uuid

    if st.session_state.get("fg_groups_data") is None:
        st.session_state["fg_groups_data"] = _fgg_load()

    groups: list[dict] = st.session_state["fg_groups_data"]

    # ------------------------------------------------------------------
    # Header: Add Group + Save All
    # ------------------------------------------------------------------
    _h1, _h2, _h3 = st.columns([5, 1, 1])
    with _h1:
        st.subheader("Factor Groups")
    with _h2:
        if st.button("+ Add Group", use_container_width=True):
            _new_id = f"grp_{_uuid.uuid4().hex[:8]}"
            groups.append({
                "id": _new_id, "name": f"Group {len(groups) + 1}",
                "capital": 50_000.0, "holding_period": None,
            })
            st.session_state[f"fg_entry_{_new_id}"] = []
            st.session_state[f"fg_exit_{_new_id}"]  = []
            st.session_state[f"fg_run_{_new_id}"]   = True
            _fgg_save(groups)
            st.rerun()
    with _h3:
        if st.button("Save All", use_container_width=True,
                     help="Persist group names, capital, conditions to disk"):
            _fgg_save(groups)
            st.session_state["_fgg_saved"] = True
            st.rerun()

    _saved = st.session_state.pop("_fgg_saved", None)
    if _saved:
        st.success("Groups saved.")

    if not groups:
        st.caption("No groups yet — click **+ Add Group** to create one.")

    # ------------------------------------------------------------------
    # Group cards (collapsible)
    # ------------------------------------------------------------------
    for _g in groups:
        _gid  = _g["id"]
        # Build a summary line for the expander label
        _disp_name = st.session_state.get(f"fg_name_{_gid}", _g.get("name", "Group"))
        _disp_cap  = st.session_state.get(f"fg_cap_{_gid}",  _g.get("capital", 50_000.0))
        _n_entry   = len(st.session_state.get(f"fg_entry_{_gid}", []))
        _n_exit    = len(st.session_state.get(f"fg_exit_{_gid}",  []))
        _lbl = (
            f"{_disp_name}  ·  ¥{float(_disp_cap):,.0f}"
            f"  ·  {_n_entry} entry / {_n_exit} exit"
        )

        with st.expander(_lbl, expanded=True):
            # Top row: include-in-run checkbox + delete button
            _cr1, _cr2 = st.columns([5, 1])
            with _cr1:
                st.checkbox("Include in run", value=True, key=f"fg_run_{_gid}")
            with _cr2:
                if st.button("Delete", key=f"fg_del_{_gid}", use_container_width=True):
                    groups.remove(_g)
                    _fgg_save(groups)
                    st.rerun()

            # Settings row: name | capital | max hold
            _s1, _s2, _s3 = st.columns([3, 2, 2])
            with _s1:
                st.text_input(
                    "Name", value=_g.get("name", "Group"),
                    key=f"fg_name_{_gid}", label_visibility="collapsed",
                    placeholder="Group name",
                )
            with _s2:
                st.number_input(
                    "Capital (¥)", value=float(_g.get("capital", 50_000.0)),
                    min_value=1_000.0, step=1_000.0, format="%.0f",
                    key=f"fg_cap_{_gid}", label_visibility="collapsed",
                )
            with _s3:
                st.number_input(
                    "Max hold bars (0 = no limit)",
                    value=int(_g.get("holding_period") or 0),
                    min_value=0, max_value=500, step=1,
                    key=f"fg_hp_{_gid}", label_visibility="collapsed",
                    help="Force-close after this many bars. 0 = hold until exit condition or end of data.",
                )
            st.caption("Name  ·  Capital per trade (¥)  ·  Max hold bars")

            st.write("")
            _render_condition_builder(f"fg_entry_{_gid}", "Entry — all must fire")
            st.write("")
            _render_condition_builder(
                f"fg_exit_{_gid}",
                "Exit — any fires  *(optional — leave empty to rely on max hold)*",
            )

    # ------------------------------------------------------------------
    # Run section
    # ------------------------------------------------------------------
    st.divider()

    _selected  = [g for g in groups if st.session_state.get(f"fg_run_{g['id']}", True)]
    _runnable  = [g for g in _selected if st.session_state.get(f"fg_entry_{g['id']}")]
    _btn_label = (
        f"Run {len(_runnable)} selected group(s)"
        if _runnable else "No groups selected with entry conditions"
    )

    _rs1, _rs2, _rs3, _rs4 = st.columns([2, 2, 2, 1])
    with _rs1:
        st.text_input(
            "Symbol", key="fg_symbol",
            placeholder="e.g. 600519",
            help="6-digit A-share code. Must already be in cache.",
            label_visibility="collapsed",
        )
    with _rs2:
        st.date_input("Start date", key="fg_date_start",
                      value=(pd.Timestamp.today() - pd.Timedelta(days=3 * 365)).date(),
                      help="Backtest start date.")
    with _rs3:
        st.date_input("End date", key="fg_date_end",
                      value=pd.Timestamp.today().date(),
                      help="Backtest end date.")
    with _rs4:
        _run_btn = st.button(
            _btn_label, type="primary",
            use_container_width=True, key="fg_run_btn",
            disabled=not _runnable,
        )

    st.checkbox(
        "Strict mode — only exit when exit condition fires (ignore group holding period)",
        key="fg_strict_mode",
        help="When on, each group's max holding period is ignored. "
             "The position is held until an exit condition fires. "
             "Sell quantity equals buy quantity.",
    )

    if _run_btn:
        _sym = st.session_state.get("fg_symbol", "").strip()
        if not _sym:
            st.error("Enter a symbol.")
        else:
            _fgg_save(groups)
            _provider = AKShareCNProvider()

            _fg_start = str(st.session_state.get("fg_date_start",
                            (pd.Timestamp.today() - pd.Timedelta(days=3 * 365)).date()))
            _fg_end   = str(st.session_state.get("fg_date_end", pd.Timestamp.today().date()))

            # Fetch B&H reference data once — shared across all groups
            _px = _provider.get_daily_history(
                _sym, start=_fg_start, end=_fg_end, adjust="hfq", cache_only=True,
            )
            _bh_shared = None
            if _px is not None and not _px.empty and len(_px) > 1:
                _px_sub = _px[["date", "close"]].copy()
                _px_sub["bh"] = _px_sub["close"] / _px_sub["close"].iloc[0] * 100
                _bh_shared = _px_sub.reset_index(drop=True)

            _fg_strict = st.session_state.get("fg_strict_mode", False)
            _fg_group_results: dict = {}
            for _rg in _runnable:
                _rgid    = _rg["id"]
                _rg_name = st.session_state.get(f"fg_name_{_rgid}", _rg.get("name", "Group"))
                _rg_cap  = float(st.session_state.get(f"fg_cap_{_rgid}", _rg.get("capital", 50_000.0)))
                _rg_hp   = None if _fg_strict else (int(st.session_state.get(f"fg_hp_{_rgid}", 0) or 0) or None)
                _rg_entry = [Condition(**c) for c in st.session_state.get(f"fg_entry_{_rgid}", [])]
                _rg_exit  = [Condition(**c) for c in st.session_state.get(f"fg_exit_{_rgid}",  [])]
                if not _rg_entry:
                    continue
                _rg_spec = SignalSpec(
                    mode="backtest", market="CN", universe=[_sym],
                    conditions=[], logic="AND",
                    entry=_rg_entry,
                    exit=_rg_exit or None,
                    holding_period=_rg_hp,
                    date_range=(_fg_start, _fg_end),
                )
                with st.spinner(f"Running '{_rg_name}' on {_sym_label(_sym)}…"):
                    _rg_stats = run_backtest(_rg_spec, _sym, _provider, cash=_rg_cap)
                _fg_group_results[_rgid] = {
                    "name":    _rg_name,
                    "capital": _rg_cap,
                    "stats":   _rg_stats,
                    "bh":      _bh_shared,
                }

            st.session_state["fg_results"] = {"symbol": _sym, "groups": _fg_group_results}
            st.rerun()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    _fg_results = st.session_state.get("fg_results")
    if not _fg_results:
        return

    _fg_sym = _fg_results.get("symbol", "")
    st.subheader(f"Results — {_sym_label(_fg_sym)}")
    st.caption("Each group uses its own capital allocation. "
               "This is a hypothesis-testing tool — not financial advice.")

    _fg_groups = _fg_results.get("groups", {})

    # Comparison table when multiple groups ran
    if len(_fg_groups) > 1:
        _cmp = []
        for _cgdata in _fg_groups.values():
            _cs  = _cgdata.get("stats")
            _cap = _cgdata.get("capital", 0)
            _ret_pct = _stat(_cs, "Return [%]") if _cs is not None else None
            _final   = _cap * (1 + _ret_pct / 100) if _ret_pct is not None else None
            _cmp.append({
                "Group":         _cgdata.get("name", "—"),
                "Capital":       f"¥{_cap:,.0f}",
                "Final Value":   f"¥{_final:,.0f}" if _final is not None else "—",
                "Net PnL":       f"¥{_final - _cap:+,.0f}" if _final is not None else "—",
                "Return":        f"{_ret_pct:+.1f}%" if _ret_pct is not None else "—",
                "vs B&H":        f"{_ret_pct - _stat(_cs, 'Buy & Hold Return [%]'):+.1f}%" if _cs is not None else "—",
                "Sharpe":        f"{_stat(_cs, 'Sharpe Ratio'):.2f}" if _cs is not None else "—",
                "Max DD":        f"{_stat(_cs, 'Max. Drawdown [%]'):.1f}%" if _cs is not None else "—",
                "# Trades":      int(_stat(_cs, "# Trades")) if _cs is not None else "—",
                "Win Rate":      f"{_stat(_cs, 'Win Rate [%]'):.1f}%" if _cs is not None else "—",
            })
        st.dataframe(pd.DataFrame(_cmp), use_container_width=True, hide_index=True)
        st.write("")

    # Per-group detailed results — identical to signal backtest
    for _gdata in _fg_groups.values():
        _gname   = _gdata.get("name", "Group")
        _gcap    = _gdata.get("capital", 50_000.0)
        _gstats  = _gdata.get("stats")
        _gret    = _stat(_gstats, "Return [%]") if _gstats is not None else None
        _gfinal  = _gcap * (1 + _gret / 100) if _gret is not None else None
        _val_str = f"  →  ¥{_gfinal:,.0f}" if _gfinal is not None else ""
        st.subheader(f"{_gname}  ·  ¥{_gcap:,.0f}{_val_str}")
        _render_one_backtest_result(_fg_sym, _gstats, _gdata.get("bh"))
        st.divider()



# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

_tab1, _tab2 = st.tabs(["Signal Backtest", "Factor Groups"])
with _tab1:
    _render_backtest_tab()
with _tab2:
    _render_factor_groups_tab()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Monkey Tester is a research tool for testing ideas. "
    "Nothing here is financial advice. "
    "Past performance does not predict future results."
)
