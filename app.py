import streamlit as st

from templates import get_templates, UNVALIDATED_NOTE

st.set_page_config(page_title="Monkey", layout="wide")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

_DEFAULTS: dict = {
    "directive":          "",
    "market":             "CN",
    "mode":               "Screen",
    "loaded_template_id": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

_templates      = get_templates()
_template_by_id = {t["id"]: t for t in _templates}

# ---------------------------------------------------------------------------
# Template selectbox callback — fires before the script body re-renders
# ---------------------------------------------------------------------------

def _on_template_change() -> None:
    tid = st.session_state.get("loaded_template_id")
    if tid and tid in _template_by_id:
        t = _template_by_id[tid]
        st.session_state["directive"] = t["description"]
        # Sync mode toggle to the template's mode
        st.session_state["mode"] = t["spec"].mode.capitalize()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Monkey")
st.caption("A place to test your thinking, not a black box that thinks for you.")
st.divider()

# ---------------------------------------------------------------------------
# Controls row — market + mode toggles
# ---------------------------------------------------------------------------

ctrl_col1, ctrl_col2, _ = st.columns([1.5, 1.5, 5])

with ctrl_col1:
    st.radio(
        "Market",
        options=["CN", "US"],
        key="market",
        horizontal=True,
        help="CN = China A-shares via AKShare. US market coming in Phase 6.",
    )

with ctrl_col2:
    st.radio(
        "Mode",
        options=["Screen", "Backtest"],
        key="mode",
        horizontal=True,
        help=(
            "Screen: find stocks matching a signal right now.  "
            "Backtest: test a rule over historical data."
        ),
    )

st.write("")

# ---------------------------------------------------------------------------
# Framework template loader
# ---------------------------------------------------------------------------

st.selectbox(
    "Load a framework template",
    options=[t["id"] for t in _templates],
    format_func=lambda tid: _template_by_id[tid]["name"],
    index=None,
    placeholder="— or select a framework template to start from —",
    key="loaded_template_id",
    on_change=_on_template_change,
)

_loaded = _template_by_id.get(st.session_state.get("loaded_template_id"))
if _loaded:
    with st.container(border=True):
        st.markdown(f"**{_loaded['name']}**")
        st.write(_loaded["rationale"])
        st.warning(UNVALIDATED_NOTE)

st.write("")

# ---------------------------------------------------------------------------
# Directive text area
# ---------------------------------------------------------------------------

st.text_area(
    "Describe a signal or strategy to test",
    key="directive",
    height=120,
    placeholder=(
        "e.g.  find stocks that hit limit-up yesterday with turnover rate above 10%\n"
        "e.g.  buy when 5-day MA crosses above 20-day MA, hold 10 days"
    ),
)

# ---------------------------------------------------------------------------
# Example chips
# ---------------------------------------------------------------------------

_EXAMPLES = [
    (
        "Limit-up with high turnover",
        "Stocks that hit limit-up yesterday with turnover rate above 10%",
    ),
    (
        "Volume surge on an up day",
        "Stocks where volume today is more than 3x the 20-day average and price is up",
    ),
    (
        "Range compression",
        "Stocks with an intraday range tighter than 60% of their 20-day average range",
    ),
    (
        "Gap-up with volume",
        "Stocks that opened at least 2% above yesterday's close with above-average volume",
    ),
]

st.caption("Try an example:")
_chip_cols = st.columns(len(_EXAMPLES))
for _col, (_label, _text) in zip(_chip_cols, _EXAMPLES):
    with _col:
        if st.button(_label, use_container_width=True, key=f"chip__{_label}"):
            st.session_state["directive"]          = _text
            st.session_state["loaded_template_id"] = None
            st.rerun()

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------

st.write("")
_run_col, _ = st.columns([1, 5])
with _run_col:
    _run = st.button(
        "Run",
        type="primary",
        use_container_width=True,
        disabled=not st.session_state.get("directive", "").strip(),
    )

if _run:
    if st.session_state.get("loaded_template_id"):
        # A validated SignalSpec already exists — run wiring comes in Phase 4 Step 3
        st.info(
            "Template loaded and spec is ready. "
            "Run wiring (run_screen / run_backtest) is added in Phase 4 Step 3.",
        )
    else:
        # Typed directive — needs Phase 3 NL parser
        st.info(
            "Natural-language parsing is built in Phase 3. "
            "Until then, select a framework template from the dropdown above to run a pre-built spec.",
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Monkey is a research tool for testing ideas. "
    "Nothing here is financial advice. "
    "Past performance does not predict future results."
)
