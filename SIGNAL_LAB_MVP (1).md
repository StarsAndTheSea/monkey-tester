# Monkey — MVP Build Specification

> **For Claude Code.** This is a complete build brief for an MVP. Read it top to bottom before
> writing code. Build incrementally, in the phase order given at the bottom. Ask the user before
> introducing any paid service. Everything in the default stack is free.

---

## 0. What we are building (one paragraph)

A **signal research tool** for the China A-share market (with US stocks as a secondary market). The
user types a directive in plain language — e.g. *"find stocks that hit a limit-up yesterday and have
turnover rate above 10%"* or *"test a strategy that buys when the 5-day moving average crosses above
the 20-day"*. The app translates that directive into a structured **signal spec**, runs it against
historical price data (a screen and/or a backtest), returns transparent statistics, and then layers
in **current news and sentiment** so the user can interpret the result against real-world context.
The product is a **hypothesis-testing / "test your thinking" tool**, NOT an automated stock picker
and NOT financial advice. That framing is a core design constraint, not a disclaimer afterthought.

The engine is **fully generic** — it tests any signal the user can describe. On top of that generic
engine, the app ships with a **preloaded library of signal templates derived from one specific
A-share trading framework** (the "catalyst / smart-money-flow" method documented in §4a). These
templates are not hardcoded opinions baked into the engine — they are ordinary, fully-editable
`SignalSpec` examples that load into the same input/confirm/run loop as anything the user types.
Their purpose is twofold: (a) teach the interaction by example, and (b) let the framework's own author
**stress-test his discretionary rules against history** — turning intuitions like "sell on the second
day of 30%+ turnover" into backtestable hypotheses. The generic "test your thinking" identity is
preserved; the framework is the worked example, not a black box.

**Primary user:** semi-pro / quant-curious, and specifically a discretionary A-share trader who wants
to **validate or refute his own discretionary signals** with transparent stats. They want
transparency — to see the generated signal logic, the backtest assumptions, and the raw stats — not a
black box. Design every screen for that.

---

## 1. Hard constraints (do not violate)

- **Budget:** Free / near-free only (target $0/mo, hard ceiling ~$50/mo). Do NOT add a paid API
  without asking the user first. The default stack below is entirely free.
- **Market priority:** China A-shares first and primary; US stocks secondary. Architecture must treat
  "market" as a pluggable provider so US can be added without rewrites.
- **Data freshness:** 15-minute-delayed or end-of-day data is acceptable. Do NOT build a real-time
  streaming scanner — it is unnecessary, expensive, and out of scope. This is a scan-and-backtest
  tool over cached/daily data.
- **Coding level:** User can read and tweak code. Write clear, commented, modular code. Explain
  non-obvious decisions in comments. Avoid clever one-liners where a readable block is clearer.
- **Compliance posture:** Not selling yet. No payments, no user accounts, no auth in the MVP. DO
  include the "research tool, not financial advice" framing prominently in the UI.

---

## 2. Recommended tech stack (all free)

### Backend / language
- **Python 3.13+**. Use a virtual environment (`venv`) and pin versions in `requirements.txt`.
- **No separate web framework.** This is a single-process Streamlit app — Streamlit IS the server.
  The UI calls the `core/` Python functions directly. Do not add FastAPI/Flask/uvicorn; a second
  server is needless complexity for a self-contained MVP. (If a public HTTP API is ever needed, that's
  a post-MVP concern — note it as future work, don't build it now.)

### Data providers (the most important decisions)

| Purpose | Library / API | Cost | Why |
|---|---|---|---|
| **China A-share prices & metrics** | **AKShare** (`pip install akshare`) | Free, no key | Daily-history endpoint returns `涨跌幅` (% change → limit-up detection) and `换手率` (turnover rate) directly. Broadest free China coverage. |
| China A-share (optional upgrade) | **Tushare** (`pip install tushare`) | Free token w/ points; some data paid | Cleaner point-in-time data, money-flow, limit-up stats. Wire as OPTIONAL provider, off by default. |
| **US stock prices** | **Twelve Data** | Free tier: 800 req/day | Clean JSON, simple API-key auth. Secondary market. |
| **News + sentiment (both markets)** | **Marketaux** | Free tier: limited daily req | Sentiment scored per ticker, entity recognition, filter by symbol/sector. |
| China-specific news | **AKShare** news endpoints | Free | Supplement for A-share news the global APIs miss. |

> **Critical architecture note:** SEPARATE the two data paths.
> 1. **Bulk historical prices** (for screening + backtesting) → fetched once, then **cached locally**.
>    Even tight rate limits don't matter because you're not re-pulling.
> 2. **News/sentiment** (fresh interpretation layer) → called on demand, infrequently.
>
> This keeps the app on free tiers indefinitely. Do not call price APIs live on every request.

> **Known risk to handle:** AKShare is scraping-based. If an upstream site changes layout, a function
> can break until the community patches it. Mitigations to BUILD IN: (a) local caching so cached data
> survives an upstream outage; (b) pin the AKShare version in requirements.txt; (c) wrap every AKShare
> call in try/except that returns a clear, user-facing "data source temporarily unavailable" message
> rather than a stack trace.

### Signal / backtest engine
- **backtesting.py** (`pip install backtesting`) — the default engine. Chosen for simplicity and
  self-containment: it is pure Python (Pandas + NumPy + Bokeh), installs in seconds with no
  compilation step, and its entire API fits on one page — easy to read and tweak. Despite being
  lightweight it still does what this app needs: it produces **interactive zoomable trade charts**
  (price with entry/exit markers) and can run **parameter optimization that yields a heatmap** of
  hundreds of variants — so you keep both standout visuals from §7.
- **Important architectural split** (this keeps things simple):
  - **Backtesting** = per-symbol, runs through backtesting.py one asset at a time. (The library is
    single-asset by design; that's fine — backtests are naturally per-symbol here.)
  - **Screening** = multi-symbol point-in-time filtering. This needs NO backtest engine at all — it's
    a plain pandas boolean-mask operation over the latest bar of each symbol's cached data. Implement
    screening directly in pandas in `engine.py`; do not route it through backtesting.py.
- **Optional upgrade path (do NOT install by default):** vectorbt is faster for sweeping *thousands*
  of combinations across many symbols at once, but it depends on Numba and is notably harder to
  install. Leave it as a documented future option for power users; the MVP does not need it.
- **pandas** for all data wrangling and for screening. **pandas-ta** (`pip install pandas-ta`) for
  the technical indicator library (RSI, MACD, moving averages, etc.) so the user can reference common
  indicators by name in their directive. backtesting.py is compatible with pandas-ta.

### LLM layer (natural-language → signal spec, and interpretation)
- **DeepSeek API** for two jobs:
  1. **Parse** the user's plain-language directive into a structured JSON signal spec (see §4).
  2. **Interpret** backtest results + fetched news into a plain-language readout for the user.
- **Why DeepSeek here:** it is the cheapest frontier-class API on the market and handles
  Chinese-language financial text natively — directly relevant since the primary market is China
  A-shares and much of the news/sentiment will be in Chinese. This is the only recurring cost in the
  app and it is very small.
- **Model selection:**
  - Use **`deepseek-v4-flash`** (fast, cheap) for the directive-parsing step — it is plenty for
    turning a sentence into a JSON spec. Run it in **non-thinking mode** for parsing.
  - Use **`deepseek-v4-flash` in thinking mode**, or escalate to **`deepseek-v4-pro`** only if the
    interpretation quality needs it, for the results+news readout step. Default to Flash; make the
    interpretation model a config setting so the user can switch without code changes.
  - **Do NOT use the legacy `deepseek-chat` / `deepseek-reasoner` aliases** — they are scheduled for
    deprecation (hard cutoff, no fallback) and would break the app. Use the explicit `deepseek-v4-*`
    IDs only.
- **API format:** DeepSeek is **OpenAI-compatible**. Use the official `openai` Python SDK pointed at
  DeepSeek's base URL — no separate SDK needed:
  ```python
  from openai import OpenAI
  client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
  resp = client.chat.completions.create(
      model="deepseek-v4-flash",
      messages=[...],
      response_format={"type": "json_object"},  # use JSON mode for the spec-parsing call
  )
  ```
- **Cost tip:** DeepSeek gives a steep cache discount on repeated input prefixes. Keep the system
  prompt / instructions identical across calls so the static prefix is cached — this cuts the
  parsing cost dramatically at volume. Verify current model names and pricing at
  api-docs.deepseek.com at build time rather than trusting any hardcoded figure, since DeepSeek
  adjusts both periodically.

### Storage
- Keep it dead simple and file-based — no database server, no ORM.
- **Parquet files** (`pyarrow`) for cached bulk price history — fast columnar reads, one file per
  symbol or per market. This is the primary cache.
- **stdlib `sqlite3`** (no SQLAlchemy) only if you need a tiny table for saved user signals. A single
  JSON file works too and is even simpler — prefer whichever is less code. Do not introduce an ORM.

### Frontend
- **Streamlit** (`pip install streamlit`) for the MVP UI, and it's the *only* server (see Backend
  above). Rationale: a quant-curious user wants tables, charts, and editable parameters fast;
  Streamlit gives interactive dataframes, metric cards, and charts with almost no frontend code, so
  Claude Code can ship the whole UX in one file. (If the user later wants a polished public product,
  migrating to a React + FastAPI split is future work — do not build it now.)
- **Charting — use the simplest tool that fits each chart, in this priority order:**
  1. **backtesting.py's built-in plot** for the equity curve, trade markers, and the optimization
     **heatmap** — these come free from the engine, zoomable, no extra charting code. Embed in
     Streamlit via the Bokeh figure it returns.
  2. **Streamlit native** (`st.bar_chart`, `st.line_chart`, `st.metric`) for simple distributions and
     stat cards — zero extra dependencies.
  3. **Plotly** (`st.plotly_chart`) only where 1 and 2 don't suffice (e.g. a custom diverging
     sentiment bar). Don't reach for Plotly when a native chart already does the job.

---

## 3. Architecture overview

```
signal-lab/
├── README.md                  # setup + run instructions (generate this)
├── requirements.txt           # pinned versions
├── .env.example               # API keys template (never commit real .env)
├── app.py                     # Streamlit entry point + UI (the only server)
├── core/
│   ├── signal_spec.py         # the structured signal-spec schema (Pydantic models)
│   ├── nl_parser.py           # DeepSeek: plain language -> signal_spec JSON
│   ├── interpreter.py         # DeepSeek: results + news -> plain-language readout
│   ├── engine.py              # backtesting.py (per-symbol backtests) + pandas screening
│   ├── indicators.py          # pandas-ta wrappers, custom signal builders, §4a framework metrics
│   └── templates.py           # §4a preloaded framework signal templates (or templates.json)
├── data/
│   ├── providers/
│   │   ├── base.py            # MarketDataProvider abstract base class
│   │   ├── akshare_cn.py      # China A-share provider (PRIMARY)
│   │   ├── tushare_cn.py      # optional China provider (off by default)
│   │   └── twelvedata_us.py   # US provider (SECONDARY)
│   ├── news.py                # Marketaux + AKShare news fetch
│   └── cache.py               # parquet cache (+ optional tiny sqlite/JSON for saved signals)
├── cache/                     # local cached price data (gitignored)
└── tests/                     # pytest: spec parsing, engine math, cache behavior
```

> **Self-containment note for Claude Code:** the module tree above is the *clean* target, but this
> MVP is small enough that you may start with everything in `app.py` plus one `core.py` and split
> out modules only once a file gets unwieldy. Fewer files = easier to paste, run, and reason about.
> Either way: no second server, no database server, no build step. One `pip install`, one
> `streamlit run`.

**Provider abstraction.** Define `MarketDataProvider` with methods like
`get_daily_history(symbol, start, end)`, `list_symbols(market)`, `get_snapshot(symbols)`. Each
provider returns a **normalized DataFrame** with standard English column names
(`date, open, high, close, low, volume, pct_change, turnover_rate`) regardless of source language.
The AKShare provider is responsible for renaming `涨跌幅 → pct_change`, `换手率 → turnover_rate`, etc.
This normalization is what lets the same signal spec run on either market.

---

## 4. The signal-spec schema (the heart of the app)

The user's plain-language directive is parsed by DeepSeek into a structured, validated spec. This is
what makes the app **flexible** — any signal the user can describe maps onto this schema, and the
engine executes it generically.

Define with Pydantic. Suggested shape:

```python
class Condition(BaseModel):
    metric: str        # e.g. "pct_change", "turnover_rate", "rsi", "ma_cross", "volume",
                       # plus framework metrics (see §4a): "consecutive_limit_ups",
                       # "turnover_rate_Nd", "volume_vs_avg", "range_compression",
                       # "gap_open_pct", "days_since_limit_up"
    operator: str      # ">", ">=", "<", "<=", "==", "crosses_above", "crosses_below"
    value: float | str # threshold, or another metric name for cross/compare
    window: int | None # lookback in days where relevant (e.g. RSI period, MA length)
    timeframe: str = "1d"

class SignalSpec(BaseModel):
    mode: str                  # "screen" (point-in-time) or "backtest" (historical rule)
    market: str                # "CN" or "US"
    universe: str | list[str]  # "all_a_shares", "csi300", or explicit ticker list
    conditions: list[Condition]
    logic: str = "AND"         # how conditions combine: "AND" / "OR" / custom expression
    # backtest-only fields:
    entry: list[Condition] | None
    exit: list[Condition] | None
    holding_period: int | None # max bars to hold
    date_range: tuple[str, str] | None
```

**Always show the user the generated spec** before running, and let them edit it (raw JSON editor or
form fields). Transparency is the product. The semi-pro user needs to trust and tune the logic.

**Example mappings to internalize:**
- *"stocks that hit limit-up yesterday with turnover over 10%"* →
  `mode=screen, conditions=[{metric:pct_change, op:>=, value:9.9}, {metric:turnover_rate, op:>, value:10}], logic=AND`
  (Note: A-share main-board daily limit is ±10%; use `>= 9.9` to catch limit-ups robustly. STAR/ChiNext
  boards use ±20% — handle board-specific limits in the engine, see §6.)
- *"buy when 5-day MA crosses above 20-day MA, hold 10 days"* →
  `mode=backtest, entry=[{metric:ma_cross, op:crosses_above, value:"ma20", window:5}], holding_period:10`

---

## 4a. The built-in framework & signal-template library

This is the alignment layer. The app ships with a small **preloaded library** of saved `SignalSpec`
templates that encode one specific discretionary A-share method. They are loadable, inspectable, and
**fully editable** — identical in kind to anything the user types. Store them as a `templates.json`
(or a `core/templates.py` list of `SignalSpec` instances) and surface them in the UI as a "Load a
framework template" dropdown beside the free-text box (see §5). Each template carries a plain-language
`description` and a `rationale` field so the user sees *why* the rule exists, not just the JSON.

> **Why this matters for the product's identity:** the engine stays generic. The framework is
> expressed entirely as data (specs + notes), never as hardcoded engine branches. A user who doesn't
> care about this method ignores the templates and types their own directive. The framework's author
> uses the templates to **test whether his discretionary rules actually hold up** — which is the whole
> point of building this privately before trusting or teaching the method.

### The framework in one paragraph (for context, so the templates make sense)

The method is **catalyst-momentum / smart-money-flow trading** on A-shares: enter when institutional
("smart money") capital is flowing into a stock around an identifiable catalyst, and exit on the first
clear sign that flow is reversing — prioritizing **securing profit and reducing risk over maximizing
the top tick**. Position size encodes conviction. The T+1 rule (cannot sell shares bought the same
day) shapes every risk decision. Three distinct **trade types** recur:

1. **Momentum / rumor** — a hot thematic catalyst (e.g. a sector narrative) drives consecutive
   limit-ups; enter on confirmed momentum + a bullish pre-market call-auction, size larger (~25%).
2. **Macro-thesis** — a macro/geopolitical catalyst (e.g. a tariff shock driving capital into hard
   assets) supports a sector *before* technical confirmation; enter on conviction, size smaller
   (~15%), and apply a **time-stop** (exit if the thesis hasn't paid within ~5 days).
3. **Stealth-accumulation detection** — a dormant, low-turnover stock with sound fundamentals shows an
   unexplained limit-up on a volume spike, consistent with a large holder quietly accumulating; scale
   in gradually (5% → add) to spread risk where there is no public catalyst.

> **Framing discipline (important — keep this in the doc and the UI copy):** describe trade type 3 as
> **detecting abnormal institutional accumulation footprints**, never as coordinating with or endorsing
> market manipulation. The tool *observes and reacts to* unusual volume/price behavior; it does not
> facilitate it. This distinction is both accurate and prudent given A-share regulatory sensitivity.

### New metrics the engine must support for these templates

The current schema covers `pct_change`, `turnover_rate`, `rsi`, `ma_cross`, `volume`. The framework
needs a few additions (implement in `indicators.py`; document each clearly):

| Metric | Meaning | Notes for implementation |
|---|---|---|
| `consecutive_limit_ups` | count of consecutive prior days closing at limit-up | board-aware (±10/20/5%); see §6 |
| `turnover_rate_Nd` | turnover rate sustained over the last N days | parameterize N; used for both entry interest and exit distribution |
| `volume_vs_avg` | today's volume ÷ trailing N-day avg volume | flags the "unexplained volume spike" on dormant names |
| `range_compression` | recent intraday range vs. historical (tight-range detection) | proxy for a controlled/accumulating stock |
| `gap_open_pct` | open vs. prior close, % | the call-auction outflow signal (e.g. a sharp negative open after a rally) |
| `days_since_limit_up` | bars since the last limit-up | for staged-entry and exit timing |

> **Call-auction signals are a known data limitation.** Free EOD/15-min-delayed sources (AKShare) do
> **not** reliably expose intraday pre-market call-auction microstructure (e.g. "price surges in the
> last 6 seconds" or "rigid drop in the final seconds"). Do NOT fake these. Approximate what you can
> from daily data — `gap_open_pct` is the honest EOD proxy for an auction-strength read — and clearly
> label in the UI that true call-auction signals require an intraday data source (a documented future
> upgrade, not an MVP feature). Honesty about what the data can and cannot test is part of the product.

### The template specs (encode these in templates.json)

These are **starting points to be backtested and refined**, not validated rules. Each should carry a
`status: "unvalidated"` flag and a UI note: *"Framework template — a hypothesis to test, not a proven
edge. Backtest it before trusting it."*

- **T1 · Momentum/rumor entry** —
  `mode=backtest, entry=[{consecutive_limit_ups, >=, 2}, {turnover_rate, >, 20}], exit=[{turnover_rate_Nd(2), >, 30}], holding_period=4`
  *rationale:* ride confirmed momentum; exit when two days of >30% turnover signal distribution.
- **T1-exit-test · "sell one day earlier"** — same as T1 but `exit` fires on the **first** day of
  >30% turnover instead of the second. *rationale:* directly backtests the GCL reflection ("I should
  have sold a day earlier") — does exiting earlier actually improve the average outcome, or just
  reduce variance?
- **T2 · Macro-thesis entry with time-stop** —
  `mode=backtest, entry=[{volume_vs_avg, >, 1.5}], exit=[{gap_open_pct, <, -3}], holding_period=5`
  *rationale:* enter on early accumulation under a macro catalyst; hard time-stop at 5 days; exit on a
  sharp negative open (the North Copper -4.57% open signal, proxied by `gap_open_pct`).
- **T3 · Stealth-accumulation screen** —
  `mode=screen, conditions=[{consecutive_limit_ups, >=, 1}, {volume_vs_avg, >, 3}, {turnover_rate, <, 5, window:20}, {range_compression, >, threshold}], logic=AND`
  *rationale:* a dormant, tight-range, low-turnover name that suddenly limit-ups on a volume spike —
  the "unexplained limit-up" footprint. A **screen**, not a backtest: it surfaces candidates to
  examine, deliberately not a buy signal.
- **T3-exit-test · "false-breakdown filter"** — a backtest variant comparing a naive exit (sell on a
  big down-open) against a **volume-confirmed** exit (only exit if the down-move comes on *high*
  volume; ignore low-volume drops as possible shakeouts). *rationale:* tests the Yabo reflection
  directly — "I sold too early on a low-volume retreat and missed 25%." Does the volume filter recover
  that?

These five templates turn the three case-study reflections into concrete, falsifiable backtests. That
is the single most valuable thing this tool can do for the framework's author.

---

## 5. Core user flow (build the UX around this)

1. **Input.** A prominent text box: *"Describe a signal or strategy to test."* Plus a market toggle
   (China / US, China default) and a mode toggle (Screen now / Backtest). Show 3–4 example directives
   as clickable chips to teach the interaction. **Also include a "Load a framework template"
   dropdown** (the §4a library): selecting one populates the input/spec with that template's
   `SignalSpec` and shows its `rationale` + the "unvalidated hypothesis — backtest before trusting"
   note. A loaded template is fully editable before running, exactly like a typed directive.
2. **Parse + confirm.** DeepSeek turns the directive into a `SignalSpec`. Render it back as an editable
   form/JSON so the user sees and can correct the interpretation. ("Here's how I read that — adjust
   anything.")
3. **Run.** The engine executes against cached data: screens run as a pandas filter (instant);
   backtests run through backtesting.py per symbol. Show a progress indicator for backtests over many
   symbols or parameter sweeps.
4. **Results.** Lead with visuals, not raw tables — see §7 for the full visualization spec.
   - *Screen mode:* a row of summary stat cards (count of matches, avg metric values) above a
     sortable results table, plus a distribution chart of the key metric across matches.
   - *Backtest mode:* summary stat cards (total return, win rate, Sharpe, max drawdown, trades,
     avg holding period) → equity curve vs. benchmark → drawdown chart → parameter-sweep heatmap →
     a sample symbol's price chart with entry/exit markers.
5. **News/context layer.** For the top results (or the backtested symbol), fetch recent
   news + sentiment and have DeepSeek write a short, neutral interpretation: what the signal found,
   how it performed, and what current news context surrounds these names. Always end this readout
   with the not-advice framing.
6. **Save / iterate.** Let the user save a spec to revisit, tweak a parameter, and re-run. Fast
   iteration is the whole point — make "tweak and re-run" a one-click loop.

---

## 6. Engine implementation notes

- **Limit-up detection by board.** A-share daily price limits differ: ±10% main board (Shanghai/
  Shenzhen), ±20% STAR Market (688xxx) and ChiNext (300xxx), ±5% for ST stocks. The engine should
  detect the board from the ticker prefix and apply the correct limit threshold when the user's
  directive references "limit-up." Document this clearly.
- **Use cached, adjusted prices.** Pull forward/backward-adjusted prices (AKShare `adjust="hfq"` or
  `"qfq"`) for backtests so splits/dividends don't create false signals. Make the adjustment method
  a setting; default to `hfq` (后复权) for backtests.
- **Engine patterns (matches §2's split).**
  - *Screening* is pure pandas: compute the referenced metrics/indicators on each symbol's cached
    DataFrame, evaluate the spec's conditions as boolean masks over the latest bar, combine with the
    spec's AND/OR logic, and return the matching symbols. No backtest engine involved.
  - *Backtesting* uses backtesting.py per symbol: translate the spec's entry/exit conditions into a
    `Strategy` subclass's `next()` logic, run `Backtest(df, MyStrategy, cash=..., commission=...)`,
    and read `stats` for the metrics panel. For parameter ranges (e.g. "MA window from 5 to 50"), use
    `Backtest.optimize(...)` which returns the grid of results that feeds the §7 heatmap.
- **Keep the spec→engine mapping in one place.** Write a single translator that turns a `SignalSpec`
  into either the pandas mask (screen) or the Strategy class (backtest). This is the heart of the
  flexibility; keep it readable and well-commented so you can extend it with new metrics later.
- **Framework metrics (§4a) live in `indicators.py`.** Implement `consecutive_limit_ups`,
  `turnover_rate_Nd`, `volume_vs_avg`, `range_compression`, `gap_open_pct`, and `days_since_limit_up`
  as ordinary indicator functions over the cached daily DataFrame, so the §4a templates run through
  the exact same translator as any user directive. Do NOT special-case them in the engine.
- **Backtest honesty.** Include transaction cost + slippage parameters (default to realistic A-share
  values: ~0.1% commission, stamp duty on sells). Never present a frictionless backtest as if real.
  Surface assumptions in the stats panel.
- **Guard against lookahead bias.** Signals at bar *t* must only use data available at *t*. Add a test
  that fails if any indicator leaks future data.

---

## 7. Visualization & graphing (the output is mostly visual)

The quant-curious user reads charts faster than tables. Every results view should **lead with a
visual** and keep raw numbers as supporting detail. Build all of these; they are core, not optional.

### Charting library
- **Follow the §2 charting priority order — do not default to a heavy library.** In short:
  1. **backtesting.py's built-in plot** gives you the equity curve, trade markers on price, and the
     optimization **heatmap** for free (interactive Bokeh) — use it for those.
  2. **Streamlit native** (`st.line_chart`, `st.bar_chart`, `st.metric`) for stat cards and simple
     distributions — zero extra dependencies.
  3. **Plotly** only for the few visuals the first two can't do cleanly (e.g. a custom diverging
     sentiment bar).
- Keep a consistent visual language: one color for the strategy, a muted/dashed line for any
  benchmark, green for gains, red for losses/drawdown. Round every displayed number sensibly
  (integers for counts, 1–2 decimals for percentages and ratios).
- Make charts theme-aware (light/dark) and responsive to container width.

### Backtest-mode visuals (build in this order, top to bottom)
1. **Summary stat cards** — a horizontal row of cards: total return, win rate, Sharpe ratio, max
   drawdown, number of trades, avg holding period. Color the return card green/red by sign. These are
   the at-a-glance verdict.
2. **Equity curve vs. benchmark** — the strategy's cumulative return over the test period as a filled
   line, overlaid with a benchmark (CSI 300 for CN, S&P 500 for US) as a muted dashed line so the user
   sees whether the signal actually beat buy-and-hold. This is the single most important chart.
3. **Drawdown chart** — an underwater plot beneath the equity curve showing peak-to-trough declines,
   so risk is visible, not just return. Shade the area red.
4. **Parameter-sweep heatmap** — when the directive includes a parameter range (e.g. "MA window from
   5 to 50"), render a 2-D heatmap where each cell is one parameter combination colored by its result
   (return or Sharpe), darker = better. backtesting.py's `optimize()` returns this grid directly, and
   it can plot the heatmap for you — a genuine differentiator, so expose it prominently. If you build
   a custom version, let the user click a cell to load that combination.
5. **Sample trade chart** — a candlestick/price chart for one representative matched symbol with
   entry markers (green ▲) and exit markers (red ▼) plotted on the bars, plus any referenced
   indicator (e.g. the two moving averages) overlaid, so the user can eyeball whether the rule fired
   where they expected.
6. **Trade distribution** — a histogram of per-trade returns, so the user sees whether results come
   from many small edges or a few outliers (a key honesty check for semi-pro users).

### Screen-mode visuals
1. **Summary stat cards** — number of matches, plus average/median of the key metric across matches.
2. **Sortable results table** — matched stocks with the relevant metric columns; let the user sort and
   click a row to open that symbol's price chart.
3. **Metric distribution** — a histogram or bar chart of the screening metric (e.g. turnover-rate
   distribution among limit-up names) so the user sees where the matches cluster.
4. **Optional sector/board breakdown** — a small bar chart of matches by board (主板 / 创业板 / 科创板)
   or sector, useful for spotting concentration.

### News / sentiment-layer visuals
1. **Sentiment gauge or bar** — per-symbol sentiment score from the news API as a simple
   diverging bar (negative ↔ positive) next to each name.
2. **News-volume sparkline** — recent article count over time for the symbol, so a spike in coverage
   is visible at a glance. Keep these compact; they accent the text readout, they don't replace it.

### Implementation notes
- In Streamlit: stat cards via `st.columns()` + `st.metric()`; equity curve / trade chart / heatmap
  via backtesting.py's returned figures; simple distributions via `st.line_chart` / `st.bar_chart`;
  reach for `st.plotly_chart` only for the rare custom visual. This keeps the whole results view to
  very little frontend code.
- Generate charts from the **cached** data path — never trigger live API calls just to draw a chart.
- Provide a download/export button for key figures so users can save findings.
- Empty states matter: if a screen returns zero matches or a backtest has too few trades to be
  meaningful, show a clear message instead of an empty chart, and say so plainly.

---

## 8. Branding & language (research-backed positioning)

The competitive field (Trade Ideas ~$127–254/mo, Finviz Elite ~$40/mo US-only, TrendSpider ~$39/mo)
is crowded with **preset-algorithm, "AI picks winners" black boxes**. The genuinely underdeveloped
lane is **natural-language signal definition + transparent testing + live-news interpretation**,
especially for the **China A-share** market which the Western tools largely ignore. Position there.

**Positioning statement (use as north star):**
> *"Describe any signal in plain language, test it rigorously against real market history, and see it
> in the context of today's news — built for the China A-share market. A place to test your thinking,
> not a black box that thinks for you."*

**Naming direction.** Lean into the *lab / hypothesis / proving-ground* vocabulary (signals you test),
NOT the saturated *alpha / oracle / genie / picks* vocabulary (predictions you trust). Candidate names
(verify trademark/domain availability before committing — do not assume any are free):
- **Monkey** (clear, on-strategy; used as the working title here)
- **Hypothesis** / **Hypothesize**
- **Backtest Bench** / **The Bench**
- **Litmus** (signal → test → result)
- **量验 (Liàngyàn)** or a bilingual mark, given the China focus — "quant + verify." Worth testing
  with target users since the primary market is Chinese.

**In-app language rules:**
- Use **testing/research verbs**: "test," "screen," "explore," "what if," "evaluate." Avoid
  "guaranteed," "winning," "best stocks," "you should buy."
- Frame outputs as **findings about a hypothesis**, not recommendations. "This signal would have…"
  not "Buy these."
- Keep a persistent, plain footer: *"Monkey is a research tool for testing ideas. Nothing here is
  financial advice. Past performance does not predict future results."*
- For the Chinese market, ensure metric labels are bilingual where helpful
  (Turnover Rate / 换手率, % Change / 涨跌幅) — the quant-curious CN user will expect the native terms.

---

## 9. Setup steps (put these in README.md, written for a read-and-tweak coder)

1. `python -m venv venv && source venv/bin/activate` (Windows: `venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill keys:
   - `DEEPSEEK_API_KEY` (required — get from platform.deepseek.com; new accounts get free trial tokens, no card needed)
   - `TWELVEDATA_API_KEY` (only if using US market — free signup)
   - `MARKETAUX_API_KEY` (only if using news layer — free signup)
   - `TUSHARE_TOKEN` (optional, leave blank to skip)
   - AKShare needs NO key for core endpoints.
4. First run will warm the cache (downloads recent A-share history). This may take a few minutes.
5. `streamlit run app.py` → opens at `localhost:8501`.

`.env.example` should list every key with a comment on where to get it and whether it's required.

**Minimal `requirements.txt` (China-only happy path — keep it this short):**
```
streamlit
akshare
backtesting
pandas
pandas-ta
pyarrow
openai          # used as the DeepSeek client (OpenAI-compatible)
python-dotenv
plotly          # only needed for the few custom visuals; safe to keep
```
Add `tushare` only if enabling the optional provider, and `requests` is pulled in transitively. Pin
exact versions once the app runs cleanly (especially `akshare`, given the scraping-fragility note in
§2). That is the entire dependency surface — no database driver, no web framework, no Numba.

---

## 10. Build phases (do these in order; ship each before the next)

**Phase 1 — Skeleton + China data.** Provider abstraction, AKShare CN provider with column
normalization, parquet cache, and a hardcoded screen ("limit-up + turnover > X") proving the
data path end to end. No LLM, no UI yet — a CLI or notebook is fine. Verify the numbers against a
known stock manually.

**Phase 2 — Engine.** pandas screening (boolean masks over the latest bar) and backtesting.py
per-symbol backtests against cached data. Implement the `SignalSpec` schema and the single translator
(spec → pandas mask or Strategy class). Add board-aware limit-up logic, adjusted prices,
costs/slippage, and the lookahead-bias test. **Also implement the §4a framework metrics** in
`indicators.py` (`consecutive_limit_ups`, `turnover_rate_Nd`, `volume_vs_avg`, `range_compression`,
`gap_open_pct`, `days_since_limit_up`) and verify each against a known stock by hand.

**Phase 2a — Framework template library.** Encode the §4a templates in `templates.py`/`templates.json`
with their `description`, `rationale`, and `status: "unvalidated"` fields. Confirm each loads, parses,
and runs through the same translator as a typed directive. This is small but high-value: it's what
lets the framework's reflections (GCL "sell a day earlier", Yabo "false-breakdown filter") become
runnable backtests.

**Phase 3 — Natural language.** `nl_parser.py`: DeepSeek (`deepseek-v4-flash`, JSON mode) turns a
directive into a validated `SignalSpec`. Round-trip it: directive → spec → editable → run. Handle
parse failures gracefully (show the user what was ambiguous).

**Phase 4 — UI + visualizations.** Streamlit front end implementing the §5 flow: input box,
**framework-template dropdown (§4a)**, spec confirm/edit, then the §7 visualization stack (stat cards,
equity curve vs. benchmark, drawdown, parameter-sweep heatmap, sample trade chart) and save/iterate
loop. Apply the §8 language rules. Build the backtest equity curve and stat cards first (highest
value), then the heatmap, then the remaining charts. Use backtesting.py's built-in plots +
Streamlit-native charts per the §7 priority order; reach for Plotly only where needed.

**Phase 5 — News + interpretation.** `news.py` (Marketaux + AKShare) and `interpreter.py` (DeepSeek
reads results + news → neutral readout with not-advice framing). Add the §7 sentiment/news visuals
(sentiment bar, news-volume sparkline) alongside the text readout.

**Phase 6 — US market + polish.** Add the Twelve Data US provider behind the same abstraction.
Tighten error handling, caching, empty-states, and the bilingual labels.

**Out of scope for MVP (note as future work, do not build):** real-time streaming, user accounts /
auth, payments, automated trade execution, broker integration, React migration, Tushare-paid tiers.

---

## 11. Quality bar / definition of done

- A user can type a plain-language directive, see it correctly parsed into an editable spec, run a
  screen or backtest on real A-share data, and read a transparent stats panel — without touching code.
- The §4a framework templates load from the dropdown, parse into editable specs, and run through the
  same engine as a typed directive — each clearly marked as an unvalidated hypothesis to be tested.
- Backtest results lead with visuals: stat cards, an equity curve vs. benchmark, a drawdown chart,
  and (when a parameter range is given) a sweep heatmap — all interactive and theme-aware.
- Backtest stats include costs and pass the no-lookahead test.
- All AKShare calls are cached and fail soft (clear message, never a raw stack trace).
- The not-financial-advice framing is visible on every results view.
- README lets a read-and-tweak coder go from clone to running app in under 15 minutes.
- No paid *subscription* is required to run the China-only happy path. The only runtime cost is
  DeepSeek API usage, which is pay-per-token and very small (cents-scale for typical MVP use), and
  new DeepSeek accounts include free trial tokens — so early development can run at effectively $0.
