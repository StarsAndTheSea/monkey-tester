# Signal Lab — Build Guide (Step-by-Step)

> **What this is.** A companion to `SIGNAL_LAB_MVP.md`. The spec says *what* to build; this guide
> breaks every phase into ordered, concrete steps you can follow and return to at any time. Each step
> says what to do, what to watch out for, and how you know it's done.
>
> **How to use it.** Work top to bottom. Finish and verify a phase before starting the next — a
> half-built data layer will silently poison everything downstream. When handing work to Claude Code,
> paste the relevant phase from here *plus* the matching section from the spec.
>
> **Time budget (you, driving Claude Code, reading/tweaking Python):**
> - Phases 0–2a (minimum useful path, runs from a notebook): **~15–25 hrs**
> - Through Phase 4 (NL parsing + Streamlit UI): **~30–50 hrs**
> - All six phases polished: **~60–90 hrs**
> The data layer (Phase 1–2) and *verifying* it are the longest poles, not the UI.
>
> **The wildcard.** AKShare is scraping-based. If an upstream endpoint breaks the day you build, you
> lose unpredictable time. Build the caching and fail-soft handling early so a broken endpoint never
> blocks you on data you already pulled.

---

## Phase 0 — Setup & environment (≈1–2 hrs)

Goal: a clean project that runs, before any real logic exists.

1. **Create the project folder and a virtual environment.**
   `python -m venv venv` then activate it (`source venv/bin/activate`, or `venv\Scripts\activate` on
   Windows). Confirm `python --version` is 3.13+.
2. **Start minimal.** Per the spec's self-containment note, don't build the full module tree yet.
   Begin with one `app.py` and one `core.py`; split into the `core/` and `data/` tree only when a file
   gets unwieldy. Fewer files = easier to run and reason about early on.
3. **Create `requirements.txt`** with the China-only happy path from the spec §9:
   `streamlit, akshare, backtesting, pandas, pandas-ta, pyarrow, openai, python-dotenv, plotly`.
   Install with `pip install -r requirements.txt`. Do **not** pin versions yet — pin them at the end
   of Phase 1 once it runs cleanly (especially `akshare`).
4. **Create `.env.example` and `.env`.** List every key with a comment on where to get it and whether
   it's required: `DEEPSEEK_API_KEY` (required, from platform.deepseek.com), `TWELVEDATA_API_KEY`
   (US only), `MARKETAUX_API_KEY` (news only), `TUSHARE_TOKEN` (optional). Note AKShare needs no key.
   Add `.env` to `.gitignore` immediately — never commit real keys.
5. **Create a `cache/` folder and gitignore it.** This holds downloaded price data.
6. **Sanity check.** A one-line `streamlit run app.py` that shows "Hello" should open at
   `localhost:8501`. If that works, the environment is sound.

**Done when:** the venv installs everything without error, a placeholder Streamlit app opens locally,
and `.env`/`cache/` are gitignored.

---

## Phase 1 — Skeleton + China data (≈5–9 hrs; the real foundation)

Goal: prove the data path end to end with no LLM and no UI. A notebook or CLI is fine here.

1. **Define the provider abstraction.** Create `MarketDataProvider` (an abstract base) with
   `get_daily_history(symbol, start, end)`, `list_symbols(market)`, `get_snapshot(symbols)`. Every
   provider must return a **normalized DataFrame** with standard English columns:
   `date, open, high, close, low, volume, pct_change, turnover_rate`.
2. **Build the AKShare CN provider.** It fetches A-share daily history and renames the Chinese columns
   to the normalized English ones: `涨跌幅 → pct_change`, `换手率 → turnover_rate`, etc. This renaming
   is what later lets the same signal run on either market.
3. **Pull adjusted prices.** Use AKShare's `adjust="hfq"` (后复权 / back-adjusted) by default for
   backtests so splits/dividends don't create false signals. Make the adjustment method a setting.
4. **Wrap every AKShare call in try/except.** On failure, return a clear "data source temporarily
   unavailable" message — never a raw stack trace. This is non-negotiable given the scraping fragility.
5. **Build the parquet cache.** Write fetched history to `cache/` as parquet (one file per symbol or
   per market) using `pyarrow`. On every fetch: check the cache first, only hit AKShare for what's
   missing. This is what keeps you on free tiers and survives upstream outages.
6. **Write a cache-warming routine.** A function that downloads recent history for a batch of symbols
   into the cache. Expect the first run to take a few minutes. Don't warm the entire market yet — a
   few hundred liquid names is plenty for development.
7. **Prove it with a hardcoded screen.** Write a throwaway filter: "stocks that hit limit-up yesterday
   AND turnover rate > X." This isn't the real engine — it's a proof that the data path works.
8. **Verify against reality by hand.** Pick one stock you know, pull its recent data, and manually
   confirm the `pct_change` and `turnover_rate` numbers match what you'd see on your broker/quote
   source. **Do not skip this.** Every downstream result depends on these numbers being right.

**Watch out for:** AKShare endpoint quirks (column names and formats change between functions);
timezone/date handling; symbols needing exchange prefixes; the difference between adjusted and raw
prices. Cache aggressively so you're not re-pulling while debugging.

**Done when:** you can call the provider for any A-share symbol, get a normalized cached DataFrame,
run the hardcoded screen, and you've hand-verified the numbers against a known stock.

---

## Phase 2 — The engine (≈5–9 hrs; the heart of the app)

Goal: the generic screen + backtest engine, plus your framework's custom metrics.

1. **Implement the `SignalSpec` schema** (Pydantic) exactly as in spec §4: `mode, market, universe,
   conditions, logic`, plus backtest fields `entry, exit, holding_period, date_range`. Each
   `Condition` has `metric, operator, value, window, timeframe`.
2. **Write the single spec→engine translator.** One function that turns a `SignalSpec` into either a
   pandas boolean mask (screen) or a `backtesting.py` `Strategy` subclass (backtest). Keep it readable
   and well-commented — this one function is the source of the app's flexibility.
3. **Build screening as pure pandas.** Compute the referenced metrics on each symbol's cached
   DataFrame, evaluate each condition as a boolean mask over the **latest bar**, combine with the
   spec's AND/OR logic, return matching symbols. No backtest engine involved in screening.
4. **Build backtesting per symbol.** Translate entry/exit conditions into a `Strategy.next()`, run
   `Backtest(df, MyStrategy, cash=..., commission=...)`, read `stats` for the metrics panel. For
   parameter ranges, use `Backtest.optimize(...)` — it returns the grid that later feeds the heatmap.
5. **Add board-aware limit-up logic.** Detect the board from the ticker prefix and apply the right
   daily limit: ±10% main board, ±20% STAR (688xxx) and ChiNext (300xxx), ±5% for ST stocks. Use
   `pct_change >= 9.9` (not exactly 10) to catch limit-ups robustly.
6. **Add realistic costs.** Default to ~0.1% commission plus A-share stamp duty on sells. Never
   present a frictionless backtest as if it were real; surface these assumptions in the stats output.
7. **Guard against lookahead bias.** Signals at bar *t* must only use data available at *t*. Write a
   test that deliberately fails if any indicator leaks future data. This is the most important
   correctness check in the whole project — a leaky backtest will tell you a losing rule is a winner.
8. **Implement the six framework metrics** (spec §4a) in `indicators.py`, each as an ordinary
   indicator function over the cached daily DataFrame:
   - `consecutive_limit_ups` — count of consecutive prior days closing at limit-up (board-aware).
   - `turnover_rate_Nd` — turnover sustained over the last N days (parameterize N).
   - `volume_vs_avg` — today's volume ÷ trailing N-day average volume.
   - `range_compression` — recent intraday range vs. historical (tight-range detection).
   - `gap_open_pct` — open vs. prior close, as a %. (Your EOD proxy for call-auction strength.)
   - `days_since_limit_up` — bars since the last limit-up.
   Do **not** special-case these in the engine — they run through the same translator as any metric.
9. **Hand-verify each metric** against a known stock, same as Phase 1. A wrong metric silently
   corrupts every template that uses it.

**Watch out for:** lookahead bias (test it explicitly); off-by-one errors in "consecutive" and "days
since" counts; `backtesting.py` being single-asset by design (that's fine — backtests are per-symbol).

**Done when:** you can hand the engine a `SignalSpec` (in code) and get back either a correct screen
result or a backtest stats panel with costs applied, the lookahead test passes, and all six framework
metrics are hand-verified.

---

## Phase 2a — Framework template library (≈2–4 hrs; your highest-value step)

Goal: turn your three trade types and case-study reflections into runnable, editable backtests. This
is the step that directly serves *refining your method* — prioritize it.

1. **Create `templates.py` (or `templates.json`).** Encode the five §4a templates as `SignalSpec`
   instances, each with a `description`, a `rationale`, and `status: "unvalidated"`.
2. **Encode the five templates** (from spec §4a — copy the exact condition values from there):
   - **T1 · Momentum/rumor entry** — enter on confirmed consecutive limit-ups + high turnover, exit on
     the *second* day of >30% turnover, hold ~4 days.
   - **T1-exit-test · "sell one day earlier"** — same as T1 but exit on the *first* day of >30%
     turnover. (Tests your GCL reflection: does exiting earlier actually help, or just cut variance?)
   - **T2 · Macro-thesis entry with time-stop** — enter on early volume surge, hard 5-day time-stop,
     exit on a sharp negative open (`gap_open_pct < -3`). (Your North Copper setup.)
   - **T3 · Stealth-accumulation screen** — a *screen*, not a backtest: dormant + low-turnover +
     sudden volume-spike limit-up + tight range. Surfaces candidates to examine, not a buy signal.
   - **T3-exit-test · "false-breakdown filter"** — compares a naive down-open exit against a
     volume-confirmed exit (only exit if the drop comes on high volume). (Tests your Yabo reflection:
     does ignoring low-volume shakeouts recover the 25% you left behind?)
3. **Confirm each template loads, parses, and runs** through the same translator as a typed spec.
4. **Run them against your case-study windows first.** Before anything else, backtest T1 over the GCL
   window, T2 over North Copper, T3 over Yabo. Two purposes: (a) a smoke test that the templates are
   wired correctly, and (b) your first real look at whether the rules generalize beyond the single
   trades you remember.
5. **Treat results as hypotheses, not verdicts.** Three trades in one hot tape is not an edge.
   Anything green here is a starting point to test over more history and other market regimes.

**Done when:** all five templates run from code/notebook, you've backtested them over your three
case-study windows, and the "sell earlier" and "false-breakdown" tests produce comparable numbers you
can actually reason about.

> **If your goal is purely to refine the method, you can stop here.** Phases 3–6 add convenience (plain
> language, a UI, news, US stocks) but not new analytical power. Run templates from a notebook, read
> the stats, tweak, re-run. That's the whole refinement loop and it works without any UI.

---

## Phase 3 — Natural language → spec (≈3–5 hrs)

Goal: type a directive in plain language, get back an editable `SignalSpec`.

1. **Set up the DeepSeek client.** It's OpenAI-compatible — use the `openai` SDK pointed at
   `https://api.deepseek.com`. Key from `DEEPSEEK_API_KEY`. **Verify current model names and pricing at
   api-docs.deepseek.com at build time** rather than trusting any hardcoded value.
2. **Write `nl_parser.py`.** Use `deepseek-v4-flash` in **non-thinking mode** with JSON mode
   (`response_format={"type": "json_object"}`) to turn a directive into a `SignalSpec`. Use the
   explicit `deepseek-v4-*` IDs only — the legacy `deepseek-chat`/`deepseek-reasoner` aliases are
   being deprecated and will break the app.
3. **Keep the system prompt identical across calls.** DeepSeek gives a steep cache discount on
   repeated input prefixes — a stable prompt cuts parsing cost dramatically at volume.
4. **Validate the LLM output against the Pydantic schema.** Never trust raw model JSON — parse it into
   `SignalSpec` and catch validation errors.
5. **Round-trip it.** directive → spec → (editable) → run. Show the user the parsed spec and let them
   correct it before running. Transparency is the product.
6. **Handle parse failures gracefully.** When the directive is ambiguous, show the user *what* was
   ambiguous rather than failing silently or guessing wildly.

**Done when:** a plain sentence like "stocks that hit limit-up yesterday with turnover over 10%"
parses into a correct, schema-valid, editable spec — and a deliberately vague sentence produces a
clear "here's what I couldn't pin down" message.

---

## Phase 4 — UI + visualizations (≈6–10 hrs)

Goal: the whole flow without touching code — input, confirm, run, read results visually.

1. **Build the input screen** (spec §5 step 1): a prominent "Describe a signal or strategy to test"
   text box, a market toggle (China default), a mode toggle (Screen / Backtest), 3–4 clickable example
   chips, and the **"Load a framework template" dropdown** (your §4a library). A loaded template
   populates the spec, shows its `rationale` and the "unvalidated hypothesis" note, and stays editable.
2. **Build the parse + confirm step.** Render the `SignalSpec` back as an editable form or JSON
   ("Here's how I read that — adjust anything") before running.
3. **Wire the run step.** Screens run as an instant pandas filter; backtests run per symbol through
   the engine. Show a progress indicator for backtests over many symbols or parameter sweeps. Always
   draw charts from **cached** data — never trigger live API calls just to render a chart.
4. **Build the backtest visuals in this priority order** (highest value first):
   1. **Summary stat cards** (`st.columns` + `st.metric`): total return, win rate, Sharpe, max
      drawdown, # trades, avg holding period. Color the return card green/red by sign.
   2. **Equity curve vs. benchmark** — strategy cumulative return vs. CSI 300 (muted dashed line). The
      single most important chart: it shows whether the signal actually beat buy-and-hold.
   3. **Drawdown chart** — underwater plot, shaded red, so risk is visible, not just return.
   4. **Parameter-sweep heatmap** — when the directive includes a range, render the grid from
      `optimize()`; darker = better. A genuine differentiator — expose it prominently.
   5. **Sample trade chart** — one matched symbol's price with entry (green ▲) / exit (red ▼) markers
      and any referenced indicator overlaid.
   6. **Trade distribution** — histogram of per-trade returns (honesty check: many small edges vs. a
      few outliers).
5. **Build the screen visuals:** summary stat cards (match count, avg/median of key metric), a
   sortable results table (click a row → that symbol's chart), and a metric distribution histogram.
   Optionally a board breakdown (主板 / 创业板 / 科创板).
6. **Use the charting priority order** (spec §7): backtesting.py's built-in Bokeh plots for equity
   curve / trade markers / heatmap; Streamlit-native for stat cards and simple distributions; Plotly
   only for the rare custom visual. Don't reach for a heavy library when a native chart does the job.
7. **Add the save / iterate loop.** Save a spec to revisit; make "tweak a parameter and re-run" a
   one-click action. Fast iteration is the whole point.
8. **Handle empty states.** Zero matches or too-few-trades-to-be-meaningful should show a clear
   message, not an empty chart.
9. **Apply the §8 language rules** (see Phase 6 polish too): testing/research verbs only, findings not
   recommendations, persistent not-financial-advice footer.

**Done when:** you can go from typing (or loading a template) → confirming the spec → running →
reading a visual results panel, entirely in the browser, with the save/re-run loop working.

---

## Phase 5 — News + interpretation (≈4–7 hrs; optional for refinement)

Goal: layer real-world context on top of the stats. Skippable if you only want to test rules.

1. **Build `news.py`.** Fetch recent news + sentiment from Marketaux (per-ticker sentiment, entity
   recognition) and supplement A-share-specific news from AKShare's news endpoints. Call these **on
   demand and infrequently** — they're the fresh interpretation layer, not the price path. Keep them
   on free tiers by not over-calling.
2. **Build `interpreter.py`.** Feed backtest results + fetched news to DeepSeek (Flash in thinking
   mode, or escalate to `deepseek-v4-pro` only if quality needs it — make it a config setting). It
   writes a short, **neutral** readout: what the signal found, how it performed, what current news
   surrounds the names. Always end with the not-advice framing.
3. **Add the news visuals:** a per-symbol diverging sentiment bar (negative ↔ positive) and a compact
   news-volume sparkline (article count over time). These accent the text readout; they don't replace
   it.
4. **Keep the interpretation honest.** It interprets; it never recommends. "This signal would have…"
   never "buy these."

**Done when:** for a backtested symbol or top screen results, you get a neutral plain-language readout
plus sentiment/news-volume visuals, always closing with the not-advice line.

---

## Phase 6 — US market + polish (≈4–8 hrs; optional)

Goal: a second market behind the same abstraction, plus production hardening.

1. **Add the Twelve Data US provider** behind the same `MarketDataProvider` interface (free tier: 800
   req/day). Because the abstraction normalizes columns, the same specs and engine should "just work"
   on US data with no rewrites — that's the test of whether your abstraction was clean.
2. **Tighten error handling** everywhere: every external call fails soft with a user-facing message.
3. **Harden caching** and empty states across all views.
4. **Add bilingual labels** for the CN user: Turnover Rate / 换手率, % Change / 涨跌幅, board names.
5. **Apply the full §8 branding/language pass:** lab/hypothesis vocabulary (not alpha/oracle/picks),
   findings-not-recommendations framing, persistent advice footer on every results view. (Verify any
   product name's trademark/domain availability before committing — don't assume any candidate is
   free.)
6. **Pin exact versions in `requirements.txt`**, especially `akshare`, now that everything runs
   cleanly.

**Done when:** US symbols run through the same flow, labels are bilingual where helpful, errors never
show stack traces, and the not-advice framing is visible on every results view.

---

## Things to keep true throughout (don't let these slip)

- **Cache first, always.** Never call a price API live to draw a chart or re-run a screen. The cache is
  what keeps the app free and resilient to AKShare breakage.
- **Fail soft.** Every external call wrapped so a broken endpoint shows a clear message, never a trace.
- **Verify numbers by hand** at every data/metric step. The entire point is to trust the results
  enough to change how you trade — a silent calculation bug defeats that completely.
- **Test for lookahead bias** and keep that test passing. A leaky backtest flatters bad rules.
- **Templates are unvalidated hypotheses,** labeled as such, until tested over real history across
  more than one market regime. Three winning trades is a starting point, not an edge.
- **Frame stealth-accumulation as detection, never partnership.** The tool observes abnormal
  institutional footprints; it does not facilitate manipulation. Keep this clean in code comments and
  UI copy alike.
- **Research tool, not financial advice** — visible on every results view, by design, not as an
  afterthought.

---

## Suggested order if you're short on time

If the goal is refining your method (not shipping a product), do **0 → 1 → 2 → 2a**, run your five
templates from a notebook, and stop. That's the ~15–25 hour minimum useful path and it answers your
real questions (does selling a day earlier help? does the false-breakdown filter recover the missed
gains?). Add Phase 3 (plain language) and Phase 4 (UI) only when notebook-driving gets tedious. Treat
Phases 5–6 as someday-maybe.
