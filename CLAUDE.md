# Signal Lab — Claude Code Context

## What this is

A signal research tool for the China A-share market (US secondary). The user describes a signal or strategy in plain language; the app parses it into a structured `SignalSpec`, runs it against cached historical data (screen or backtest), returns transparent stats, and layers in news/sentiment for context.

This is a **hypothesis-testing tool**, not a stock picker, not financial advice. That framing is a design constraint, not a disclaimer — it must be visible on every results view.

## Runtime environment

- **Python 3.13.13** — the venv at `.venv/` is already created and ready. Activate with `source .venv/bin/activate`.
- **No separate web framework.** Streamlit is the only server. Do not add FastAPI, Flask, or uvicorn — a second process is out of scope for this MVP.
- **Entry point:** `streamlit run app.py` → opens at `localhost:8501`.
- **API keys:** stored in `.env` (never commit). Template at `.env.example`. Required key: `DEEPSEEK_API_KEY`. Optional: `TWELVEDATA_API_KEY`, `MARKETAUX_API_KEY`, `TUSHARE_TOKEN`. AKShare needs no key.

## Build phases — work in this order, finish each before starting the next

| Phase | Goal | Done when |
|---|---|---|
| **0** | Setup (venv, requirements, .env, cache/ folder, placeholder app) | `streamlit run app.py` opens; `.env` and `cache/` are gitignored |
| **1** | China data: AKShare provider, parquet cache, hardcoded screen | Call any A-share symbol → normalized cached DataFrame; hand-verify numbers against a known stock |
| **2** | Engine: `SignalSpec` schema (Pydantic), spec→engine translator, pandas screening, backtesting.py backtests, board-aware limit-up, costs, lookahead-bias test, six framework metrics in `indicators.py` | Hand a `SignalSpec` in code → correct screen or backtest stats; lookahead test passes; all six metrics hand-verified |
| **2a** | Framework template library: five `SignalSpec` templates in `templates.py`/`templates.json` with `description`, `rationale`, `status: "unvalidated"` | All five templates load, parse, and run through the same translator as any typed spec |
| **3** | NL parsing: `nl_parser.py` — DeepSeek (`deepseek-v4-flash`, JSON mode) → validated `SignalSpec` | A plain directive produces a correct editable spec; ambiguous input shows what was unclear |
| **4** | Streamlit UI + full visualization stack (stat cards, equity curve vs. benchmark, drawdown, heatmap, sample trade chart) | Full flow in browser: input → confirm spec → run → visual results → save/re-run |
| **5** | News + interpretation: `news.py` (Marketaux + AKShare), `interpreter.py` (DeepSeek neutral readout) | Neutral plain-language readout + sentiment/news visuals on top results |
| **6** | US market (Twelve Data provider) + polish, bilingual labels, pinned requirements | US symbols run through the same flow; no stack traces anywhere |

A **minimum useful path** for method refinement only: phases 0 → 1 → 2 → 2a, run templates from a notebook, stop there.

## Rules that must not slip — ever

**Cache first.** Never call a price API live to draw a chart, run a screen, or re-run a backtest. Pull once, write to `cache/` as parquet, read from cache on every subsequent request. This is what keeps the app on free tiers and resilient when AKShare breaks.

**Fail soft on every external API call.** Wrap every AKShare, Marketaux, Twelve Data, and DeepSeek call in `try/except`. On failure, return a clear user-facing message. Never let a raw stack trace reach the UI. This is non-negotiable — AKShare is scraping-based and will break.

**Test for lookahead bias and keep that test passing.** Signals at bar *t* must only use data available at *t*. A leaky indicator will make a losing rule look like a winner. Write the test in Phase 2; don't remove it.

**Hand-verify all metrics against a known stock.** At every data and indicator step, pull a stock you can cross-reference and confirm the numbers manually. A silent calculation bug corrupts every result downstream and defeats the entire point of building this.

**Framework templates are unvalidated hypotheses — label them as such.** Each template carries `status: "unvalidated"` and the UI note: *"Framework template — a hypothesis to test, not a proven edge. Backtest it before trusting it."* Three winning trades in one hot tape is a starting point, not an edge. Never harden these into the engine as special cases; they run through the same translator as any user directive.

## Key architecture decisions (don't reverse without reason)

- **Screening = pure pandas** (boolean mask over the latest bar of cached data). No backtest engine involved.
- **Backtesting = backtesting.py**, one symbol at a time. Single-asset by design — that's fine.
- **One spec→engine translator** turns a `SignalSpec` into either a pandas mask or a `Strategy` subclass. Keep it in one place and keep it readable.
- **DeepSeek via the `openai` SDK** pointed at `https://api.deepseek.com`. Use `deepseek-v4-flash` for parsing (non-thinking, JSON mode). Use explicit `deepseek-v4-*` IDs only — the legacy aliases are being deprecated.
- **File-based storage only.** Parquet for price cache, JSON or stdlib `sqlite3` for saved specs. No ORM, no database server.
- **Charting priority:** (1) backtesting.py's built-in Bokeh plots for equity curve / trade markers / heatmap, (2) Streamlit native for stat cards and simple distributions, (3) Plotly only where 1 and 2 don't suffice.
- **Stealth-accumulation framing:** always "detecting abnormal institutional accumulation footprints" — the tool observes and reacts to unusual volume/price behavior. Never frame it as coordinating with or endorsing manipulation. Keep this in code comments and UI copy.
