# Monkey Tester

A factor-testing tool for the China A-share market. Build a condition set from statistical factors — volume, turnover rate, price change, technical indicators — then run it as a screen (which stocks match today?) or a backtest (how would this have performed?).

Results are transparent stats. No black-box signals, no buy/sell recommendations.

I made this app to test out some strategies using historical data of stocks. There are only a few buying and selling indicators for now. If you want more, tell me and I can code it into the app or you can manually code it into indicators.py

> **This is a hypothesis-testing tool, not a stock picker. Not financial advice.**

---

## What it does

**Signal Backtest tab**
- Build entry/exit conditions from a menu of metrics
- Backtest against any A-share symbol over a custom date range
- Results: return vs buy-and-hold, Sharpe ratio, max drawdown, win rate, equity curve, drawdown chart, monthly heatmap, price+trades chart, trade distribution
- Save and reload past runs from the sidebar
- Optional: describe your strategy in plain English and let AI fill the builder (requires DeepSeek API key)

**Factor Groups tab**
- Run multiple strategies side by side with different position sizes (¥ capital per trade)
- Compare risk/return across groups in one table
- Strict mode: hold until exit condition fires, no time-limit close

---

## Requirements

- Python 3.10+
- A China VPN or mainland China network connection to fetch new stock data
  (cached data works offline indefinitely)
- [DeepSeek API key](https://platform.deepseek.com/) — optional, for AI strategy parsing

---

## Installation

```bash
git clone https://github.com/StarsAndTheSea/monkey-tester.git
cd monkey-tester

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy the environment file and add your keys:

```bash
cp .env.example .env
# Edit .env and add DEEPSEEK_API_KEY if you want AI parsing
```

Run the app:

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Adding stock data

New stock data requires a connection to EastMoney (AKShare's backend), which is only accessible from mainland China or via a China VPN.

1. Connect your China VPN
2. In the app sidebar, enter a 6-digit A-share code (e.g. `000001`) and click **Warm cache**
3. Disconnect — cached data is stored locally as parquet files and works offline

Once cached, a symbol is available indefinitely without a VPN.

---

## Available metrics

| Metric | Description |
|---|---|
| `pct_change` | Daily price change % |
| `volume_vs_avg20` | Volume ÷ 20-day average (1.0 = average, 3.0 = 3× average) |
| `turnover_rate` | Turnover rate % today |
| `turnover_rate_5d` | Turnover rate % 5-day average |
| `turnover_rate_10d` | Turnover rate % 10-day average |
| `consecutive_limit_ups` | Number of consecutive limit-up days ending today |
| `range_compression20` | Today's high-low range ÷ 20-day average range |
| `gap_open_pct` | Gap open % vs previous close |
| `days_since_limit_up` | Calendar days since last limit-up day |
| `close` | Closing price (CNY) |
| `volume` | Volume in shares |

---

## Project structure

```
app.py              Streamlit UI — two tabs
core.py             Data provider, cache, SignalSpec schema, backtest engine
indicators.py       Computed metrics (volume ratios, limit-up streaks, etc.)
nl_parser.py        DeepSeek AI → validated strategy spec
templates.py        Five pre-built strategy templates
factors.py          Factor library models + lookahead bias checker
tests/              77 tests (pytest)
cache/              Local parquet price data (gitignored)
specs/              Saved backtest runs (gitignored)
```

---

## API keys

| Key | Required | Purpose |
|---|---|---|
| `DEEPSEEK_API_KEY` | Optional | AI strategy parser (plain English → condition builder) |
| `MARKETAUX_API_KEY` | Optional | English news headlines (future feature) |

AKShare requires no API key.

---

## Running tests

```bash
source .venv/bin/activate
python -m pytest tests/ -q
```

77 tests, no network required.

---

## Limitations

- **China A-shares only** for now. US market support is planned.
- **Single-asset backtests** — one symbol at a time, no portfolio simulation.
- **Data coverage** depends on your cache. Signals that use `volume_vs_avg20` or `turnover_rate` require data fetched via EastMoney; the Tencent fallback provides price data only.
- AKShare news coverage is reliable for ~12 months back.
