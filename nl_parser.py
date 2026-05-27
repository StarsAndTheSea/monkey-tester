"""
Natural language → SignalSpec parser (Phase 3).

Sends a plain-English trading rule to DeepSeek (JSON mode) and returns a
validated SignalSpec plus notes about any assumptions or ambiguities.

Usage:
    from nl_parser import parse_nl

    result = parse_nl("enter when volume is 3× normal and pct change > 5%")
    if result["spec"]:
        spec  = result["spec"]    # validated SignalSpec, ready for run_backtest()
        notes = result["notes"]   # what was assumed / flagged
    else:
        print(result["error"])

Requires DEEPSEEK_API_KEY in .env.
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from core import Condition, SignalSpec

load_dotenv()

_API_KEY  = os.getenv("DEEPSEEK_API_KEY", "")
_BASE_URL = "https://api.deepseek.com"
_MODEL    = "deepseek-v4-flash"   # DeepSeek V4 Flash non-thinking, JSON mode

# ---------------------------------------------------------------------------
# Metric catalogue — mirrors _METRICS in app.py exactly
# ---------------------------------------------------------------------------

_METRICS: dict[str, str] = {
    "pct_change":            "Daily price change %",
    "volume_vs_avg20":       "Volume ÷ 20-day average volume  (1.0 = average, 3.0 = 3× average)",
    "turnover_rate":         "Turnover rate % — today",
    "turnover_rate_5d":      "Turnover rate % — 5-day average",
    "turnover_rate_10d":     "Turnover rate % — 10-day average",
    "consecutive_limit_ups": "Number of consecutive limit-up days ending today",
    "range_compression20":   "Today's high-low range ÷ 20-day average range  (< 1 = compressed coil)",
    "gap_open_pct":          "Gap open % vs previous close  (negative = gap down)",
    "days_since_limit_up":   "Calendar days since the last limit-up day",
    "close":                 "Closing price (CNY)",
    "volume":                "Volume in shares",
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_METRIC_LINES = "\n".join(f"  {k}: {v}" for k, v in _METRICS.items())

_SYSTEM_PROMPT = f"""You are a trading strategy parser for a China A-share market research tool.

Convert the user's plain-English rule into a JSON SignalSpec. Use ONLY the metrics below.

AVAILABLE METRICS
{_METRIC_LINES}

VALID OPERATORS: >=  >  <=  <  ==

OUTPUT FORMAT — return exactly this JSON structure, nothing else:
{{
  "mode":           "screen" or "backtest",
  "market":         "CN",
  "universe":       ["600519"] or another list of 6-digit A-share codes, or "csi300",
  "logic":          "AND" or "OR",
  "conditions":     [...],          // screen mode — list of Condition objects
  "entry":          [...],          // backtest mode — list of Condition objects
  "exit":           [...],          // backtest mode — optional, omit or use []
  "holding_period": null or integer, // max bars to hold before forced exit
  "notes":          "string"        // your assumptions and any ambiguities — always include
}}

Condition object:
{{ "metric": "<one of the available metrics>", "operator": "<operator>", "value": <number> }}

RULES
- Use mode="screen" when the user wants to find stocks matching conditions today.
- Use mode="backtest" when the user mentions entries, exits, returns, or testing over history.
- If no symbol is mentioned, default universe to ["600519"].
- Always set market to "CN".
- If something is ambiguous, make a reasonable assumption and describe it in "notes".
- If the user mentions something that has no matching metric, omit it and note it.
- Never invent metric names. Never use metric names not in the list above.
- Return valid JSON only. Do not include markdown fences or commentary outside the JSON.
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_nl(text: str) -> dict:
    """Parse a plain-English trading rule into a validated SignalSpec.

    Returns a dict:
        spec  — validated SignalSpec instance, or None on failure
        notes — string: assumptions / ambiguities DeepSeek flagged
        error — string: present only on failure, describes what went wrong
        raw   — the raw JSON string from DeepSeek (useful for debugging)
    """
    if not _API_KEY:
        return {
            "spec":  None,
            "notes": "",
            "error": "DEEPSEEK_API_KEY is not set in .env — add it and restart.",
        }

    # --- Call DeepSeek ---
    try:
        client   = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": text.strip()},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content
    except Exception as e:
        return {"spec": None, "notes": "", "error": f"DeepSeek API error: {e}"}

    # --- Parse JSON ---
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return {
            "spec":  None,
            "notes": "",
            "error": f"DeepSeek returned malformed JSON: {e}",
            "raw":   raw,
        }

    notes = str(data.pop("notes", "")).strip()

    # Ensure required fields have defaults (DeepSeek omits conditions in backtest mode)
    data.setdefault("conditions", [])
    data.setdefault("entry", [])
    data.setdefault("exit", [])

    # Convert condition dicts to Condition objects
    for field in ("conditions", "entry", "exit"):
        raw_conds = data.get(field)
        if raw_conds:
            try:
                data[field] = [Condition(**c) for c in raw_conds]
            except Exception as e:
                return {
                    "spec":  None,
                    "notes": notes,
                    "error": f"DeepSeek produced an invalid condition in '{field}': {e}",
                    "raw":   raw,
                }

    # Validate against SignalSpec schema
    try:
        spec = SignalSpec.model_validate(data)
    except Exception as e:
        return {
            "spec":  None,
            "notes": notes,
            "error": f"DeepSeek output did not match SignalSpec schema: {e}",
            "raw":   raw,
        }

    return {"spec": spec, "notes": notes, "raw": raw}
