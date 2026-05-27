"""
Tests for nl_parser.py (Phase 3).

All DeepSeek API calls are mocked — no network required.
"""

from unittest.mock import MagicMock, patch

import pytest

from core import SignalSpec
from nl_parser import parse_nl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(json_str: str):
    """Build a fake openai ChatCompletion response containing json_str."""
    msg = MagicMock()
    msg.content = json_str
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _patch_openai(json_str: str):
    """Context manager: patch OpenAI so it returns json_str."""
    return patch(
        "nl_parser.OpenAI",
        return_value=MagicMock(
            chat=MagicMock(
                completions=MagicMock(
                    create=MagicMock(return_value=_mock_response(json_str))
                )
            )
        ),
    )


# ---------------------------------------------------------------------------
# Missing API key
# ---------------------------------------------------------------------------

def test_missing_api_key_returns_error(monkeypatch):
    monkeypatch.setattr("nl_parser._API_KEY", "")
    result = parse_nl("enter when volume is high")
    assert result["spec"] is None
    assert "DEEPSEEK_API_KEY" in result["error"]


# ---------------------------------------------------------------------------
# Happy path — screen spec
# ---------------------------------------------------------------------------

_SCREEN_JSON = """{
  "mode": "screen",
  "market": "CN",
  "universe": ["600519"],
  "logic": "AND",
  "conditions": [
    {"metric": "volume_vs_avg20", "operator": ">=", "value": 3.0},
    {"metric": "pct_change",      "operator": ">",  "value": 5.0}
  ],
  "entry": [],
  "exit": [],
  "holding_period": null,
  "notes": "Assumed CN market and 600519 as default universe."
}"""

def test_screen_spec_parsed_correctly(monkeypatch):
    monkeypatch.setattr("nl_parser._API_KEY", "sk-fake")
    with _patch_openai(_SCREEN_JSON):
        result = parse_nl("find stocks where volume is 3x normal and price is up over 5%")
    assert result["spec"] is not None
    assert isinstance(result["spec"], SignalSpec)
    assert result["spec"].mode == "screen"
    assert len(result["spec"].conditions) == 2
    assert result["notes"] != ""


# ---------------------------------------------------------------------------
# Happy path — backtest spec with holding period
# ---------------------------------------------------------------------------

_BACKTEST_JSON = """{
  "mode": "backtest",
  "market": "CN",
  "universe": ["600519"],
  "logic": "AND",
  "conditions": [],
  "entry": [
    {"metric": "consecutive_limit_ups", "operator": ">=", "value": 2},
    {"metric": "turnover_rate",         "operator": ">=", "value": 10.0}
  ],
  "exit": [
    {"metric": "turnover_rate_5d", "operator": ">", "value": 20.0}
  ],
  "holding_period": 4,
  "notes": "Holding period capped at 4 bars as specified."
}"""

def test_backtest_spec_parsed_correctly(monkeypatch):
    monkeypatch.setattr("nl_parser._API_KEY", "sk-fake")
    with _patch_openai(_BACKTEST_JSON):
        result = parse_nl("enter on two consecutive limit-ups with turnover > 10, exit after 4 bars or when 5-day turnover exceeds 20")
    spec = result["spec"]
    assert spec is not None
    assert spec.mode == "backtest"
    assert len(spec.entry) == 2
    assert len(spec.exit) == 1
    assert spec.holding_period == 4


# ---------------------------------------------------------------------------
# Notes field surfaced on ambiguity
# ---------------------------------------------------------------------------

_NOTES_JSON = """{
  "mode": "screen",
  "market": "CN",
  "universe": ["600519"],
  "logic": "AND",
  "conditions": [
    {"metric": "pct_change", "operator": ">=", "value": 9.9}
  ],
  "entry": [],
  "exit": [],
  "holding_period": null,
  "notes": "'Strong momentum' is ambiguous — used pct_change >= 9.9 as a proxy for limit-up."
}"""

def test_notes_returned_when_ambiguous(monkeypatch):
    monkeypatch.setattr("nl_parser._API_KEY", "sk-fake")
    with _patch_openai(_NOTES_JSON):
        result = parse_nl("show me strong momentum stocks")
    assert result["spec"] is not None
    assert len(result["notes"]) > 0


# ---------------------------------------------------------------------------
# Error handling — malformed JSON from DeepSeek
# ---------------------------------------------------------------------------

def test_malformed_json_returns_error(monkeypatch):
    monkeypatch.setattr("nl_parser._API_KEY", "sk-fake")
    with _patch_openai("this is not json at all"):
        result = parse_nl("enter on high volume")
    assert result["spec"] is None
    assert "JSON" in result["error"]


# ---------------------------------------------------------------------------
# Error handling — schema mismatch
# ---------------------------------------------------------------------------

_BAD_SCHEMA_JSON = """{
  "mode": "invalid_mode",
  "market": "CN",
  "universe": ["600519"],
  "logic": "AND",
  "conditions": [],
  "notes": ""
}"""

def test_schema_mismatch_returns_error(monkeypatch):
    monkeypatch.setattr("nl_parser._API_KEY", "sk-fake")
    with _patch_openai(_BAD_SCHEMA_JSON):
        result = parse_nl("something")
    assert result["spec"] is None
    assert result["error"] is not None


# ---------------------------------------------------------------------------
# Error handling — network / API exception
# ---------------------------------------------------------------------------

def test_api_exception_returns_error(monkeypatch):
    monkeypatch.setattr("nl_parser._API_KEY", "sk-fake")
    with patch("nl_parser.OpenAI", side_effect=Exception("connection refused")):
        result = parse_nl("enter on limit-up")
    assert result["spec"] is None
    assert "connection refused" in result["error"]
