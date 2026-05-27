"""
Framework template library for Monkey signal research.

Five SignalSpec templates encoding the user's three trade types (GCL / North Copper / Yabo)
as runnable, editable backtests and screens. All carry status="unvalidated".

Usage:
    from templates import get_templates, get_template

    for t in get_templates():
        print(t["id"], t["name"])

    entry = get_template("t1_momentum_entry")
    spec  = entry["spec"]   # fully validated SignalSpec, ready for run_backtest()

Templates are stored as data in templates.json and parsed here into SignalSpec objects.
Edit templates.json to adjust condition values without touching Python code.
"""

import json
from pathlib import Path

from core import Condition, SignalSpec

_JSON_PATH = Path(__file__).parent / "templates.json"

# Reminder shown in the UI for every framework template
UNVALIDATED_NOTE = (
    "Framework template — a hypothesis to test, not a proven edge. "
    "Backtest it before trusting it."
)


def get_templates() -> list[dict]:
    """Return all templates as a list of dicts.

    Each dict has keys: id, name, description, rationale, status, spec.
    spec is a fully validated SignalSpec instance.
    """
    with open(_JSON_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return [_parse(t) for t in raw]


def get_template(template_id: str) -> dict | None:
    """Return a single template dict by id, or None if not found."""
    for entry in get_templates():
        if entry["id"] == template_id:
            return entry
    return None


def _parse(raw: dict) -> dict:
    """Parse one JSON entry into a dict with a validated SignalSpec."""
    spec_dict = raw["spec"].copy()

    for field in ("conditions", "entry", "exit"):
        if spec_dict.get(field):
            spec_dict[field] = [Condition(**c) for c in spec_dict[field]]

    spec = SignalSpec(**spec_dict)

    return {
        "id":          raw["id"],
        "name":        raw["name"],
        "description": raw["description"],
        "rationale":   raw["rationale"],
        "status":      raw["status"],
        "spec":        spec,
    }
