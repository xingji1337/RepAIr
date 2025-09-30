"""
Purpose:
- Post-process RAG/Search results for safety-sensitive topics and attach warnings.
- Minimal keyword/regex rules for: GAS, APPLIANCE, ELECTRICAL (risky).
- Non-blocking: we never hide results; we only annotate with flags + advisories.

How it's used:
- For each result, we scan the snippet (and optionally a title/path) for matches.
- We aggregate a global warnings list (unique advisories) for the whole response.

Extensibility:
- Add/adapt rules in HAZARD_RULES.
- Tweak advisory text in ADVISORIES.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Any

# --- Rule / Advisory configuration ------------------------------------------

@dataclass
class HazardRule:
    label: str            # short category code
    patterns: List[str]   # regex (case-insensitive)
    severity: str         # "caution", "warning", "danger"

# Keyword/regex rules (basic; refine over time)
HAZARD_RULES: List[HazardRule] = [
    HazardRule(
        label="GAS",
        severity="danger",
        patterns=[
            r"\b(natural\s+gas|gas\s+line|gas\s+leak|gas\s+valve|pilot\s+light|propane)\b",
            r"\b(smell\s+gas|odor\s+of\s+gas|rotten\s+egg\s+smell)\b",
            r"\bshut\s*off\s+gas\b",
        ],
    ),
    HazardRule(
        label="APPLIANCE",
        severity="warning",
        patterns=[
            r"\b(water\s+heater|furnace|boiler|dryer|oven|range|dishwasher)\b",
            r"\b(thermocouple|igniter|pilot|flue|vent|combustion)\b",
        ],
    ),
    HazardRule(
        label="ELECTRICAL",
        severity="danger",
        patterns=[
            r"\b(breaker\s+panel|service\s+panel|main\s+shut\s*off|live\s+wire|line\s+voltage|240\s*v|120\s*v)\b",
            r"\b(gfci|arc[-\s]?fault|ground(ed)?|neutral|hot\s+wire)\b",
            r"\b(wiring|receptacle|outlet|junction\s+box|lugs?)\b",
        ],
    ),
]

# Standardized advisory text (kept short for UI; can expand later)
ADVISORIES: Dict[str, str] = {
    "GAS": (
        "Gas work can be dangerous. If you smell gas or suspect a leak, leave the area and "
        "contact your utility/emergency services. DIY gas repairs may be illegal in some areas."
    ),
    "APPLIANCE": (
        "Appliance repairs can involve gas, high voltage, or sharp/heavy parts. "
        "Consult the manufacturer guide and consider a licensed technician."
    ),
    "ELECTRICAL": (
        "Electrical work can cause shock, fire, or death. Turn off power at the breaker, "
        "verify with a tester, and consider a licensed electrician for panel or wiring work."
    ),
}

# --- Core logic --------------------------------------------------------------

def _match_labels(text: str) -> List[Dict[str, str]]:
    """Return a list of {'label':..., 'severity':...} for rules matched in text."""
    if not text:
        return []
    matches: List[Dict[str, str]] = []
    low = text.lower()
    for rule in HAZARD_RULES:
        for pat in rule.patterns:
            if re.search(pat, low, flags=re.IGNORECASE):
                matches.append({"label": rule.label, "severity": rule.severity})
                break  # one hit per rule is enough
    return matches

def analyze_snippets(snippets: List[str]) -> Dict[str, Any]:
    """
    Analyze many snippets and aggregate:
    - per_item_flags: list[List[flag]] parallel to input order
    - warnings: unique advisories at response level
    """
    per_item_flags: List[List[Dict[str, str]]] = []
    triggered_labels: set[str] = set()

    for snip in snippets:
        flags = _match_labels(snip)
        per_item_flags.append(flags)
        for f in flags:
            triggered_labels.add(f["label"])

    warnings = []
    for label in sorted(triggered_labels):
        warnings.append({
            "label": label,
            "severity": next((r.severity for r in HAZARD_RULES if r.label == label), "caution"),
            "advisory": ADVISORIES.get(label, "Use caution."),
        })

    return {"per_item_flags": per_item_flags, "warnings": warnings}

