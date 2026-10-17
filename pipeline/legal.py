"""Legal references (legal/norms.yaml) — never generated, only looked up."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

NORMS = Path(__file__).resolve().parent.parent / "legal" / "norms.yaml"


@lru_cache(maxsize=1)
def _norms() -> dict:
    return yaml.safe_load(NORMS.read_text())


def legal_ref(ref_id: str) -> dict | None:
    n = _norms()
    r = n["refs"].get(ref_id)
    if not r:
        return None
    d = n["documents"][r["doc"]]
    clause = r.get("clause")
    title_short = d["title"].split(" в банках")[0].split(" банков")[0].strip()
    short = f"{clause} «{title_short}…» (постановление Правления НБРК {d['act'].replace('постановление ', '')})"
    return {
        "document": " ".join(d["title"].split()),
        "issuer": d["issuer"],
        "act": d["act"],
        "clause": clause,
        "requirement": " ".join(r["requirement"].split()),
        "interpretation": " ".join(r.get("interpretation", "").split()) or None,
        "url": d["url"],
        "verified": bool(r.get("verified")),
        "short": short,
    }
