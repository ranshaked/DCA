"""Parse analyst-supplied glossary files into repository-ready records."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path


FIELD_ALIASES = {
    "term": ("term", "מונח"),
    "meaning": ("meaning", "definition", "משמעות", "הגדרה"),
    "variants": ("variants", "aliases", "גרסאות", "שיבושים"),
    "notes": ("notes", "הערות"),
}


def parse_glossary_file(filename: str, data: bytes) -> list[dict[str, str]]:
    text = data.decode("utf-8-sig", errors="replace")
    suffix = Path(filename).suffix.lower()
    if suffix == ".json":
        raw = json.loads(text)
        rows = raw if isinstance(raw, list) else raw.get("terms", [])
    elif suffix in {".csv", ".tsv"}:
        rows = list(csv.DictReader(io.StringIO(text), delimiter="\t" if suffix == ".tsv" else ","))
    else:
        rows = []
        for line in text.splitlines():
            line = line.strip().lstrip("-* ")
            if not line:
                continue
            parts = [part.strip() for part in (line.split("|") if "|" in line else line.split(":", 1))]
            if len(parts) >= 2:
                rows.append({"term": parts[0], "meaning": parts[1], "variants": parts[2] if len(parts) > 2 else "", "notes": parts[3] if len(parts) > 3 else ""})
    normalized = []
    for row in rows:
        value = {field: next((str(row.get(alias, "")).strip() for alias in aliases if row.get(alias) is not None), "") for field, aliases in FIELD_ALIASES.items()}
        if value["term"] and value["meaning"]:
            normalized.append(value)
    if not normalized:
        raise ValueError("לא נמצאו רשומות מילון תקינות בקובץ.")
    return normalized
