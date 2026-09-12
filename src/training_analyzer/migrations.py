"""Non-destructive imports from the original prototype layout."""

from __future__ import annotations

from training_analyzer.gemini import PROJECT_ROOT
from training_analyzer.repository import WorkspaceRepository


def import_legacy_glossary(repository: WorkspaceRepository) -> int:
    if repository.list_glossary():
        return 0
    upload_dir = PROJECT_ROOT / "data" / "uploads"
    pointers = list(upload_dir.glob("*glossary_current.txt")) if upload_dir.exists() else []
    if not pointers:
        return 0
    target_name = pointers[0].read_text(encoding="utf-8").strip()
    candidates = list(upload_dir.glob(f"*{target_name}"))
    if not candidates:
        return 0
    imported = 0
    for line in candidates[0].read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.startswith("|---") or "Correct term" in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 2 and cells[0] and cells[1]:
            repository.upsert_glossary(cells[0], cells[1], cells[2] if len(cells) > 2 else "", cells[3] if len(cells) > 3 else "")
            imported += 1
    return imported
