"""SQLite persistence for training workspaces and analyst-reviewed state."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from training_analyzer.gemini import PROJECT_ROOT
from training_analyzer.models import ReportProposal, ReportVersion, Source, Training


DATABASE_PATH = PROJECT_ROOT / "data" / "training_analyzer.db"
GLOBAL_KNOWLEDGE_ID = "__global_knowledge__"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class WorkspaceRepository:
    """Own all durable workspace state behind a small persistence interface."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(os.getenv("TRAINING_ANALYZER_DB", DATABASE_PATH))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS trainings (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    exercise_date TEXT NOT NULL DEFAULT '', unit_name TEXT NOT NULL DEFAULT '',
                    location TEXT NOT NULL DEFAULT '', notes TEXT NOT NULL DEFAULT '',
                    stage TEXT NOT NULL DEFAULT 'ingestion', created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY, training_id TEXT NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,
                    relative_path TEXT NOT NULL, local_path TEXT NOT NULL, sha256 TEXT NOT NULL,
                    mime_type TEXT NOT NULL, size_bytes INTEGER NOT NULL, gcs_uri TEXT,
                    status TEXT NOT NULL DEFAULT 'stored', extracted_text TEXT NOT NULL DEFAULT '',
                    error TEXT, offset_seconds REAL NOT NULL DEFAULT 0,
                    UNIQUE(training_id, sha256, relative_path)
                );
                CREATE TABLE IF NOT EXISTS force_entities (
                    id TEXT PRIMARY KEY, training_id TEXT NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,
                    name TEXT NOT NULL, kind TEXT NOT NULL, role TEXT NOT NULL DEFAULT '',
                    evidence TEXT NOT NULL DEFAULT '', approved INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS timeline_events (
                    id TEXT PRIMARY KEY, training_id TEXT NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,
                    seconds REAL NOT NULL DEFAULT 0, description TEXT NOT NULL,
                    source_name TEXT NOT NULL DEFAULT '', confidence REAL NOT NULL DEFAULT 0,
                    approved INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS glossary (
                    id TEXT PRIMARY KEY, training_id TEXT, term TEXT NOT NULL, meaning TEXT NOT NULL,
                    variants TEXT NOT NULL DEFAULT '', notes TEXT NOT NULL DEFAULT '',
                    UNIQUE(training_id, term)
                );
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY, training_id TEXT NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,
                    version INTEGER NOT NULL, sections_json TEXT NOT NULL, html_path TEXT NOT NULL,
                    pdf_path TEXT, created_by TEXT NOT NULL, change_summary TEXT NOT NULL,
                    created_at TEXT NOT NULL, UNIQUE(training_id, version)
                );
                CREATE TABLE IF NOT EXISTS report_proposals (
                    id TEXT PRIMARY KEY, training_id TEXT NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,
                    section_id TEXT NOT NULL, proposed_markdown TEXT NOT NULL,
                    rationale TEXT NOT NULL, citations_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    training_id TEXT NOT NULL REFERENCES trainings(id) ON DELETE CASCADE,
                    role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
                );
                """
            )
            now = _now()
            db.execute(
                """
                INSERT OR IGNORE INTO trainings
                    (id, name, stage, created_at, updated_at)
                VALUES (?, 'General Knowledge', 'processing', ?, ?)
                """,
                (GLOBAL_KNOWLEDGE_ID, now, now),
            )

    def create_training(
        self, name: str, exercise_date: str = "", unit_name: str = "",
        location: str = "", notes: str = "",
    ) -> Training:
        name = name.strip()
        if not name:
            raise ValueError("Training name is required.")
        training_id = uuid.uuid4().hex
        now = _now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO trainings VALUES (?, ?, ?, ?, ?, ?, 'ingestion', ?, ?)",
                (training_id, name, exercise_date, unit_name, location, notes, now, now),
            )
        return self.get_training(training_id)

    def list_trainings(self) -> list[Training]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM trainings WHERE id != ? ORDER BY created_at DESC",
                (GLOBAL_KNOWLEDGE_ID,),
            ).fetchall()
        return [Training(**{key: row[key] for key in Training.__dataclass_fields__}) for row in rows]

    def get_training(self, training_id: str) -> Training:
        with self._connect() as db:
            row = db.execute("SELECT * FROM trainings WHERE id = ?", (training_id,)).fetchone()
        if row is None:
            raise KeyError(f"Training not found: {training_id}")
        return Training(**{key: row[key] for key in Training.__dataclass_fields__})

    def set_stage(self, training_id: str, stage: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE trainings SET stage = ?, updated_at = ? WHERE id = ?", (stage, _now(), training_id))

    def update_training_metadata(
        self,
        training_id: str,
        *,
        name: str,
        exercise_date: str = "",
        unit_name: str = "",
        location: str = "",
        notes: str = "",
    ) -> Training:
        name = name.strip()
        if not name:
            raise ValueError("Training name is required.")
        with self._connect() as db:
            db.execute(
                """
                UPDATE trainings
                SET name = ?, exercise_date = ?, unit_name = ?, location = ?,
                    notes = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    name,
                    exercise_date.strip(),
                    unit_name.strip(),
                    location.strip(),
                    notes.strip(),
                    _now(),
                    training_id,
                ),
            )
        return self.get_training(training_id)

    def add_source(self, training_id: str, relative_path: str, local_path: str, sha256: str,
                   mime_type: str, size_bytes: int, gcs_uri: str | None = None) -> Source:
        source_id = uuid.uuid4().hex
        with self._connect() as db:
            existing = db.execute(
                "SELECT id FROM sources WHERE training_id = ? AND sha256 = ? AND relative_path = ?",
                (training_id, sha256, relative_path),
            ).fetchone()
            if existing:
                return self.get_source(existing["id"])
            db.execute(
                "INSERT INTO sources (id, training_id, relative_path, local_path, sha256, mime_type, size_bytes, gcs_uri) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (source_id, training_id, relative_path, local_path, sha256, mime_type, size_bytes, gcs_uri),
            )
        return self.get_source(source_id)

    def get_source(self, source_id: str) -> Source:
        with self._connect() as db:
            row = db.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        if row is None:
            raise KeyError(f"Source not found: {source_id}")
        return Source(**{key: row[key] for key in Source.__dataclass_fields__})

    def list_sources(self, training_id: str) -> list[Source]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM sources WHERE training_id = ? ORDER BY relative_path", (training_id,)).fetchall()
        return [Source(**{key: row[key] for key in Source.__dataclass_fields__}) for row in rows]

    def update_source(self, source_id: str, *, status: str, extracted_text: str = "", error: str | None = None, gcs_uri: str | None = None) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE sources SET status = ?, extracted_text = ?, error = ?, gcs_uri = COALESCE(?, gcs_uri) WHERE id = ?",
                (status, extracted_text, error, gcs_uri, source_id),
            )

    def set_source_offset(self, source_id: str, offset_seconds: float) -> None:
        with self._connect() as db:
            db.execute("UPDATE sources SET offset_seconds = ? WHERE id = ?", (offset_seconds, source_id))

    def delete_source(self, source_id: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM sources WHERE id = ?", (source_id,))

    def replace_force_entities(self, training_id: str, entities: list[dict]) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM force_entities WHERE training_id = ?", (training_id,))
            db.executemany(
                "INSERT INTO force_entities VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(uuid.uuid4().hex, training_id, item["name"], item.get("kind", "unit"), item.get("role", ""), item.get("evidence", ""), int(item.get("approved", False))) for item in entities],
            )

    def list_force_entities(self, training_id: str) -> list[dict]:
        with self._connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM force_entities WHERE training_id = ? ORDER BY name", (training_id,))]

    def approve_force_entities(self, training_id: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE force_entities SET approved = 1 WHERE training_id = ?", (training_id,))

    def replace_timeline(self, training_id: str, events: list[dict]) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM timeline_events WHERE training_id = ?", (training_id,))
            db.executemany(
                "INSERT INTO timeline_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(uuid.uuid4().hex, training_id, float(item.get("seconds", 0)), item["description"], item.get("source_name", ""), float(item.get("confidence", 0)), int(item.get("approved", False))) for item in events],
            )

    def list_timeline(self, training_id: str) -> list[dict]:
        with self._connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM timeline_events WHERE training_id = ? ORDER BY seconds", (training_id,))]

    def approve_timeline(self, training_id: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE timeline_events SET approved = 1 WHERE training_id = ?", (training_id,))

    def upsert_glossary(self, term: str, meaning: str, variants: str = "", notes: str = "", training_id: str | None = None) -> None:
        with self._connect() as db:
            existing = db.execute(
                "SELECT id FROM glossary WHERE term = ? AND ((training_id IS NULL AND ? IS NULL) OR training_id = ?)",
                (term.strip(), training_id, training_id),
            ).fetchone()
            if existing:
                db.execute("UPDATE glossary SET meaning = ?, variants = ?, notes = ? WHERE id = ?", (meaning.strip(), variants.strip(), notes.strip(), existing["id"]))
            else:
                db.execute("INSERT INTO glossary VALUES (?, ?, ?, ?, ?, ?)", (uuid.uuid4().hex, training_id, term.strip(), meaning.strip(), variants.strip(), notes.strip()))

    def list_glossary(self, training_id: str | None = None) -> list[dict]:
        with self._connect() as db:
            if training_id:
                rows = db.execute("SELECT * FROM glossary WHERE training_id IS NULL OR training_id = ? ORDER BY training_id, term", (training_id,)).fetchall()
            else:
                rows = db.execute("SELECT * FROM glossary WHERE training_id IS NULL ORDER BY term").fetchall()
        values = [dict(row) for row in rows]
        if not training_id:
            return values

        # SQLite sorts NULL before training IDs, so a training-specific entry
        # naturally replaces the global meaning for the same normalized term.
        effective: dict[str, dict] = {}
        for value in values:
            effective[value["term"].casefold()] = value
        return sorted(effective.values(), key=lambda value: value["term"].casefold())

    def save_report(self, training_id: str, sections: dict[str, str], html_path: str,
                    created_by: str, change_summary: str, pdf_path: str | None = None) -> ReportVersion:
        with self._connect() as db:
            version = db.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM reports WHERE training_id = ?", (training_id,)).fetchone()[0]
            report_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO reports VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (report_id, training_id, version, json.dumps(sections, ensure_ascii=False), html_path, pdf_path, created_by, change_summary, _now()),
            )
        return self.get_report(report_id)

    def get_report(self, report_id: str) -> ReportVersion:
        with self._connect() as db:
            row = db.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        if row is None:
            raise KeyError(report_id)
        return ReportVersion(row["id"], row["training_id"], row["version"], json.loads(row["sections_json"]), row["html_path"], row["pdf_path"], row["created_by"], row["change_summary"], row["created_at"])

    def latest_report(self, training_id: str) -> ReportVersion | None:
        with self._connect() as db:
            row = db.execute("SELECT id FROM reports WHERE training_id = ? ORDER BY version DESC LIMIT 1", (training_id,)).fetchone()
        return self.get_report(row["id"]) if row else None

    def list_reports(self, training_id: str) -> list[ReportVersion]:
        with self._connect() as db:
            ids = [row[0] for row in db.execute("SELECT id FROM reports WHERE training_id = ? ORDER BY version DESC", (training_id,))]
        return [self.get_report(item) for item in ids]

    def set_report_pdf(self, report_id: str, pdf_path: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE reports SET pdf_path = ? WHERE id = ?", (pdf_path, report_id))

    def delete_training(self, training_id: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM trainings WHERE id = ?", (training_id,))

    def create_proposal(self, training_id: str, section_id: str, markdown: str, rationale: str, citations: list[str]) -> ReportProposal:
        proposal_id = uuid.uuid4().hex
        with self._connect() as db:
            db.execute("INSERT INTO report_proposals VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)", (proposal_id, training_id, section_id, markdown, rationale, json.dumps(citations, ensure_ascii=False), _now()))
        return self.get_proposal(proposal_id)

    def get_proposal(self, proposal_id: str) -> ReportProposal:
        with self._connect() as db:
            row = db.execute("SELECT * FROM report_proposals WHERE id = ?", (proposal_id,)).fetchone()
        if row is None:
            raise KeyError(proposal_id)
        return ReportProposal(row["id"], row["training_id"], row["section_id"], row["proposed_markdown"], row["rationale"], json.loads(row["citations_json"]), row["status"])

    def set_proposal_status(self, proposal_id: str, status: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE report_proposals SET status = ? WHERE id = ?", (status, proposal_id))

    def pending_proposals(self, training_id: str) -> list[ReportProposal]:
        with self._connect() as db:
            ids = [row[0] for row in db.execute("SELECT id FROM report_proposals WHERE training_id = ? AND status = 'pending' ORDER BY created_at", (training_id,))]
        return [self.get_proposal(item) for item in ids]

    def add_message(self, training_id: str, role: str, content: str) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO messages (training_id, role, content, created_at) VALUES (?, ?, ?, ?)", (training_id, role, content, _now()))

    def list_messages(self, training_id: str) -> list[dict[str, str]]:
        with self._connect() as db:
            rows = db.execute("SELECT role, content FROM messages WHERE training_id = ? ORDER BY id", (training_id,)).fetchall()
        return [dict(row) for row in rows]
