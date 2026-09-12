"""Domain records exchanged by the application's modules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class WorkflowStage(StrEnum):
    INGESTION = "ingestion"
    FORCE_MAPPING = "force_mapping"
    SYNCHRONIZATION = "synchronization"
    PROCESSING = "processing"
    RESULTS = "results"
    REVIEW = "review"
    GLOSSARY = "glossary"


@dataclass(frozen=True)
class Training:
    id: str
    name: str
    exercise_date: str
    unit_name: str
    location: str
    notes: str
    stage: str
    created_at: str


@dataclass(frozen=True)
class Source:
    id: str
    training_id: str
    relative_path: str
    local_path: str
    sha256: str
    mime_type: str
    size_bytes: int
    gcs_uri: str | None
    status: str
    extracted_text: str
    error: str | None
    offset_seconds: float


@dataclass(frozen=True)
class ReportVersion:
    id: str
    training_id: str
    version: int
    sections: dict[str, str]
    html_path: str
    pdf_path: str | None
    created_by: str
    change_summary: str
    created_at: str


@dataclass(frozen=True)
class ReportProposal:
    id: str
    training_id: str
    section_id: str
    proposed_markdown: str
    rationale: str
    citations: list[str]
    status: str
