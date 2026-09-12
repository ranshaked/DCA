"""AI-assisted force-map and synchronized-timeline drafting."""

from __future__ import annotations

from pydantic import BaseModel, Field

from training_analyzer.gemini import create_chat_model
from training_analyzer.prompts import load_prompt, render_prompt
from training_analyzer.repository import WorkspaceRepository


class ForceEntityDraft(BaseModel):
    name: str
    kind: str = Field(description=load_prompt("schema/force_entity_kind"))
    role: str = ""
    evidence: str = ""


class ForceMapDraft(BaseModel):
    entities: list[ForceEntityDraft]


class TimelineEventDraft(BaseModel):
    seconds: float = 0
    description: str
    source_name: str = ""
    confidence: float = Field(default=0.5, ge=0, le=1)


class TimelineDraft(BaseModel):
    events: list[TimelineEventDraft]


class TrainingAnalysis:
    def __init__(self, repository: WorkspaceRepository) -> None:
        self.repository = repository

    def _evidence(self, training_id: str) -> str:
        parts = [f"SOURCE {source.relative_path}\n{source.extracted_text}" for source in self.repository.list_sources(training_id) if source.extracted_text]
        if not parts:
            raise ValueError("Process at least one source before analysis.")
        return "\n\n".join(parts)[:500_000]

    def draft_force_map(self, training_id: str) -> list[dict]:
        model = create_chat_model(temperature=0).with_structured_output(ForceMapDraft)
        result = model.invoke(
            render_prompt("force_map_analysis", evidence=self._evidence(training_id))
        )
        entities = [item.model_dump() for item in result.entities]
        self.repository.replace_force_entities(training_id, entities)
        return entities

    def draft_timeline(self, training_id: str) -> list[dict]:
        model = create_chat_model(temperature=0).with_structured_output(TimelineDraft)
        result = model.invoke(
            render_prompt("timeline_analysis", evidence=self._evidence(training_id))
        )
        events = [item.model_dump() for item in result.events]
        self.repository.replace_timeline(training_id, events)
        return events
