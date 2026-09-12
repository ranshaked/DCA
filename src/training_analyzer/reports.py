"""Template-driven report generation, versioning, chat, and export."""

from __future__ import annotations

import difflib
import json
import uuid
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import bleach
import markdown
from bs4 import BeautifulSoup
from langchain.agents import create_agent
from pydantic import BaseModel, Field

from training_analyzer.chat import messages_with_retrieved_context
from training_analyzer.background import JobCancelled
from training_analyzer.gemini import PROJECT_ROOT, create_chat_model
from training_analyzer.knowledge_base import TrainingKnowledgeBase
from training_analyzer.models import ReportProposal, ReportVersion
from training_analyzer.prompts import load_prompt, render_prompt
from training_analyzer.repository import WorkspaceRepository


TEMPLATE_PATH = PROJECT_ROOT / "training_summary_report_template.html"
REPORTS_DIR = PROJECT_ROOT / "data" / "trainings"


class ReportSectionContent(BaseModel):
    markdown: str
    citations: list[str] = Field(default_factory=list)


class ChatReply(BaseModel):
    answer: str
    edit_requested: bool = False
    section_id: str | None = None
    proposed_markdown: str | None = None
    rationale: str = ""
    citations: list[str] = Field(default_factory=list)


class TrainingMetadataDraft(BaseModel):
    exercise_date: str | None = None
    unit_name: str | None = None
    location: str | None = None
    notes: str | None = None
    citations: list[str] = Field(default_factory=list)


class ReportTemplate:
    """Validate and render analyst-controlled HTML without model-owned layout."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or TEMPLATE_PATH
        path = self.path
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            raise ValueError(f"Report template is missing or empty: {path}")
        self.html = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(self.html, "html.parser")
        if soup.html is None:
            raise ValueError("The report template must contain an html root element.")
        markers = soup.select("[data-report-section]")
        self.uses_explicit_markers = bool(markers)
        if not markers:
            markers = soup.select("section[id], section[aria-labelledby]")
        if not markers:
            raise ValueError(
                "The report template needs data-report-section markers or section elements with IDs."
            )
        ids = [
            marker.get("data-report-section", marker.get("id", marker.get("aria-labelledby", ""))).strip()
            for marker in markers
        ]
        if any(not item for item in ids) or len(ids) != len(set(ids)):
            raise ValueError("Report section IDs must be non-empty and unique.")
        self.sections = []
        for marker, section_id in zip(markers, ids, strict=True):
            heading = marker.find(["h1", "h2", "h3"])
            self.sections.append({
                "id": section_id,
                "title": marker.get("data-report-title", heading.get_text(" ", strip=True) if heading else section_id),
                "instructions": marker.get("data-report-instructions")
                or self._section_instructions(section_id),
            })

    @staticmethod
    def _section_instructions(section_id: str) -> str:
        try:
            return load_prompt(f"report_sections/{section_id}")
        except FileNotFoundError:
            return load_prompt("default_report_section")

    def render(self, sections: dict[str, str]) -> str:
        soup = BeautifulSoup(self.html, "html.parser")
        soup.html["lang"] = "he"
        soup.html["dir"] = "rtl"
        allowed_tags = set(bleach.sanitizer.ALLOWED_TAGS) | {"p", "h1", "h2", "h3", "h4", "table", "thead", "tbody", "tr", "th", "td", "ul", "ol", "li", "hr", "br"}
        selector = "[data-report-section]" if self.uses_explicit_markers else "section[id], section[aria-labelledby]"
        for marker in soup.select(selector):
            section_id = marker.get("data-report-section", marker.get("id", marker.get("aria-labelledby")))
            rendered = markdown.markdown(sections.get(section_id, ""), extensions=["tables", "fenced_code"])
            safe = bleach.clean(rendered, tags=allowed_tags, attributes={}, strip=True)
            heading = marker.find(["h1", "h2", "h3"]) if not self.uses_explicit_markers else None
            if heading:
                heading.extract()
            marker.clear()
            if heading:
                marker.append(heading)
            fragment = BeautifulSoup(safe, "html.parser")
            marker.extend(list(fragment.contents))
        return str(soup)


class ReportWorkspace:
    """Own report lifecycle and preserve every analyst-approved version."""

    def __init__(self, repository: WorkspaceRepository, training_id: str, template_path: Path | None = None) -> None:
        self.repository = repository
        self.training_id = training_id
        self.template_path = template_path or TEMPLATE_PATH

    def generate(
        self,
        on_progress: Callable[[str, float], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> ReportVersion:
        notify = on_progress or (lambda _message, _fraction: None)
        is_cancelled = cancel_requested or (lambda: False)

        def checkpoint() -> None:
            if is_cancelled():
                raise JobCancelled()

        checkpoint()
        notify("קורא ומאמת את תבנית הדוח", 0.02)
        template = ReportTemplate(self.template_path)
        knowledge = TrainingKnowledgeBase(self.repository, self.training_id)
        training = self.repository.get_training(self.training_id)
        missing_metadata = [
            field
            for field in ("exercise_date", "unit_name", "location")
            if not getattr(training, field)
        ]
        if missing_metadata:
            try:
                notify("Gemini מאתר בראיות פרטי אימון חסרים", 0.04)
                evidence = knowledge.search(load_prompt("metadata_evidence_query"), include_global=False)
                metadata_model = create_chat_model(temperature=0).with_structured_output(
                    TrainingMetadataDraft
                )
                inferred = metadata_model.invoke(
                    render_prompt(
                        "metadata_inference",
                        missing_fields=", ".join(missing_metadata),
                        evidence=evidence,
                    )
                )
                checkpoint()
                inferred_notes = inferred.notes or ""
                if inferred.citations:
                    inferred_notes += "\nמקורות לפרטי האימון: " + ", ".join(inferred.citations)
                training = self.repository.update_training_metadata(
                    self.training_id,
                    name=training.name,
                    exercise_date=training.exercise_date or inferred.exercise_date or "",
                    unit_name=training.unit_name or inferred.unit_name or "",
                    location=training.location or inferred.location or "",
                    notes=training.notes or inferred_notes,
                )
            except JobCancelled:
                raise
            except Exception:
                notify("לא ניתן היה להשלים מטא-דאטה חסר; הדוח ממשיך עם הפרטים הקיימים", 0.05)
        model = create_chat_model(temperature=0.1).with_structured_output(ReportSectionContent)
        sections: dict[str, str] = {}
        total = len(template.sections)
        for position, section in enumerate(template.sections):
            checkpoint()
            notify(
                f"מחפש ראיות עבור הסעיף: {section['title']}",
                position / total,
            )
            evidence = knowledge.search(
                render_prompt(
                    "report_evidence_query",
                    section_title=section["title"],
                    section_instructions=section["instructions"],
                )
            )
            notify(
                f"Gemini מנסח את הסעיף: {section['title']}",
                (position + 0.45) / total,
            )
            prompt = render_prompt(
                "report_section",
                training_name=training.name,
                exercise_date=training.exercise_date,
                unit_name=training.unit_name,
                location=training.location,
                notes=training.notes,
                section_title=section["title"],
                section_instructions=section["instructions"],
                evidence=evidence,
            )
            result = model.invoke(prompt)
            checkpoint()
            body = result.markdown
            if result.citations and not any(citation in body for citation in result.citations):
                body += "\n\n**מקורות:** " + ", ".join(f"[{citation}]" for citation in result.citations)
            sections[section["id"]] = body
            notify(
                f"הסעיף הושלם: {section['title']}",
                (position + 1) / total,
            )
        checkpoint()
        notify("מרכיב את הסעיפים בתוך תבנית ה-HTML ושומר גרסה", 1.0)
        return self._save(sections, "agent", "Initial grounded report")

    def _save(self, sections: dict[str, str], created_by: str, summary: str) -> ReportVersion:
        rendered = ReportTemplate(self.template_path).render(sections)
        output_dir = REPORTS_DIR / self.training_id / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / f"report-{uuid.uuid4().hex}.html"
        html_path.write_text(rendered, encoding="utf-8")
        return self.repository.save_report(self.training_id, sections, str(html_path), created_by, summary)

    def apply(self, proposal: ReportProposal) -> ReportVersion:
        current = self.repository.latest_report(self.training_id)
        if current is None:
            raise ValueError("Generate a report before applying edits.")
        sections = dict(current.sections)
        if proposal.section_id not in sections:
            raise ValueError(f"Unknown report section: {proposal.section_id}")
        sections[proposal.section_id] = proposal.proposed_markdown
        result = self._save(sections, "analyst-approved", proposal.rationale)
        self.repository.set_proposal_status(proposal.id, "applied")
        return result

    def reject(self, proposal: ReportProposal) -> None:
        self.repository.set_proposal_status(proposal.id, "rejected")

    def restore(self, report: ReportVersion) -> ReportVersion:
        return self._save(dict(report.sections), "analyst-approved", f"Restored version {report.version}")

    def proposal_diff(self, proposal: ReportProposal) -> str:
        current = self.repository.latest_report(self.training_id)
        old = current.sections.get(proposal.section_id, "") if current else ""
        return "\n".join(difflib.unified_diff(old.splitlines(), proposal.proposed_markdown.splitlines(), fromfile="current", tofile="proposed", lineterm=""))

    def export_pdf(self, report: ReportVersion) -> Path:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError("Install Playwright and its Chromium browser to export PDF.") from exc
        target = Path(report.html_path).with_suffix(".pdf")
        with sync_playwright() as runtime:
            browser = runtime.chromium.launch()
            page = browser.new_page()
            page.set_content(Path(report.html_path).read_text(encoding="utf-8"), wait_until="load")
            page.emulate_media(media="screen")
            page.pdf(path=str(target), format="A4", print_background=True, prefer_css_page_size=True)
            browser.close()
        return target


class TrainingChat:
    """Ground every turn and turn report edits into reviewable proposals."""

    def __init__(self, repository: WorkspaceRepository, training_id: str) -> None:
        self.repository = repository
        self.training_id = training_id
        self.knowledge = TrainingKnowledgeBase(repository, training_id)

    def ask(self, question: str) -> ChatReply:
        report = self.repository.latest_report(self.training_id)
        training = self.repository.get_training(self.training_id)
        section_text = json.dumps(report.sections, ensure_ascii=False) if report else "No report exists yet."
        context = self.knowledge.search(question)
        history = self.repository.list_messages(self.training_id)
        history.append({"role": "user", "content": question})
        grounded = messages_with_retrieved_context(history, context)
        grounded[-1]["content"] += "\n\n" + render_prompt(
            "chat_workspace_context",
            training_metadata=json.dumps(asdict(training), ensure_ascii=False),
            report_sections=section_text,
        )
        agent = create_agent(
            model=create_chat_model(temperature=0.2),
            tools=[self.knowledge.as_tool()],
            response_format=ChatReply,
            system_prompt=load_prompt("chat_system"),
        )
        result = agent.invoke({"messages": grounded})
        reply: ChatReply = result["structured_response"]
        self.repository.add_message(self.training_id, "user", question)
        self.repository.add_message(self.training_id, "assistant", reply.answer)
        if reply.edit_requested and reply.section_id and reply.proposed_markdown:
            self.repository.create_proposal(self.training_id, reply.section_id, reply.proposed_markdown, reply.rationale, reply.citations)
        return reply
