import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from training_analyzer.ingestion import TrainingIngestion
from training_analyzer.repository import GLOBAL_KNOWLEDGE_ID, WorkspaceRepository
from training_analyzer.reports import ReportTemplate, ReportWorkspace


class WorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = WorkspaceRepository(self.root / "app.db")
        self.training = self.repo.create_training("Alpha")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_training_sources_are_isolated(self) -> None:
        other = self.repo.create_training("Bravo")
        self.repo.add_source(self.training.id, "a.log", "/tmp/a.log", "a", "text/plain", 1)
        self.assertEqual(len(self.repo.list_sources(self.training.id)), 1)
        self.assertEqual(self.repo.list_sources(other.id), [])

    def test_source_can_be_deleted(self) -> None:
        source = self.repo.add_source(
            self.training.id,
            "old.log",
            "/tmp/old.log",
            "old",
            "text/plain",
            1,
        )

        self.repo.delete_source(source.id)

        self.assertEqual(self.repo.list_sources(self.training.id), [])

    def test_deleting_reference_preserves_external_file(self) -> None:
        external = self.root / "external.log"
        external.write_text("evidence", encoding="utf-8")
        source = self.repo.add_source(
            self.training.id,
            "external.log",
            str(external),
            "external",
            "text/plain",
            external.stat().st_size,
        )

        TrainingIngestion(self.repo).delete_source(source)

        self.assertTrue(external.exists())
        self.assertEqual(self.repo.list_sources(self.training.id), [])

    def test_text_processing_reports_incremental_progress(self) -> None:
        path = self.root / "server.rpt"
        path.write_text("training evidence", encoding="utf-8")
        self.repo.add_source(
            self.training.id,
            path.name,
            str(path),
            "progress",
            "application/octet-stream",
            path.stat().st_size,
        )
        updates: list[tuple[str, float]] = []

        processed = TrainingIngestion(self.repo).process_all(
            self.training.id,
            on_progress=lambda message, fraction: updates.append((message, fraction)),
        )

        self.assertEqual(processed[0].status, "processed")
        self.assertGreaterEqual(len(updates), 3)
        self.assertEqual(updates[-1][1], 1.0)
        self.assertIn("חילוץ המקורות הסתיים", updates[-1][0])
        self.assertIn("1/1 הושלמו", updates[-1][0])
        self.assertIn("0 נותרו", updates[-1][0])

    def test_source_processing_runs_in_parallel_and_reports_on_caller_thread(self) -> None:
        for position in range(4):
            self.repo.add_source(
                self.training.id,
                f"source-{position}.log",
                f"/tmp/source-{position}.log",
                f"parallel-{position}",
                "text/plain",
                1,
            )
        lock = threading.Lock()
        active = 0
        maximum_active = 0
        callback_threads: list[int] = []
        caller_thread = threading.get_ident()

        class ProbeIngestion(TrainingIngestion):
            def _process(self, source, notify=None) -> None:
                nonlocal active, maximum_active
                with lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                if notify:
                    notify(f"processing {source.relative_path}", 0.5)
                time.sleep(0.04)
                self.repository.update_source(source.id, status="processed", extracted_text="done")
                with lock:
                    active -= 1

        ProbeIngestion(self.repo).process_all(
            self.training.id,
            on_progress=lambda _message, _fraction: callback_threads.append(threading.get_ident()),
        )

        self.assertGreaterEqual(maximum_active, 2)
        self.assertTrue(callback_threads)
        self.assertEqual(set(callback_threads), {caller_thread})

    def test_slow_source_processing_emits_progress_heartbeats(self) -> None:
        self.repo.add_source(
            self.training.id,
            "video.mp4",
            "/tmp/video.mp4",
            "slow-progress",
            "video/mp4",
            1,
        )
        updates: list[str] = []

        class SlowIngestion(TrainingIngestion):
            def _process(self, source, notify=None) -> None:
                if notify:
                    notify("Gemini מנתח וידאו", 0.55)
                time.sleep(0.12)
                self.repository.update_source(source.id, status="processed", extracted_text="done")

        with patch("training_analyzer.ingestion.PROGRESS_HEARTBEAT_SECONDS", 0.03):
            SlowIngestion(self.repo).process_all(
                self.training.id,
                on_progress=lambda message, _fraction: updates.append(message),
            )

        self.assertTrue(any("העיבוד נמשך" in message for message in updates))

    def test_cancellation_skips_sources_that_have_not_started(self) -> None:
        for position in range(3):
            self.repo.add_source(
                self.training.id,
                f"source-{position}.log",
                f"/tmp/source-{position}.log",
                f"cancel-{position}",
                "text/plain",
                1,
            )
        cancelled = threading.Event()
        started: list[str] = []

        class CancellableIngestion(TrainingIngestion):
            def _process(self, source, notify=None) -> None:
                started.append(source.id)
                cancelled.set()

        with patch.dict("os.environ", {"TRAINING_PROCESS_WORKERS": "1"}):
            processed = CancellableIngestion(self.repo).process_all(
                self.training.id,
                cancel_requested=cancelled.is_set,
            )

        self.assertEqual(len(started), 1)
        self.assertTrue(all(source.status == "stored" for source in processed))

    def test_general_knowledge_workspace_is_hidden(self) -> None:
        self.assertEqual(self.repo.get_training(GLOBAL_KNOWLEDGE_ID).name, "General Knowledge")
        self.assertNotIn(GLOBAL_KNOWLEDGE_ID, [item.id for item in self.repo.list_trainings()])

    def test_global_glossary_upsert_does_not_duplicate(self) -> None:
        self.repo.upsert_glossary("קודקוד", "מפקד")
        self.repo.upsert_glossary("קודקוד", "מפקד השדה")
        rows = self.repo.list_glossary()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["meaning"], "מפקד השדה")

    def test_training_glossary_overrides_global_meaning(self) -> None:
        self.repo.upsert_glossary("קודקוד", "מפקד")
        self.repo.upsert_glossary("קודקוד", "מפקד התרגיל", training_id=self.training.id)

        rows = self.repo.list_glossary(self.training.id)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["meaning"], "מפקד התרגיל")

    def test_training_name_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "required"):
            self.repo.create_training("  ")

    def test_training_metadata_can_be_updated(self) -> None:
        updated = self.repo.update_training_metadata(
            self.training.id,
            name="Alpha updated",
            exercise_date="2026-09-12",
            unit_name="Unit 7",
            location="South range",
            notes="Night exercise",
        )

        self.assertEqual(updated.name, "Alpha updated")
        self.assertEqual(updated.exercise_date, "2026-09-12")
        self.assertEqual(updated.unit_name, "Unit 7")
        self.assertEqual(updated.location, "South range")

    def test_report_proposal_requires_apply_to_create_version(self) -> None:
        template_path = self.root / "template.html"
        template_path.write_text('<html><div data-report-section="summary"></div></html>', encoding="utf-8")
        workspace = ReportWorkspace(self.repo, self.training.id, template_path)
        initial = workspace._save({"summary": "Old"}, "agent", "initial")
        proposal = self.repo.create_proposal(self.training.id, "summary", "New", "update", ["a.log"])
        self.assertEqual(self.repo.latest_report(self.training.id).id, initial.id)
        updated = workspace.apply(proposal)
        self.assertEqual(updated.sections["summary"], "New")
        self.assertEqual(updated.version, 2)

    def test_template_rejects_duplicate_markers(self) -> None:
        path = self.root / "bad.html"
        path.write_text('<html><div data-report-section="x"></div><div data-report-section="x"></div></html>', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unique"):
            ReportTemplate(path)

    def test_report_render_enforces_hebrew_rtl_root(self) -> None:
        path = self.root / "template.html"
        path.write_text('<html lang="en" dir="ltr"><div data-report-section="summary"></div></html>', encoding="utf-8")

        rendered = ReportTemplate(path).render({"summary": "סיכום"})

        self.assertIn('dir="rtl"', rendered)
        self.assertIn('lang="he"', rendered)

    def test_template_can_infer_sections_from_existing_ids(self) -> None:
        path = self.root / "template.html"
        path.write_text('<html><section id="summary"><h2>Summary</h2><p>old</p></section></html>', encoding="utf-8")

        template = ReportTemplate(path)
        rendered = template.render({"summary": "New content"})

        self.assertEqual(template.sections[0]["id"], "summary")
        self.assertIn("New content", rendered)
        self.assertNotIn("old", rendered)


if __name__ == "__main__":
    unittest.main()
