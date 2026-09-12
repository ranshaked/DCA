import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from training_analyzer.repository import WorkspaceRepository
from training_analyzer.ui import get_background_jobs, get_repository


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingFormTests(unittest.TestCase):
    def setUp(self) -> None:
        get_repository.clear()
        get_background_jobs.clear()

    def test_new_training_form_can_submit_after_typing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "app.db"
            with patch.dict(os.environ, {"TRAINING_ANALYZER_DB": str(database_path)}):
                app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=30)

                create_button = next(button for button in app.button if button.label == "צור סביבת אימון")
                self.assertFalse(create_button.disabled)

                create_button.click().run(timeout=30)
                self.assertTrue(any("יש להזין שם אימון" in error.value for error in app.error))

                name = next(field for field in app.text_input if field.label == "שם האימון")
                name.input("תרג״ד")
                next(button for button in app.button if button.label == "צור סביבת אימון").click().run(timeout=30)

                self.assertTrue(any(title.value == "תרג״ד" for title in app.title))

    def test_active_report_job_has_a_working_stop_button(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "app.db"
            with patch.dict(os.environ, {"TRAINING_ANALYZER_DB": str(database_path)}):
                repository = WorkspaceRepository(database_path)
                training = repository.create_training("Stop control")
                repository.add_source(
                    training.id,
                    "evidence.log",
                    "/tmp/evidence.log",
                    "stop-control",
                    "text/plain",
                    1,
                )
                job = get_background_jobs().for_training(training.id)

                def worker(active_job) -> None:
                    while True:
                        active_job.checkpoint()
                        time.sleep(0.005)

                job.start(worker)
                app = AppTest.from_file(PROJECT_ROOT / "app.py").run(timeout=30)
                stop_button = next(button for button in app.button if button.label == "עצור את העיבוד")
                stop_button.click().run(timeout=30)

                deadline = time.monotonic() + 1
                while job.snapshot().active and time.monotonic() < deadline:
                    time.sleep(0.005)

                self.assertEqual(job.snapshot().state, "cancelled")
                self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
