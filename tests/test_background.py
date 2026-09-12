import time
import unittest

from training_analyzer.background import BackgroundJob


class BackgroundJobTests(unittest.TestCase):
    def test_cancelled_job_stops_at_checkpoint(self) -> None:
        job = BackgroundJob()

        def worker(active_job: BackgroundJob) -> None:
            while True:
                active_job.checkpoint()
                time.sleep(0.005)

        job.start(worker)
        job.cancel()
        deadline = time.monotonic() + 1
        while job.snapshot().active and time.monotonic() < deadline:
            time.sleep(0.005)

        snapshot = job.snapshot()
        self.assertEqual(snapshot.state, "cancelled")
        self.assertIn("הופסק", snapshot.message)

    def test_completed_job_reaches_full_progress(self) -> None:
        job = BackgroundJob()
        job.start(lambda active_job: active_job.update("working", 0.5))
        deadline = time.monotonic() + 1
        while job.snapshot().active and time.monotonic() < deadline:
            time.sleep(0.005)

        snapshot = job.snapshot()
        self.assertEqual(snapshot.state, "complete")
        self.assertEqual(snapshot.progress, 1.0)


if __name__ == "__main__":
    unittest.main()
