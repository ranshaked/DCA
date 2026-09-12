"""Training-source import, cloud staging, and multimodal extraction."""

from __future__ import annotations

import hashlib
import mimetypes
import os
import queue
import re
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from google import genai
from google.cloud import storage
from google.genai.types import HttpOptions, Part

from training_analyzer.gemini import PROJECT_ROOT, load_credentials, location, model_name
from training_analyzer.concurrency import worker_count
from training_analyzer.models import Source
from training_analyzer.prompts import load_prompt
from training_analyzer.repository import WorkspaceRepository


TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".tsv", ".json", ".log", ".rpt", ".xml", ".yaml", ".yml", ".ini", ".toml", ".py", ".js", ".ts", ".html", ".css", ".sql"}
IGNORED_FILE_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}
SUPPORTED_MEDIA_PREFIXES = ("image/", "audio/", "video/")
ProgressCallback = Callable[[str, float], None]
PROGRESS_POLL_SECONDS = 0.25
PROGRESS_HEARTBEAT_SECONDS = 2.0


def _safe_relative_path(value: str) -> Path:
    parts = [re.sub(r"[^\w. -]+", "_", part).strip(". ") or "file" for part in Path(value).parts if part not in {"/", "..", "."}]
    return Path(*parts) if parts else Path("file")


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class GCSMediaStore:
    def __init__(self) -> None:
        bucket_name = os.getenv("TRAINING_MEDIA_BUCKET", "").strip()
        if not bucket_name:
            raise ValueError("TRAINING_MEDIA_BUCKET is not configured.")
        credentials, project_id = load_credentials()
        self.bucket = storage.Client(project=project_id, credentials=credentials).bucket(bucket_name)

    def validate(self) -> None:
        if not self.bucket.exists():
            raise ValueError(f"GCS bucket is unavailable: {self.bucket.name}")

    def upload(self, training_id: str, source_id: str, path: Path, relative_path: str) -> str:
        object_name = f"trainings/{training_id}/{source_id}/{_safe_relative_path(relative_path).as_posix()}"
        blob = self.bucket.blob(object_name)
        blob.upload_from_filename(path, timeout=900)
        return f"gs://{self.bucket.name}/{object_name}"

    def delete_training(self, training_id: str) -> None:
        for blob in self.bucket.list_blobs(prefix=f"trainings/{training_id}/"):
            blob.delete()

    def delete_source(self, source: Source) -> None:
        prefix = f"trainings/{source.training_id}/{source.id}/"
        for blob in self.bucket.list_blobs(prefix=prefix):
            blob.delete(timeout=120)


class TrainingIngestion:
    """Import and process a folder without exposing storage details to the UI."""

    def __init__(self, repository: WorkspaceRepository, media_store: GCSMediaStore | None = None) -> None:
        self.repository = repository
        self.media_store = media_store

    def import_local_folder(self, training_id: str, folder: Path) -> list[Source]:
        root = folder.expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("The selected path is not a directory.")
        imported = []
        for path in sorted(
            item
            for item in root.rglob("*")
            if item.is_file() and not item.is_symlink() and item.name not in IGNORED_FILE_NAMES
        ):
            resolved = path.resolve(strict=True)
            if resolved.is_relative_to(root):
                imported.append(self._register(training_id, resolved.relative_to(root).as_posix(), resolved))
        return imported

    def import_uploaded_file(self, training_id: str, relative_path: str, data: bytes) -> Source:
        safe_path = _safe_relative_path(relative_path)
        destination = PROJECT_ROOT / "data" / "trainings" / training_id / "sources" / safe_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return self.register_stored_file(training_id, safe_path.as_posix(), destination)

    def register_stored_file(self, training_id: str, relative_path: str, path: Path) -> Source:
        """Register a file already copied into this training's managed storage."""
        return self._register(training_id, relative_path, path)

    def _register(self, training_id: str, relative_path: str, path: Path) -> Source:
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return self.repository.add_source(training_id, relative_path, str(path), _digest(path), mime_type, path.stat().st_size)

    def process_all(
        self,
        training_id: str,
        on_progress: ProgressCallback | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> list[Source]:
        is_cancelled = cancel_requested or (lambda: False)
        sources = self.repository.list_sources(training_id)
        source_count = len(sources)
        total = max(len(sources), 1)
        progress_by_source = {
            source.id: 1.0 if source.status == "processed" else 0.0
            for source in sources
        }
        pending = [source for source in sources if source.status != "processed"]
        events: queue.Queue[tuple[str, str, float]] = queue.Queue()

        def notify_progress(message: str, fraction: float | None = None) -> None:
            if not on_progress:
                return
            completed_count = sum(value >= 1.0 for value in progress_by_source.values())
            remaining_count = source_count - completed_count
            file_progress = (
                f"התקדמות קבצים: {completed_count}/{source_count} הושלמו · "
                f"{remaining_count} נותרו"
            )
            aggregate = sum(progress_by_source.values()) / total if fraction is None else fraction
            on_progress(f"{message} | {file_progress}", aggregate)

        def publish(source_id: str, message: str, fraction: float) -> None:
            events.put((source_id, message, min(1.0, max(0.0, fraction))))

        class SourceCancelled(Exception):
            pass

        def process(source: Source) -> tuple[bool, Exception | None]:
            if is_cancelled():
                return True, None

            def worker_progress(message: str, fraction: float = 0.5) -> None:
                if is_cancelled():
                    raise SourceCancelled()
                publish(source.id, message, fraction)

            try:
                self._process(source, worker_progress)
                return is_cancelled(), None
            except SourceCancelled:
                return True, None
            except Exception as exc:
                return is_cancelled(), exc

        def report_events() -> int:
            reported = 0
            while True:
                try:
                    source_id, message, fraction = events.get_nowait()
                except queue.Empty:
                    return reported
                progress_by_source[source_id] = max(progress_by_source[source_id], fraction)
                notify_progress(message)
                reported += 1

        workers = worker_count("TRAINING_PROCESS_WORKERS", 4, len(pending))
        if pending:
            notify_progress(f"מעבד {len(pending)} מקורות במקביל באמצעות {workers} workers")
        started_at = time.monotonic()
        last_visible_update = started_at
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="training-source") as executor:
            futures = {executor.submit(process, source): source for source in pending}
            while futures:
                completed, _ = wait(
                    futures,
                    timeout=min(PROGRESS_POLL_SECONDS, PROGRESS_HEARTBEAT_SECONDS),
                    return_when=FIRST_COMPLETED,
                )
                if report_events():
                    last_visible_update = time.monotonic()
                for future in completed:
                    source = futures.pop(future)
                    cancelled, error = future.result()
                    if cancelled:
                        continue
                    if error is not None:
                        self.repository.update_source(source.id, status="failed", error=str(error))
                        message = f"המקור נכשל וממשיכים לבא: {source.relative_path}"
                    else:
                        message = f"המקור מוכן: {source.relative_path}"
                    progress_by_source[source.id] = 1.0
                    if on_progress:
                        notify_progress(message)
                        last_visible_update = time.monotonic()
                now = time.monotonic()
                if on_progress and futures and now - last_visible_update >= PROGRESS_HEARTBEAT_SECONDS:
                    elapsed_seconds = max(1, int(now - started_at))
                    notify_progress(
                        f"העיבוד נמשך: {len(futures)} מקורות פעילים · {elapsed_seconds} שניות",
                    )
                    last_visible_update = now
            report_events()
        if is_cancelled():
            notify_progress("חילוץ המקורות נעצר; מקורות שטרם התחילו לא יעובדו")
        else:
            notify_progress("חילוץ המקורות הסתיים", 1.0)
        return self.repository.list_sources(training_id)

    def delete_source(self, source: Source) -> None:
        """Delete managed copies while preserving external local-folder originals."""
        is_cloud_media = source.gcs_uri or source.mime_type.startswith(SUPPORTED_MEDIA_PREFIXES) or source.mime_type == "application/pdf"
        if self.media_store is not None and is_cloud_media:
            self.media_store.delete_source(source)
        managed_root = (PROJECT_ROOT / "data" / "trainings" / source.training_id / "sources").resolve()
        local_path = Path(source.local_path).resolve()
        if local_path.is_relative_to(managed_root):
            local_path.unlink(missing_ok=True)
        self.repository.delete_source(source.id)

    def _process(self, source: Source, notify: ProgressCallback | None = None) -> None:
        update = notify or (lambda _message, _fraction=0.0: None)
        path = Path(source.local_path)
        if path.name in IGNORED_FILE_NAMES:
            self.repository.update_source(source.id, status="processed", extracted_text="")
            return
        is_text = source.mime_type.startswith("text/") or path.suffix.lower() in TEXT_EXTENSIONS
        if is_text:
            update(f"קורא טקסט מקומי: {source.relative_path}", 0.35)
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            self.repository.update_source(source.id, status="processed", extracted_text=text)
            return
        supported = source.mime_type.startswith(SUPPORTED_MEDIA_PREFIXES) or source.mime_type == "application/pdf"
        if not supported:
            raise ValueError(f"Unsupported binary format: {source.mime_type}")
        if self.media_store is None:
            raise ValueError("Private GCS media staging is not configured.")
        update(f"מעלה מדיה ל-GCS פרטי: {source.relative_path}", 0.2)
        gcs_uri = source.gcs_uri or self.media_store.upload(source.training_id, source.id, path, source.relative_path)
        if not source.gcs_uri:
            self.repository.update_source(source.id, status="processing", gcs_uri=gcs_uri)
        credentials, project_id = load_credentials()
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location(),
            credentials=credentials,
            http_options=HttpOptions(api_version="v1", timeout=900_000),
        )
        update(f"Gemini מנתח תמונה, שמע או וידאו: {source.relative_path}", 0.55)
        prompt = load_prompt("media_extraction")
        response = client.models.generate_content(model=model_name(), contents=[Part.from_uri(file_uri=gcs_uri, mime_type=source.mime_type), prompt])
        update(f"שומר את הראיות שחולצו: {source.relative_path}", 0.9)
        text = (response.text or "").strip()
        if not text:
            raise ValueError("Gemini returned no searchable evidence.")
        self.repository.update_source(source.id, status="processed", extracted_text=text, gcs_uri=gcs_uri)
