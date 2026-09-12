"""Recursive Google Drive folder import through the configured service account."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from training_analyzer.gemini import PROJECT_ROOT, load_credentials
from training_analyzer.ingestion import IGNORED_FILE_NAMES, TrainingIngestion, _safe_relative_path
from training_analyzer.models import Source


DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
SHORTCUT_MIME_TYPE = "application/vnd.google-apps.shortcut"
GOOGLE_APPS_PREFIX = "application/vnd.google-apps."
PDF_EXPORT_TYPES = {
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
    "application/vnd.google-apps.drawing",
}


def folder_id_from_url(value: str) -> str:
    """Extract a Drive folder ID from supported sharing URL shapes."""
    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.netloc not in {
        "drive.google.com",
        "docs.google.com",
    }:
        raise ValueError("Enter a valid Google Drive folder URL.")
    match = re.search(r"/folders/([A-Za-z0-9_-]+)", parsed.path)
    if match:
        return match.group(1)
    query_id = parse_qs(parsed.query).get("id", [""])[0]
    if re.fullmatch(r"[A-Za-z0-9_-]+", query_id):
        return query_id
    raise ValueError("The Google Drive URL does not contain a folder ID.")


class GoogleDriveFolderImporter:
    """Download a Drive folder tree into managed Source storage."""

    def __init__(self, ingestion: TrainingIngestion) -> None:
        credentials, _ = load_credentials([DRIVE_READONLY_SCOPE])
        self.drive = build("drive", "v3", credentials=credentials, cache_discovery=False)
        self.ingestion = ingestion

    def import_folder(self, training_id: str, folder_url: str) -> tuple[list[Source], list[str]]:
        folder_id = folder_id_from_url(folder_url)
        root = self._metadata(folder_id)
        if root["mimeType"] != FOLDER_MIME_TYPE:
            raise ValueError("The URL must point to a Google Drive folder.")

        sources: list[Source] = []
        warnings: list[str] = []
        visited: set[str] = set()
        root_name = _safe_relative_path(root["name"]).name
        self._walk(training_id, folder_id, Path(root_name), visited, sources, warnings)
        return sources, warnings

    def _metadata(self, file_id: str) -> dict:
        return self.drive.files().get(
            fileId=file_id,
            fields="id,name,mimeType,shortcutDetails",
            supportsAllDrives=True,
        ).execute()

    def _children(self, folder_id: str) -> list[dict]:
        children: list[dict] = []
        page_token = None
        while True:
            response = self.drive.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                spaces="drive",
                pageSize=1000,
                pageToken=page_token,
                orderBy="folder,name_natural",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                fields="nextPageToken,files(id,name,mimeType,shortcutDetails)",
            ).execute()
            children.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                return children

    def _walk(
        self,
        training_id: str,
        folder_id: str,
        relative_folder: Path,
        visited: set[str],
        sources: list[Source],
        warnings: list[str],
    ) -> None:
        if folder_id in visited:
            warnings.append(f"Skipped repeated folder: {relative_folder.as_posix()}")
            return
        visited.add(folder_id)
        for item in self._children(folder_id):
            display_name = item["name"]
            if display_name in IGNORED_FILE_NAMES:
                continue
            if item["mimeType"] == SHORTCUT_MIME_TYPE:
                target_id = item.get("shortcutDetails", {}).get("targetId")
                if not target_id:
                    warnings.append(f"Skipped unresolved shortcut: {display_name}")
                    continue
                item = self._metadata(target_id)
                item["name"] = display_name
            safe_name = _safe_relative_path(item["name"]).name
            relative_path = relative_folder / safe_name
            if item["mimeType"] == FOLDER_MIME_TYPE:
                self._walk(training_id, item["id"], relative_path, visited, sources, warnings)
                continue
            try:
                source = self._download(training_id, item, relative_path)
                sources.append(source)
            except Exception as exc:
                warnings.append(f"{relative_path.as_posix()}: {exc}")

    def _download(self, training_id: str, item: dict, relative_path: Path) -> Source:
        mime_type = item["mimeType"]
        if mime_type.startswith(GOOGLE_APPS_PREFIX):
            if mime_type not in PDF_EXPORT_TYPES:
                raise ValueError(f"Unsupported Google Workspace type: {mime_type}")
            request = self.drive.files().export_media(fileId=item["id"], mimeType="application/pdf")
            if relative_path.suffix.lower() != ".pdf":
                relative_path = relative_path.with_name(relative_path.name + ".pdf")
        else:
            request = self.drive.files().get_media(fileId=item["id"], supportsAllDrives=True)

        relative_path = _safe_relative_path(relative_path.as_posix())
        destination = PROJECT_ROOT / "data" / "trainings" / training_id / "sources" / "google-drive" / relative_path
        if destination.exists():
            destination = destination.with_name(f"{destination.stem}-{item['id'][:8]}{destination.suffix}")
            relative_path = relative_path.with_name(destination.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            downloader = MediaIoBaseDownload(stream, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        managed_relative = (Path("google-drive") / relative_path).as_posix()
        return self.ingestion.register_stored_file(training_id, managed_relative, destination)
