"""Shared Gemini configuration for the Streamlit pages."""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Sequence

from google.auth.credentials import Credentials
from google.oauth2 import service_account
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVICE_ACCOUNT_FILE = PROJECT_ROOT / "service_account.json"
GOOGLE_CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def location() -> str:
    return os.getenv("GOOGLE_CLOUD_LOCATION", "global")


def load_credentials(scopes: Sequence[str] | None = None) -> tuple[Credentials, str]:
    if not SERVICE_ACCOUNT_FILE.is_file():
        raise FileNotFoundError(
            f"Service-account credentials were not found at {SERVICE_ACCOUNT_FILE}."
        )

    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=list(scopes or [GOOGLE_CLOUD_SCOPE]),
    )
    project_id = credentials.project_id
    if not project_id:
        raise ValueError("The service-account file does not contain a project_id.")
    return credentials, project_id


def create_chat_model(*, temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    credentials, project_id = load_credentials()
    return ChatGoogleGenerativeAI(
        model=model_name(),
        credentials=credentials,
        project=project_id,
        location=location(),
        vertexai=True,
        temperature=temperature,
    )


def create_embedding_model() -> GoogleGenerativeAIEmbeddings:
    credentials, project_id = load_credentials()
    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        credentials=credentials,
        project=project_id,
        location=location(),
        vertexai=True,
    )
