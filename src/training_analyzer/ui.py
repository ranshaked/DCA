"""Shared Streamlit shell for the three user-facing pages."""

from __future__ import annotations

import os
from datetime import date

import streamlit as st

from training_analyzer.background import BackgroundJobRegistry
from training_analyzer.ingestion import GCSMediaStore
from training_analyzer.migrations import import_legacy_glossary
from training_analyzer.models import Training
from training_analyzer.repository import WorkspaceRepository


def configure_page(title: str) -> None:
    st.set_page_config(page_title=title, page_icon="◼", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp { background: #090d0e; color: #f4f4f1; direction: rtl; }
        [data-testid="stSidebar"] { background: #101516; border-left: 1px solid #273033; }
        h1, h2, h3 { color: #f4f4f1; }
        .dca-kicker { color: #f07822; font-weight: 800; letter-spacing: .08em; }
        .dca-brand { font-size: 1.45rem; font-weight: 900; color: #f07822; direction: ltr; text-align: left; }
        div[data-testid="stButton"] button[kind="primary"] { background:#ef7622; color:#111; border-color:#ef7622; font-weight:800; }
        [data-testid="stMetricValue"] { color:#ef7622; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_repository() -> WorkspaceRepository:
    repository = WorkspaceRepository()
    import_legacy_glossary(repository)
    return repository


@st.cache_resource
def get_background_jobs() -> BackgroundJobRegistry:
    return BackgroundJobRegistry()


def get_media_store() -> GCSMediaStore | None:
    try:
        return GCSMediaStore()
    except Exception:
        return None


def render_brand_sidebar() -> None:
    with st.sidebar:
        st.markdown('<div class="dca-brand">DCA<br><small>TRAIN AS YOU FIGHT</small></div>', unsafe_allow_html=True)
        if os.getenv("TRAINING_MEDIA_BUCKET"):
            st.success("אחסון המדיה בענן מוגדר")
        else:
            st.warning("אחסון מדיה בענן אינו מוגדר; קובצי טקסט יעובדו כרגיל")


def render_sidebar(*, require_training: bool = True) -> Training | None:
    repository = get_repository()
    trainings = repository.list_trainings()
    with st.sidebar:
        st.markdown('<div class="dca-brand">DCA<br><small>TRAIN AS YOU FIGHT</small></div>', unsafe_allow_html=True)
        st.subheader("סביבת אימון")
        selected_id = None
        if trainings:
            options = {item.id: item.name for item in trainings}
            current = st.session_state.get("training_id")
            index = list(options).index(current) if current in options else 0
            selected_id = st.selectbox("אימון פעיל", list(options), format_func=lambda key: options[key], index=index)
            st.session_state.training_id = selected_id
            selected_training = repository.get_training(selected_id)
            with st.expander("פרטי האימון הפעיל"):
                with st.form(f"edit-training-{selected_id}"):
                    current_date = (
                        date.fromisoformat(selected_training.exercise_date)
                        if selected_training.exercise_date
                        else None
                    )
                    edit_name = st.text_input("שם האימון", value=selected_training.name, key="edit-name")
                    edit_date = st.date_input("תאריך", value=current_date, key="edit-date")
                    edit_unit = st.text_input("יחידה", value=selected_training.unit_name, key="edit-unit")
                    edit_location = st.text_input("מיקום", value=selected_training.location, key="edit-location")
                    edit_notes = st.text_area("הערות", value=selected_training.notes, key="edit-notes")
                    if st.form_submit_button("שמור פרטים"):
                        if not edit_name.strip():
                            st.error("יש להזין שם אימון.")
                        else:
                            repository.update_training_metadata(
                                selected_id,
                                name=edit_name,
                                exercise_date=edit_date.isoformat() if edit_date else "",
                                unit_name=edit_unit,
                                location=edit_location,
                                notes=edit_notes,
                            )
                            st.rerun()
        with st.expander("פתיחת אימון חדש", expanded=not trainings):
            with st.form("new-training"):
                name = st.text_input("שם האימון")
                exercise_date = st.date_input("תאריך", value=None)
                unit_name = st.text_input("יחידה")
                location_name = st.text_input("מיקום")
                notes = st.text_area("הערות")
                if st.form_submit_button("צור סביבת אימון", type="primary"):
                    if not name.strip():
                        st.error("יש להזין שם אימון.")
                    else:
                        created = repository.create_training(
                            name,
                            exercise_date.isoformat() if exercise_date else "",
                            unit_name,
                            location_name,
                            notes,
                        )
                        st.session_state.training_id = created.id
                        st.rerun()
        if os.getenv("TRAINING_MEDIA_BUCKET"):
            st.success("אחסון המדיה בענן מוגדר")
        else:
            st.warning("אחסון מדיה בענן אינו מוגדר; קובצי טקסט יעובדו כרגיל")
    if selected_id:
        return repository.get_training(selected_id)
    if require_training:
        st.info("צור סביבת אימון בסרגל הצד כדי להתחיל.")
        st.stop()
    return None
