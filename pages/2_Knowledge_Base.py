"""Manage the glossary and general evidence shared by every training."""

from pathlib import Path

import streamlit as st

from training_analyzer.glossary import parse_glossary_file
from training_analyzer.google_drive import GoogleDriveFolderImporter
from training_analyzer.ingestion import IGNORED_FILE_NAMES, TrainingIngestion
from training_analyzer.knowledge_base import TrainingKnowledgeBase
from training_analyzer.repository import GLOBAL_KNOWLEDGE_ID
from training_analyzer.ui import configure_page, get_media_store, get_repository, render_brand_sidebar


configure_page("DCA · ידע כללי ומילון")
repository = get_repository()
render_brand_sidebar()
ingestion = TrainingIngestion(repository, get_media_store())

st.markdown('<p class="dca-kicker">SHARED KNOWLEDGE</p>', unsafe_allow_html=True)
st.title("ידע כללי ומילון")
st.write("המידע במסך הזה זמין בכל דוחות האימון ובכל השיחות.")

glossary_tab, knowledge_tab = st.tabs(["מילון מונחים", "מאגר ידע כללי"])

with glossary_tab:
    st.subheader("העלאת מילון")
    glossary_file = st.file_uploader("CSV, TSV, JSON, TXT או Markdown", type=["csv", "tsv", "json", "txt", "md"])
    if st.button("ייבא מילון", type="primary", disabled=glossary_file is None):
        try:
            records = parse_glossary_file(glossary_file.name, glossary_file.getvalue())
            for record in records:
                repository.upsert_glossary(**record)
            st.success(f"יובאו {len(records)} מונחים")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    with st.expander("הוספת מונח ידנית"):
        with st.form("manual-glossary", clear_on_submit=True):
            term = st.text_input("מונח")
            meaning = st.text_input("משמעות")
            variants = st.text_input("גרסאות ושיבושים")
            notes = st.text_input("הערות")
            if st.form_submit_button("שמור"):
                if not term.strip() or not meaning.strip():
                    st.error("יש להזין מונח ומשמעות.")
                else:
                    repository.upsert_glossary(term, meaning, variants, notes)
                    st.rerun()
    st.dataframe(repository.list_glossary(), width="stretch", hide_index=True)

with knowledge_tab:
    st.subheader("מקורות כלליים")
    uploads = st.file_uploader("העלה קבצים", accept_multiple_files=True, key="general-files")
    if st.button("שמור מקורות", disabled=not uploads):
        kept = [item for item in uploads if Path(item.name).name not in IGNORED_FILE_NAMES]
        for item in kept:
            ingestion.import_uploaded_file(GLOBAL_KNOWLEDGE_ID, item.name, item.getvalue())
        st.success(f"נשמרו {len(kept)} קבצים")
        st.rerun()
    drive_url = st.text_input("קישור לתיקיית Google Drive", key="general-drive")
    if st.button("ייבא תיקיית ידע מ-Google Drive", disabled=not drive_url.strip()):
        try:
            imported, warnings = GoogleDriveFolderImporter(ingestion).import_folder(GLOBAL_KNOWLEDGE_ID, drive_url)
            st.success(f"נשמרו {len(imported)} קבצים")
            for warning in warnings:
                st.warning(warning)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    sources = repository.list_sources(GLOBAL_KNOWLEDGE_ID)
    st.metric("מקורות כלליים", len(sources))
    if sources:
        st.dataframe(
            [{"קובץ": source.relative_path, "סוג": source.mime_type, "מצב": source.status} for source in sources],
            width="stretch",
            hide_index=True,
        )

st.divider()
if st.button("עבד ועדכן את מאגר הידע", type="primary", width="stretch"):
    try:
        with st.status("מעדכן את מאגר הידע הכללי", expanded=True) as status:
            progress = st.progress(0.0, text="בודק את מקורות הידע")

            def extraction_progress(message: str, fraction: float) -> None:
                progress.progress(0.6 * fraction, text=message)

            def indexing_progress(message: str, fraction: float) -> None:
                progress.progress(0.6 + 0.4 * fraction, text=message)

            status.write("מחלץ תוכן חדש ומשתמש מחדש במקורות שכבר עובדו.")
            processed = ingestion.process_all(GLOBAL_KNOWLEDGE_ID, on_progress=extraction_progress)
            status.write("כלי פעיל: Chroma עם Gemini Embeddings.")
            count = TrainingKnowledgeBase(repository, GLOBAL_KNOWLEDGE_ID).rebuild(on_progress=indexing_progress)
            progress.progress(1.0, text="מאגר הידע מוכן")
            status.update(label="מאגר הידע עודכן", state="complete")
        failures = [source for source in processed if source.status == "failed"]
        st.success(f"מאגר הידע עודכן עם {count} מקטעים")
        for source in failures:
            st.warning(f"{source.relative_path}: {source.error}")
    except Exception as exc:
        st.error(str(exc))
