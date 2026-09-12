"""Upload training data and generate the evidence-grounded report."""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from training_analyzer.google_drive import GoogleDriveFolderImporter
from training_analyzer.ingestion import IGNORED_FILE_NAMES, TrainingIngestion
from training_analyzer.knowledge_base import TrainingKnowledgeBase, remove_source_from_index
from training_analyzer.reports import ReportTemplate, ReportWorkspace
from training_analyzer.ui import (
    configure_page,
    get_background_jobs,
    get_media_store,
    get_repository,
    render_sidebar,
)
from dotenv import load_dotenv

load_dotenv(override=False)
configure_page("DCA · יצירת דוח אימון")
repository = get_repository()
training = render_sidebar()
training_id = training.id
ingestion = TrainingIngestion(repository, get_media_store())
report_job = get_background_jobs().for_training(training_id)

st.markdown('<p class="dca-kicker">DCA TRAINING ANALYZER</p>', unsafe_allow_html=True)
st.title(training.name)
st.write("העלה את חומרי האימון ולאחר מכן הפק דוח אחד המבוסס על התבנית ועל מאגרי הידע.")

st.header("חומרי האימון")
browser_tab, drive_tab, local_tab = st.tabs(["העלאת תיקייה", "Google Drive", "תיקייה מקומית"])

with browser_tab:
    uploads = st.file_uploader("בחר תיקייה", accept_multiple_files="directory")
    if st.button("שמור את הקבצים", type="primary", disabled=not uploads):
        kept = [item for item in uploads if Path(item.name).name not in IGNORED_FILE_NAMES]
        for item in kept:
            ingestion.import_uploaded_file(training_id, item.name, item.getvalue())
        st.success(f"נשמרו {len(kept)} קבצים")
        st.rerun()

with drive_tab:
    drive_url = st.text_input(
        "קישור לתיקיית Google Drive",
        placeholder="https://drive.google.com/drive/folders/...",
    )
    st.caption("יש לשתף את התיקייה עם כתובת הדוא״ל של חשבון השירות.")
    if st.button("ייבא מ-Google Drive", disabled=not drive_url.strip()):
        try:
            with st.spinner("מוריד את התיקייה..."):
                imported, warnings = GoogleDriveFolderImporter(ingestion).import_folder(training_id, drive_url)
            st.success(f"נשמרו {len(imported)} קבצים")
            for warning in warnings:
                st.warning(warning)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

with local_tab:
    local_path = st.text_input("נתיב מלא לתיקייה המקומית")
    if st.button("קרא תיקייה מקומית", disabled=not local_path.strip()):
        try:
            imported = ingestion.import_local_folder(training_id, Path(local_path))
            st.success(f"נרשמו {len(imported)} קבצים")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

sources = repository.list_sources(training_id)
st.metric("קבצים באימון", len(sources))
if sources:
    with st.expander("הצג קבצים", expanded=False):
        st.dataframe(
            [{"קובץ": source.relative_path, "סוג": source.mime_type, "מצב": source.status} for source in sources],
            width="stretch",
            hide_index=True,
        )
        source_options = {source.id: source.relative_path for source in sources}
        selected_sources = st.multiselect(
            "קבצים למחיקה",
            list(source_options),
            format_func=lambda source_id: source_options[source_id],
            placeholder="בחר קובץ אחד או יותר",
        )
        if st.button(
            "מחק קבצים נבחרים",
            disabled=not selected_sources,
            type="secondary",
        ):
            try:
                for source_id in selected_sources:
                    source = repository.get_source(source_id)
                    remove_source_from_index(training_id, source.id)
                    ingestion.delete_source(source)
                st.success(f"נמחקו {len(selected_sources)} קבצים")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

st.divider()
st.header("דוח סיכום אימון")
try:
    template = ReportTemplate()
    st.caption(f"התבנית מוכנה וכוללת {len(template.sections)} סעיפים.")
    template_ready = True
except Exception as exc:
    st.error(str(exc))
    template_ready = False

job_snapshot = report_job.snapshot()
if st.button(
    "עבד את החומרים והפק דוח",
    type="primary",
    disabled=not sources or not template_ready or job_snapshot.active,
    width="stretch",
):
    def generate_report(job) -> None:
        def phase(start: float, end: float):
            def update(message: str, fraction: float) -> None:
                job.update(message, min(1.0, start + (end - start) * fraction))

            return update

        job.update("בודק אילו מקורות כבר עובדו ואילו מקורות דורשים ניסיון נוסף", 0.0)
        processed_sources = ingestion.process_all(
            training_id,
            on_progress=phase(0.0, 0.45),
            cancel_requested=job.cancel_requested,
        )
        job.checkpoint()
        usable = [source for source in processed_sources if source.status == "processed"]
        if not usable:
            raise ValueError("לא נמצא אף מקור שניתן לעבד.")
        job.update(
            f"נמצאו {len(usable)} מקורות זמינים; בונה מאגר Chroma עם Gemini Embeddings",
            0.45,
        )
        TrainingKnowledgeBase(repository, training_id).rebuild(on_progress=phase(0.45, 0.68))
        job.checkpoint()
        job.update("מאגר הראיות מוכן; Gemini מנסח את סעיפי הדוח לפי התבנית", 0.68)
        ReportWorkspace(repository, training_id).generate(
            on_progress=phase(0.68, 1.0),
            cancel_requested=job.cancel_requested,
        )

    report_job.start(generate_report)
    st.rerun()


@st.fragment(run_every=1)
def render_report_job() -> None:
    snapshot = report_job.snapshot()
    if snapshot.state == "idle":
        return
    state = "running" if snapshot.active else "complete" if snapshot.state == "complete" else "error"
    label = {
        "running": "מכין את דוח האימון",
        "cancelling": "עוצר את הפקת הדוח",
        "complete": "הדוח מוכן",
        "cancelled": "הפקת הדוח הופסקה",
        "failed": "הפקת הדוח נכשלה",
    }.get(snapshot.state, snapshot.message)
    with st.status(label, expanded=snapshot.active, state=state):
        st.progress(snapshot.progress, text=snapshot.message)
        for message in snapshot.history[-5:]:
            st.write(message)
        if snapshot.state == "running":
            if st.button("עצור את העיבוד", type="secondary", key=f"stop-report-{snapshot.run_id}"):
                report_job.cancel()
        elif snapshot.state == "cancelling":
            st.warning("בקשת העצירה התקבלה. הפעולה הפעילה תסתיים, ושלבים נוספים לא יתחילו.")
        elif snapshot.state == "failed":
            st.error(snapshot.error)

    completion_key = f"report-job-rendered-{training_id}"
    if snapshot.state == "complete" and st.session_state.get(completion_key) != snapshot.run_id:
        st.session_state[completion_key] = snapshot.run_id
        st.rerun()


render_report_job()

failed_sources = [source for source in repository.list_sources(training_id) if source.status == "failed"]
if failed_sources and not report_job.snapshot().active:
    with st.expander(f"קבצים שלא עובדו ({len(failed_sources)})"):
        for source in failed_sources:
            st.warning(f"{source.relative_path}: {source.error}")

report = repository.latest_report(training_id)
if report:
    st.success(f"הדוח האחרון: גרסה {report.version}")
    components.html(Path(report.html_path).read_text(encoding="utf-8"), height=680, scrolling=True)
    left, right = st.columns(2)
    left.download_button(
        "הורד HTML",
        Path(report.html_path).read_bytes(),
        file_name=f"{training.name}-v{report.version}.html",
        mime="text/html",
        width="stretch",
    )
    if right.button("פתח שיחה על הדוח", type="primary", width="stretch"):
        st.switch_page("pages/1_Chat.py")
