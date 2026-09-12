"""Discuss the generated report and approve report changes."""

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from training_analyzer.reports import ReportWorkspace, TrainingChat
from training_analyzer.ui import configure_page, get_repository, render_sidebar


configure_page("DCA · שיחה על הדוח")
repository = get_repository()
training = render_sidebar()
report = repository.latest_report(training.id)

st.markdown('<p class="dca-kicker">REPORT REVIEW</p>', unsafe_allow_html=True)
st.title("שיחה על דוח האימון")

if report is None:
    st.warning("עדיין לא הופק דוח עבור האימון הזה.")
    if st.button("חזרה להפקת הדוח", type="primary"):
        st.switch_page("app.py")
    st.stop()

report_workspace = ReportWorkspace(repository, training.id)
report_column, chat_column = st.columns([1.35, 1], gap="large")

with report_column:
    st.subheader(f"הדוח · גרסה {report.version}")
    components.html(Path(report.html_path).read_text(encoding="utf-8"), height=760, scrolling=True)
    html_column, pdf_column = st.columns(2)
    html_column.download_button(
        "הורד HTML",
        Path(report.html_path).read_bytes(),
        file_name=f"{training.name}-v{report.version}.html",
        mime="text/html",
        width="stretch",
    )
    if pdf_column.button("צור PDF", width="stretch"):
        try:
            pdf_path = report_workspace.export_pdf(report)
            repository.set_report_pdf(report.id, str(pdf_path))
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if report.pdf_path and Path(report.pdf_path).exists():
        st.download_button(
            "הורד PDF",
            Path(report.pdf_path).read_bytes(),
            file_name=f"{training.name}-v{report.version}.pdf",
            mime="application/pdf",
        )
    versions = repository.list_reports(training.id)
    if len(versions) > 1:
        with st.expander("גרסאות קודמות"):
            restore_id = st.selectbox(
                "גרסה לשחזור",
                [item.id for item in versions],
                format_func=lambda value: f"גרסה {next(item.version for item in versions if item.id == value)}",
            )
            if st.button("שחזר כגרסה חדשה"):
                report_workspace.restore(repository.get_report(restore_id))
                st.rerun()

with chat_column:
    st.subheader("שאל את מנתח האימון")
    st.caption("אפשר לשאול על הראיות או לבקש שינוי בדוח. שינוי ימתין לאישור שלך.")
    for message in repository.list_messages(training.id):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("שאלה או בקשת שינוי בדוח")
    if question:
        try:
            with st.spinner("מחפש בראיות ובדוח..."):
                TrainingChat(repository, training.id).ask(question)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    for proposal in repository.pending_proposals(training.id):
        with st.container(border=True):
            st.markdown(f"**הצעת שינוי לסעיף `{proposal.section_id}`**")
            st.write(proposal.rationale)
            st.code(report_workspace.proposal_diff(proposal), language="diff")
            apply_column, reject_column = st.columns(2)
            if apply_column.button("החל שינוי", key=f"apply-{proposal.id}", type="primary"):
                report_workspace.apply(proposal)
                st.rerun()
            if reject_column.button("דחה", key=f"reject-{proposal.id}"):
                report_workspace.reject(proposal)
                st.rerun()
