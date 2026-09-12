# DCA Training Analyzer

A local, single-user Streamlit workspace for military-training analysis. Each
training is isolated in SQLite and its own persistent Chroma collection. Gemini
on Vertex AI extracts evidence from media, drafts analysis, produces an HTML/PDF
summary report, and powers a report-aware LangChain chat.

## Pages

1. **Training report** — create a training, upload a browser folder, select a
   local directory, or import a shared Google Drive folder. One button extracts
   the evidence, updates the training's Chroma collection, and generates the
   report from the HTML template. Uploaded sources can be selected and deleted;
   managed local/cloud copies and their indexed chunks are removed with them.
2. **Report chat** — inspect the current report and ask questions about the
   report, training evidence, glossary, or general knowledge. Requested report
   changes appear as diffs and create a new version only after explicit Apply.
3. **Knowledge base** — upload shared reference material and import a global
   glossary. This information is retrieved alongside every training collection.

The chat always injects Chroma retrieval results into the model input. The agent
also receives the retrieval tool for follow-up searches, so answering does not
depend on Gemini deciding to make the first tool call.

## Prerequisites

- Python 3.14 and `uv`
- a Google Cloud project with Vertex AI enabled
- a private Google Cloud Storage bucket for image/audio/video/PDF staging
- a service account allowed to use Vertex AI and read/write/delete objects in
  that bucket
- the Google Drive API enabled when Drive-folder import is used

Place the service-account key at:

```text
./service_account.json
```

It is ignored by Git.

## Report template

The application does not replace `training_summary_report_template.html`.
Gemini writes section content only, while the template continues to own layout
and styling. Existing templates work when their report sections have unique
HTML `id` attributes. For more control, use explicit markers:

```html
<section
  data-report-section="executive_summary"
  data-report-title="Executive summary">
</section>
```

Store the model instruction for that section in
`prompts/report_sections/executive_summary.md`. All other model-facing
instructions are editable Markdown files under `prompts/` and are loaded at
runtime by `training_analyzer.prompts`.

To start from the bundled RTL example:

```bash
cp training_summary_report_template.example.html training_summary_report_template.html
```

## Configure and run

```bash
uv sync
uv run playwright install chromium

export TRAINING_MEDIA_BUCKET=your-private-bucket
export GEMINI_MODEL=gemini-2.5-flash
export GEMINI_EMBEDDING_MODEL=gemini-embedding-001
export GOOGLE_CLOUD_LOCATION=global

# Optional bounded parallelism (defaults shown)
export TRAINING_PROCESS_WORKERS=4
export TRAINING_EMBED_WORKERS=4

uv run streamlit run app.py
```

Optional database override:

```bash
export TRAINING_ANALYZER_DB=/absolute/path/to/training-analyzer.db
```

## Storage and privacy

- Metadata, approvals, chat, proposals, and report versions are stored in
  `data/training_analyzer.db`.
- Browser uploads and generated reports are stored under
  `data/trainings/<training-id>/`.
- Embeddings are stored in `data/chroma/`, one collection per training.
- Text files are extracted locally. Images, audio, video, and PDFs are uploaded
  to `gs://$TRAINING_MEDIA_BUCKET/trainings/<training-id>/...` and referenced by
  Gemini through their private GCS URI.
- Local-folder sources are referenced in place; the app does not modify them.
- Google Drive imports are downloaded recursively through the service account.
  Share the source folder with the service account's `client_email` first.
  Google Docs, Sheets, Slides, and Drawings are exported to PDF.
- `data/` is ignored by Git because it may contain sensitive material.

## Verification

```bash
uv run python -m unittest discover -s tests -v
uv run streamlit run app.py
```

PDF export uses Playwright's Chromium browser. If export reports that Chromium
is unavailable, rerun `uv run playwright install chromium`.
