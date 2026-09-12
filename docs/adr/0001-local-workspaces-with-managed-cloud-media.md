# Local workspaces with managed cloud media

The application is a single-user local Streamlit system with SQLite and per-training Chroma collections, while large media is staged in a preconfigured private GCS bucket for Gemini analysis. This preserves local workflow simplicity and training isolation without forcing multi-gigabyte media through inline model requests; deleting a Training Workspace must also delete its managed GCS prefix.
