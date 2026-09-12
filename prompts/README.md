# Application prompts

Every model-facing instruction is stored as editable Markdown in this directory.
The application loads prompts at runtime through `training_analyzer.prompts`.

## Prompt groups

- `media_extraction.md`: multimodal source extraction
- `force_map_analysis.md`: force-map analysis
- `timeline_analysis.md`: chronological event analysis
- `metadata_evidence_query.md`: metadata retrieval query
- `metadata_inference.md`: training metadata inference
- `report_evidence_query.md`: per-section retrieval query
- `report_section.md`: report-writing prompt
- `chat_system.md`: LangChain agent system prompt
- `chat_grounding.md`: retrieved-context wrapper for chat questions
- `chat_workspace_context.md`: active training and report context
- `search_training_evidence_tool.md`: LangChain retrieval-tool description
- `default_report_section.md`: fallback instruction for custom report sections
- `report_sections/*.md`: instructions keyed by each HTML `data-report-section`
- `schema/*.md`: model-visible structured-output field descriptions

Dynamic values use Python `string.Template` placeholders such as `$evidence`.
Keep placeholder names unchanged when editing a prompt.
