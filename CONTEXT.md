# Training Analysis

This context turns heterogeneous military-training evidence into an analyst-reviewed timeline, force map, report, and grounded conversation.

## Language

**Training Workspace**:
A named, isolated analysis of one military exercise, containing its sources, evidence, review state, report versions, and chat history.
_Avoid_: Project, session, global knowledge base

**Source**:
An original file or read-only local file reference supplied for a Training Workspace.
_Avoid_: Document, upload

**Evidence**:
A searchable, source-linked observation or text span extracted from a Source.
_Avoid_: Chunk, context

**Force Entity**:
A unit, callsign, person, or operational role identified in training Evidence and reviewed by the analyst.
_Avoid_: Actor, participant

**Timeline Event**:
An observed training occurrence positioned on the synchronized exercise timeline and linked to Evidence.
_Avoid_: Log entry, incident

**Global Glossary**:
Reusable military terminology and transcription corrections applied to every Training Workspace.

**Training Glossary Override**:
A terminology rule belonging to one Training Workspace that takes precedence over the Global Glossary.

**Report Version**:
An immutable set of report sections rendered through the report template.
_Avoid_: Draft file, current HTML

**Report Proposal**:
A pending, source-aware change to one report section that requires analyst approval before becoming a Report Version.
_Avoid_: Edit, patch
