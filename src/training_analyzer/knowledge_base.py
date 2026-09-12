"""Per-training persistent Chroma evidence store and retrieval tool."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from training_analyzer.gemini import PROJECT_ROOT, create_embedding_model
from training_analyzer.concurrency import worker_count
from training_analyzer.prompts import load_prompt
from training_analyzer.repository import GLOBAL_KNOWLEDGE_ID, WorkspaceRepository


CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"


def collection_name_for(training_id: str) -> str:
    """Map internal workspace IDs to names accepted by Chroma."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", f"training_{training_id}")
    name = re.sub(r"[_-]{2,}", "_", name).strip("._-")
    if len(name) < 3:
        name = f"training_{hashlib.sha256(training_id.encode()).hexdigest()[:12]}"
    return name[:512].rstrip("._-")


def remove_source_from_index(training_id: str, source_id: str) -> int:
    """Remove source chunks without constructing an embedding client."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = client.get_collection(collection_name_for(training_id))
    except Exception:
        return 0
    result = collection.get(where={"source_id": source_id}, include=[])
    ids = result.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)


class TrainingKnowledgeBase:
    """Own indexing and retrieval for one isolated Training Workspace."""

    def __init__(self, repository: WorkspaceRepository, training_id: str) -> None:
        self.repository = repository
        self.training_id = training_id
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.store = Chroma(
            collection_name=collection_name_for(training_id),
            embedding_function=create_embedding_model(),
            persist_directory=str(CHROMA_DIR),
        )
        self.global_store = self.store if training_id == GLOBAL_KNOWLEDGE_ID else Chroma(
            collection_name=collection_name_for(GLOBAL_KNOWLEDGE_ID),
            embedding_function=create_embedding_model(),
            persist_directory=str(CHROMA_DIR),
        )

    def rebuild(
        self,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> int:
        notify = on_progress or (lambda _message, _fraction: None)
        notify("Chroma מחלק את הראיות למקטעים", 0.05)
        documents: list[Document] = []
        ids: list[str] = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=1_200, chunk_overlap=200)
        for source in self.repository.list_sources(self.training_id):
            if source.status != "processed" or not source.extracted_text:
                continue
            for position, text in enumerate(splitter.split_text(source.extracted_text)):
                documents.append(Document(page_content=text, metadata={"source": source.relative_path, "source_id": source.id, "mime_type": source.mime_type, "offset_seconds": source.offset_seconds}))
                ids.append(f"{source.id}:{source.sha256}:{position}")
        for item in self.repository.list_glossary(self.training_id):
            text = f"מונח: {item['term']}\nמשמעות: {item['meaning']}\nגרסאות: {item['variants']}\nהערות: {item['notes']}"
            documents.append(Document(page_content=text, metadata={"source": "glossary", "source_id": item["id"], "mime_type": "text/glossary", "offset_seconds": 0}))
            ids.append(f"glossary:{hashlib.sha256((self.training_id + item['id']).encode()).hexdigest()}")
        for entity in self.repository.list_force_entities(self.training_id):
            documents.append(Document(page_content=f"מיפוי כוח: {entity['name']} | סוג: {entity['kind']} | תפקיד: {entity['role']} | ראיה: {entity['evidence']}", metadata={"source": "force-map", "source_id": entity["id"], "mime_type": "application/force-map", "offset_seconds": 0}))
            ids.append(f"force:{entity['id']}")
        for event in self.repository.list_timeline(self.training_id):
            documents.append(Document(page_content=f"אירוע בזמן {event['seconds']} שניות: {event['description']}", metadata={"source": event["source_name"] or "timeline", "source_id": event["id"], "mime_type": "application/timeline-event", "offset_seconds": event["seconds"]}))
            ids.append(f"timeline:{event['id']}")
        self.store.reset_collection()
        if documents:
            batch_size = 32
            batches = [
                (start, min(start + batch_size, len(documents)))
                for start in range(0, len(documents), batch_size)
            ]
            workers = worker_count("TRAINING_EMBED_WORKERS", 4, len(batches))
            notify(
                f"יוצר הטמעות במקביל באמצעות {workers} workers",
                0.12,
            )

            def embed_batch(start: int, end: int) -> tuple[int, int, list[list[float]]]:
                embedding_model = create_embedding_model()
                vectors = embedding_model.embed_documents(
                    [document.page_content for document in documents[start:end]],
                    batch_size=batch_size,
                )
                return start, end, vectors

            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            collection = client.get_collection(collection_name_for(self.training_id))
            completed_documents = 0
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="training-embed") as executor:
                futures = [executor.submit(embed_batch, start, end) for start, end in batches]
                for future in as_completed(futures):
                    start, end, vectors = future.result()
                    batch_documents = documents[start:end]
                    collection.add(
                        ids=ids[start:end],
                        embeddings=vectors,
                        documents=[document.page_content for document in batch_documents],
                        metadatas=[document.metadata for document in batch_documents],
                    )
                    completed_documents += end - start
                    notify(
                        f"שומר ב-Chroma: {completed_documents} מתוך {len(documents)} מקטעים",
                        0.15 + 0.85 * completed_documents / len(documents),
                    )
        notify(f"מאגר הראיות מוכן עם {len(documents)} מקטעים", 1.0)
        return len(documents)

    def search(self, query: str, k: int = 8, *, include_global: bool = True) -> str:
        matches_with_scores = []
        stores = [self.store]
        if include_global and self.global_store is not self.store:
            stores.append(self.global_store)
        for store in stores:
            if store.get(include=[])["ids"]:
                matches_with_scores.extend(store.similarity_search_with_score(query, k=k))
        if not matches_with_scores:
            return "No indexed evidence is available for this training."
        matches = [document for document, _ in sorted(matches_with_scores, key=lambda item: item[1])[:k]]
        return "\n\n".join(
            f"Source: {item.metadata['source']} @ +{item.metadata.get('offset_seconds', 0):g}s\n{item.page_content}"
            for item in matches
        )

    def as_tool(self):
        knowledge_base = self

        @tool(description=load_prompt("search_training_evidence_tool"))
        def search_training_evidence(query: str) -> str:
            return knowledge_base.search(query)

        return search_training_evidence

    def delete(self) -> None:
        self.store.delete_collection()

    def remove_source(self, source_id: str) -> int:
        """Remove every indexed evidence chunk belonging to one Source."""
        return remove_source_from_index(self.training_id, source_id)
