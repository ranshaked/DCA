import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from training_analyzer.knowledge_base import TrainingKnowledgeBase
from training_analyzer.repository import GLOBAL_KNOWLEDGE_ID, WorkspaceRepository


class KnowledgeBaseCollectionTests(unittest.TestCase):
    def test_global_knowledge_uses_valid_chroma_collection_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = WorkspaceRepository(root / "app.db")
            with patch("training_analyzer.knowledge_base.CHROMA_DIR", root / "chroma"):
                knowledge = TrainingKnowledgeBase(repository, GLOBAL_KNOWLEDGE_ID)

            self.assertEqual(knowledge.store._collection.name, "training_global_knowledge")

    def test_embedding_batches_run_in_parallel_while_chroma_writes_are_serial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = WorkspaceRepository(root / "app.db")
            training = repository.create_training("Parallel embedding")
            for position in range(40):
                source = repository.add_source(
                    training.id,
                    f"source-{position}.txt",
                    f"/tmp/source-{position}.txt",
                    f"embedding-{position}",
                    "text/plain",
                    10,
                )
                repository.update_source(source.id, status="processed", extracted_text=f"evidence {position}")

            lock = threading.Lock()
            active = 0
            maximum_active = 0
            stored: list[str] = []

            class FakeEmbeddingModel:
                def embed_documents(self, texts, **_kwargs):
                    nonlocal active, maximum_active
                    with lock:
                        active += 1
                        maximum_active = max(maximum_active, active)
                    time.sleep(0.04)
                    with lock:
                        active -= 1
                    return [[float(index), 0.0] for index, _ in enumerate(texts)]

            class FakeStore:
                def reset_collection(self):
                    return None

            class FakeCollection:
                def add(self, *, ids, **_kwargs):
                    stored.extend(ids)

            class FakeClient:
                def get_collection(self, _name):
                    return FakeCollection()

            knowledge = object.__new__(TrainingKnowledgeBase)
            knowledge.repository = repository
            knowledge.training_id = training.id
            knowledge.store = FakeStore()
            with (
                patch("training_analyzer.knowledge_base.create_embedding_model", return_value=FakeEmbeddingModel()),
                patch("training_analyzer.knowledge_base.chromadb.PersistentClient", return_value=FakeClient()),
            ):
                count = knowledge.rebuild()

            self.assertEqual(count, 40)
            self.assertEqual(len(stored), 40)
            self.assertGreaterEqual(maximum_active, 2)


if __name__ == "__main__":
    unittest.main()
