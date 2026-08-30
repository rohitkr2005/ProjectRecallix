from app.embeddings.embedding_engine import EmbeddingEngine
from app.memory.memory_store import MemoryStore


class EmbeddingBackfill:

    def __init__(self, memory_store=None, embedding_engine=None):

        self.memory_store = memory_store or MemoryStore()
        self.embedding_engine = embedding_engine or EmbeddingEngine()

    def backfill(self):

        memories = self.memory_store.get_all_memories()

        updated = 0
        skipped = 0

        for memory in memories:

            if memory.embedding is not None:
                skipped += 1
                continue

            text = (
                f"{memory.subject} "
                f"{memory.relation} "
                f"{memory.value}"
            )

            embedding = self.embedding_engine.generate_embedding(text)

            success = self.memory_store.update_embedding(
                memory.id,
                embedding
            )

            if success:
                updated += 1

        return {
            "updated": updated,
            "skipped": skipped,
            "total": len(memories)
        }

    def close(self):

        self.memory_store.close()