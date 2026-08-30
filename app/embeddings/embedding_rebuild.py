from app.embeddings.embedding_engine import EmbeddingEngine
from app.memory.memory_store import MemoryStore


class EmbeddingRebuild:

    def __init__(self, memory_store=None, embedding_engine=None):

        self.memory_store = memory_store or MemoryStore()
        self.embedding_engine = embedding_engine or EmbeddingEngine()

    def _build_memory_text(self, memory):

        relation_phrases = {
            "likes": "likes",
            "lives_in": "lives in",
            "studies": "studies",
            "works_on": "works on the project",
            "knows": "knows",
            "wants_to_learn": "wants to learn",
            "wants_to_become": "wants to become",
            "wants_to_build": "wants to build",
        }

        relation = relation_phrases.get(
            memory.relation,
            memory.relation.replace("_", " ")
        )

        category = (
            memory.category.replace("_", " ")
            if memory.category
            else ""
        )

        return (
            f"The user {relation} {memory.value}. "
            f"This is a {category} memory."
        )

    def rebuild(self):

        memories = self.memory_store.get_all_memories()

        rebuilt = 0

        for memory in memories:

            text = self._build_memory_text(memory)

            embedding = (
                self.embedding_engine.generate_embedding(text)
            )

            success = self.memory_store.update_embedding(
                memory.id,
                embedding
            )

            if success:
                rebuilt += 1

        return {
            "total": len(memories),
            "rebuilt": rebuilt
        }

    def close(self):

        self.memory_store.close()