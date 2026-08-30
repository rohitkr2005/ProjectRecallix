import numpy as np

from app.embeddings.embedding_engine import EmbeddingEngine
from app.memory.memory_store import MemoryStore


class RetrievalEngine:

    def __init__(self, memory_store=None, embedding_engine=None):

        self.memory_store = memory_store or MemoryStore()
        self.embedding_engine = embedding_engine or EmbeddingEngine()

    def _deserialize_embedding(self, embedding):

        if embedding is None:
            return None

        if isinstance(embedding, np.ndarray):

            vector = embedding.astype(
                np.float32,
                copy=False
            )

        elif isinstance(embedding, (bytes, bytearray, memoryview)):

            try:
                vector = np.frombuffer(
                    embedding,
                    dtype=np.float32
                ).copy()

            except (TypeError, ValueError):
                return None

        else:

            try:
                vector = np.asarray(
                    embedding,
                    dtype=np.float32
                )

            except (TypeError, ValueError):
                return None

        if vector.size != 384:
            return None

        return vector.reshape(384)

    def _cosine_similarity(self, vector_a, vector_b):

        vector_a = self._deserialize_embedding(vector_a)
        vector_b = self._deserialize_embedding(vector_b)

        if vector_a is None or vector_b is None:
            return 0.0

        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return float(
            np.dot(vector_a, vector_b)
            / (norm_a * norm_b)
        )

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

    def _detect_query_intent(self, query):

        query_lower = query.lower()

        if any(
            word in query_lower
            for word in [
                "project",
                "projects",
                "working on",
                "building"
            ]
        ):
            return "PROJECT"

        if any(
            word in query_lower
            for word in [
                "study",
                "studying",
                "learn",
                "learning",
                "education"
            ]
        ):
            return "EDUCATION"

        if any(
            word in query_lower
            for word in [
                "live",
                "lives",
                "location",
                "where"
            ]
        ):
            return "LOCATION"

        if any(
            word in query_lower
            for word in [
                "like",
                "likes",
                "love",
                "prefer",
                "enjoy",
                "sports"
            ]
        ):
            return "PREFERENCE"

        if any(
            word in query_lower
            for word in [
                "skill",
                "skills",
                "know",
                "knows",
                "use"
            ]
        ):
            return "SKILL"

        if any(
            word in query_lower
            for word in [
                "goal",
                "want",
                "wants",
                "wish"
            ]
        ):
            return "GOAL"

        return None

    def _calculate_final_score(
        self,
        semantic_score,
        memory,
        query_intent
    ):

        final_score = semantic_score

        if query_intent is not None:

            if memory.category == query_intent:
                final_score += 0.15

            elif (
                query_intent == "PROJECT"
                and memory.relation == "works_on"
            ):
                final_score += 0.20

            elif (
                query_intent == "LOCATION"
                and memory.relation == "lives_in"
            ):
                final_score += 0.20

        return final_score

    def search(
        self,
        query,
        top_k=5,
        min_score=0.0
    ):

        if not query or not query.strip():
            return []

        if top_k <= 0:
            return []

        query = query.strip()

        query_embedding = (
            self.embedding_engine.generate_embedding(query)
        )

        query_intent = self._detect_query_intent(query)

        memories = self.memory_store.get_all_memories()

        results = []

        for memory in memories:

            if memory.embedding is None:
                continue

            memory_embedding = (
                self.memory_store.get_embedding(memory)
            )

            if memory_embedding is None:
                continue

            semantic_score = self._cosine_similarity(
                query_embedding,
                memory_embedding
            )

            final_score = self._calculate_final_score(
                semantic_score,
                memory,
                query_intent
            )

            if final_score >= min_score:

                results.append(
                    {
                        "memory": memory,
                        "score": final_score,
                        "semantic_score": semantic_score
                    }
                )

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]

    def close(self):

        self.memory_store.close()