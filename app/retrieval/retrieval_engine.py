import math
from datetime import datetime

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

        # Goal-related phrases are checked first because
        # phrases such as "want to learn" also contain "learn".
        if any(
            phrase in query_lower
            for phrase in [
                "want to learn",
                "want to become",
                "want to build",
                "wants to learn",
                "wants to become",
                "wants to build",
                "goal",
                "want",
                "wants",
                "wish"
            ]
        ):
            return "GOAL"

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

        return None

    def _calculate_importance_score(self, memory):

        importance = memory.importance

        if importance is None:
            importance = 5

        try:
            importance = float(importance)

        except (TypeError, ValueError):
            importance = 5.0

        importance = min(
            max(importance, 1.0),
            10.0
        )

        return (importance - 1.0) / 9.0

    def _calculate_recency_score(self, memory):

        timestamp = memory.updated_at or memory.created_at

        if timestamp is None:
            return 0.0

        now = datetime.utcnow()

        if timestamp.tzinfo is not None and now.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=None)

        age_seconds = max(
            0.0,
            (now - timestamp).total_seconds()
        )

        age_days = age_seconds / 86400.0

        # A 30-day half-life keeps recent memories slightly favored
        # without allowing old but important memories to disappear.
        half_life_days = 30.0

        return math.exp(
            -math.log(2.0) * age_days / half_life_days
        )

    def _calculate_relationship_score(
        self,
        memory,
        query_intent
    ):

        if query_intent is None:
            return 0.0

        intent_relations = {
            "PROJECT": {
                "works_on"
            },
            "EDUCATION": {
                "studies"
            },
            "LOCATION": {
                "lives_in"
            },
            "PREFERENCE": {
                "likes"
            },
            "SKILL": {
                "knows"
            },
            "GOAL": {
                "wants_to_learn",
                "wants_to_become",
                "wants_to_build"
            },
        }

        if memory.relation in intent_relations.get(
            query_intent,
            set()
        ):
            return 1.0

        if memory.category == query_intent:
            return 0.5

        return 0.0

    def _calculate_final_score(
        self,
        semantic_score,
        memory,
        query_intent
    ):

        importance_score = (
            self._calculate_importance_score(memory)
        )

        recency_score = (
            self._calculate_recency_score(memory)
        )

        relationship_score = (
            self._calculate_relationship_score(
                memory,
                query_intent
            )
        )

        # Semantic similarity remains the strongest signal.
        # Additional signals provide intelligent tie-breaking
        # and contextual ranking.
        return (
            semantic_score
            + (0.10 * importance_score)
            + (0.10 * recency_score)
            + (0.20 * relationship_score)
        )

    def search(
        self,
        query,
        top_k=5,
        min_score=0.0,
        semantic_threshold=0.0
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
            
            if semantic_score < semantic_threshold:
                continue

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