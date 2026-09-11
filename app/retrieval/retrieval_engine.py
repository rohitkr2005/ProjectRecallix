import math
import re
from datetime import datetime

import numpy as np

from app.embeddings.embedding_engine import EmbeddingEngine
from app.memory.memory_store import MemoryStore


class RetrievalEngine:

    INTENT_PATTERNS = {
        "GOAL": [
            r"\bwant(?:s)?\s+to\s+(?:learn|become|build)\b",
            r"\b(?:my|future)\s+goals?\b",
            r"\bwhat\s+do\s+i\s+want\b",
            r"\bwhat\s+are\s+my\s+(?:plans|ambitions|aspirations)\b",
            r"\bwhat\s+am\s+i\s+planning\s+to\s+build\b",
            r"\b(?:goal|goals|wish|wishes|aspiration|aspirations)\b",
        ],
        "PROJECT": [
            r"\bwhat\s+(?:project|projects)\b",
            r"\bwhich\s+(?:project|projects)\b",
            r"\b(?:working|work)\s+on\b",
            r"\bwhat\s+am\s+i\s+(?:working|building)\b",
        ],
        "EDUCATION": [
            r"\bwhat\s+(?:am\s+i\s+)?stud(?:y|ying)\b",
            r"\bwhat\s+do\s+i\s+stud(?:y|ying)\b",
            r"\bwhat\s+(?:am\s+i\s+)?learn(?:ing)?\b",
            r"\bwhat\s+do\s+i\s+learn(?:ing)?\b",
            r"\bwhere\s+do\s+i\s+stud(?:y|ying)\b",
            r"\bwhat\s+is\s+my\s+(?:education|degree|course|college|university)\b",
            r"\b(?:education|degree|course|college|university)\b",
        ],
        "LOCATION": [
            r"\bwhere\s+do\s+i\s+(?:live|stay)\b",
            r"\bwhere\s+(?:am\s+i|do\s+i\s+live)\b",
            r"\bwhat\s+is\s+my\s+(?:location|city)\b",
            r"\b(?:location|city|live|lives|living)\b",
        ],
        "PREFERENCE": [
            r"\bwhat\s+do\s+i\s+(?:like|love|enjoy|prefer)\b",
            r"\bwhat\s+are\s+my\s+preferences\b",
            r"\bwhat\s+do\s+i\s+like\b",
            r"\b(?:like|likes|love|prefer|enjoy|favorite|favourite|sports)\b",
        ],
        "SKILL": [
            r"\bwhat\s+(?:skills?|technologies|tools)\s+(?:do\s+i\s+)?(?:know|use|have)\b",
            r"\bwhat\s+(?:can|do)\s+i\s+(?:do|use)\b",
            r"\bwhat\s+(?:am\s+i\s+)?(?:good\s+at|skilled\s+at)\b",
            r"\b(?:skill|skills|know|knows|technology|technologies|tools)\b",
        ],
    }

    def __init__(self, memory_store=None, embedding_engine=None):
        self.memory_store = memory_store or MemoryStore()
        self.embedding_engine = embedding_engine or EmbeddingEngine()

    def _deserialize_embedding(self, embedding):
        if embedding is None:
            return None
        if isinstance(embedding, np.ndarray):
            vector = embedding.astype(np.float32, copy=False)
        elif isinstance(embedding, (bytes, bytearray, memoryview)):
            try:
                vector = np.frombuffer(embedding, dtype=np.float32).copy()
            except (TypeError, ValueError):
                return None
        else:
            try:
                vector = np.asarray(embedding, dtype=np.float32)
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
        return float(np.dot(vector_a, vector_b) / (norm_a * norm_b))

    def _normalize_semantic_score(self, semantic_score):
        try:
            semantic_score = float(semantic_score)
        except (TypeError, ValueError):
            return 0.0
        semantic_score = min(max(semantic_score, -1.0), 1.0)
        return (semantic_score + 1.0) / 2.0

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
        relation = relation_phrases.get(memory.relation, memory.relation.replace("_", " "))
        category = memory.category.replace("_", " ") if memory.category else ""
        return f"The user {relation} {memory.value}. This is a {category} memory."

    def _normalize_query(self, query):
        """Normalize user wording without changing its meaning."""
        if query is None:
            return ""
        query = str(query).strip().lower()
        query = re.sub(r"[\u2018\u2019]", "'", query)
        query = re.sub(r"\s+", " ", query)
        query = re.sub(r"[!?]+$", "?", query)
        return query

    def _intent_scores(self, query):
        normalized_query = self._normalize_query(query)
        scores = {intent: 0 for intent in self.INTENT_PATTERNS}
        if not normalized_query:
            return scores
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, normalized_query):
                    scores[intent] += 1
        return scores

    def _has_ambiguous_intent_signals(self, query):
        """Detect mixed topic cues that make strict intent filtering unsafe."""
        normalized_query = self._normalize_query(query)
        if not normalized_query:
            return False

        preference_signal = bool(
            re.search(r"\b(?:like|likes|love|enjoy|prefer|favorite|favourite)\b", normalized_query)
        )
        education_signal = bool(
            re.search(r"\b(?:learn|learning|study|studying|education)\b", normalized_query)
        )
        project_signal = bool(
            re.search(r"\bprojects?\b", normalized_query)
        )

        return sum((preference_signal, education_signal, project_signal)) >= 2

    def analyze_query_intent(self, query):
        """Return intent, confidence and whether strict filtering is safe."""
        normalized_query = self._normalize_query(query)
        scores = self._intent_scores(normalized_query)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_intent, best_score = ranked[0]
        second_score = ranked[1][1]

        if best_score == 0:
            return {
                "intent": None,
                "confidence": 0.0,
                "strict": False,
                "normalized_query": normalized_query,
                "scores": scores,
            }

        total = sum(scores.values())
        confidence = best_score / total if total else 0.0
        margin = (best_score - second_score) / best_score
        ambiguous = self._has_ambiguous_intent_signals(normalized_query)
        strict = confidence >= 0.60 and margin >= 0.50 and not ambiguous

        return {
            "intent": best_intent,
            "confidence": round(confidence, 3),
            "strict": strict,
            "normalized_query": normalized_query,
            "scores": scores,
        }

    def _detect_query_intent(self, query):
        """Return the best intent while preserving the Phase 6.1-6.5 API."""
        return self.analyze_query_intent(query)["intent"]

    def _calculate_importance_score(self, memory):
        importance = memory.importance
        if importance is None:
            importance = 5
        try:
            importance = float(importance)
        except (TypeError, ValueError):
            importance = 5.0
        importance = min(max(importance, 1.0), 10.0)
        return (importance - 1.0) / 9.0

    def _calculate_recency_score(self, memory):
        timestamp = memory.updated_at or memory.created_at
        if timestamp is None:
            return 0.0
        now = datetime.utcnow()
        if timestamp.tzinfo is not None and now.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=None)
        age_seconds = max(0.0, (now - timestamp).total_seconds())
        age_days = age_seconds / 86400.0
        half_life_days = 30.0
        return math.exp(-math.log(2.0) * age_days / half_life_days)

    def _calculate_relationship_score(self, memory, query_intent):
        if query_intent is None:
            return 0.0
        intent_relations = {
            "PROJECT": {"works_on"},
            "EDUCATION": {"studies"},
            "LOCATION": {"lives_in"},
            "PREFERENCE": {"likes"},
            "SKILL": {"knows"},
            "GOAL": {"wants_to_learn", "wants_to_become", "wants_to_build"},
        }
        if memory.relation in intent_relations.get(query_intent, set()):
            return 1.0
        if memory.category == query_intent:
            return 0.5
        return 0.0

    def _calculate_score_components(self, semantic_score, memory, query_intent):
        """Return the normalized components used by the final ranking score."""
        return {
            "semantic": self._normalize_semantic_score(semantic_score),
            "importance": self._calculate_importance_score(memory),
            "recency": self._calculate_recency_score(memory),
            "relationship": self._calculate_relationship_score(memory, query_intent),
        }

    def _calculate_final_score(self, semantic_score, memory, query_intent):
        components = self._calculate_score_components(
            semantic_score,
            memory,
            query_intent
        )
        weighted_score = (
            components["semantic"]
            + (0.10 * components["importance"])
            + (0.10 * components["recency"])
            + (0.20 * components["relationship"])
        )
        return weighted_score / 1.40

    def _build_retrieval_explanation(
        self,
        semantic_score,
        memory,
        query_intent,
        final_score=None
    ):
        """Build internal/debug metadata explaining why a memory ranked."""
        components = self._calculate_score_components(
            semantic_score,
            memory,
            query_intent
        )

        if final_score is None:
            final_score = (
                components["semantic"]
                + (0.10 * components["importance"])
                + (0.10 * components["recency"])
                + (0.20 * components["relationship"])
            ) / 1.40

        reasons = []

        if components["semantic"] >= 0.75:
            reasons.append("high semantic similarity")
        elif components["semantic"] >= 0.50:
            reasons.append("moderate semantic similarity")
        else:
            reasons.append("low semantic similarity")

        if components["relationship"] == 1.0:
            reasons.append("direct intent relationship match")
        elif components["relationship"] == 0.5:
            reasons.append("intent category match")

        if components["importance"] >= 0.75:
            reasons.append("high importance")
        elif components["importance"] < 0.25:
            reasons.append("low importance")

        if components["recency"] >= 0.75:
            reasons.append("recent memory")
        elif components["recency"] < 0.25:
            reasons.append("older memory")

        return {
            "query_intent": query_intent,
            "semantic_similarity": float(semantic_score),
            "score_components": components,
            "final_score": float(final_score),
            "reasons": reasons,
        }

    def _sort_results(self, results):
        def sort_key(item):
            memory = item["memory"]
            semantic_score = item.get("semantic_score", 0.0)
            importance_score = self._calculate_importance_score(memory)
            timestamp = memory.updated_at or memory.created_at
            timestamp_value = timestamp.timestamp() if timestamp is not None else 0.0
            memory_id = memory.id if memory.id is not None else float("inf")
            return (item["score"], semantic_score, importance_score, timestamp_value, -memory_id)
        return sorted(results, key=sort_key, reverse=True)

    def search(self, query, top_k=5, min_score=0.0, semantic_threshold=0.0):
        if not query or not str(query).strip():
            return []
        if top_k <= 0:
            return []

        query = self._normalize_query(query)
        query_embedding = self.embedding_engine.generate_embedding(query)
        query_intent = self._detect_query_intent(query)
        memories = self.memory_store.get_all_memories()
        results = []

        for memory in memories:
            if memory.embedding is None:
                continue
            memory_embedding = self.memory_store.get_embedding(memory)
            if memory_embedding is None:
                continue
            deserialized_embedding = self._deserialize_embedding(memory_embedding)
            if deserialized_embedding is None:
                continue
            semantic_score = self._cosine_similarity(query_embedding, deserialized_embedding)
            if semantic_score < semantic_threshold:
                continue
            final_score = self._calculate_final_score(
                semantic_score,
                memory,
                query_intent
            )
            if final_score >= min_score:
                explanation = self._build_retrieval_explanation(
                    semantic_score=semantic_score,
                    memory=memory,
                    query_intent=query_intent,
                    final_score=final_score
                )
                results.append({
                    "memory": memory,
                    "score": final_score,
                    "semantic_score": semantic_score,
                    "explanation": explanation,
                })

        return self._sort_results(results)[:top_k]

    def close(self):
        self.memory_store.close()
