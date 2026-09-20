from datetime import datetime
import math
from typing import Dict, List, Optional, Any

import numpy as np

from app.memory.graph import MemoryGraph
from app.memory.importance_learning import ImportanceLearner
from app.memory.memory_store import MemoryStore
from app.memory.temporal import TemporalState, detect_query_temporal_intent
from app.retrieval.retrieval_engine import RetrievalEngine


class AdvancedRetrievalEngine:
    """
    10.8 Advanced Retrieval

    Combines 6 ranking signals:
    1. Semantic Similarity
    2. Relationship / Graph Retrieval
    3. Temporal Alignment
    4. Intent Alignment
    5. Learned Importance & Reinforcement
    6. Recency Decay
    """

    WEIGHTS = {
        "semantic": 0.35,
        "relationship": 0.20,
        "temporal": 0.15,
        "intent": 0.10,
        "importance": 0.10,
        "recency": 0.10,
    }

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        embedding_engine=None,
        importance_learner: Optional[ImportanceLearner] = None,
        memory_graph: Optional[MemoryGraph] = None,
    ):
        self.memory_store = memory_store or MemoryStore()
        self.base_retrieval = RetrievalEngine(
            memory_store=self.memory_store,
            embedding_engine=embedding_engine,
        )
        self.embedding_engine = getattr(self.base_retrieval, "embedding_engine", None)
        self.learner = importance_learner or ImportanceLearner(session=getattr(self.memory_store, "session", None))
        self.graph = memory_graph or MemoryGraph(session=getattr(self.memory_store, "session", None))

    def _detect_query_intent(self, query: str):
        return self.base_retrieval._detect_query_intent(query)

    def _calculate_semantic_score(self, query_vector: np.ndarray, memory_embedding: Optional[bytes]) -> float:
        if memory_embedding is None or len(memory_embedding) == 0:
            return 0.0

        try:
            mem_vec = np.frombuffer(memory_embedding, dtype=np.float32)
            if mem_vec.shape != query_vector.shape:
                return 0.0

            dot = float(np.dot(query_vector, mem_vec))
            norm_q = float(np.linalg.norm(query_vector))
            norm_m = float(np.linalg.norm(mem_vec))

            if norm_q == 0.0 or norm_m == 0.0:
                return 0.0

            cosine = dot / (norm_q * norm_m)
            # Map [-1, 1] to [0, 1]
            return max(0.0, min(1.0, (cosine + 1.0) / 2.0))
        except Exception:
            return 0.0

    def _calculate_temporal_score(
        self,
        query_temporal: Optional[TemporalState],
        memory_temporal: Optional[str],
    ) -> float:
        mem_temp = (memory_temporal or "PRESENT").upper()

        if query_temporal is not None:
            # Explicit temporal query
            if query_temporal.value == mem_temp:
                return 1.0
            return 0.15

        # Neutral query: favor current facts, then future, then historical
        if mem_temp == "PRESENT":
            return 1.0
        elif mem_temp == "FUTURE":
            return 0.80
        elif mem_temp == "PAST":
            return 0.60
        return 0.50

    def _calculate_relationship_score(
        self,
        query: str,
        intent: Optional[str],
        memory: Any,
    ) -> float:
        # Base relationship score from intent/relation alignment
        rel = getattr(memory, "relation", "")
        cat = getattr(memory, "category", "")
        val = str(getattr(memory, "value", "")).lower()
        sub = str(getattr(memory, "subject", "")).lower()

        base_rel_score = 0.0
        if intent:
            if cat == intent:
                base_rel_score = 0.8
            elif rel in {"works_on", "studies", "lives_in", "likes", "knows", "wants_to_learn"}:
                base_rel_score = 0.6

        # Check graph proximity to any entity extracted from query
        graph_prox = 0.0
        q_words = [w for w in query.lower().split() if len(w) > 3]
        for w in q_words:
            p1 = self.graph.calculate_proximity(w, val)
            p2 = self.graph.calculate_proximity(w, sub)
            graph_prox = max(graph_prox, p1, p2)

        return max(base_rel_score, graph_prox)

    def _calculate_importance_score(self, memory: Any) -> float:
        base_imp = getattr(memory, "importance", 5) or 5
        acc = getattr(memory, "access_count", 0) or 0
        help_cnt = getattr(memory, "helpful_count", 0) or 0
        reinf = getattr(memory, "reinforcement_score", 0.0) or 0.0

        effective = self.learner.calculate_effective_importance(
            base_importance=base_imp,
            access_count=acc,
            helpful_count=help_cnt,
            reinforcement_score=reinf,
        )
        return effective / 10.0

    def _calculate_recency_score(self, memory: Any, now: datetime) -> float:
        created = getattr(memory, "created_at", None) or now
        accessed = getattr(memory, "last_accessed_at", None) or created
        ref_time = max(created, accessed)

        days_ago = max(0.0, (now - ref_time).total_seconds() / 86400.0)
        # Half-life of 60 days
        decay = math.exp(-0.0115 * days_ago)
        return max(0.0, min(1.0, decay))

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        semantic_threshold: float = 0.0,
        now: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute multi-signal hybrid search combining all 6 signals.
        """
        if not query or not query.strip():
            return []

        query = query.strip()
        ref_now = now or datetime.utcnow()

        # Generate query vector
        query_vector = None
        if self.embedding_engine and hasattr(self.embedding_engine, "generate_embedding"):
            try:
                emb = self.embedding_engine.generate_embedding(query)
                if emb is not None:
                    query_vector = np.frombuffer(emb, dtype=np.float32)
            except Exception:
                query_vector = None

        intent = self._detect_query_intent(query)
        query_temporal = detect_query_temporal_intent(query)

        # Get active memories
        active_memories = self.memory_store.get_all_memories()
        if not active_memories:
            return []

        # Ensure graph is synced
        if not self.graph._synced and self.memory_store.session:
            self.graph.sync_from_database()
        elif not self.graph.edges:
            self.graph.build_from_memories(active_memories)

        scored_results = []
        for mem in active_memories:
            if not getattr(mem, "active", True):
                continue

            sem_score = 0.0
            if query_vector is not None:
                sem_score = self._calculate_semantic_score(query_vector, getattr(mem, "embedding", None))

            if sem_score < semantic_threshold and semantic_threshold > 0.0:
                continue

            rel_score = self._calculate_relationship_score(query, intent, mem)
            temp_score = self._calculate_temporal_score(query_temporal, getattr(mem, "temporal_state", "PRESENT"))
            intent_score = 1.0 if getattr(mem, "category", "") == intent else 0.0
            imp_score = self._calculate_importance_score(mem)
            rec_score = self._calculate_recency_score(mem, ref_now)

            final_score = (
                self.WEIGHTS["semantic"] * sem_score
                + self.WEIGHTS["relationship"] * rel_score
                + self.WEIGHTS["temporal"] * temp_score
                + self.WEIGHTS["intent"] * intent_score
                + self.WEIGHTS["importance"] * imp_score
                + self.WEIGHTS["recency"] * rec_score
            )

            if final_score < min_score:
                continue

            reasons = []
            if intent_score > 0.5:
                reasons.append(f"intent match ({getattr(mem, 'category', '')})")
            if temp_score > 0.8 and query_temporal:
                reasons.append(f"temporal alignment ({query_temporal.value})")
            if rel_score > 0.5:
                reasons.append("graph/relationship match")
            if imp_score > 0.7:
                reasons.append("high learned importance")
            if sem_score > 0.7:
                reasons.append("high semantic similarity")

            if not reasons:
                reasons.append(f"composite score of {final_score:.2f}")

            explanation = {
                "relevance": round(final_score, 4),
                "reasons": reasons,
                "signals": {
                    "semantic": round(sem_score, 4),
                    "relationship": round(rel_score, 4),
                    "temporal": round(temp_score, 4),
                    "intent": round(intent_score, 4),
                    "importance": round(imp_score, 4),
                    "recency": round(rec_score, 4),
                },
            }

            scored_results.append({
                "memory": mem,
                "score": round(final_score, 4),
                "explanation": explanation,
            })

        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k] if top_k is not None and top_k > 0 else scored_results

    def close(self):
        if hasattr(self.base_retrieval, "close") and callable(self.base_retrieval.close):
            self.base_retrieval.close()
