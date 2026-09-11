from datetime import datetime

import numpy as np

from app.database.models import Memory
from app.retrieval.retrieval_engine import RetrievalEngine


class DummyMemoryStore:
    def __init__(self, memories=None):
        self.memories = memories or []

    def get_all_memories(self):
        return self.memories

    def get_embedding(self, memory):
        return memory.embedding

    def close(self):
        pass


class DummyEmbeddingEngine:
    def __init__(self, embedding):
        self.embedding = embedding
        self.last_text = None

    def generate_embedding(self, text):
        self.last_text = text
        return self.embedding


def create_memory(
    relation="works_on",
    category="PROJECT",
    importance=10,
    embedding=None,
):
    memory = Memory(
        subject="User",
        relation=relation,
        value="Project Recallix",
        category=category,
        importance=importance,
        active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    memory.embedding = embedding
    return memory


def create_engine(memories=None, query_embedding=None):
    if query_embedding is None:
        query_embedding = np.ones(384, dtype=np.float32)

    return RetrievalEngine(
        memory_store=DummyMemoryStore(memories),
        embedding_engine=DummyEmbeddingEngine(query_embedding),
    )


def test_score_components_match_final_score():
    engine = create_engine()
    memory = create_memory()

    components = engine._calculate_score_components(0.5, memory, "PROJECT")
    final_score = engine._calculate_final_score(0.5, memory, "PROJECT")

    expected = (
        components["semantic"]
        + 0.10 * components["importance"]
        + 0.10 * components["recency"]
        + 0.20 * components["relationship"]
    ) / 1.40

    assert set(components) == {
        "semantic",
        "importance",
        "recency",
        "relationship",
    }
    assert final_score == expected


def test_retrieval_explanation_contains_all_ranking_signals():
    engine = create_engine()
    memory = create_memory()

    explanation = engine._build_retrieval_explanation(
        semantic_score=1.0,
        memory=memory,
        query_intent="PROJECT",
    )

    assert explanation["query_intent"] == "PROJECT"
    assert explanation["semantic_similarity"] == 1.0
    assert explanation["final_score"] > 0.0
    assert set(explanation["score_components"]) == {
        "semantic",
        "importance",
        "recency",
        "relationship",
    }
    assert "high semantic similarity" in explanation["reasons"]
    assert "direct intent relationship match" in explanation["reasons"]
    assert "high importance" in explanation["reasons"]
    assert "recent memory" in explanation["reasons"]


def test_retrieval_explanation_uses_low_signal_reasons():
    engine = create_engine()
    memory = create_memory(importance=1)

    old = datetime.utcnow()
    memory.created_at = old
    memory.updated_at = old

    explanation = engine._build_retrieval_explanation(
        semantic_score=-0.5,
        memory=memory,
        query_intent=None,
    )

    assert "low semantic similarity" in explanation["reasons"]
    assert "low importance" in explanation["reasons"]


def test_search_attaches_explanation_without_changing_ranking():
    query = np.zeros(384, dtype=np.float32)
    query[0] = 1.0

    strong = np.zeros(384, dtype=np.float32)
    strong[0] = 1.0

    weak = np.zeros(384, dtype=np.float32)
    weak[1] = 1.0

    first = create_memory(embedding=weak)
    second = create_memory(embedding=strong)

    engine = create_engine(
        memories=[first, second],
        query_embedding=query,
    )

    results = engine.search("What projects am I working on?")

    assert results[0]["memory"] is second
    assert results[1]["memory"] is first
    assert results[0]["score"] > results[1]["score"]

    for result in results:
        explanation = result["explanation"]
        assert explanation["final_score"] == result["score"]
        assert explanation["semantic_similarity"] == result["semantic_score"]


def test_explanation_metadata_is_debug_only_and_optional_in_assistant():
    from app.assistant.assistant_engine import AssistantEngine

    memory = create_memory()
    retrieval_result = {
        "memory": memory,
        "score": 0.9,
        "semantic_score": 0.8,
        "explanation": {
            "query_intent": "PROJECT",
            "semantic_similarity": 0.8,
            "score_components": {
                "semantic": 0.9,
                "importance": 1.0,
                "recency": 1.0,
                "relationship": 1.0,
            },
            "final_score": 0.9,
            "reasons": ["high semantic similarity"],
        },
    }

    class DummyRetrieval:
        def search(self, **kwargs):
            return [retrieval_result]

        def analyze_query_intent(self, query):
            return {
                "intent": "PROJECT",
                "confidence": 1.0,
                "strict": True,
                "normalized_query": query.lower(),
                "scores": {"PROJECT": 1},
            }

        def close(self):
            pass

    class DummyLLM:
        def generate_with_memories(self, **kwargs):
            for item in kwargs["memories"]:
                assert "explanation" in item
            return "Project Recallix."

        def close(self):
            pass

    assistant = AssistantEngine(
        retrieval_engine=DummyRetrieval(),
        llm_engine=DummyLLM(),
    )

    normal_result = assistant.respond("What projects am I working on?")
    assert "retrieval_explanations" not in normal_result

    debug_result = assistant.respond(
        "What projects am I working on?",
        include_explanations=True,
    )
    assert debug_result["retrieval_explanations"]
    assert debug_result["retrieval_explanations"][0]["query_intent"] == "PROJECT"
