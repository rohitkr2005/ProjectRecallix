from types import SimpleNamespace

from app.assistant.assistant_engine import AssistantEngine
from app.retrieval.retrieval_engine import RetrievalEngine


class DummyRetrievalEngine:
    def __init__(self, memories):
        self.memories = memories
        self.search_calls = []

    def search(self, query, top_k=5, min_score=0.0, semantic_threshold=0.0):
        self.search_calls.append({
            "query": query,
            "top_k": top_k,
            "min_score": min_score,
            "semantic_threshold": semantic_threshold,
        })
        return self.memories

    def analyze_query_intent(self, query):
        return RetrievalEngine().analyze_query_intent(query)

    def close(self):
        pass


class DummyLLMEngine:
    def __init__(self):
        self.calls = []

    def generate_with_memories(self, user_message, memories, temperature=0.2, max_tokens=500):
        self.calls.append(memories)
        return "Dummy Recallix response"

    def is_available(self):
        return True

    def close(self):
        pass


def memory(relation, value, category, score=0.9, active=True):
    return {
        "memory": SimpleNamespace(
            subject="User",
            relation=relation,
            value=value,
            category=category,
            active=active,
        ),
        "score": score,
    }


def test_query_normalization_is_consistent():
    engine = RetrievalEngine()
    assert engine._normalize_query("  WHAT am I working on!!!  ") == "what am i working on?"
    engine.close()


def test_project_phrase_variations_detect_project():
    engine = RetrievalEngine()
    for query in [
        "What projects am I working on?",
        "What am I building these days?",
        "Which project am I working on",
        "What have I been working on?",
    ]:
        assert engine._detect_query_intent(query) == "PROJECT"
    engine.close()


def test_location_phrase_variations_detect_location():
    engine = RetrievalEngine()
    for query in [
        "Where do I currently live?",
        "What is my current city?",
        "Where am I living?",
    ]:
        assert engine._detect_query_intent(query) == "LOCATION"
    engine.close()


def test_goal_phrase_variations_detect_goal():
    engine = RetrievalEngine()
    for query in [
        "What are my future goals?",
        "What do I want to become?",
        "What am I planning to build?",
    ]:
        assert engine._detect_query_intent(query) == "GOAL"
    engine.close()


def test_intent_confidence_is_high_for_clear_query():
    engine = RetrievalEngine()
    analysis = engine.analyze_query_intent("What project am I working on?")
    assert analysis["intent"] == "PROJECT"
    assert analysis["confidence"] >= 0.60
    assert analysis["strict"] is True
    engine.close()


def test_unknown_query_has_zero_confidence_and_non_strict_mode():
    engine = RetrievalEngine()
    analysis = engine.analyze_query_intent("Tell me something interesting.")
    assert analysis["intent"] is None
    assert analysis["confidence"] == 0.0
    assert analysis["strict"] is False
    engine.close()


def test_ambiguous_query_is_not_strict():
    engine = RetrievalEngine()
    analysis = engine.analyze_query_intent("What do I like to learn about projects?")
    assert analysis["intent"] in {"PROJECT", "EDUCATION", "PREFERENCE"}
    assert analysis["strict"] is False
    engine.close()


def test_low_confidence_intent_uses_semantic_fallback():
    memories = [
        memory("likes", "Football", "PREFERENCE"),
    ]
    assistant = AssistantEngine(
        retrieval_engine=DummyRetrievalEngine(memories),
        llm_engine=DummyLLMEngine(),
    )
    result = assistant.respond("Tell me something about me")
    assert result["intent"] is None
    assert result["intent_strict"] is False
    assert result["supported"] is True
    assert len(result["memories"]) == 1
    assistant.close()


def test_clear_intent_still_uses_strict_filtering():
    memories = [
        memory("works_on", "Project Recallix", "PROJECT"),
        memory("likes", "Football", "PREFERENCE"),
    ]
    assistant = AssistantEngine(
        retrieval_engine=DummyRetrievalEngine(memories),
        llm_engine=DummyLLMEngine(),
    )
    result = assistant.respond("What project am I working on?")
    assert result["intent"] == "PROJECT"
    assert result["intent_strict"] is True
    assert [item["memory"].value for item in result["memories"]] == ["Project Recallix"]
    assistant.close()


def test_normalized_query_is_used_for_retrieval():
    memories = [memory("likes", "Football", "PREFERENCE")]
    retrieval = DummyRetrievalEngine(memories)
    assistant = AssistantEngine(retrieval_engine=retrieval, llm_engine=DummyLLMEngine())
    assistant.respond("   WHAT DO I LIKE!!!   ")
    assert retrieval.search_calls[0]["query"] == "WHAT DO I LIKE!!!"
    assistant.close()
