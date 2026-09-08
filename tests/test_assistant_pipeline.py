from types import SimpleNamespace

import pytest

from app.assistant.assistant_engine import AssistantEngine
from app.llm.llm_engine import LLMEngine


class DummyRetrievalEngine:
    def __init__(self, memories):
        self.memories = memories
        self.search_calls = []

    def search(
        self,
        query,
        top_k=5,
        min_score=0.0,
        semantic_threshold=0.0,
    ):
        self.search_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "min_score": min_score,
                "semantic_threshold": semantic_threshold,
            }
        )

        return self.memories

    def _detect_query_intent(self, query):
        query_lower = query.lower()

        if "project" in query_lower:
            return "PROJECT"

        if (
            "live" in query_lower
            or "location" in query_lower
        ):
            return "LOCATION"

        if "like" in query_lower:
            return "PREFERENCE"

        if (
            "learn" in query_lower
            or "education" in query_lower
        ):
            return "EDUCATION"

        return None

    def close(self):
        pass


class DummyLLMEngine:
    def __init__(self):
        self.calls = []

    def generate_with_memories(
        self,
        user_message,
        memories,
        temperature=0.2,
        max_tokens=500,
    ):
        self.calls.append(
            {
                "user_message": user_message,
                "memories": memories,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )

        return "Dummy Recallix response"

    def is_available(self):
        return True

    def close(self):
        pass


def create_memory(
    relation,
    value,
    category,
    score=0.9,
    active=True,
):
    memory = SimpleNamespace(
        subject="User",
        relation=relation,
        value=value,
        category=category,
        active=active,
    )

    return {
        "memory": memory,
        "score": score,
    }


# ---------------------------------------------------------
# 6.5.1 — End-to-End Retrieval Pipeline
# ---------------------------------------------------------


def test_assistant_pipeline_connects_retrieval_to_llm():

    retrieved_memories = [
        create_memory(
            relation="works_on",
            value="Project Recallix",
            category="PROJECT",
        )
    ]

    retrieval_engine = DummyRetrievalEngine(
        retrieved_memories
    )

    llm_engine = DummyLLMEngine()

    assistant = AssistantEngine(
        retrieval_engine=retrieval_engine,
        llm_engine=llm_engine,
    )

    result = assistant.respond(
        "What project am I working on?"
    )

    assert result["response"] == (
        "Dummy Recallix response"
    )

    assert result["intent"] == "PROJECT"

    assert result["retrieved_memories"] == (
        retrieved_memories
    )

    assert result["memories"] == (
        retrieved_memories
    )

    assert result["supported"] is True

    assert len(llm_engine.calls) == 1

    assistant.close()


def test_assistant_pipeline_passes_retrieval_parameters():

    retrieval_engine = DummyRetrievalEngine(
        memories=[]
    )

    llm_engine = DummyLLMEngine()

    assistant = AssistantEngine(
        retrieval_engine=retrieval_engine,
        llm_engine=llm_engine,
    )

    assistant.respond(
        "What do I like?",
        top_k=3,
        min_score=0.4,
        semantic_threshold=0.6,
    )

    call = retrieval_engine.search_calls[0]

    assert call["query"] == "What do I like?"
    assert call["top_k"] == 3
    assert call["min_score"] == 0.4
    assert call["semantic_threshold"] == 0.6

    assistant.close()


# ---------------------------------------------------------
# 6.5.2 — Memory → Context Integration
# ---------------------------------------------------------


def test_intent_filtering_removes_unrelated_memories():

    memories = [
        create_memory(
            relation="works_on",
            value="Project Recallix",
            category="PROJECT",
        ),
        create_memory(
            relation="likes",
            value="Football",
            category="PREFERENCE",
        ),
    ]

    retrieval_engine = DummyRetrievalEngine(
        memories
    )

    llm_engine = DummyLLMEngine()

    assistant = AssistantEngine(
        retrieval_engine=retrieval_engine,
        llm_engine=llm_engine,
    )

    result = assistant.respond(
        "Which project am I working on?"
    )

    assert result["intent"] == "PROJECT"

    assert len(result["memories"]) == 1

    assert (
        result["memories"][0]["memory"].value
        == "Project Recallix"
    )

    assert (
        llm_engine.calls[0]["memories"]
        == result["memories"]
    )

    assistant.close()


def test_intent_filtering_supports_location_relation():

    memories = [
        create_memory(
            relation="lives_in",
            value="Delhi",
            category="LOCATION",
        ),
        create_memory(
            relation="likes",
            value="Football",
            category="PREFERENCE",
        ),
    ]

    retrieval_engine = DummyRetrievalEngine(
        memories
    )

    llm_engine = DummyLLMEngine()

    assistant = AssistantEngine(
        retrieval_engine=retrieval_engine,
        llm_engine=llm_engine,
    )

    result = assistant.respond(
        "Where do I live?"
    )

    assert result["intent"] == "LOCATION"

    assert len(result["memories"]) == 1

    assert (
        result["memories"][0]["memory"].value
        == "Delhi"
    )

    assistant.close()


# ---------------------------------------------------------
# 6.5.3 — LLM-Ready Context
# ---------------------------------------------------------


def test_memory_context_contains_only_supported_memory():

    engine = LLMEngine()

    memories = [
        create_memory(
            relation="works_on",
            value="Project Recallix",
            category="PROJECT",
            score=0.987,
        )
    ]

    context = engine._build_memory_context(
        memories
    )

    assert (
        "User works on Project Recallix."
        in context
    )

    # Internal metadata must not leak to the LLM.
    assert "0.987" not in context
    assert "PROJECT" not in context

    engine.close()


def test_memory_context_skips_archived_memory():

    engine = LLMEngine()

    memories = [
        create_memory(
            relation="lives_in",
            value="Jaipur",
            category="LOCATION",
            active=False,
        ),
        create_memory(
            relation="lives_in",
            value="Delhi",
            category="LOCATION",
            active=True,
        ),
    ]

    context = engine._build_memory_context(
        memories
    )

    assert "Jaipur" not in context
    assert "Delhi" in context

    engine.close()


def test_empty_memory_context_is_explicit():

    engine = LLMEngine()

    context = engine._build_memory_context([])

    assert context == (
        "No relevant memories found."
    )

    engine.close()


# ---------------------------------------------------------
# 6.5.4 — Unsupported Answer Prevention
# ---------------------------------------------------------


def test_assistant_does_not_call_llm_without_supported_memory():

    retrieval_engine = DummyRetrievalEngine(
        memories=[]
    )

    llm_engine = DummyLLMEngine()

    assistant = AssistantEngine(
        retrieval_engine=retrieval_engine,
        llm_engine=llm_engine,
    )

    result = assistant.respond(
        "What is my favorite programming language?"
    )

    assert result["supported"] is False

    assert result["memories"] == []

    assert (
        result["response"]
        == "I don't have enough information "
           "in my memory to answer that."
    )

    assert len(llm_engine.calls) == 0

    assistant.close()


def test_archived_memory_cannot_support_answer():

    memories = [
        create_memory(
            relation="lives_in",
            value="Jaipur",
            category="LOCATION",
            active=False,
        )
    ]

    retrieval_engine = DummyRetrievalEngine(
        memories
    )

    llm_engine = DummyLLMEngine()

    assistant = AssistantEngine(
        retrieval_engine=retrieval_engine,
        llm_engine=llm_engine,
    )

    result = assistant.respond(
        "Where do I live?"
    )

    assert result["supported"] is False
    assert result["memories"] == []
    assert len(llm_engine.calls) == 0

    assistant.close()


def test_low_relevance_memory_cannot_support_answer():

    memories = [
        create_memory(
            relation="lives_in",
            value="Delhi",
            category="LOCATION",
            score=0.10,
        )
    ]

    retrieval_engine = DummyRetrievalEngine(
        memories
    )

    llm_engine = DummyLLMEngine()

    assistant = AssistantEngine(
        retrieval_engine=retrieval_engine,
        llm_engine=llm_engine,
    )

    result = assistant.respond(
        "Where do I live?",
        memory_relevance=0.25,
    )

    assert result["supported"] is False
    assert result["memories"] == []
    assert len(llm_engine.calls) == 0

    assistant.close()


# ---------------------------------------------------------
# 6.5.5 — Conflict-Aware E2E Retrieval
# ---------------------------------------------------------


def test_conflicting_archived_memory_is_never_sent_to_llm():

    memories = [
        create_memory(
            relation="lives_in",
            value="Jaipur",
            category="LOCATION",
            score=0.99,
            active=False,
        ),
        create_memory(
            relation="lives_in",
            value="Delhi",
            category="LOCATION",
            score=0.90,
            active=True,
        ),
    ]

    retrieval_engine = DummyRetrievalEngine(
        memories
    )

    llm_engine = DummyLLMEngine()

    assistant = AssistantEngine(
        retrieval_engine=retrieval_engine,
        llm_engine=llm_engine,
    )

    result = assistant.respond(
        "Where do I live?"
    )

    assert result["supported"] is True

    assert len(result["memories"]) == 1

    assert (
        result["memories"][0]["memory"].value
        == "Delhi"
    )

    assert (
        llm_engine.calls[0]["memories"]
        == result["memories"]
    )

    assistant.close()


def test_multiple_active_memories_can_support_multi_value_relation():

    memories = [
        create_memory(
            relation="likes",
            value="Python",
            category="PREFERENCE",
            score=0.95,
        ),
        create_memory(
            relation="likes",
            value="SQL",
            category="PREFERENCE",
            score=0.90,
        ),
    ]

    retrieval_engine = DummyRetrievalEngine(
        memories
    )

    llm_engine = DummyLLMEngine()

    assistant = AssistantEngine(
        retrieval_engine=retrieval_engine,
        llm_engine=llm_engine,
    )

    result = assistant.respond(
        "What do I like?"
    )

    assert result["supported"] is True

    assert len(result["memories"]) == 2

    values = {
        item["memory"].value
        for item in result["memories"]
    }

    assert values == {
        "Python",
        "SQL",
    }

    assistant.close()


# ---------------------------------------------------------
# 6.5.6 — Edge / Pipeline Tests
# ---------------------------------------------------------


def test_empty_user_message_is_rejected():

    retrieval_engine = DummyRetrievalEngine(
        memories=[]
    )

    llm_engine = DummyLLMEngine()

    assistant = AssistantEngine(
        retrieval_engine=retrieval_engine,
        llm_engine=llm_engine,
    )

    with pytest.raises(
        ValueError,
        match="User message cannot be empty",
    ):
        assistant.respond("")

    assistant.close()


def test_whitespace_user_message_is_rejected():

    retrieval_engine = DummyRetrievalEngine(
        memories=[]
    )

    llm_engine = DummyLLMEngine()

    assistant = AssistantEngine(
        retrieval_engine=retrieval_engine,
        llm_engine=llm_engine,
    )

    with pytest.raises(
        ValueError,
        match="User message cannot be empty",
    ):
        assistant.respond("   ")

    assistant.close()


def test_user_message_is_stripped_before_retrieval():

    retrieval_engine = DummyRetrievalEngine(
        memories=[]
    )

    llm_engine = DummyLLMEngine()

    assistant = AssistantEngine(
        retrieval_engine=retrieval_engine,
        llm_engine=llm_engine,
    )

    assistant.respond(
        "   What do I like?   "
    )

    assert (
        retrieval_engine.search_calls[0]["query"]
        == "What do I like?"
    )

    assistant.close()