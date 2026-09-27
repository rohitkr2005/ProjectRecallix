import pytest
from unittest.mock import MagicMock, patch
from app.assistant.assistant_engine import AssistantEngine, InputType
from app.memory.memory_store import MemoryStore
from app.retrieval.retrieval_engine import RetrievalEngine


def create_mock_assistant():
    mock_retrieval = MagicMock()
    mock_retrieval.search.return_value = []
    mock_retrieval._detect_query_intent.return_value = None
    mock_retrieval.analyze_query_intent.return_value = {
        "intent": None,
        "confidence": 0.0,
        "strict": False,
        "normalized_query": "",
        "scores": {},
    }

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.generate.return_value = "Intelligent AI response."
    mock_llm.generate_with_memories.return_value = "Intelligent AI response."
    mock_llm.verify_answer_grounding.return_value = {
        "grounded": True,
        "supported_values": [],
        "grounding_score": 1.0,
        "details": "General conversational answer.",
    }

    mock_store = MagicMock()
    saved_mock = MagicMock()
    saved_mock.id = 1
    mock_store.save_memory_with_semantics.return_value = (saved_mock, "created", {})
    mock_store.list_memories.return_value = []

    assistant = AssistantEngine(
        retrieval_engine=mock_retrieval,
        llm_engine=mock_llm,
        memory_store=mock_store,
    )
    return assistant, mock_store, mock_retrieval, mock_llm


def test_assistant_self_awareness_who_are_you():
    """Verify Recallix introduces itself warmly when asked 'who are you?'."""
    assistant, _, _, _ = create_mock_assistant()
    result = assistant.respond("who are you?")

    assert result["supported"] is True
    assert result["grounded"] is True
    assert "Recallix" in result["response"]
    assert "Second Brain" in result["response"]
    assert result["llm_status"] == "assistant_identity"


def test_assistant_self_awareness_what_can_you_do():
    """Verify Recallix describes its capabilities when asked 'what can you do?'."""
    assistant, _, _, _ = create_mock_assistant()
    result = assistant.respond("what can you do?")

    assert result["supported"] is True
    assert "Recallix" in result["response"]
    assert "remember" in result["response"]


def test_user_introduction_and_identity_saving():
    """Verify 'I am Rohit do you know me?' saves name 'Rohit' and greets warmly."""
    assistant, store, _, _ = create_mock_assistant()
    result = assistant.respond("I am Rohit do you know me?")

    assert result["supported"] is True
    assert result["grounded"] is True
    assert "Rohit" in result["response"]
    assert len(result["extracted_memories"]) >= 1

    extracted_relations = [r["relation"] for r in result["extracted_memories"]]
    extracted_values = [r["value"] for r in result["extracted_memories"]]
    assert "name" in extracted_relations
    assert "Rohit" in extracted_values
    assert store.save_memory_with_semantics.called


def test_owner_identity_who_am_i_with_saved_name():
    """Verify 'who am I?' recalls saved owner name."""
    assistant, store, _, _ = create_mock_assistant()
    mem = MagicMock()
    mem.relation = "name"
    mem.value = "Rohit"
    mem.active = True
    store.list_memories.return_value = [mem]

    result = assistant.respond("who am I?")
    assert result["supported"] is True
    assert "Rohit" in result["response"]


def test_general_question_uses_llm_mind():
    """Verify general questions without personal facts use the LLM brain rather than refusing."""
    assistant, _, _, mock_llm = create_mock_assistant()
    mock_llm.generate_with_memories.return_value = "Python is a high-level programming language."

    result = assistant.respond("Can you explain what Python is?")
    assert result["supported"] is True
    assert result["response"] == "Python is a high-level programming language."
    assert mock_llm.generate_with_memories.called

    # Verify fallback_on_empty is False for general questions
    call_kwargs = mock_llm.generate_with_memories.call_args[1]
    assert call_kwargs.get("fallback_on_empty") is False


def test_conversation_history_passed_to_llm():
    """Verify multi-turn conversation history is forwarded to LLMEngine."""
    assistant, _, _, mock_llm = create_mock_assistant()
    history = [
        {"role": "user", "content": "I am building a Flutter app."},
        {"role": "assistant", "content": "That sounds great! What kind of app is it?"},
    ]

    assistant.respond(
        "What database should I use for it?",
        conversation_history=history,
    )

    call_kwargs = mock_llm.generate_with_memories.call_args[1]
    assert call_kwargs.get("history") == history
