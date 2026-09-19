import io
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from app.assistant.assistant_engine import AssistantEngine
from app.llm.llm_engine import (
    GROUNDED_SYSTEM_PROMPT,
    UNSUPPORTED_RESPONSE,
    LLMEngine,
)


def make_memory(
    subject="User",
    relation="likes",
    value="Python",
    category="PREFERENCE",
    active=True,
    score=0.9,
):
    mem = SimpleNamespace(
        id=1,
        subject=subject,
        relation=relation,
        value=value,
        category=category,
        active=active,
        importance=5,
        created_at="2026-09-19T00:00:00",
    )
    return {
        "memory": mem,
        "score": score,
        "explanation": {"relevance": score, "reasons": ["test match"]},
    }


# =====================================================================
# 8.1 — LLM Interface (LLMEngine)
# =====================================================================


def test_llm_engine_initialization_defaults_and_env():
    """Verify default initialization and custom overrides via parameters and env vars."""
    engine = LLMEngine()
    assert engine.model == "qwen2.5:3b"
    assert engine.base_url == "http://localhost:11434"
    assert engine.timeout == 120.0
    assert engine.connect_timeout == 3.0

    custom = LLMEngine(
        model="llama3:8b",
        base_url="http://custom-host:8000/",
        timeout=60,
        connect_timeout=5,
    )
    assert custom.model == "llama3:8b"
    assert custom.base_url == "http://custom-host:8000"
    assert custom.timeout == 60.0
    assert custom.connect_timeout == 5.0

    with patch.dict(
        os.environ,
        {
            "RECALLIX_LLM_MODEL": "mistral:7b",
            "RECALLIX_LLM_BASE_URL": "http://ollama-env:11434/",
            "RECALLIX_LLM_TIMEOUT": "45",
            "RECALLIX_LLM_CONNECT_TIMEOUT": "2",
        },
    ):
        env_engine = LLMEngine()
        assert env_engine.model == "mistral:7b"
        assert env_engine.base_url == "http://ollama-env:11434"
        assert env_engine.timeout == 45.0
        assert env_engine.connect_timeout == 2.0


def test_llm_engine_is_available_success():
    """Verify is_available returns True when Ollama responds with 200."""
    engine = LLMEngine()
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.__enter__.return_value = mock_response

    with patch("app.llm.llm_engine.urlopen", return_value=mock_response):
        assert engine.is_available() is True


def test_llm_engine_is_available_failure():
    """Verify is_available returns False when Ollama is unreachable."""
    engine = LLMEngine()

    with patch("app.llm.llm_engine.urlopen", side_effect=URLError("Connection refused")):
        assert engine.is_available() is False

    with patch(
        "app.llm.llm_engine.urlopen",
        side_effect=HTTPError("http://localhost:11434/api/tags", 500, "Internal Error", {}, None),
    ):
        assert engine.is_available() is False

    with patch("app.llm.llm_engine.urlopen", side_effect=OSError("Network down")):
        assert engine.is_available() is False


def test_llm_engine_generate_success():
    """Verify generate sends proper chat payload and extracts response content."""
    engine = LLMEngine()
    chat_payload = {
        "message": {
            "role": "assistant",
            "content": "Recallix is working properly.",
        }
    }
    response_bytes = json.dumps(chat_payload).encode("utf-8")

    mock_response = MagicMock()
    mock_response.read.return_value = response_bytes
    mock_response.__enter__.return_value = mock_response

    with patch("app.llm.llm_engine.urlopen", return_value=mock_response) as mock_urlopen:
        result = engine.generate(
            prompt="Hello Recallix",
            system_prompt="You are an assistant.",
            temperature=0.3,
            max_tokens=250,
        )

        assert result == "Recallix is working properly."
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.get_method() == "POST"
        assert req.full_url == "http://localhost:11434/api/chat"

        sent_body = json.loads(req.data.decode("utf-8"))
        assert sent_body["model"] == "qwen2.5:3b"
        assert sent_body["messages"][0]["role"] == "system"
        assert sent_body["messages"][0]["content"] == "You are an assistant."
        assert sent_body["messages"][1]["role"] == "user"
        assert sent_body["messages"][1]["content"] == "Hello Recallix"
        assert sent_body["options"]["temperature"] == 0.3
        assert sent_body["options"]["num_predict"] == 250


def test_llm_engine_generate_validation_errors():
    """Verify generate rejects empty or whitespace-only prompts."""
    engine = LLMEngine()

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        engine.generate("")

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        engine.generate("   \n\t  ")


def test_llm_engine_generate_http_error_handling():
    """Verify generate handles HTTP errors with meaningful diagnostics."""
    engine = LLMEngine()
    error_io = io.BytesIO(b"Model not found")
    http_err = HTTPError(
        url="http://localhost:11434/api/chat",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=error_io,
    )

    with patch("app.llm.llm_engine.urlopen", side_effect=http_err):
        with pytest.raises(RuntimeError, match="Ollama request failed: Model not found"):
            engine.generate("test")


def test_llm_engine_generate_connection_error_handling():
    """Verify generate handles connection failure with actionable guidance."""
    engine = LLMEngine()

    with patch("app.llm.llm_engine.urlopen", side_effect=URLError("Connection refused")):
        with pytest.raises(RuntimeError, match="Unable to connect to Ollama"):
            engine.generate("test")


def test_llm_engine_generate_empty_content_handling():
    """Verify generate handles empty response content from Ollama."""
    engine = LLMEngine()
    chat_payload = {"message": {"role": "assistant", "content": ""}}
    response_bytes = json.dumps(chat_payload).encode("utf-8")

    mock_response = MagicMock()
    mock_response.read.return_value = response_bytes
    mock_response.__enter__.return_value = mock_response

    with patch("app.llm.llm_engine.urlopen", return_value=mock_response):
        with pytest.raises(RuntimeError, match="Ollama returned an empty response"):
            engine.generate("test")


# =====================================================================
# 8.2 — Memory Context Construction
# =====================================================================


def test_build_memory_context_natural_language_formatting():
    """Verify build_memory_context produces natural English sentences."""
    engine = LLMEngine()
    memories = [
        make_memory(subject="User", relation="works_on", value="Project Recallix"),
        make_memory(subject="User", relation="likes", value="Python"),
        make_memory(subject="User", relation="lives_in", value="Delhi"),
        make_memory(subject="User", relation="studies_at", value="Tribhuvan College"),
        make_memory(subject="User", relation="wants_to_learn", value="PyTorch"),
        make_memory(subject="User", relation="current_role", value="AI Engineer"),
    ]

    context = engine.build_memory_context(memories)

    expected_lines = [
        "User works on Project Recallix.",
        "User likes Python.",
        "User lives in Delhi.",
        "User studies at Tribhuvan College.",
        "User wants to learn PyTorch.",
        "User has the current role AI Engineer.",
    ]
    assert context == "\n".join(expected_lines)


def test_build_memory_context_strips_internal_metadata():
    """Verify context never leaks internal IDs, scores, categories, or timestamps."""
    engine = LLMEngine()
    memories = [
        make_memory(
            subject="User",
            relation="works_on",
            value="Project Recallix",
            category="PROJECT",
            score=0.987,
        )
    ]

    context = engine.build_memory_context(memories)
    assert "0.987" not in context
    assert "PROJECT" not in context
    assert "score" not in context
    assert "relevance" not in context
    assert "explanation" not in context
    assert "2026" not in context
    assert context == "User works on Project Recallix."


def test_build_memory_context_filters_inactive_and_empty():
    """Verify inactive memories and empty values are excluded from context."""
    engine = LLMEngine()
    memories = [
        make_memory(subject="User", relation="likes", value="Python", active=True),
        make_memory(subject="User", relation="lives_in", value="Old City", active=False),
        {"memory": None, "score": 0.5},
        make_memory(subject="User", relation="knows", value="   ", active=True),
    ]

    context = engine.build_memory_context(memories)
    assert context == "User likes Python."


def test_build_memory_context_empty_fallback():
    """Verify empty or entirely invalid memories produce standard fallback."""
    engine = LLMEngine()
    assert engine.build_memory_context([]) == "No relevant memories found."
    assert engine.build_memory_context(None) == "No relevant memories found."

    inactive_only = [
        make_memory(value="Archived Item", active=False),
    ]
    assert engine.build_memory_context(inactive_only) == "No relevant memories found."


def test_build_memory_context_supports_bare_objects_and_max_memories():
    """Verify build_memory_context supports bare Memory objects and max_memories limit."""
    engine = LLMEngine()
    bare_memories = [
        SimpleNamespace(subject="User", relation="knows", value="Python", active=True),
        SimpleNamespace(subject="User", relation="knows", value="SQL", active=True),
        SimpleNamespace(subject="User", relation="knows", value="Git", active=True),
    ]

    context_all = engine.build_memory_context(bare_memories)
    assert context_all == "User knows Python.\nUser knows SQL.\nUser knows Git."

    context_capped = engine.build_memory_context(bare_memories, max_memories=2)
    assert context_capped == "User knows Python.\nUser knows SQL."


# =====================================================================
# 8.3 — Grounded Reasoning
# =====================================================================


def test_grounded_system_prompt_contains_anti_hallucination_rules():
    """Verify GROUNDED_SYSTEM_PROMPT includes explicit grounding constraints."""
    prompt = GROUNDED_SYSTEM_PROMPT
    assert "explicitly provided memory context" in prompt
    assert "Do not invent facts" in prompt
    assert "Do not infer facts that are not explicitly supported" in prompt
    assert "Do not treat general world knowledge as user memory" in prompt
    assert "clearly say that you do not have enough information" in prompt
    assert "Never claim an unsupported user fact" in prompt
    assert "Do not mention memory categories, relevance scores" in prompt


def test_generate_with_memories_uses_grounded_prompt_and_context():
    """Verify generate_with_memories correctly bundles memory context and user question."""
    engine = LLMEngine()
    memories = [
        make_memory(subject="User", relation="works_on", value="Project Recallix")
    ]

    with patch.object(engine, "generate", return_value="You are working on Project Recallix.") as mock_generate:
        response = engine.generate_with_memories(
            user_message="What project am I working on?",
            memories=memories,
            temperature=0.1,
            max_tokens=300,
        )

        assert response == "You are working on Project Recallix."
        mock_generate.assert_called_once()
        kwargs = mock_generate.call_args[1]

        assert "Memory context:\nUser works on Project Recallix." in kwargs["prompt"]
        assert "User question:\nWhat project am I working on?" in kwargs["prompt"]
        assert kwargs["system_prompt"] == GROUNDED_SYSTEM_PROMPT
        assert kwargs["temperature"] == 0.1
        assert kwargs["max_tokens"] == 300


# =====================================================================
# 8.4 — Context Prioritization
# =====================================================================


def test_prioritize_memories_sorts_by_score_descending():
    """Verify prioritize_memories ranks highest-scoring memories first."""
    engine = LLMEngine()
    memories = [
        make_memory(value="C", score=0.4),
        make_memory(value="A", score=0.95),
        make_memory(value="B", score=0.75),
    ]

    ranked = engine.prioritize_memories(memories)
    assert [m["memory"].value for m in ranked] == ["A", "B", "C"]


def test_prioritize_memories_filters_min_relevance():
    """Verify prioritize_memories removes memories below min_relevance."""
    engine = LLMEngine()
    memories = [
        make_memory(value="High", score=0.8),
        make_memory(value="Borderline", score=0.3),
        make_memory(value="Low", score=0.15),
    ]

    filtered = engine.prioritize_memories(memories, min_relevance=0.25)
    assert [m["memory"].value for m in filtered] == ["High", "Borderline"]


def test_prioritize_memories_caps_max_memories():
    """Verify prioritize_memories respects max_memories limit."""
    engine = LLMEngine()
    memories = [
        make_memory(value="1st", score=0.9),
        make_memory(value="2nd", score=0.8),
        make_memory(value="3rd", score=0.7),
        make_memory(value="4th", score=0.6),
    ]

    capped = engine.prioritize_memories(memories, max_memories=2)
    assert len(capped) == 2
    assert [m["memory"].value for m in capped] == ["1st", "2nd"]


def test_prioritize_memories_excludes_inactive():
    """Verify prioritize_memories skips inactive memories."""
    engine = LLMEngine()
    memories = [
        make_memory(value="Active", score=0.8, active=True),
        make_memory(value="Inactive", score=0.99, active=False),
    ]

    result = engine.prioritize_memories(memories)
    assert len(result) == 1
    assert result[0]["memory"].value == "Active"


# =====================================================================
# 8.5 — Unsupported Question Handling
# =====================================================================


def test_unsupported_response_constant_matches():
    """Verify standard unsupported response constant across engine and assistant."""
    expected = "I don't have enough information in my memory to answer that."
    assert UNSUPPORTED_RESPONSE == expected
    assert LLMEngine.UNSUPPORTED_RESPONSE == expected
    assert AssistantEngine.UNSUPPORTED_RESPONSE == expected


def test_generate_with_memories_fallback_on_empty():
    """Verify generate_with_memories returns UNSUPPORTED_RESPONSE immediately on empty memories."""
    engine = LLMEngine()

    with patch.object(engine, "generate") as mock_generate:
        result_empty = engine.generate_with_memories(
            user_message="What is my favorite movie?",
            memories=[],
            fallback_on_empty=True,
        )
        assert result_empty == UNSUPPORTED_RESPONSE
        mock_generate.assert_not_called()

        inactive_memories = [make_memory(value="The Matrix", active=False)]
        result_inactive = engine.generate_with_memories(
            user_message="What is my favorite movie?",
            memories=inactive_memories,
            fallback_on_empty=True,
        )
        assert result_inactive == UNSUPPORTED_RESPONSE
        mock_generate.assert_not_called()


def test_assistant_engine_handles_unsupported_question():
    """Verify AssistantEngine returns unsupported response with supported=False."""
    mock_retrieval = MagicMock()
    mock_retrieval.search.return_value = []
    mock_retrieval._detect_query_intent.return_value = None

    mock_llm = MagicMock()

    assistant = AssistantEngine(
        retrieval_engine=mock_retrieval,
        llm_engine=mock_llm,
    )

    result = assistant.respond("What is my favorite movie?")
    assert result["response"] == UNSUPPORTED_RESPONSE
    assert result["supported"] is False
    assert result["memories"] == []
    mock_llm.generate_with_memories.assert_not_called()


def test_assistant_engine_handles_low_relevance_as_unsupported():
    """Verify AssistantEngine treats sub-threshold memories as unsupported."""
    mock_retrieval = MagicMock()
    # Memory score is 0.1, below default memory_relevance threshold (0.25)
    mock_retrieval.search.return_value = [
        make_memory(relation="likes", value="Sci-Fi", score=0.1)
    ]
    mock_retrieval._detect_query_intent.return_value = "PREFERENCE"

    mock_llm = MagicMock()

    assistant = AssistantEngine(
        retrieval_engine=mock_retrieval,
        llm_engine=mock_llm,
    )

    result = assistant.respond(
        "What is my favorite movie?",
        memory_relevance=0.25,
    )
    assert result["response"] == UNSUPPORTED_RESPONSE
    assert result["supported"] is False
    assert result["memories"] == []
    mock_llm.generate_with_memories.assert_not_called()


# =====================================================================
# 8.6 — Answer Grounding
# =====================================================================


def test_verify_answer_grounding_matches_memory_values():
    """Verify verify_answer_grounding confirms when response references memory values."""
    engine = LLMEngine()
    memories = [
        make_memory(subject="User", relation="works_on", value="Project Recallix"),
        make_memory(subject="User", relation="likes", value="Python"),
    ]

    response = "You are working on Project Recallix and you like Python."
    result = engine.verify_answer_grounding(response, memories)

    assert result["grounded"] is True
    assert "Project Recallix" in result["supported_values"]
    assert "Python" in result["supported_values"]
    assert result["grounding_score"] == 1.0


def test_verify_answer_grounding_detects_hallucination():
    """Verify verify_answer_grounding flags ungrounded claims not in memories."""
    engine = LLMEngine()
    memories = [
        make_memory(subject="User", relation="likes", value="Python"),
    ]

    response = "You love Java and Spring Boot."
    result = engine.verify_answer_grounding(response, memories)

    assert result["grounded"] is False
    assert result["supported_values"] == []
    assert result["grounding_score"] == 0.0
    assert "Ungrounded" in result["details"]


def test_verify_answer_grounding_unsupported_response():
    """Verify standard unsupported response is recognized as grounded."""
    engine = LLMEngine()
    result = engine.verify_answer_grounding(UNSUPPORTED_RESPONSE, [])
    assert result["grounded"] is True
    assert result["grounding_score"] == 1.0


def test_assistant_attaches_grounding_metadata():
    """Verify AssistantEngine attaches grounding metadata to responses."""
    mem = make_memory(subject="User", relation="works_on", value="Project Recallix")
    mock_retrieval = MagicMock()
    mock_retrieval.search.return_value = [mem]
    mock_retrieval._detect_query_intent.return_value = "PROJECT"

    mock_llm = MagicMock()
    mock_llm.generate_with_memories.return_value = "You are working on Project Recallix."
    mock_llm.verify_answer_grounding.return_value = {
        "grounded": True,
        "supported_values": ["Project Recallix"],
        "grounding_score": 1.0,
        "details": "Grounded: 1/1 memory values supported.",
    }

    assistant = AssistantEngine(
        retrieval_engine=mock_retrieval,
        llm_engine=mock_llm,
    )

    result = assistant.respond("What project am I working on?")
    assert result["supported"] is True
    assert result["grounded"] is True
    assert result["grounding_details"]["grounded"] is True
    assert result["llm_status"] == "success"


# =====================================================================
# 8.7 — Multi-Memory Reasoning
# =====================================================================


def test_multi_memory_context_synthesis():
    """Verify context construction synthesizes multiple diverse memories cleanly."""
    engine = LLMEngine()
    memories = [
        make_memory(subject="User", relation="works_on", value="Project Recallix"),
        make_memory(subject="User", relation="works_on", value="Antigravity"),
        make_memory(subject="User", relation="knows", value="Python"),
        make_memory(subject="User", relation="wants_to_learn", value="Rust"),
    ]

    context = engine.build_memory_context(memories)
    lines = context.split("\n")
    assert len(lines) == 4
    assert "User works on Project Recallix." in lines
    assert "User works on Antigravity." in lines
    assert "User knows Python." in lines
    assert "User wants to learn Rust." in lines


def test_multi_memory_context_deduplication():
    """Verify duplicate formatted memories are deduplicated in context."""
    engine = LLMEngine()
    memories = [
        make_memory(subject="User", relation="knows", value="Python"),
        make_memory(subject="User", relation="knows", value="Python"),
    ]

    context = engine.build_memory_context(memories)
    assert context == "User knows Python."


def test_multi_memory_conflicting_values_prefers_active():
    """Verify inactive conflicting memories are excluded from multi-memory context."""
    engine = LLMEngine()
    memories = [
        make_memory(subject="User", relation="lives_in", value="Delhi", active=False),
        make_memory(subject="User", relation="lives_in", value="Mumbai", active=True),
    ]

    context = engine.build_memory_context(memories)
    assert context == "User lives in Mumbai."


# =====================================================================
# 8.8 — LLM Failure Handling
# =====================================================================


def test_assistant_handles_llm_connection_failure_with_fallback():
    """Verify AssistantEngine falls back to extractive memory summary when LLM fails."""
    mem = make_memory(subject="User", relation="works_on", value="Project Recallix")
    mock_retrieval = MagicMock()
    mock_retrieval.search.return_value = [mem]
    mock_retrieval._detect_query_intent.return_value = "PROJECT"

    mock_llm = MagicMock()
    mock_llm.generate_with_memories.side_effect = RuntimeError("Unable to connect to Ollama.")
    mock_llm.build_memory_context.return_value = "User works on Project Recallix."
    mock_llm.verify_answer_grounding.return_value = {
        "grounded": True,
        "supported_values": ["Project Recallix"],
        "grounding_score": 1.0,
        "details": "Grounded fallback.",
    }

    assistant = AssistantEngine(
        retrieval_engine=mock_retrieval,
        llm_engine=mock_llm,
    )

    result = assistant.respond("What project am I working on?")
    assert result["supported"] is True
    assert "unable to reach the local LLM" in result["response"]
    assert "User works on Project Recallix." in result["response"]
    assert "fallback: Unable to connect to Ollama." in result["llm_status"]
    assert result["grounded"] is True


def test_assistant_handles_llm_timeout_with_fallback():
    """Verify AssistantEngine handles timeout with extractive fallback."""
    mem = make_memory(subject="User", relation="likes", value="Python")
    mock_retrieval = MagicMock()
    mock_retrieval.search.return_value = [mem]
    mock_retrieval._detect_query_intent.return_value = "PREFERENCE"

    mock_llm = MagicMock()
    mock_llm.generate_with_memories.side_effect = TimeoutError("Request timed out")
    mock_llm.build_memory_context.return_value = "User likes Python."
    mock_llm.verify_answer_grounding.return_value = {
        "grounded": True,
        "supported_values": ["Python"],
        "grounding_score": 1.0,
        "details": "Grounded fallback.",
    }

    assistant = AssistantEngine(
        retrieval_engine=mock_retrieval,
        llm_engine=mock_llm,
    )

    result = assistant.respond("What do I like?")
    assert result["supported"] is True
    assert "User likes Python." in result["response"]
    assert "fallback: Request timed out" in result["llm_status"]


def test_assistant_disables_fallback_when_configured():
    """Verify AssistantEngine raises error if fallback_to_extractive=False."""
    mem = make_memory(subject="User", relation="likes", value="Python")
    mock_retrieval = MagicMock()
    mock_retrieval.search.return_value = [mem]
    mock_retrieval._detect_query_intent.return_value = "PREFERENCE"

    mock_llm = MagicMock()
    mock_llm.generate_with_memories.side_effect = RuntimeError("Ollama crashed")

    assistant = AssistantEngine(
        retrieval_engine=mock_retrieval,
        llm_engine=mock_llm,
    )

    with pytest.raises(RuntimeError, match="Ollama crashed"):
        assistant.respond("What do I like?", fallback_to_extractive=False)

