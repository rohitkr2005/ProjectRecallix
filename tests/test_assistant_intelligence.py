from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.assistant.assistant_engine import AssistantEngine, InputType
from app.memory.memory_extractor import ExtractedMemory, MemoryExtractor
from app.memory.memory_store import MemoryStore


def create_in_memory_assistant(mock_llm=None):
    """Helper to create a fully wired AssistantEngine with an isolated in-memory DB."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database.models import Base

    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    session = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)()
    store = MemoryStore(session=session)

    # Mock embedding engine so tests don't spend time encoding sentence transformers
    mock_embedding = MagicMock()
    mock_embedding.generate_embedding.return_value = b"\x00" * 1536
    mock_embedding.generate_memory_embedding.return_value = b"\x00" * 1536

    mock_retrieval = MagicMock()
    mock_retrieval.memory_store = store
    mock_retrieval.embedding_engine = mock_embedding
    mock_retrieval.search.return_value = []
    mock_retrieval._detect_query_intent.return_value = None

    llm = mock_llm or MagicMock()
    llm.is_available.return_value = True
    llm.generate_with_memories.return_value = "Mocked LLM answer."
    llm.build_memory_context.return_value = "Mocked context."
    llm.verify_answer_grounding.return_value = {
        "grounded": True,
        "supported_values": [],
        "grounding_score": 1.0,
        "details": "Grounded.",
    }

    assistant = AssistantEngine(
        retrieval_engine=mock_retrieval,
        llm_engine=llm,
        memory_store=store,
        memory_extractor=MemoryExtractor(),
    )
    return assistant, store, mock_retrieval, llm


# =====================================================================
# 9.1 — Unified Assistant Engine
# =====================================================================


def test_unified_assistant_initialization_defaults():
    """Verify AssistantEngine initializes with all integrated subsystems."""
    assistant, store, retrieval, llm = create_in_memory_assistant()

    assert assistant.memory_store is store
    assert assistant.retrieval_engine is retrieval
    assert assistant.llm_engine is llm
    assert isinstance(assistant.memory_extractor, MemoryExtractor)
    assert assistant.is_available() is True


def test_unified_assistant_process_alias():
    """Verify process() is an alias to respond()."""
    assistant, _, _, _ = create_in_memory_assistant()
    assert assistant.process == assistant.respond


def test_unified_assistant_close():
    """Verify close() cleanly calls close on all subsystems."""
    assistant, store, retrieval, llm = create_in_memory_assistant()
    assistant.close()
    retrieval.close.assert_called_once()
    llm.close.assert_called_once()


# =====================================================================
# 9.2 — Memory + Question Handling (Input Classification)
# =====================================================================


def test_classify_statement():
    """Verify factual statements to remember are classified as STATEMENT."""
    assistant, _, _, _ = create_in_memory_assistant()

    assert assistant.classify_input("I like Python.") == InputType.STATEMENT
    assert assistant.classify_input("I live in Delhi.") == InputType.STATEMENT
    assert assistant.classify_input("I am studying Data Science at Tribhuvan College.") == InputType.STATEMENT
    assert assistant.classify_input("I work on Project Recallix.") == InputType.STATEMENT


def test_classify_question():
    """Verify questions without new facts are classified as QUESTION."""
    assistant, _, _, _ = create_in_memory_assistant()

    assert assistant.classify_input("What is my current city?") == InputType.QUESTION
    assert assistant.classify_input("Where do I live?") == InputType.QUESTION
    assert assistant.classify_input("What projects am I working on?") == InputType.QUESTION
    assert assistant.classify_input("Tell me what programming languages I know") == InputType.QUESTION
    assert assistant.classify_input("Do I like Python?") == InputType.QUESTION


def test_classify_both():
    """Verify compound input with new facts AND questions is classified as BOTH."""
    assistant, _, _, _ = create_in_memory_assistant()

    compound_1 = "I moved to Berlin. Where do I live now?"
    assert assistant.classify_input(compound_1) == InputType.BOTH

    compound_2 = "I am working on Project Recallix. What projects am I building?"
    assert assistant.classify_input(compound_2) == InputType.BOTH


def test_classify_neither():
    """Verify casual greetings and chit-chat are classified as NEITHER."""
    assistant, _, _, _ = create_in_memory_assistant()

    assert assistant.classify_input("Hello there!") == InputType.NEITHER
    assert assistant.classify_input("Hi") == InputType.NEITHER
    assert assistant.classify_input("Thank you very much") == InputType.NEITHER
    assert assistant.classify_input("Goodbye") == InputType.NEITHER
    assert assistant.classify_input("") == InputType.NEITHER


# =====================================================================
# 9.3 — Memory-Aware Conversation
# =====================================================================


def test_conversation_statement_saves_memory_and_acknowledges():
    """Verify statement input extracts and persists memory and returns confirmation."""
    assistant, store, _, _ = create_in_memory_assistant()

    result = assistant.respond("I am learning PyTorch.")
    assert result["input_type"] == InputType.STATEMENT
    assert "remembered" in result["response"].lower()
    assert len(result["extracted_memories"]) == 1

    saved = result["extracted_memories"][0]
    assert saved["relation"] == "studies"
    assert saved["value"] == "PyTorch"
    assert saved["category"] == "EDUCATION"

    # Verify memory is actually in the database
    memories_in_db = store.get_all_memories()
    assert len(memories_in_db) == 1
    assert memories_in_db[0].value == "PyTorch"
    assert memories_in_db[0].active is True


def test_conversation_turn_by_turn_statement_then_question():
    """Verify turn-by-turn interaction: store memory in turn 1, answer from it in turn 2."""
    assistant, store, retrieval, llm = create_in_memory_assistant()

    # Turn 1: Save statement
    turn_1 = assistant.respond("I live in Mumbai.")
    assert turn_1["input_type"] == InputType.STATEMENT
    assert len(store.get_all_memories()) == 1

    # Turn 2: Ask question
    db_mem = store.get_all_memories()[0]
    retrieval.search.return_value = [
        {
            "memory": db_mem,
            "score": 0.95,
            "explanation": {"relevance": 0.95, "reasons": ["LOCATION match"]},
        }
    ]
    retrieval._detect_query_intent.return_value = "LOCATION"
    llm.generate_with_memories.return_value = "You live in Mumbai."
    llm.verify_answer_grounding.return_value = {
        "grounded": True,
        "supported_values": ["Mumbai"],
        "grounding_score": 1.0,
        "details": "Grounded.",
    }

    turn_2 = assistant.respond("Where do I live?")
    assert turn_2["input_type"] == InputType.QUESTION
    assert turn_2["response"] == "You live in Mumbai."
    assert turn_2["supported"] is True
    assert turn_2["grounded"] is True
    assert len(turn_2["memories"]) == 1


def test_conversation_both_input_saves_and_answers():
    """Verify BOTH input saves memory first, then answers the question."""
    assistant, store, retrieval, llm = create_in_memory_assistant()

    llm.generate_with_memories.return_value = "You just started learning Rust."
    llm.verify_answer_grounding.return_value = {
        "grounded": True,
        "supported_values": ["Rust"],
        "grounding_score": 1.0,
        "details": "Grounded.",
    }

    def search_side_effect(query, **kwargs):
        # When search is called, the memory should already be saved in DB
        active_mems = store.get_all_memories()
        return [{"memory": m, "score": 0.9} for m in active_mems]

    retrieval.search.side_effect = search_side_effect
    retrieval._detect_query_intent.return_value = "GOAL"

    result = assistant.respond("I want to learn Rust. What do I want to learn?")
    assert result["input_type"] == InputType.BOTH
    assert len(result["extracted_memories"]) == 1
    assert result["extracted_memories"][0]["value"] == "Rust"
    assert len(store.get_all_memories()) == 1
    assert result["response"] == "You just started learning Rust."


# =====================================================================
# 9.4 — Multi-Memory Questions
# =====================================================================


def test_multi_memory_question_retrieval_and_reasoning():
    """Verify multi-memory question aggregates multiple memories into context."""
    assistant, store, retrieval, llm = create_in_memory_assistant()

    mem1 = SimpleNamespace(id=1, subject="User", relation="works_on", value="Project Recallix", category="PROJECT", active=True)
    mem2 = SimpleNamespace(id=2, subject="User", relation="works_on", value="Antigravity", category="PROJECT", active=True)

    retrieval.search.return_value = [
        {"memory": mem1, "score": 0.95, "explanation": {"relevance": 0.95, "reasons": ["project 1"]}},
        {"memory": mem2, "score": 0.90, "explanation": {"relevance": 0.90, "reasons": ["project 2"]}},
    ]
    retrieval._detect_query_intent.return_value = "PROJECT"
    llm.generate_with_memories.return_value = "You are working on Project Recallix and Antigravity."
    llm.verify_answer_grounding.return_value = {
        "grounded": True,
        "supported_values": ["Project Recallix", "Antigravity"],
        "grounding_score": 1.0,
        "details": "Both projects supported.",
    }

    result = assistant.respond("What projects am I working on?")
    assert result["input_type"] == InputType.QUESTION
    assert len(result["memories"]) == 2
    assert result["response"] == "You are working on Project Recallix and Antigravity."
    assert result["grounded"] is True


# =====================================================================
# 9.5 — Explainable Responses
# =====================================================================


def test_explainable_responses_exposes_why_used_without_internals():
    """Verify include_explanations=True provides human-readable rationales."""
    assistant, store, retrieval, llm = create_in_memory_assistant()

    mem = SimpleNamespace(
        id=42,
        subject="User",
        relation="lives_in",
        value="Delhi",
        category="LOCATION",
        active=True,
    )
    retrieval.search.return_value = [
        {
            "memory": mem,
            "score": 0.92,
            "explanation": {
                "relevance": 0.92,
                "reasons": ["direct intent match", "high semantic similarity"],
            },
        }
    ]
    retrieval._detect_query_intent.return_value = "LOCATION"

    result = assistant.respond(
        "Where do I live?",
        include_explanations=True,
    )

    assert "explanations" in result
    assert "why_used" in result
    assert len(result["why_used"]) == 1

    explanation = result["why_used"][0]
    assert "Memory 'User lives in Delhi' was used because:" in explanation
    assert "direct intent match" in explanation
    assert "high semantic similarity" in explanation

    # Ensure no internal SQL IDs or BLOB pointers are in the user explanation
    assert "id=42" not in explanation
    assert "vector" not in explanation
    assert "blob" not in explanation


# =====================================================================
# 9.6 — Interactive Chat Interface (ChatCLI)
# =====================================================================


def test_chat_cli_commands():
    """Verify ChatCLI handles slash commands (/help, /memories, /explain, /exit)."""
    from io import StringIO
    from app.chat.cli import ChatCLI

    assistant, store, _, _ = create_in_memory_assistant()
    out = StringIO()
    cli = ChatCLI(assistant=assistant, out_stream=out)

    # Test /help
    assert cli.handle_command("/help") is True
    assert "PROJECT RECALLIX" in out.getvalue()

    # Test /memories on empty
    out.truncate(0)
    out.seek(0)
    assert cli.handle_command("/memories") is True
    assert "No active memories" in out.getvalue()

    # Test /explain toggle
    out.truncate(0)
    out.seek(0)
    assert cli.include_explanations is False
    assert cli.handle_command("/explain") is True
    assert cli.include_explanations is True
    assert "ENABLED" in out.getvalue()

    # Test /exit
    out.truncate(0)
    out.seek(0)
    assert cli.handle_command("/exit") is False
    assert "Goodbye" in out.getvalue()


def test_chat_cli_process_input():
    """Verify ChatCLI.process_input formats assistant output cleanly."""
    from io import StringIO
    from app.chat.cli import ChatCLI

    assistant, store, _, _ = create_in_memory_assistant()
    out = StringIO()
    cli = ChatCLI(assistant=assistant, out_stream=out)

    output = cli.process_input("I work on Project Recallix.")
    assert "Recallix > Got it." in output
    assert "works on -> Project Recallix" in output

    # Check /memories after storing
    out.truncate(0)
    out.seek(0)
    cli.handle_command("/memories")
    assert "Project Recallix" in out.getvalue()


# =====================================================================
# 9.7 — Conversational Lifecycle Hardening
# =====================================================================


def test_conversation_lifecycle_superseding():
    """Verify multi-turn flow where memory is updated/superseded and subsequent question uses new fact."""
    assistant, store, retrieval, llm = create_in_memory_assistant()

    # Turn 1: User says they live in Delhi
    turn_1 = assistant.respond("I live in Delhi.")
    assert turn_1["input_type"] == InputType.STATEMENT
    assert "remembered" in turn_1["response"].lower()
    delhi_mems = store.get_all_memories()
    assert len(delhi_mems) == 1
    assert delhi_mems[0].value == "Delhi"
    assert delhi_mems[0].active is True

    # Turn 2: User says they moved to Mumbai
    turn_2 = assistant.respond("I moved to Mumbai.")
    assert turn_2["input_type"] == InputType.STATEMENT
    assert "updated" in turn_2["response"].lower()

    # Verify Delhi is now inactive and Mumbai is active
    active_mems = store.get_all_memories()
    inactive_mems = store.get_archived_memories()

    assert len(active_mems) == 1
    assert active_mems[0].value == "Mumbai"
    assert len(inactive_mems) == 1
    assert inactive_mems[0].value == "Delhi"

    # Turn 3: User asks where they live
    retrieval.search.return_value = [
        {
            "memory": active_mems[0],
            "score": 0.95,
            "explanation": {"relevance": 0.95, "reasons": ["LOCATION match"]},
        }
    ]
    retrieval._detect_query_intent.return_value = "LOCATION"
    llm.generate_with_memories.return_value = "You live in Mumbai."
    llm.verify_answer_grounding.return_value = {
        "grounded": True,
        "supported_values": ["Mumbai"],
        "grounding_score": 1.0,
        "details": "Grounded to active location.",
    }

    turn_3 = assistant.respond("Where do I live?")
    assert turn_3["input_type"] == InputType.QUESTION
    assert turn_3["response"] == "You live in Mumbai."
    assert turn_3["supported"] is True
    assert turn_3["grounded"] is True


def test_conversation_unsupported_question():
    """Verify conversational question with no relevant memories is safely declined."""
    assistant, store, retrieval, llm = create_in_memory_assistant()
    retrieval.search.return_value = []
    retrieval._detect_query_intent.return_value = None

    result = assistant.respond("What is my favorite movie?")
    assert result["input_type"] == InputType.QUESTION
    assert result["supported"] is False
    assert result["grounded"] is True
    assert result["response"] == assistant.UNSUPPORTED_RESPONSE


def test_conversation_empty_and_whitespace_validation():
    """Verify empty or whitespace-only inputs raise ValueError."""
    assistant, _, _, _ = create_in_memory_assistant()

    with pytest.raises(ValueError, match="User message cannot be empty"):
        assistant.respond("")

    with pytest.raises(ValueError, match="User message cannot be empty"):
        assistant.respond("   \n\t  ")

