from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.assistant.assistant_engine import AssistantEngine, InputType
from app.database.models import Base, Memory, MemoryEdge
from app.memory.forgetting import MemoryForgetter
from app.memory.graph import MemoryGraph
from app.memory.importance_learning import ImportanceLearner
from app.memory.memory_store import MemoryStore
from app.memory.temporal import (
    TemporalState,
    detect_query_temporal_intent,
    detect_temporal_state,
)
from app.retrieval.advanced_retrieval import AdvancedRetrievalEngine


def create_in_memory_db():
    """Helper to create an isolated in-memory DB session."""
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    session = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)()
    store = MemoryStore(session=session)
    return session, store


# =====================================================================
# 10.1, 10.2, 10.3 — Importance Learning, Feedback & Reinforcement
# =====================================================================


def test_importance_learning_effective_importance_calculation():
    """Verify effective importance calculation balances base importance, boost, and penalty."""
    learner = ImportanceLearner()

    # Baseline with no access
    assert learner.calculate_effective_importance(base_importance=5, access_count=0, helpful_count=0) == 5.0

    # Helpful reinforcement boosts importance
    boosted = learner.calculate_effective_importance(
        base_importance=5,
        access_count=4,
        helpful_count=4,
        reinforcement_score=2.0,
    )
    assert boosted > 5.0
    assert boosted <= 10.0

    # Unhelpful access penalty reduces importance
    penalized = learner.calculate_effective_importance(
        base_importance=5,
        access_count=10,
        helpful_count=0,
        reinforcement_score=0.0,
    )
    assert penalized < 5.0
    assert penalized >= 1.0


def test_retrieval_feedback_record_access_updates_attributes():
    """Verify record_access updates access counts, reinforcement, and last_accessed_at."""
    session, store = create_in_memory_db()
    learner = ImportanceLearner(session=session)

    mem, _, _ = store.save_memory_with_semantics(
        subject="User",
        relation="knows",
        value="Python",
        category="SKILL",
        importance=6,
    )

    # Record helpful access
    res1 = learner.record_access(mem, was_helpful=True)
    assert res1["access_count"] == 1
    assert res1["helpful_count"] == 1
    assert res1["reinforcement_score"] == 0.5
    assert mem.access_count == 1
    assert mem.helpful_count == 1
    assert mem.last_accessed_at is not None

    # Record unhelpful access
    res2 = learner.record_access(mem, was_helpful=False)
    assert res2["access_count"] == 2
    assert res2["helpful_count"] == 1
    assert res2["reinforcement_score"] == 0.4


def test_apply_retrieval_feedback_bulk():
    """Verify apply_retrieval_feedback accurately identifies cited vs uncited memories."""
    session, store = create_in_memory_db()
    learner = ImportanceLearner(session=session)

    mem1, _, _ = store.save_memory_with_semantics(subject="User", relation="works_on", value="Recallix", category="PROJECT")
    mem2, _, _ = store.save_memory_with_semantics(subject="User", relation="knows", value="Java", category="SKILL")

    retrieved = [{"memory": mem1}, {"memory": mem2}]
    # Answer grounded in Recallix only
    feedback = learner.apply_retrieval_feedback(retrieved, supported_values=["Recallix"])

    assert feedback["total_processed"] == 2
    assert mem1.id in feedback["helpful_ids"]
    assert mem2.id in feedback["unhelpful_ids"]
    assert mem1.helpful_count == 1
    assert mem2.helpful_count == 0


# =====================================================================
# 10.4 — Memory Forgetting
# =====================================================================


def test_forgetting_protects_high_importance_and_reinforced():
    """Verify high importance or reinforced memories are protected from forgetting."""
    session, store = create_in_memory_db()
    forgetter = MemoryForgetter(session=session)

    mem, _, _ = store.save_memory_with_semantics(
        subject="User",
        relation="works_on",
        value="Core Project",
        category="PROJECT",
        importance=8,
    )
    mem.last_accessed_at = datetime.utcnow() - timedelta(days=60)

    should_forget, reason, details = forgetter.evaluate_forgetting(mem, decay_days=30, min_importance=2)
    assert should_forget is False
    assert "protected" in reason


def test_forgetting_archives_dormant_low_importance_memory():
    """Verify dormant, low-importance memories are marked active=False non-destructively."""
    session, store = create_in_memory_db()
    forgetter = MemoryForgetter(session=session)

    mem, _, _ = store.save_memory_with_semantics(
        subject="User",
        relation="likes",
        value="Casual Hobby",
        category="PREFERENCE",
        importance=1,
    )
    # Simulate 45 days dormant
    mem.created_at = datetime.utcnow() - timedelta(days=50)
    mem.last_accessed_at = datetime.utcnow() - timedelta(days=45)

    should_forget, reason, _ = forgetter.evaluate_forgetting(mem, decay_days=30, min_importance=2)
    assert should_forget is True
    assert reason == "low_importance_and_dormant"

    # Execute non-destructive forget
    forgotten, status, meta = forgetter.forget_memory(mem)
    assert status == "archived"
    assert forgotten.active is False
    assert meta["reason"] == "FORGOTTEN_CONTROLLED_DECAY"

    # Verify memory is in archived list, not active
    assert len(store.get_all_memories()) == 0
    assert len(store.get_archived_memories()) == 1


# =====================================================================
# 10.5 — Temporal Memory
# =====================================================================


def test_detect_temporal_state_past_present_future():
    """Verify temporal state detection identifies past, present, and future statements."""
    assert detect_temporal_state("Previously lived in Delhi.") == TemporalState.PAST
    assert detect_temporal_state("I used to work at Google.") == TemporalState.PAST
    assert detect_temporal_state("I lived in Boston before.") == TemporalState.PAST

    assert detect_temporal_state("Currently lives in Mumbai.") == TemporalState.PRESENT
    assert detect_temporal_state("I work on Project Recallix.") == TemporalState.PRESENT
    assert detect_temporal_state("I am learning Python now.") == TemporalState.PRESENT

    assert detect_temporal_state("Plans to move to Bangalore.") == TemporalState.FUTURE
    assert detect_temporal_state("I want to learn Rust.") == TemporalState.FUTURE
    assert detect_temporal_state("I will build an AI agent in the future.") == TemporalState.FUTURE


def test_detect_query_temporal_intent():
    """Verify query temporal intent detection identifies target time horizons."""
    assert detect_query_temporal_intent("Where did I live before?") == TemporalState.PAST
    assert detect_query_temporal_intent("Where did I use to work?") == TemporalState.PAST

    assert detect_query_temporal_intent("Where do I currently live?") == TemporalState.PRESENT
    assert detect_query_temporal_intent("What is my current role?") == TemporalState.PRESENT

    assert detect_query_temporal_intent("Where do I plan to move?") == TemporalState.FUTURE
    assert detect_query_temporal_intent("What do I want to learn?") == TemporalState.FUTURE

    assert detect_query_temporal_intent("What projects am I working on?") is None


# =====================================================================
# 10.6 & 10.7 — Memory Relationships & Memory Graph
# =====================================================================


def test_memory_graph_construction_and_traversal():
    """Verify MemoryGraph connects memories and explicit edges for multi-hop discovery."""
    graph = MemoryGraph()

    # Direct memory edges
    graph.add_edge(source="User", relation="works_on", target="Project Recallix", category="PROJECT")
    graph.add_edge(source="User", relation="knows", target="Python", category="SKILL")

    # Explicit concept edges (10.6)
    graph.add_edge(source="Project Recallix", relation="uses", target="Python")
    graph.add_edge(source="Python", relation="related_to", target="Data Science")

    # Verify neighbors
    neighbors = graph.get_neighbors("Python", direction="both")
    assert len(neighbors) == 3

    # Verify path finding: User -> works_on -> Recallix -> uses -> Python -> related_to -> Data Science
    paths = graph.find_paths("User", "Data Science", max_depth=3)
    assert len(paths) >= 1
    shortest = min(paths, key=len)
    assert len(shortest) == 2  # User -[knows]-> Python -[related_to]-> Data Science

    # Verify proximity
    assert graph.calculate_proximity("User", "Python") == 1.0
    assert graph.calculate_proximity("User", "Data Science") == 0.67
    assert graph.calculate_proximity("User", "UnrelatedEntity") == 0.0

    # Verify connected entities
    connected = graph.get_connected_entities("Project Recallix", max_depth=2)
    assert "python" in connected
    assert "data science" in connected


# =====================================================================
# 10.8 — Advanced Retrieval
# =====================================================================


def test_advanced_retrieval_temporal_filtering():
    """Verify AdvancedRetrievalEngine boosts past memories when a past query is asked."""
    session, store = create_in_memory_db()

    # Past location
    mem_past, _, _ = store.save_memory_with_semantics(
        subject="User",
        relation="lives_in",
        value="Delhi",
        category="LOCATION",
        temporal_state="PAST",
    )
    # Present location
    mem_pres, _, _ = store.save_memory_with_semantics(
        subject="User",
        relation="lives_in",
        value="Mumbai",
        category="LOCATION",
        temporal_state="PRESENT",
    )

    retrieval = AdvancedRetrievalEngine(memory_store=store)

    # Query targeting past: "Where did I live before?"
    past_results = retrieval.search("Where did I live before?", top_k=2)
    assert len(past_results) == 2
    # Delhi (PAST) must rank above Mumbai (PRESENT) for past query
    assert past_results[0]["memory"].value == "Delhi"

    # Query targeting present: "Where do I currently live?"
    pres_results = retrieval.search("Where do I currently live?", top_k=2)
    assert len(pres_results) == 2
    # Mumbai (PRESENT) must rank above Delhi (PAST) for present query
    assert pres_results[0]["memory"].value == "Mumbai"


def test_advanced_retrieval_relationship_graph_boost():
    """Verify AdvancedRetrievalEngine uses graph proximity to boost related memories."""
    session, store = create_in_memory_db()

    mem_proj, _, _ = store.save_memory_with_semantics(
        subject="User",
        relation="works_on",
        value="Recallix",
        category="PROJECT",
    )
    mem_skill, _, _ = store.save_memory_with_semantics(
        subject="User",
        relation="knows",
        value="Python",
        category="SKILL",
    )

    graph = MemoryGraph(session=session)
    graph.build_from_memories([mem_proj, mem_skill])
    # Connect Recallix to Python
    graph.add_edge(source="Recallix", relation="uses", target="Python")

    retrieval = AdvancedRetrievalEngine(memory_store=store, memory_graph=graph)

    # Search query mentioning Recallix should elevate connected Python memory
    results = retrieval.search("Recallix technologies and tools", top_k=2)
    values = [r["memory"].value for r in results]
    assert "Recallix" in values
    assert "Python" in values


# =====================================================================
# Full Assistant Intelligence with Phase 10 Integration
# =====================================================================


def test_assistant_engine_temporal_statement_and_query_flow():
    """Verify AssistantEngine stores temporal state and applies feedback in full conversational flow."""
    session, store = create_in_memory_db()

    mock_llm = MagicMock()
    mock_llm.is_available.return_value = True
    mock_llm.generate_with_memories.return_value = "You previously lived in Delhi."
    mock_llm.verify_answer_grounding.return_value = {
        "grounded": True,
        "supported_values": ["Delhi"],
        "grounding_score": 1.0,
        "details": "Grounded to past location.",
    }

    assistant = AssistantEngine(
        memory_store=store,
        llm_engine=mock_llm,
    )

    # Turn 1: Past statement
    res1 = assistant.respond("Previously lived in Delhi.")
    assert res1["input_type"] == InputType.STATEMENT
    assert len(res1["extracted_memories"]) == 1
    assert res1["extracted_memories"][0]["temporal_state"] == "PAST"

    # Turn 2: Present statement
    res2 = assistant.respond("Currently lives in Mumbai.")
    assert res2["input_type"] == InputType.STATEMENT
    assert len(res2["extracted_memories"]) == 1
    assert res2["extracted_memories"][0]["temporal_state"] == "PRESENT"

    # Turn 3: Ask about past
    res3 = assistant.respond("Where did I live before?")
    assert res3["input_type"] == InputType.QUESTION
    assert res3["response"] == "You previously lived in Delhi."

    # Verify reinforcement feedback was applied to Delhi memory
    delhi_mem = [m for m in store.get_all_memories() if m.value == "Delhi"][0]
    assert delhi_mem.access_count >= 1
    assert delhi_mem.helpful_count >= 1
    assert delhi_mem.reinforcement_score > 0.0

    assistant.close()
