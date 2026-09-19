import uuid
import pytest
from app.memory.lifecycle import (
    MemoryLifecycleState,
    determine_lifecycle_state,
    is_active,
    is_archived,
    is_superseded,
    validate_restoration_safety,
)
from app.memory.memory_store import MemoryStore
from app.retrieval.retrieval_engine import RetrievalEngine


def test_lifecycle_state_enum_values():
    assert MemoryLifecycleState.ACTIVE == "ACTIVE"
    assert MemoryLifecycleState.ARCHIVED == "ARCHIVED"
    assert MemoryLifecycleState.SUPERSEDED == "SUPERSEDED"


def test_determine_lifecycle_state_active():
    class DummyMemory:
        active = True
        subject = "User"
        relation = "lives_in"
        value = "Mumbai"

    mem = DummyMemory()
    assert determine_lifecycle_state(mem) == MemoryLifecycleState.ACTIVE
    assert is_active(mem) is True
    assert is_archived(mem) is False
    assert is_superseded(mem) is False


def test_determine_lifecycle_state_archived_when_no_active_supersession():
    class DummyMemory:
        active = False
        subject = "User"
        relation = "likes"
        value = "Python"

    mem = DummyMemory()
    # likes is a multi-value relation, deactivating it makes it ARCHIVED
    assert determine_lifecycle_state(mem, is_single_value_fn=lambda r: r == "lives_in") == MemoryLifecycleState.ARCHIVED
    assert is_archived(mem, is_single_value_fn=lambda r: r == "lives_in") is True
    assert is_superseded(mem, is_single_value_fn=lambda r: r == "lives_in") is False


def test_determine_lifecycle_state_superseded_when_active_replacement_exists():
    class InactiveMemory:
        active = False
        subject = "User"
        relation = "lives_in"
        value = "Delhi"

    class ActiveMemory:
        active = True
        subject = "User"
        relation = "lives_in"
        value = "Mumbai"

    old_mem = InactiveMemory()
    active_mems = [ActiveMemory()]

    is_single = lambda r: r == "lives_in"

    assert determine_lifecycle_state(old_mem, active_memories=active_mems, is_single_value_fn=is_single) == MemoryLifecycleState.SUPERSEDED
    assert is_superseded(old_mem, active_memories=active_mems, is_single_value_fn=is_single) is True
    assert is_archived(old_mem, active_memories=active_mems, is_single_value_fn=is_single) is False


def test_memory_store_lifecycle_tracking():
    store = MemoryStore()
    user_id = f"LifecycleUser_{uuid.uuid4()}"

    # 1. Create first location memory -> ACTIVE
    m1, status1 = store.save_memory(
        subject=user_id,
        relation="lives_in",
        value="Delhi",
        category="LOCATION",
        importance=8,
    )
    assert status1 == "created"
    assert store.get_lifecycle_state(m1.id) == MemoryLifecycleState.ACTIVE

    # 2. Update location to Mumbai -> m1 becomes SUPERSEDED, m2 becomes ACTIVE
    m2, status2 = store.save_memory(
        subject=user_id,
        relation="lives_in",
        value="Mumbai",
        category="LOCATION",
        importance=9,
    )
    assert status2 == "created"
    assert store.get_lifecycle_state(m2.id) == MemoryLifecycleState.ACTIVE
    assert store.get_lifecycle_state(m1.id) == MemoryLifecycleState.SUPERSEDED

    # 3. Create a preference memory and explicitly archive it -> ARCHIVED
    m3, status3 = store.save_memory(
        subject=user_id,
        relation="likes",
        value="Tennis",
        category="PREFERENCE",
        importance=6,
    )
    assert status3 == "created"
    assert store.get_lifecycle_state(m3.id) == MemoryLifecycleState.ACTIVE

    archive_result = store.archive_memory(m3.id)
    assert archive_result is True
    assert store.get_lifecycle_state(m3.id) == MemoryLifecycleState.ARCHIVED

    store.close()


def test_restoration_safety_prevents_single_value_conflict():
    store = MemoryStore()
    user_id = f"RestoreSafetyUser_{uuid.uuid4()}"

    # Create Delhi, then Mumbai
    m_delhi, _ = store.save_memory(
        subject=user_id,
        relation="lives_in",
        value="Delhi",
        category="LOCATION",
    )
    m_mumbai, _ = store.save_memory(
        subject=user_id,
        relation="lives_in",
        value="Mumbai",
        category="LOCATION",
    )

    assert store.get_lifecycle_state(m_delhi.id) == MemoryLifecycleState.SUPERSEDED
    assert store.get_lifecycle_state(m_mumbai.id) == MemoryLifecycleState.ACTIVE

    # Attempting to restore m_delhi while m_mumbai is active must be rejected
    can_restore = store.restore_memory(m_delhi.id, allow_conflict=False)
    assert can_restore is False
    assert store.get_lifecycle_state(m_delhi.id) == MemoryLifecycleState.SUPERSEDED

    # If we deactivate Mumbai, then restoring Delhi must succeed
    store.deactivate_memory(m_mumbai.id)
    can_restore_now = store.restore_memory(m_delhi.id, allow_conflict=False)
    assert can_restore_now is True
    assert store.get_lifecycle_state(m_delhi.id) == MemoryLifecycleState.ACTIVE

    store.close()


def test_get_superseded_memories():
    store = MemoryStore()
    user_id = f"SupersededQueryUser_{uuid.uuid4()}"

    m1, _ = store.save_memory(
        subject=user_id,
        relation="works_at",
        value="CompanyA",
        category="WORK",
    )
    m2, _ = store.save_memory(
        subject=user_id,
        relation="works_at",
        value="CompanyB",
        category="WORK",
    )

    superseded = store.get_superseded_memories()
    superseded_ids = {m.id for m in superseded}

    assert m1.id in superseded_ids
    assert m2.id not in superseded_ids

    store.close()


def test_get_lifecycle_history():
    store = MemoryStore()
    user_id = f"HistoryQueryUser_{uuid.uuid4()}"

    store.save_memory(subject=user_id, relation="current_city", value="Pune", category="LOCATION")
    store.save_memory(subject=user_id, relation="current_city", value="Bangalore", category="LOCATION")

    history = store.get_lifecycle_history(subject=user_id, relation="current_city")
    assert len(history) == 2

    assert history[0]["memory"].value == "Pune"
    assert history[0]["lifecycle_state"] == MemoryLifecycleState.SUPERSEDED

    assert history[1]["memory"].value == "Bangalore"
    assert history[1]["lifecycle_state"] == MemoryLifecycleState.ACTIVE

    store.close()


def test_retrieval_excludes_archived_and_superseded_memories():
    store = MemoryStore()
    user_id = f"RetrievalExclusionUser_{uuid.uuid4()}"

    # Create dummy embedding
    fake_embedding = bytes([1] * 384)

    # 1. Superseded memory
    m_old, _ = store.save_memory(
        subject=user_id,
        relation="lives_in",
        value="OldCity",
        category="LOCATION",
        embedding=fake_embedding,
    )
    # 2. Active memory
    m_new, _ = store.save_memory(
        subject=user_id,
        relation="lives_in",
        value="NewCity",
        category="LOCATION",
        embedding=fake_embedding,
    )
    # 3. Explicitly archived memory
    m_archived, _ = store.save_memory(
        subject=user_id,
        relation="likes",
        value="ArchivedTopic",
        category="PREFERENCE",
        embedding=fake_embedding,
    )
    store.archive_memory(m_archived.id)

    engine = RetrievalEngine(memory_store=store)
    active_mems = store.get_all_memories()
    active_ids = {m.id for m in active_mems}

    assert m_new.id in active_ids
    assert m_old.id not in active_ids
    assert m_archived.id not in active_ids

    engine.close()
