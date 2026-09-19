import uuid
import pytest
from app.memory.lifecycle import (
    MemoryLifecycleState,
    UpdateSemanticsDecision,
    evaluate_update_semantics,
)
from app.memory.memory_store import MemoryStore


def test_update_semantics_decision_enum_values():
    assert UpdateSemanticsDecision.DUPLICATE == "DUPLICATE"
    assert UpdateSemanticsDecision.CONFLICT_SUPERSEDE == "CONFLICT_SUPERSEDE"
    assert UpdateSemanticsDecision.NEW_MEMORY == "NEW_MEMORY"


def test_evaluate_update_semantics_new_memory_empty():
    decision, matched, reason = evaluate_update_semantics(
        subject="User",
        relation="likes",
        value="Python",
        active_memories=[],
        is_single_value_fn=lambda r: False,
    )
    assert decision == UpdateSemanticsDecision.NEW_MEMORY
    assert matched is None
    assert reason == "no_active_memories"


def test_evaluate_update_semantics_duplicate():
    class DummyMemory:
        id = 101
        active = True
        subject = "User"
        relation = "likes"
        value = "Python"

    mem = DummyMemory()
    decision, matched, reason = evaluate_update_semantics(
        subject="User",
        relation="likes",
        value="Python",
        active_memories=[mem],
        is_single_value_fn=lambda r: False,
    )
    assert decision == UpdateSemanticsDecision.DUPLICATE
    assert matched.id == 101
    assert "exact_duplicate_of:101" in reason


def test_evaluate_update_semantics_conflict_supersede():
    class DummyMemory:
        id = 102
        active = True
        subject = "User"
        relation = "lives_in"
        value = "Delhi"

    mem = DummyMemory()
    decision, matched, reason = evaluate_update_semantics(
        subject="User",
        relation="lives_in",
        value="Mumbai",
        active_memories=[mem],
        is_single_value_fn=lambda r: r == "lives_in",
    )
    assert decision == UpdateSemanticsDecision.CONFLICT_SUPERSEDE
    assert matched.id == 102
    assert "conflicts_with:102" in reason


def test_evaluate_update_semantics_multi_value_allows_new_memory():
    class DummyMemory:
        id = 103
        active = True
        subject = "User"
        relation = "likes"
        value = "Python"

    mem = DummyMemory()
    decision, matched, reason = evaluate_update_semantics(
        subject="User",
        relation="likes",
        value="SQL",
        active_memories=[mem],
        is_single_value_fn=lambda r: False,
    )
    assert decision == UpdateSemanticsDecision.NEW_MEMORY
    assert matched is None
    assert reason == "new_memory"


def test_duplicate_updates_importance_when_higher():
    store = MemoryStore()
    user_id = f"DuplicateImpUser_{uuid.uuid4()}"

    m1, status1 = store.save_memory(
        subject=user_id,
        relation="likes",
        value="Python",
        category="PREFERENCE",
        importance=5,
    )
    assert status1 == "created"
    assert m1.importance == 5

    m2, status2 = store.save_memory(
        subject=user_id,
        relation="likes",
        value="Python",
        category="PREFERENCE",
        importance=8,
    )
    assert status2 == "duplicate"
    assert m2.id == m1.id
    assert m2.importance == 8

    store.close()


def test_save_memory_with_semantics_metadata():
    store = MemoryStore()
    user_id = f"SemanticsUser_{uuid.uuid4()}"

    # 1. New memory
    m1, status1, meta1 = store.save_memory_with_semantics(
        subject=user_id,
        relation="lives_in",
        value="Delhi",
        category="LOCATION",
        importance=7,
    )
    assert status1 == "created"
    assert meta1["decision"] == UpdateSemanticsDecision.NEW_MEMORY
    assert meta1["superseded_memory_id"] is None

    # 2. Conflict supersession
    m2, status2, meta2 = store.save_memory_with_semantics(
        subject=user_id,
        relation="lives_in",
        value="Mumbai",
        category="LOCATION",
        importance=9,
    )
    assert status2 == "created"
    assert meta2["decision"] == UpdateSemanticsDecision.CONFLICT_SUPERSEDE
    assert meta2["superseded_memory_id"] == m1.id
    assert store.get_lifecycle_state(m1.id) == MemoryLifecycleState.SUPERSEDED
    assert store.get_lifecycle_state(m2.id) == MemoryLifecycleState.ACTIVE

    # 3. Duplicate
    m3, status3, meta3 = store.save_memory_with_semantics(
        subject=user_id,
        relation="lives_in",
        value="Mumbai",
        category="LOCATION",
        importance=9,
    )
    assert status3 == "duplicate"
    assert meta3["decision"] == UpdateSemanticsDecision.DUPLICATE
    assert meta3["matched_memory_id"] == m2.id

    store.close()


def test_update_memory_in_place_success():
    store = MemoryStore()
    user_id = f"InPlaceUser_{uuid.uuid4()}"

    mem, _ = store.save_memory(
        subject=user_id,
        relation="current_role",
        value="Engineer",
        category="WORK",
        importance=6,
    )

    updated, status = store.update_memory(
        memory_id=mem.id,
        value="Senior Engineer",
        importance=9,
        category="CAREER",
    )
    assert status == "updated"
    assert updated.id == mem.id
    assert updated.value == "Senior Engineer"
    assert updated.importance == 9
    assert updated.category == "CAREER"

    store.close()


def test_update_memory_rejects_single_value_conflict():
    store = MemoryStore()
    user_id = f"InPlaceConflictUser_{uuid.uuid4()}"

    m1, _ = store.save_memory(
        subject=user_id,
        relation="current_city",
        value="Delhi",
        category="LOCATION",
    )
    # Deactivate m1 so we can save m2 with different value
    store.deactivate_memory(m1.id)
    m2, _ = store.save_memory(
        subject=user_id,
        relation="current_city",
        value="Mumbai",
        category="LOCATION",
    )
    # Reactivate m1
    store.restore_memory(m1.id, allow_conflict=True)

    # Now if we try to in-place update m1's value to Mumbai, it should detect conflict
    result, error = store.update_memory(
        memory_id=m1.id,
        value="Mumbai",
    )
    assert result is None
    assert "single_value_conflict" in error

    store.close()


def test_update_memory_not_found():
    store = MemoryStore()
    result, error = store.update_memory(memory_id=9999999, value="Test")
    assert result is None
    assert error == "memory_not_found"
    store.close()
