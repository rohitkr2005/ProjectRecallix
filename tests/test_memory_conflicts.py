import uuid

from app.memory.memory_store import (
    MemoryStore,
    SINGLE_VALUE_RELATIONS
)


def test_single_value_relations_are_defined():
    expected_relations = {
        "lives_in",
        "works_at",
        "studies_at",
        "current_role",
        "current_city",
    }

    assert SINGLE_VALUE_RELATIONS == expected_relations


def test_single_value_relation_is_detected():
    store = MemoryStore()

    assert store.is_single_value_relation("lives_in") is True
    assert store.is_single_value_relation("works_at") is True
    assert store.is_single_value_relation("studies_at") is True
    assert store.is_single_value_relation("current_role") is True
    assert store.is_single_value_relation("current_city") is True

    store.close()


def test_multi_value_relation_is_not_single_value():
    store = MemoryStore()

    assert store.is_single_value_relation("likes") is False
    assert store.is_single_value_relation("knows") is False
    assert store.is_single_value_relation("works_on") is False
    assert store.is_single_value_relation("wants_to_learn") is False

    store.close()


def test_conflicting_memory_is_detected():
    store = MemoryStore()

    store.save_memory(
        subject="ConflictTestUser",
        relation="lives_in",
        value="Jaipur",
        category="LOCATION"
    )

    conflict = store._find_conflicting_memory(
        subject="ConflictTestUser",
        relation="lives_in",
        value="Delhi"
    )

    assert conflict is not None
    assert conflict.value == "Jaipur"
    assert conflict.active is True

    store.close()


def test_same_value_is_not_a_conflict():
    store = MemoryStore()

    store.save_memory(
        subject="SameValueTestUser",
        relation="lives_in",
        value="Jaipur",
        category="LOCATION"
    )

    conflict = store._find_conflicting_memory(
        subject="SameValueTestUser",
        relation="lives_in",
        value="Jaipur"
    )

    assert conflict is None

    store.close()


def test_multi_value_relationship_has_no_conflict():
    store = MemoryStore()

    store.save_memory(
        subject="MultiValueTestUser",
        relation="likes",
        value="Python",
        category="PREFERENCE"
    )

    conflict = store._find_conflicting_memory(
        subject="MultiValueTestUser",
        relation="likes",
        value="SQL"
    )

    assert conflict is None

    store.close()


def test_conflicting_memory_is_deactivated_when_new_value_is_saved():
    store = MemoryStore()

    subject = f"DeactivateTestUser_{uuid.uuid4()}"

    old_memory, old_status = store.save_memory(
        subject=subject,
        relation="lives_in",
        value="Jaipur",
        category="LOCATION"
    )

    assert old_status == "created"
    assert old_memory.active is True

    new_memory, new_status = store.save_memory(
        subject=subject,
        relation="lives_in",
        value="Delhi",
        category="LOCATION"
    )

    assert new_status == "created"
    assert new_memory.active is True

    archived = store.get_archived_memories()

    old_memory_from_db = next(
        memory
        for memory in archived
        if memory.id == old_memory.id
    )

    assert old_memory_from_db.value == "Jaipur"
    assert old_memory_from_db.active is False

    store.close()
    
def test_old_memory_is_preserved_as_history():
    store = MemoryStore()

    subject = f"HistoryTestUser_{uuid.uuid4()}"

    old_memory, old_status = store.save_memory(
        subject=subject,
        relation="lives_in",
        value="Jaipur",
        category="LOCATION"
    )

    assert old_status == "created"

    new_memory, new_status = store.save_memory(
        subject=subject,
        relation="lives_in",
        value="Delhi",
        category="LOCATION"
    )

    assert new_status == "created"

    archived = store.get_archived_memories()

    old_history = next(
        memory
        for memory in archived
        if memory.id == old_memory.id
    )

    assert old_history.value == "Jaipur"
    assert old_history.relation == "lives_in"
    assert old_history.subject == subject
    assert old_history.active is False

    assert new_memory.id != old_memory.id
    assert new_memory.value == "Delhi"
    assert new_memory.active is True

    store.close()
    
def test_retrieval_prefers_active_memory():
    store = MemoryStore()

    subject = f"RetrievalConflictUser_{uuid.uuid4()}"

    embedding = bytes([1] * 384)

    old_memory, _ = store.save_memory(
        subject=subject,
        relation="lives_in",
        value="Jaipur",
        category="LOCATION",
        embedding=embedding
    )

    new_memory, _ = store.save_memory(
        subject=subject,
        relation="lives_in",
        value="Delhi",
        category="LOCATION",
        embedding=embedding
    )

    assert old_memory.active is False
    assert new_memory.active is True

    active_memories = store.get_all_memories()

    active_ids = {
        memory.id
        for memory in active_memories
    }

    assert new_memory.id in active_ids
    assert old_memory.id not in active_ids

    store.close()
    
def test_multiple_updates_preserve_history():
    store = MemoryStore()

    subject = f"MultipleUpdateUser_{uuid.uuid4()}"

    first, _ = store.save_memory(
        subject=subject,
        relation="lives_in",
        value="Jaipur",
        category="LOCATION"
    )

    second, _ = store.save_memory(
        subject=subject,
        relation="lives_in",
        value="Delhi",
        category="LOCATION"
    )

    third, _ = store.save_memory(
        subject=subject,
        relation="lives_in",
        value="Mumbai",
        category="LOCATION"
    )

    assert first.active is False
    assert second.active is False
    assert third.active is True

    archived = store.get_archived_memories()

    history_values = {
        memory.value
        for memory in archived
        if memory.subject == subject
        and memory.relation == "lives_in"
    }

    assert "Jaipur" in history_values
    assert "Delhi" in history_values

    active_memories = [
        memory
        for memory in store.get_all_memories()
        if memory.subject == subject
        and memory.relation == "lives_in"
    ]

    assert len(active_memories) == 1
    assert active_memories[0].value == "Mumbai"

    store.close()


def test_duplicate_does_not_create_conflict():
    store = MemoryStore()

    subject = f"DuplicateConflictUser_{uuid.uuid4()}"

    first, first_status = store.save_memory(
        subject=subject,
        relation="lives_in",
        value="Jaipur",
        category="LOCATION"
    )

    second, second_status = store.save_memory(
        subject=subject,
        relation="lives_in",
        value="Jaipur",
        category="LOCATION"
    )

    assert first_status == "created"
    assert second_status == "duplicate"

    assert first.id == second.id
    assert first.active is True

    memories = [
        memory
        for memory in store.get_all_memories()
        if memory.subject == subject
    ]

    assert len(memories) == 1

    store.close()


def test_multi_value_relation_allows_multiple_values():
    store = MemoryStore()

    subject = f"MultiValueConflictUser_{uuid.uuid4()}"

    first, first_status = store.save_memory(
        subject=subject,
        relation="likes",
        value="Python",
        category="PREFERENCE"
    )

    second, second_status = store.save_memory(
        subject=subject,
        relation="likes",
        value="SQL",
        category="PREFERENCE"
    )

    assert first_status == "created"
    assert second_status == "created"

    assert first.active is True
    assert second.active is True

    memories = [
        memory
        for memory in store.get_all_memories()
        if memory.subject == subject
        and memory.relation == "likes"
    ]

    assert len(memories) == 2

    values = {memory.value for memory in memories}

    assert values == {"Python", "SQL"}

    store.close()