import time
import uuid
from datetime import datetime
import pytest
from app.memory.lifecycle import (
    MemoryLifecycleState,
    RestorationStrategy,
    validate_restoration_safety,
)
from app.memory.memory_store import MemoryStore


def test_restoration_strategy_enum_values():
    assert RestorationStrategy.REJECT == "REJECT"
    assert RestorationStrategy.SUPERSEDE_ACTIVE == "SUPERSEDE_ACTIVE"


def test_restore_archived_multi_value_memory():
    store = MemoryStore()
    user_id = f"RestoreMultiUser_{uuid.uuid4()}"

    mem, _ = store.save_memory(
        subject=user_id,
        relation="likes",
        value="Python",
        category="PREFERENCE",
    )
    store.archive_memory(mem.id)
    assert mem.active is False

    time.sleep(0.01)
    original_updated_at = mem.updated_at

    # Check can_restore
    can_res, reason = store.can_restore(mem.id)
    assert can_res is True
    assert reason == "safe_to_restore"

    # Restore
    restored = store.restore_memory(mem.id)
    assert restored is True
    assert mem.active is True
    assert store.get_lifecycle_state(mem.id) == MemoryLifecycleState.ACTIVE
    assert mem.updated_at >= original_updated_at

    store.close()


def test_restore_rejects_single_value_conflict_by_default():
    store = MemoryStore()
    user_id = f"RestoreConflictUser_{uuid.uuid4()}"

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

    # Check can_restore with default REJECT
    can_res, reason = store.can_restore(m_delhi.id, strategy=RestorationStrategy.REJECT)
    assert can_res is False
    assert "conflict_with_active_memory" in reason

    # Attempt restore with default strategy -> fails
    restored = store.restore_memory(m_delhi.id, strategy=RestorationStrategy.REJECT)
    assert restored is False
    assert store.get_lifecycle_state(m_delhi.id) == MemoryLifecycleState.SUPERSEDED
    assert store.get_lifecycle_state(m_mumbai.id) == MemoryLifecycleState.ACTIVE

    store.close()


def test_restore_supersede_active_strategy():
    store = MemoryStore()
    user_id = f"RestoreSupersedeUser_{uuid.uuid4()}"

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

    # Check can_restore with SUPERSEDE_ACTIVE
    can_res, reason = store.can_restore(m_delhi.id, strategy=RestorationStrategy.SUPERSEDE_ACTIVE)
    assert can_res is True
    assert "supersede_active_conflict" in reason

    # Restore with SUPERSEDE_ACTIVE
    success, status, details = store.restore_memory_with_details(
        m_delhi.id,
        strategy=RestorationStrategy.SUPERSEDE_ACTIVE,
    )
    assert success is True
    assert status == "restored"
    assert details["restored_memory_id"] == m_delhi.id
    assert details["superseded_memory_id"] == m_mumbai.id

    # Verify states inverted
    assert store.get_lifecycle_state(m_delhi.id) == MemoryLifecycleState.ACTIVE
    assert store.get_lifecycle_state(m_mumbai.id) == MemoryLifecycleState.SUPERSEDED

    store.close()


def test_restore_rejects_active_duplicate():
    store = MemoryStore()
    user_id = f"RestoreDuplicateUser_{uuid.uuid4()}"

    m1, _ = store.save_memory(
        subject=user_id,
        relation="likes",
        value="Rust",
        category="PREFERENCE",
    )
    # Deactivate m1, create m2 with same value, then try to restore m1
    store.deactivate_memory(m1.id)
    m2, _ = store.save_memory(
        subject=user_id,
        relation="likes",
        value="Rust",
        category="PREFERENCE",
    )
    assert m2.active is True
    assert m1.active is False

    can_res, reason = store.can_restore(m1.id)
    assert can_res is False
    assert f"duplicate_of_active_memory:{m2.id}" in reason

    restored = store.restore_memory(m1.id)
    assert restored is False
    assert m1.active is False

    store.close()


def test_restore_already_active_is_rejected():
    store = MemoryStore()
    user_id = f"RestoreActiveUser_{uuid.uuid4()}"

    mem, _ = store.save_memory(
        subject=user_id,
        relation="knows",
        value="SQL",
        category="SKILL",
    )
    assert mem.active is True

    can_res, reason = store.can_restore(mem.id)
    assert can_res is False
    assert reason == "already_active"

    restored = store.restore_memory(mem.id)
    assert restored is False

    store.close()
