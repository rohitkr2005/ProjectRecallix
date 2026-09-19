import uuid
from datetime import datetime, timedelta
import pytest
from app.memory.lifecycle import (
    build_conflict_lineage,
    consolidate_memories,
)
from app.memory.memory_store import MemoryStore


def test_consolidate_memories_structure_and_lineage():
    store = MemoryStore()
    user_id = f"ConsolidateUser_{uuid.uuid4()}"

    m1, _ = store.save_memory(
        subject=user_id,
        relation="knows",
        value="Python",
        category="SKILL",
        importance=8,
    )
    m2, _ = store.save_memory(
        subject=user_id,
        relation="knows",
        value="SQL",
        category="SKILL",
        importance=7,
    )
    m3, _ = store.save_memory(
        subject=user_id,
        relation="works_on",
        value="Recallix",
        category="PROJECT",
        importance=9,
    )

    profile = store.consolidate_memories(subject=user_id)

    assert profile["total_active_memories"] == 3
    assert user_id in profile["subjects"]
    user_profile = profile["subjects"][user_id]

    assert set(user_profile["memory_ids"]) == {m1.id, m2.id, m3.id}
    assert "SKILL" in user_profile["categories"]
    assert "PROJECT" in user_profile["categories"]

    skill_values = {item["value"] for item in user_profile["categories"]["SKILL"]}
    assert skill_values == {"Python", "SQL"}

    store.close()


def test_cleanup_duplicates():
    store = MemoryStore()
    user_id = f"CleanupUser_{uuid.uuid4()}"

    # Force create 3 identical active memories for (user_id, "likes", "Python")
    m1, _ = store.save_memory(
        subject=user_id,
        relation="likes",
        value="Python",
        category="PREFERENCE",
        importance=4,
    )
    # Deactivate m1 temporarily to create m2, then m3
    store.deactivate_memory(m1.id)
    m2, _ = store.save_memory(
        subject=user_id,
        relation="likes",
        value="Python",
        category="PREFERENCE",
        importance=8,
    )
    store.deactivate_memory(m2.id)
    m3, _ = store.save_memory(
        subject=user_id,
        relation="likes",
        value="Python",
        category="PREFERENCE",
        importance=6,
    )

    # Force all 3 active
    store.restore_memory(m1.id, allow_conflict=True)
    store.restore_memory(m2.id, allow_conflict=True)
    assert m1.active is True
    assert m2.active is True
    assert m3.active is True

    # 1. Dry run
    dry_result = store.cleanup_duplicates(subject=user_id, dry_run=True)
    assert dry_result["cleaned_groups"] == 1
    assert dry_result["duplicates_deactivated"] == 2
    # Still all active
    assert m1.active is True
    assert m2.active is True
    assert m3.active is True

    # 2. Execution run
    run_result = store.cleanup_duplicates(subject=user_id, dry_run=False)
    assert run_result["cleaned_groups"] == 1
    assert run_result["duplicates_deactivated"] == 2

    active_mems = [
        m for m in store.get_all_memories()
        if m.subject == user_id and m.relation == "likes"
    ]
    assert len(active_mems) == 1
    canonical = active_mems[0]
    # Canonical should have max importance (8)
    assert canonical.importance == 8

    store.close()


def test_get_conflict_history_and_lineage():
    store = MemoryStore()
    user_id = f"ConflictHistoryUser_{uuid.uuid4()}"

    m1, _ = store.save_memory(
        subject=user_id,
        relation="lives_in",
        value="Delhi",
        category="LOCATION",
    )
    m2, _ = store.save_memory(
        subject=user_id,
        relation="lives_in",
        value="Mumbai",
        category="LOCATION",
    )
    m3, _ = store.save_memory(
        subject=user_id,
        relation="lives_in",
        value="Bangalore",
        category="LOCATION",
    )

    history = store.get_conflict_history(subject=user_id, relation="lives_in")
    assert len(history) == 3

    assert history[0]["value"] == "Delhi"
    assert history[0]["state"] == "SUPERSEDED"
    assert history[0]["superseded_by"] == "Mumbai"

    assert history[1]["value"] == "Mumbai"
    assert history[1]["state"] == "SUPERSEDED"
    assert history[1]["superseded_by"] == "Bangalore"

    assert history[2]["value"] == "Bangalore"
    assert history[2]["state"] == "ACTIVE"
    assert history[2]["superseded_by"] is None

    store.close()
