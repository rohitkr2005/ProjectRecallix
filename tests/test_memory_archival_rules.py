import uuid
from datetime import datetime, timedelta
import pytest
from app.memory.lifecycle import (
    ArchivalReason,
    DEFAULT_PROTECTED_CATEGORIES,
    evaluate_archival_eligibility,
)
from app.memory.memory_store import MemoryStore


def test_archival_reason_enum_values():
    assert ArchivalReason.EXPLICIT_USER_REQUEST == "EXPLICIT_USER_REQUEST"
    assert ArchivalReason.CONFLICT_SUPERSEDED == "CONFLICT_SUPERSEDED"
    assert ArchivalReason.LOW_IMPORTANCE_DEPRECATED == "LOW_IMPORTANCE_DEPRECATED"
    assert ArchivalReason.POLICY_ARCHIVED == "POLICY_ARCHIVED"


def test_old_is_not_wrong_principle():
    class OldImportantMemory:
        id = 1
        active = True
        subject = "User"
        relation = "studies_at"
        value = "Tribhuvan College"
        category = "EDUCATION"
        importance = 9
        created_at = datetime.utcnow() - timedelta(days=500)

    mem = OldImportantMemory()

    # Even though it's 500 days old, it's not eligible for policy/deprecation archival
    can_archive_policy, reason_policy = evaluate_archival_eligibility(
        mem,
        reason=ArchivalReason.POLICY_ARCHIVED,
    )
    assert can_archive_policy is False
    assert "protected" in reason_policy

    can_archive_deprecate, reason_deprecate = evaluate_archival_eligibility(
        mem,
        reason=ArchivalReason.LOW_IMPORTANCE_DEPRECATED,
    )
    assert can_archive_deprecate is False


def test_high_importance_protection():
    store = MemoryStore()
    user_id = f"ImportanceProtectUser_{uuid.uuid4()}"

    mem, _ = store.save_memory(
        subject=user_id,
        relation="works_on",
        value="Project Recallix",
        category="PROJECT",
        importance=9,
    )

    # Policy archival should be blocked
    can_arch, reason = store.can_archive(mem.id, reason=ArchivalReason.POLICY_ARCHIVED)
    assert can_arch is False
    assert "protected_high_importance:9" in reason

    # Attempting to archive by policy should fail
    archived = store.archive_memory(mem.id, reason=ArchivalReason.POLICY_ARCHIVED)
    assert archived is False
    assert mem.active is True

    # Explicit user action should succeed
    user_archived = store.archive_memory(mem.id, reason=ArchivalReason.EXPLICIT_USER_REQUEST)
    assert user_archived is True
    assert mem.active is False

    store.close()


def test_protected_category_prevents_policy_archival():
    store = MemoryStore()
    user_id = f"CategoryProtectUser_{uuid.uuid4()}"

    mem, _ = store.save_memory(
        subject=user_id,
        relation="studies",
        value="Data Science",
        category="EDUCATION",
        importance=5,  # Moderate importance, but category is protected
    )

    can_arch, reason = store.can_archive(mem.id, reason=ArchivalReason.POLICY_ARCHIVED)
    assert can_arch is False
    assert "protected_category:EDUCATION" in reason

    # Cannot be archived by policy
    archived = store.archive_memory(mem.id, reason=ArchivalReason.POLICY_ARCHIVED)
    assert archived is False
    assert mem.active is True

    store.close()


def test_low_importance_candidate_archival():
    store = MemoryStore()
    user_id = f"LowImpUser_{uuid.uuid4()}"

    m_low, _ = store.save_memory(
        subject=user_id,
        relation="likes",
        value="TemporaryTopic",
        category="PREFERENCE",
        importance=2,
    )
    m_high, _ = store.save_memory(
        subject=user_id,
        relation="likes",
        value="Python",
        category="PREFERENCE",
        importance=9,
    )

    candidates = store.get_archival_candidates(subject=user_id, max_importance=3)
    candidate_ids = {m.id for m in candidates}

    assert m_low.id in candidate_ids
    assert m_high.id not in candidate_ids

    # Low importance memory can be archived under deprecation
    can_arch, reason = store.can_archive(m_low.id, reason=ArchivalReason.LOW_IMPORTANCE_DEPRECATED)
    assert can_arch is True
    assert reason == "eligible_low_importance"

    archived = store.archive_memory(m_low.id, reason=ArchivalReason.LOW_IMPORTANCE_DEPRECATED)
    assert archived is True
    assert m_low.active is False

    store.close()


def test_already_inactive_cannot_be_archived():
    class InactiveMemory:
        active = False
        subject = "User"
        relation = "likes"
        value = "OldTopic"
        category = "PREFERENCE"
        importance = 3

    can_arch, reason = evaluate_archival_eligibility(InactiveMemory())
    assert can_arch is False
    assert reason == "already_inactive"
