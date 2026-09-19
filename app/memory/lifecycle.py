from datetime import datetime
from enum import Enum
import math
from typing import Callable, Iterable, Optional, Tuple



class MemoryLifecycleState(str, Enum):
    """
    Formal lifecycle states for Recallix memories.
    
    States:
        ACTIVE: Operative, valid memory actively used in retrieval and reasoning.
        ARCHIVED: Inactive memory archived or deactivated intentionally.
        SUPERSEDED: Inactive memory replaced by a newer conflicting value for a single-value relation.
    """
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    SUPERSEDED = "SUPERSEDED"


def determine_lifecycle_state(
    memory,
    active_memories: Optional[Iterable] = None,
    is_single_value_fn: Optional[Callable[[str], bool]] = None,
) -> MemoryLifecycleState:
    """
    Determine the lifecycle state of a memory without requiring schema changes.
    
    Args:
        memory: The Memory object or namespace to inspect.
        active_memories: Optional collection of currently active memories to detect supersession.
        is_single_value_fn: Optional function to check if a relation is single-value.
        
    Returns:
        MemoryLifecycleState (ACTIVE, ARCHIVED, or SUPERSEDED).
    """
    if getattr(memory, "active", False):
        return MemoryLifecycleState.ACTIVE

    # If inactive, determine if it was superseded by a newer active memory
    if is_single_value_fn and is_single_value_fn(getattr(memory, "relation", "")):
        if active_memories:
            for active_mem in active_memories:
                if (
                    getattr(active_mem, "active", False)
                    and getattr(active_mem, "subject", None) == getattr(memory, "subject", None)
                    and getattr(active_mem, "relation", None) == getattr(memory, "relation", None)
                    and getattr(active_mem, "value", None) != getattr(memory, "value", None)
                ):
                    return MemoryLifecycleState.SUPERSEDED

    return MemoryLifecycleState.ARCHIVED


def is_active(memory) -> bool:
    """Check if the memory is in ACTIVE state."""
    return bool(getattr(memory, "active", False))


def is_superseded(
    memory,
    active_memories: Optional[Iterable] = None,
    is_single_value_fn: Optional[Callable[[str], bool]] = None,
) -> bool:
    """Check if the memory is in SUPERSEDED state."""
    return determine_lifecycle_state(memory, active_memories, is_single_value_fn) == MemoryLifecycleState.SUPERSEDED


def is_archived(
    memory,
    active_memories: Optional[Iterable] = None,
    is_single_value_fn: Optional[Callable[[str], bool]] = None,
) -> bool:
    """Check if the memory is in ARCHIVED state."""
    return determine_lifecycle_state(memory, active_memories, is_single_value_fn) == MemoryLifecycleState.ARCHIVED


class RestorationStrategy(str, Enum):
    """
    Strategies for restoring an inactive memory.
    
    REJECT: Safely rejects restoration if an active conflicting memory exists.
    SUPERSEDE_ACTIVE: Deactivates the conflicting active memory and restores the target memory.
    FORCE: Bypasses conflict checks and restores the memory directly.
    """
    REJECT = "REJECT"
    SUPERSEDE_ACTIVE = "SUPERSEDE_ACTIVE"
    FORCE = "FORCE"


def validate_restoration_safety(
    memory,
    active_memories: Optional[Iterable] = None,
    is_single_value_fn: Optional[Callable[[str], bool]] = None,
    strategy: RestorationStrategy = RestorationStrategy.REJECT,
) -> Tuple[bool, str, Optional[any]]:
    """
    Validate whether an inactive memory can safely be restored without violating
    single-value constraints or active duplicate constraints.
    
    Returns:
        (can_restore: bool, reason: str, conflicting_memory: Optional[Memory])
    """
    if getattr(memory, "active", False):
        return False, "already_active", None

    if strategy == RestorationStrategy.FORCE:
        return True, "forced_restoration", None

    if active_memories:
        for active_mem in active_memories:
            if not getattr(active_mem, "active", False):
                continue
            if (
                getattr(active_mem, "subject", None) == getattr(memory, "subject", None)
                and getattr(active_mem, "relation", None) == getattr(memory, "relation", None)
            ):
                # Check for active duplicate
                if getattr(active_mem, "value", None) == getattr(memory, "value", None):
                    return False, f"duplicate_of_active_memory:{getattr(active_mem, 'id', None)}", active_mem

                # Check for single-value conflict
                if is_single_value_fn and is_single_value_fn(getattr(memory, "relation", "")):
                    if strategy == RestorationStrategy.SUPERSEDE_ACTIVE:
                        return True, f"supersede_active_conflict:{getattr(active_mem, 'id', None)}", active_mem
                    return False, f"conflict_with_active_memory:{getattr(active_mem, 'id', None)}:{getattr(active_mem, 'value', None)}", active_mem

    return True, "safe_to_restore", None




class UpdateSemanticsDecision(str, Enum):
    """
    Decisions when new information arrives in Recallix.
    
    DUPLICATE: The identical memory already exists as active.
    CONFLICT_SUPERSEDE: An active single-value memory exists with a different value.
    NEW_MEMORY: A new memory that does not conflict or duplicate any active memory.
    """
    DUPLICATE = "DUPLICATE"
    CONFLICT_SUPERSEDE = "CONFLICT_SUPERSEDE"
    NEW_MEMORY = "NEW_MEMORY"


def evaluate_update_semantics(
    subject: str,
    relation: str,
    value: str,
    active_memories: Optional[Iterable] = None,
    is_single_value_fn: Optional[Callable[[str], bool]] = None,
) -> Tuple[UpdateSemanticsDecision, Optional[any], str]:
    """
    Determine whether incoming information is a duplicate, a conflict requiring supersession,
    or a brand-new memory.
    
    Returns:
        (decision: UpdateSemanticsDecision, matched_memory: Optional[Memory], reason: str)
    """
    if not active_memories:
        return UpdateSemanticsDecision.NEW_MEMORY, None, "no_active_memories"

    # 1. Check for exact active duplicate
    for mem in active_memories:
        if (
            getattr(mem, "active", False)
            and getattr(mem, "subject", None) == subject
            and getattr(mem, "relation", None) == relation
            and getattr(mem, "value", None) == value
        ):
            return UpdateSemanticsDecision.DUPLICATE, mem, f"exact_duplicate_of:{getattr(mem, 'id', None)}"

    # 2. Check for single-value conflict
    if is_single_value_fn and is_single_value_fn(relation):
        for mem in active_memories:
            if (
                getattr(mem, "active", False)
                and getattr(mem, "subject", None) == subject
                and getattr(mem, "relation", None) == relation
                and getattr(mem, "value", None) != value
            ):
                return UpdateSemanticsDecision.CONFLICT_SUPERSEDE, mem, f"conflicts_with:{getattr(mem, 'id', None)}"

    # 3. New memory (novel fact or multi-value addition)
    return UpdateSemanticsDecision.NEW_MEMORY, None, "new_memory"


class ArchivalReason(str, Enum):
    """
    Reasons for archiving/deactivating a memory in Recallix.
    """
    EXPLICIT_USER_REQUEST = "EXPLICIT_USER_REQUEST"
    CONFLICT_SUPERSEDED = "CONFLICT_SUPERSEDED"
    LOW_IMPORTANCE_DEPRECATED = "LOW_IMPORTANCE_DEPRECATED"
    POLICY_ARCHIVED = "POLICY_ARCHIVED"


DEFAULT_PROTECTED_CATEGORIES = frozenset({"EDUCATION", "IDENTITY", "CORE_PROFILE"})


def evaluate_archival_eligibility(
    memory,
    reason: ArchivalReason = ArchivalReason.EXPLICIT_USER_REQUEST,
    protected_importance: int = 7,
    protected_categories: Optional[Iterable[str]] = None,
    force: bool = False,
) -> Tuple[bool, str]:
    """
    Evaluate whether a memory can be archived according to Recallix rules.
    
    Rule: OLD != WRONG.
    Age alone never justifies archiving a memory.
    Memories with high importance or in protected categories cannot be
    archived by policy/automation unless explicitly forced or requested by the user.
    
    Returns:
        (can_archive: bool, explanation: str)
    """
    if not getattr(memory, "active", False):
        return False, "already_inactive"

    if force or reason in (ArchivalReason.EXPLICIT_USER_REQUEST, ArchivalReason.CONFLICT_SUPERSEDED):
        return True, "allowed_by_explicit_action_or_supersession"

    categories = set(protected_categories) if protected_categories is not None else DEFAULT_PROTECTED_CATEGORIES
    mem_category = getattr(memory, "category", None)
    if mem_category and mem_category.upper() in categories:
        return False, f"protected_category:{mem_category}"

    mem_importance = getattr(memory, "importance", 5)
    if mem_importance is not None and mem_importance >= protected_importance:
        return False, f"protected_high_importance:{mem_importance}"

    if reason == ArchivalReason.LOW_IMPORTANCE_DEPRECATED:
        if mem_importance is not None and mem_importance <= 3:
            return True, "eligible_low_importance"
        return False, f"importance_too_high_for_deprecation:{mem_importance}"

    if reason == ArchivalReason.POLICY_ARCHIVED:
        return True, "eligible_by_policy"

    return False, "requires_explicit_user_action"


def validate_importance(importance: Optional[int]) -> int:
    """
    Validate and clamp memory importance within the formal range [1, 10].
    Defaults to 5 if None or unparseable.
    """
    if importance is None:
        return 5
    try:
        val = int(importance)
    except (TypeError, ValueError):
        return 5
    return max(1, min(10, val))


def calculate_freshness_score(
    memory,
    reference_time: Optional[datetime] = None,
    half_life_days: float = 30.0,
    importance_damping: bool = True,
) -> float:
    """
    Calculate memory freshness using exponential decay.
    
    If importance_damping is True, applies a floor proportional to memory importance
    so that highly important or foundational memories do not decay to insignificance.
    """
    timestamp = getattr(memory, "updated_at", None) or getattr(memory, "created_at", None)
    if timestamp is None:
        return 0.0

    ref = reference_time or datetime.utcnow()
    if timestamp.tzinfo is not None and ref.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=None)

    age_seconds = max(0.0, (ref - timestamp).total_seconds())
    age_days = age_seconds / 86400.0

    decay = math.exp(-math.log(2.0) * age_days / half_life_days)

    if importance_damping:
        importance = validate_importance(getattr(memory, "importance", 5))
        # High importance (e.g. 10) provides a 0.30 floor; low importance provides near zero floor
        importance_floor = ((importance - 1.0) / 9.0) * 0.30
        decay = max(decay, importance_floor)

    return min(1.0, max(0.0, float(decay)))


def consolidate_memories(memories: Iterable) -> dict:
    """
    Consolidate active memories into structured, grounded knowledge groupings
    by subject, category, and relation without synthesizing unsupported facts.
    """
    active = [m for m in memories if getattr(m, "active", False)]
    by_subject = {}

    for m in active:
        s = getattr(m, "subject", "Unknown")
        cat = getattr(m, "category", "GENERAL")
        rel = getattr(m, "relation", "related_to")
        val = getattr(m, "value", "")
        m_id = getattr(m, "id", None)
        imp = getattr(m, "importance", 5)

        if s not in by_subject:
            by_subject[s] = {
                "subject": s,
                "categories": {},
                "relations": {},
                "memory_ids": [],
            }
        by_subject[s]["memory_ids"].append(m_id)

        if cat not in by_subject[s]["categories"]:
            by_subject[s]["categories"][cat] = []
        by_subject[s]["categories"][cat].append({
            "relation": rel,
            "value": val,
            "importance": imp,
            "memory_id": m_id,
        })

        if rel not in by_subject[s]["relations"]:
            by_subject[s]["relations"][rel] = []
        by_subject[s]["relations"][rel].append({
            "value": val,
            "category": cat,
            "importance": imp,
            "memory_id": m_id,
        })

    return {
        "total_active_memories": len(active),
        "subjects": by_subject,
    }


def build_conflict_lineage(memories: Iterable, relation: str) -> list[dict]:
    """
    Construct a chronological succession lineage for a single-value relation,
    showing active state, supersession links, and timestamps.
    """
    matching = [
        m for m in memories
        if getattr(m, "relation", None) == relation
    ]
    sorted_mems = sorted(
        matching,
        key=lambda m: (
            getattr(m, "created_at", None) or datetime.min,
            getattr(m, "id", 0) or 0,
        )
    )

    lineage = []
    for i, mem in enumerate(sorted_mems):
        next_mem = sorted_mems[i + 1] if i + 1 < len(sorted_mems) else None
        state = "ACTIVE" if getattr(mem, "active", False) else "SUPERSEDED"
        superseded_by = getattr(next_mem, "value", None) if state == "SUPERSEDED" and next_mem else None
        lineage.append({
            "memory_id": getattr(mem, "id", None),
            "subject": getattr(mem, "subject", None),
            "relation": relation,
            "value": getattr(mem, "value", None),
            "state": state,
            "created_at": getattr(mem, "created_at", None),
            "updated_at": getattr(mem, "updated_at", None),
            "superseded_by": superseded_by,
        })

    return lineage


