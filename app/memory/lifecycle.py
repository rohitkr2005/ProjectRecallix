from enum import Enum
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


def validate_restoration_safety(
    memory,
    active_memories: Optional[Iterable] = None,
    is_single_value_fn: Optional[Callable[[str], bool]] = None,
) -> Tuple[bool, str]:
    """
    Validate whether an inactive memory can safely be restored without violating
    single-value constraints or active duplicate constraints.
    
    Returns:
        (can_restore: bool, reason: str)
    """
    if getattr(memory, "active", False):
        return False, "already_active"

    if is_single_value_fn and is_single_value_fn(getattr(memory, "relation", "")):
        if active_memories:
            for active_mem in active_memories:
                if (
                    getattr(active_mem, "active", False)
                    and getattr(active_mem, "subject", None) == getattr(memory, "subject", None)
                    and getattr(active_mem, "relation", None) == getattr(memory, "relation", None)
                ):
                    if getattr(active_mem, "value", None) == getattr(memory, "value", None):
                        return False, f"duplicate_of_active_memory:{getattr(active_mem, 'id', None)}"
                    return False, f"conflict_with_active_memory:{getattr(active_mem, 'id', None)}"

    return True, "safe_to_restore"


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

