from sqlalchemy import select

from app.database.database import SessionLocal
from app.database.models import Memory
from app.memory.lifecycle import (
    ArchivalReason,
    MemoryLifecycleState,
    UpdateSemanticsDecision,
    determine_lifecycle_state,
    evaluate_archival_eligibility,
    evaluate_update_semantics,
    validate_restoration_safety,
)


SINGLE_VALUE_RELATIONS = {
    "lives_in",
    "works_at",
    "studies_at",
    "current_role",
    "current_city",
}


class MemoryStore:

    def __init__(self):
        self.session = SessionLocal()

    def is_single_value_relation(self, relation):
        return relation in SINGLE_VALUE_RELATIONS

    def _find_conflicting_memory(
        self,
        subject,
        relation,
        value
    ):
        if not self.is_single_value_relation(relation):
            return None

        statement = select(Memory).where(
            Memory.subject == subject,
            Memory.relation == relation,
            Memory.value != value,
            Memory.active.is_(True)
        )

        return (
            self.session.execute(statement)
            .scalars()
            .first()
        )

    def save_memory(
        self,
        subject,
        relation,
        value,
        category,
        importance=5,
        embedding=None,
        update_duplicate_importance=True
    ):
        active_memories = self.get_all_memories()
        decision, matched, reason = evaluate_update_semantics(
            subject=subject,
            relation=relation,
            value=value,
            active_memories=active_memories,
            is_single_value_fn=self.is_single_value_relation
        )

        if decision == UpdateSemanticsDecision.DUPLICATE:
            if update_duplicate_importance and importance is not None and matched.importance is not None:
                if importance > matched.importance:
                    matched.importance = importance
                    self.session.commit()
                    self.session.refresh(matched)
            return matched, "duplicate"

        if decision == UpdateSemanticsDecision.CONFLICT_SUPERSEDE:
            matched.active = False

        # Create new memory
        memory = Memory(
            subject=subject,
            relation=relation,
            value=value,
            category=category,
            importance=importance,
            active=True,
            embedding=embedding
        )

        self.session.add(memory)
        self.session.commit()
        self.session.refresh(memory)

        return memory, "created"

    def save_memory_with_semantics(
        self,
        subject,
        relation,
        value,
        category,
        importance=5,
        embedding=None,
        update_duplicate_importance=True
    ):
        """
        Save memory and return explainable update semantics metadata.
        Returns: (memory, status, metadata_dict)
        """
        active_memories = self.get_all_memories()
        decision, matched, reason = evaluate_update_semantics(
            subject=subject,
            relation=relation,
            value=value,
            active_memories=active_memories,
            is_single_value_fn=self.is_single_value_relation
        )

        superseded_memory_id = None
        if decision == UpdateSemanticsDecision.DUPLICATE:
            if update_duplicate_importance and importance is not None and matched.importance is not None:
                if importance > matched.importance:
                    matched.importance = importance
                    self.session.commit()
                    self.session.refresh(matched)
            return matched, "duplicate", {
                "decision": decision,
                "reason": reason,
                "matched_memory_id": matched.id,
                "superseded_memory_id": None,
            }

        if decision == UpdateSemanticsDecision.CONFLICT_SUPERSEDE:
            matched.active = False
            superseded_memory_id = matched.id

        memory = Memory(
            subject=subject,
            relation=relation,
            value=value,
            category=category,
            importance=importance,
            active=True,
            embedding=embedding
        )

        self.session.add(memory)
        self.session.commit()
        self.session.refresh(memory)

        return memory, "created", {
            "decision": decision,
            "reason": reason,
            "matched_memory_id": matched.id if matched else None,
            "superseded_memory_id": superseded_memory_id,
        }

    def update_memory(
        self,
        memory_id,
        value=None,
        importance=None,
        category=None,
        embedding=None
    ):
        """
        Perform an explicit in-place update on an existing memory.
        Validates single-value uniqueness when value is modified.
        
        Returns:
            (memory, "updated") on success, or (None, error_reason) on failure.
        """
        memory = self.session.get(Memory, memory_id)
        if memory is None:
            return None, "memory_not_found"

        if value is not None and value != memory.value:
            if memory.active and self.is_single_value_relation(memory.relation):
                statement = select(Memory).where(
                    Memory.subject == memory.subject,
                    Memory.relation == memory.relation,
                    Memory.id != memory.id,
                    Memory.active.is_(True)
                )
                other_active = self.session.execute(statement).scalars().first()
                if other_active:
                    return None, f"single_value_conflict_with_active_memory:{other_active.id}"

            memory.value = value

        if importance is not None:
            memory.importance = importance

        if category is not None:
            memory.category = category

        if embedding is not None:
            memory.embedding = embedding

        self.session.commit()
        self.session.refresh(memory)
        return memory, "updated"


    def _find_active_memory(
        self,
        subject,
        relation,
        value
    ):
        statement = select(Memory).where(
            Memory.subject == subject,
            Memory.relation == relation,
            Memory.value == value,
            Memory.active.is_(True)
        )

        return (
            self.session.execute(statement)
            .scalars()
            .first()
        )

    def get_all_memories(self):
        statement = select(Memory).where(
            Memory.active.is_(True)
        ).order_by(Memory.id)

        return (
            self.session.execute(statement)
            .scalars()
            .all()
        )

    def get_archived_memories(self):
        statement = select(Memory).where(
            Memory.active.is_(False)
        ).order_by(Memory.id)

        return (
            self.session.execute(statement)
            .scalars()
            .all()
        )

    def get_embedding(self, memory):
        if memory.embedding is None:
            return None

        return memory.embedding

    def update_embedding(self, memory_id, embedding):
        memory = self.session.get(Memory, memory_id)

        if memory is None:
            return False

        memory.embedding = embedding

        self.session.commit()
        self.session.refresh(memory)

        return True

    def deactivate_memory(self, memory_id):
        memory = self.session.get(Memory, memory_id)

        if memory is None:
            return False

        memory.active = False

        self.session.commit()
        self.session.refresh(memory)

        return True

    def archive_memory(
        self,
        memory_id,
        reason=ArchivalReason.EXPLICIT_USER_REQUEST,
        force=False,
        protected_importance=7,
        protected_categories=None,
    ):
        """
        Archive a memory respecting Recallix archival rules.
        Explicit user action or force=True always succeeds.
        Policy-based or automatic archival respects importance and category protections.
        """
        memory = self.session.get(Memory, memory_id)
        if memory is None:
            return False

        can_archive, _ = evaluate_archival_eligibility(
            memory=memory,
            reason=reason,
            protected_importance=protected_importance,
            protected_categories=protected_categories,
            force=force,
        )
        if not can_archive:
            return False

        memory.active = False
        self.session.commit()
        self.session.refresh(memory)
        return True

    def can_archive(
        self,
        memory_id,
        reason=ArchivalReason.POLICY_ARCHIVED,
        protected_importance=7,
        protected_categories=None,
    ):
        """
        Check if a memory is eligible to be archived under a specific reason or policy.
        Returns: (can_archive: bool, reason: str)
        """
        memory = self.session.get(Memory, memory_id)
        if memory is None:
            return False, "memory_not_found"

        return evaluate_archival_eligibility(
            memory=memory,
            reason=reason,
            protected_importance=protected_importance,
            protected_categories=protected_categories,
        )

    def get_archival_candidates(self, subject=None, max_importance=3, protected_categories=None):
        """
        Find active memories that are potential candidates for policy/manual archival
        (low importance, non-protected).
        """
        active_memories = self.get_all_memories()
        candidates = []
        for mem in active_memories:
            if subject is not None and mem.subject != subject:
                continue
            can_archive, _ = evaluate_archival_eligibility(
                memory=mem,
                reason=ArchivalReason.LOW_IMPORTANCE_DEPRECATED,
                protected_categories=protected_categories,
            )
            if can_archive and (mem.importance is not None and mem.importance <= max_importance):
                candidates.append(mem)
        return candidates


    def get_lifecycle_state(self, memory_id):
        """
        Compute the formal lifecycle state (ACTIVE, ARCHIVED, SUPERSEDED)
        for the given memory.
        """
        memory = self.session.get(Memory, memory_id)
        if memory is None:
            return None

        active_memories = self.get_all_memories()
        return determine_lifecycle_state(
            memory=memory,
            active_memories=active_memories,
            is_single_value_fn=self.is_single_value_relation,
        )

    def restore_memory(self, memory_id, allow_conflict=False):
        """
        Safely restore an inactive memory. If allow_conflict is False,
        prevents restoring a single-value memory when an active conflict exists.
        """
        memory = self.session.get(Memory, memory_id)

        if memory is None:
            return False

        if not allow_conflict:
            active_memories = self.get_all_memories()
            can_restore, _ = validate_restoration_safety(
                memory=memory,
                active_memories=active_memories,
                is_single_value_fn=self.is_single_value_relation,
            )
            if not can_restore:
                return False

        memory.active = True

        self.session.commit()
        self.session.refresh(memory)

        return True

    def get_superseded_memories(self):
        """Return all memories that have been superseded by newer active values."""
        archived = self.get_archived_memories()
        active = self.get_all_memories()
        return [
            m for m in archived
            if determine_lifecycle_state(m, active, self.is_single_value_relation) == MemoryLifecycleState.SUPERSEDED
        ]

    def get_lifecycle_history(self, subject=None, relation=None):
        """
        Retrieve memory history with lifecycle states for a subject and optional relation.
        """
        statement = select(Memory).order_by(Memory.created_at.asc(), Memory.id.asc())
        if subject is not None:
            statement = statement.where(Memory.subject == subject)
        if relation is not None:
            statement = statement.where(Memory.relation == relation)

        memories = self.session.execute(statement).scalars().all()
        active_memories = [m for m in memories if m.active]

        history = []
        for memory in memories:
            state = determine_lifecycle_state(
                memory=memory,
                active_memories=active_memories,
                is_single_value_fn=self.is_single_value_relation,
            )
            history.append({
                "memory": memory,
                "lifecycle_state": state,
            })
        return history

    def close(self):
        self.session.close()