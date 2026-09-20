from datetime import datetime
from sqlalchemy import select

from app.database.database import SessionLocal
from app.database.models import Memory
from app.memory.lifecycle import (
    ArchivalReason,
    MemoryLifecycleState,
    RestorationStrategy,
    UpdateSemanticsDecision,
    build_conflict_lineage,
    consolidate_memories as consolidate_memories_fn,
    determine_lifecycle_state,
    evaluate_archival_eligibility,
    evaluate_update_semantics,
    validate_importance,
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

    def __init__(self, session=None):
        self.session = session or SessionLocal()

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
        update_duplicate_importance=True,
        temporal_state=None,
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
            is_single_value_fn=self.is_single_value_relation,
            temporal_state=temporal_state,
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

        temporal = str(temporal_state or "PRESENT").upper()
        memory = Memory(
            subject=subject,
            relation=relation,
            value=value,
            category=category,
            importance=importance,
            active=True,
            embedding=embedding,
            temporal_state=temporal,
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

    def restore_memory(
        self,
        memory_id,
        strategy=RestorationStrategy.REJECT,
        allow_conflict=False,
    ):
        """
        Safely restore an inactive memory.
        If allow_conflict is True, strategy is treated as SUPERSEDE_ACTIVE.
        If strategy is SUPERSEDE_ACTIVE, conflicting active memory is deactivated.
        """
        memory = self.session.get(Memory, memory_id)
        if memory is None:
            return False

        if allow_conflict:
            strategy = RestorationStrategy.FORCE

        active_memories = self.get_all_memories()
        can_restore, reason, conflicting_mem = validate_restoration_safety(
            memory=memory,
            active_memories=active_memories,
            is_single_value_fn=self.is_single_value_relation,
            strategy=strategy,
        )

        if not can_restore:
            return False

        if conflicting_mem and strategy == RestorationStrategy.SUPERSEDE_ACTIVE:
            conflicting_mem.active = False

        memory.active = True
        memory.updated_at = datetime.utcnow()

        self.session.commit()
        self.session.refresh(memory)
        return True

    def restore_memory_with_details(
        self,
        memory_id,
        strategy=RestorationStrategy.REJECT,
        allow_conflict=False,
    ):
        """
        Restore an inactive memory and return rich transition details.
        Returns: (success: bool, status: str, details: dict)
        """
        memory = self.session.get(Memory, memory_id)
        if memory is None:
            return False, "memory_not_found", {}

        if allow_conflict:
            strategy = RestorationStrategy.FORCE


        active_memories = self.get_all_memories()
        can_restore, reason, conflicting_mem = validate_restoration_safety(
            memory=memory,
            active_memories=active_memories,
            is_single_value_fn=self.is_single_value_relation,
            strategy=strategy,
        )

        if not can_restore:
            return False, reason, {
                "restored_memory_id": memory.id,
                "conflicting_memory_id": conflicting_mem.id if conflicting_mem else None,
                "strategy_used": strategy,
            }

        superseded_id = None
        if conflicting_mem and strategy == RestorationStrategy.SUPERSEDE_ACTIVE:
            conflicting_mem.active = False
            superseded_id = conflicting_mem.id

        memory.active = True
        memory.updated_at = datetime.utcnow()

        self.session.commit()
        self.session.refresh(memory)

        return True, "restored", {
            "restored_memory_id": memory.id,
            "superseded_memory_id": superseded_id,
            "strategy_used": strategy,
        }

    def can_restore(
        self,
        memory_id,
        strategy=RestorationStrategy.REJECT,
    ):
        """
        Check if an inactive memory can be restored under a given strategy.
        Returns: (can_restore: bool, reason: str)
        """
        memory = self.session.get(Memory, memory_id)
        if memory is None:
            return False, "memory_not_found"

        active_memories = self.get_all_memories()
        can_restore, reason, _ = validate_restoration_safety(
            memory=memory,
            active_memories=active_memories,
            is_single_value_fn=self.is_single_value_relation,
            strategy=strategy,
        )
        return can_restore, reason


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

    def update_importance(self, memory_id, importance):
        """
        Validate and update memory importance in-place.
        """
        memory = self.session.get(Memory, memory_id)
        if memory is None:
            return False

        memory.importance = validate_importance(importance)
        memory.updated_at = datetime.utcnow()
        self.session.commit()
        self.session.refresh(memory)
        return True

    def cleanup_duplicates(self, subject=None, dry_run=False):
        """
        Detect and resolve active duplicates for the same (subject, relation, value).
        Designates the canonical memory (highest importance, latest updated_at),
        sets its importance to max(importances), and deactivates redundant duplicates.
        """
        statement = select(Memory).where(Memory.active.is_(True))
        if subject is not None:
            statement = statement.where(Memory.subject == subject)

        active_memories = self.session.execute(statement).scalars().all()

        # Group by (subject, relation, value)
        groups = {}
        for m in active_memories:
            key = (m.subject, m.relation, m.value)
            groups.setdefault(key, []).append(m)

        cleaned_groups = 0
        duplicates_deactivated = 0
        details = []

        for key, mem_list in groups.items():
            if len(mem_list) <= 1:
                continue

            cleaned_groups += 1
            # Sort by importance descending, updated_at descending, id ascending
            sorted_mems = sorted(
                mem_list,
                key=lambda m: (
                    m.importance if m.importance is not None else 5,
                    m.updated_at if m.updated_at is not None else datetime.min,
                    -m.id if m.id is not None else 0,
                ),
                reverse=True,
            )

            canonical = sorted_mems[0]
            redundant = sorted_mems[1:]

            max_imp = max(
                (m.importance for m in mem_list if m.importance is not None),
                default=5,
            )

            if not dry_run:
                canonical.importance = max_imp
                for red in redundant:
                    red.active = False
                self.session.commit()

            duplicates_deactivated += len(redundant)
            details.append({
                "subject": key[0],
                "relation": key[1],
                "value": key[2],
                "canonical_id": canonical.id,
                "deactivated_ids": [r.id for r in redundant],
                "canonical_importance": max_imp,
            })

        return {
            "cleaned_groups": cleaned_groups,
            "duplicates_deactivated": duplicates_deactivated,
            "dry_run": dry_run,
            "details": details,
        }

    def get_conflict_history(self, subject, relation):
        """
        Retrieve chronological conflict and supersession history for a single-value relation.
        """
        statement = select(Memory).where(
            Memory.subject == subject,
            Memory.relation == relation,
        ).order_by(Memory.created_at.asc(), Memory.id.asc())

        memories = self.session.execute(statement).scalars().all()
        return build_conflict_lineage(memories, relation)

    def consolidate_memories(self, subject=None):
        """
        Consolidate active memories into a grounded, structured knowledge profile.
        """
        statement = select(Memory).where(Memory.active.is_(True))
        if subject is not None:
            statement = statement.where(Memory.subject == subject)

        active_memories = self.session.execute(statement).scalars().all()
        return consolidate_memories_fn(active_memories)

    def close(self):
        self.session.close()
