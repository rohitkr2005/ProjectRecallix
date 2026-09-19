from sqlalchemy import select

from app.database.database import SessionLocal
from app.database.models import Memory
from app.memory.lifecycle import (
    MemoryLifecycleState,
    determine_lifecycle_state,
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
        embedding=None
    ):
        # Check for exact duplicate
        existing = self._find_active_memory(
            subject=subject,
            relation=relation,
            value=value
        )

        if existing:
            return existing, "duplicate"

        # Check for conflicting active memory
        conflict = self._find_conflicting_memory(
            subject=subject,
            relation=relation,
            value=value
        )

        # Deactivate old conflicting memory
        if conflict:
            conflict.active = False

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

    def archive_memory(self, memory_id):
        """Semantic alias for deactivating/archiving a memory."""
        return self.deactivate_memory(memory_id)

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