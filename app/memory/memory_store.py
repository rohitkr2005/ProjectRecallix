from sqlalchemy import select

from app.database.database import SessionLocal
from app.database.models import Memory


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

    def restore_memory(self, memory_id):
        memory = self.session.get(Memory, memory_id)

        if memory is None:
            return False

        memory.active = True

        self.session.commit()
        self.session.refresh(memory)

        return True

    def close(self):
        self.session.close()