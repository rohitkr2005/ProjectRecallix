from sqlalchemy import select

from app.database.database import SessionLocal
from app.database.models import Memory


class MemoryStore:

    def __init__(self):
        self.session = SessionLocal()

    def save_memory(
        self,
        subject,
        relation,
        value,
        category,
        importance=5,
        embedding=None
    ):
        existing = self._find_active_memory(
            subject=subject,
            relation=relation,
            value=value
        )

        if existing:
            return existing, "duplicate"

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