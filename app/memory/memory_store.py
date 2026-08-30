import pickle

from sqlalchemy import select

from app.database.database import SessionLocal
from app.database.models import Memory
from app.embeddings.embedding_engine import EmbeddingEngine


class MemoryStore:

    def __init__(self):

        self.session = SessionLocal()
        self.embedding_engine = EmbeddingEngine()

    def save_memory(
        self,
        subject,
        relation,
        value,
        category,
        importance=5
    ):

        # -------------------------------------------------
        # 1. Check for an exact active duplicate
        # -------------------------------------------------

        duplicate_statement = select(Memory).where(
            Memory.subject == subject,
            Memory.relation == relation,
            Memory.value == value,
            Memory.active.is_(True)
        )

        existing_memory = (
            self.session.execute(
                duplicate_statement
            )
            .scalars()
            .first()
        )

        if existing_memory:

            return existing_memory, "duplicate"

        # -------------------------------------------------
        # 2. Generate embedding
        # -------------------------------------------------

        embedding = (
            self.embedding_engine.generate_memory_embedding(
                subject=subject,
                relation=relation,
                value=value,
                category=category
            )
        )

        # -------------------------------------------------
        # 3. Convert NumPy array to bytes
        # -------------------------------------------------

        embedding_bytes = pickle.dumps(
            embedding
        )

        # -------------------------------------------------
        # 4. Create memory
        # -------------------------------------------------

        memory = Memory(
            subject=subject,
            relation=relation,
            value=value,
            category=category,
            importance=importance,
            active=True,
            embedding=embedding_bytes
        )

        self.session.add(memory)

        self.session.commit()

        self.session.refresh(memory)

        return memory, "created"

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

    def deactivate_memory(self, memory_id):

        statement = select(Memory).where(
            Memory.id == memory_id
        )

        memory = (
            self.session.execute(statement)
            .scalars()
            .first()
        )

        if memory is None:

            return False

        memory.active = False

        self.session.commit()

        return True

    def restore_memory(self, memory_id):

        statement = select(Memory).where(
            Memory.id == memory_id
        )

        memory = (
            self.session.execute(statement)
            .scalars()
            .first()
        )

        if memory is None:

            return False

        memory.active = True

        self.session.commit()

        return True

    def get_embedding(self, memory):

        if memory.embedding is None:

            return None

        return pickle.loads(
            memory.embedding
        )

    def close(self):

        self.session.close()