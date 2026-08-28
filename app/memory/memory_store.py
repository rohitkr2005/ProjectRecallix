from sqlalchemy import select

from app.database.database import SessionLocal
from app.database.models import Memory


class MemoryStore:

    # Relations where only one current value should exist
    SINGLE_VALUE_RELATIONS = {
        "lives_in",
        "located_in",
        "works_at",
        "studies_at",
        "studies",
        "job",
        "occupation",
        "age",
        "name",
    }

    def __init__(self):
        self.session = SessionLocal()

    def save_memory(
        self,
        subject,
        relation,
        value,
        category,
        importance=5
    ):
        # Normalize text for consistent comparison
        subject = subject.strip()
        relation = relation.strip().lower()
        value = value.strip()

        # ---------------------------------------------------------
        # 1. Check for an exact duplicate
        # ---------------------------------------------------------

        duplicate_statement = select(Memory).where(
            Memory.subject == subject,
            Memory.relation == relation,
            Memory.value == value,
            Memory.active.is_(True)
        )

        existing_memory = (
            self.session.execute(duplicate_statement)
            .scalars()
            .first()
        )

        if existing_memory:

            # Update importance if the new information is more important
            if importance > existing_memory.importance:
                existing_memory.importance = importance

            self.session.commit()
            self.session.refresh(existing_memory)

            return existing_memory, "duplicate"

        # ---------------------------------------------------------
        # 2. Check whether this is a single-value relation
        # ---------------------------------------------------------

        if relation in self.SINGLE_VALUE_RELATIONS:

            existing_statement = select(Memory).where(
                Memory.subject == subject,
                Memory.relation == relation,
                Memory.active.is_(True)
            )

            existing_memory = (
                self.session.execute(existing_statement)
                .scalars()
                .first()
            )

            if existing_memory:

                # Replace the old value
                existing_memory.value = value
                existing_memory.category = category
                existing_memory.importance = importance

                self.session.commit()
                self.session.refresh(existing_memory)

                return existing_memory, "updated"

        # ---------------------------------------------------------
        # 3. No duplicate and no existing single-value memory
        #    → create a new memory
        # ---------------------------------------------------------

        memory = Memory(
            subject=subject,
            relation=relation,
            value=value,
            category=category,
            importance=importance,
            active=True
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

        memory = self.session.get(Memory, memory_id)

        if memory is None:
            return False

        memory.active = False

        self.session.commit()

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