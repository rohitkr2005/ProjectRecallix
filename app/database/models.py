from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    inspect,
    text,
)
from sqlalchemy.orm import relationship

from app.database.database import Base, engine


class User(Base):
    """15.3 Multi-user authentication entity."""

    __tablename__ = "users"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    username = Column(
        String,
        unique=True,
        index=True,
        nullable=False,
    )

    email = Column(
        String,
        unique=True,
        index=True,
        nullable=False,
    )

    hashed_password = Column(
        String,
        nullable=False,
    )

    is_active = Column(
        Boolean,
        default=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    memories = relationship(
        "Memory",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class Memory(Base):

    __tablename__ = "memories"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    # 15.3: User ownership (nullable for backward compatibility with existing tests)
    user_id = Column(
        Integer,
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )

    subject = Column(
        String,
        nullable=False,
    )

    relation = Column(
        String,
        nullable=False,
    )

    value = Column(
        String,
        nullable=False,
    )

    category = Column(
        String,
        nullable=False,
    )

    importance = Column(
        Integer,
        default=5,
    )

    active = Column(
        Boolean,
        default=True,
    )

    embedding = Column(
        LargeBinary,
        nullable=True,
    )

    # 10.1 - 10.3: Usage, feedback & reinforcement
    access_count = Column(
        Integer,
        default=0,
        nullable=True,
    )

    helpful_count = Column(
        Integer,
        default=0,
        nullable=True,
    )

    last_accessed_at = Column(
        DateTime,
        nullable=True,
    )

    reinforcement_score = Column(
        Float,
        default=0.0,
        nullable=True,
    )

    # 10.5: Temporal Memory state (PAST, PRESENT, FUTURE)
    temporal_state = Column(
        String,
        default="PRESENT",
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    user = relationship(
        "User",
        back_populates="memories",
    )


class MemoryEdge(Base):
    """10.6 - 10.7: Explicit entity/concept graph relationships."""

    __tablename__ = "memory_edges"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    source_entity = Column(
        String,
        nullable=False,
        index=True,
    )

    relation = Column(
        String,
        nullable=False,
    )

    target_entity = Column(
        String,
        nullable=False,
        index=True,
    )

    category = Column(
        String,
        nullable=True,
    )

    weight = Column(
        Float,
        default=1.0,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )


def init_db(target_engine=None):
    """Initialize tables and ensure all columns exist via safe migration."""
    e = target_engine or engine
    Base.metadata.create_all(bind=e)

    # Safe migration for existing SQLite DBs: add new columns if missing
    try:
        inspector = inspect(e)
        existing_cols = {col["name"] for col in inspector.get_columns("memories")}

        columns_to_add = [
            ("user_id", "INTEGER REFERENCES users(id)"),
            ("access_count", "INTEGER DEFAULT 0"),
            ("helpful_count", "INTEGER DEFAULT 0"),
            ("last_accessed_at", "DATETIME"),
            ("reinforcement_score", "FLOAT DEFAULT 0.0"),
            ("temporal_state", "VARCHAR DEFAULT 'PRESENT'"),
        ]

        with e.connect() as conn:
            for col_name, col_type in columns_to_add:
                if col_name not in existing_cols:
                    conn.execute(text(f"ALTER TABLE memories ADD COLUMN {col_name} {col_type};"))
            conn.commit()
    except Exception:
        # In-memory or non-SQLite engines during tests can ignore ALTER errors
        pass