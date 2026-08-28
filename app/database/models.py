from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    LargeBinary,
    DateTime
)

from app.database.database import Base


class Memory(Base):

    __tablename__ = "memories"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    subject = Column(
        String,
        nullable=False
    )

    relation = Column(
        String,
        nullable=False
    )

    value = Column(
        String,
        nullable=False
    )

    category = Column(
        String,
        nullable=False
    )

    importance = Column(
        Integer,
        default=5
    )

    active = Column(
        Boolean,
        default=True
    )

    embedding = Column(
        LargeBinary,
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )