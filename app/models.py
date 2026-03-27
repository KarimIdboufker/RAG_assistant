import uuid
from datetime import datetime

from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

from app.database import Base
from app.config import settings


class Paper(Base):
    __tablename__ = "papers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String, unique=True, nullable=False)
    title = Column(Text)
    authors = Column(Text)
    ingested_at = Column(DateTime, default=datetime.utcnow)


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)

    # Element type: abstract | body
    chunk_type = Column(String, nullable=False)
    section = Column(Text)       # e.g. "3.2 Attention Functions"
    chunk_index = Column(Integer)
    page_num = Column(Integer)

    # content: raw text shown to the user / LLM
    # contextualized_content: [Paper: X][Section: Y]\n content — what we embed
    content = Column(Text, nullable=False)
    contextualized_content = Column(Text, nullable=False)

    embedding = Column(Vector(settings.embedding_dim))
