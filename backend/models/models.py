from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from backend.database.connection import Base


class FileModel(Base):
    """
    SQLAlchemy model representing the 'files' table.
    Stores metadata about uploaded files.
    """
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(1024), nullable=False, unique=True)
    filetype = Column(String(50), nullable=False)
    filesize = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    # cascade="all, delete-orphan" ensures when a file is deleted, all its related chunks and tags are also deleted
    chunks = relationship("ChunkModel", back_populates="file", cascade="all, delete-orphan")
    tags = relationship("TagModel", back_populates="file", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<File(id={self.id}, filename='{self.filename}', type='{self.filetype}')>"


class ChunkModel(Base):
    """
    SQLAlchemy model representing the 'chunks' table.
    Stores text slices from indexed documents.
    """
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(Integer, ForeignKey("files.id", ondelete="CASCADE"), nullable=False)
    chunk_text = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)  # Nullable for non-paged files (like voice notes, txt)
    chunk_order = Column(Integer, nullable=False)
    vector_id = Column(Integer, nullable=True)  # References index location in FAISS index file

    # Relationships
    file = relationship("FileModel", back_populates="chunks")

    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, file_id={self.file_id}, order={self.chunk_order}, vector_id={self.vector_id})>"


# Create an index on vector_id for quick lookups during search mapping
Index("idx_chunks_vector_id", ChunkModel.vector_id)


class TagModel(Base):
    """
    SQLAlchemy model representing the 'tags' table.
    Stores categories or tags generated for files.
    """
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_id = Column(Integer, ForeignKey("files.id", ondelete="CASCADE"), nullable=False)
    tag = Column(String(100), nullable=False)

    # Relationships
    file = relationship("FileModel", back_populates="tags")

    def __repr__(self) -> str:
        return f"<Tag(id={self.id}, file_id={self.file_id}, tag='{self.tag}')>"
