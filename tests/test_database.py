import pytest
from backend.models.models import FileModel, ChunkModel, TagModel


def test_schema_creation_and_relationships(db_session):
    """
    Verify database schema creation, insertion, relationship mappings, and metadata.
    """
    # 1. Insert a mock file record
    test_file = FileModel(
        filename="test_document.pdf",
        filepath="/absolute/path/to/test_document.pdf",
        filetype="pdf",
        filesize=20480
    )
    db_session.add(test_file)
    db_session.commit()
    db_session.refresh(test_file)

    assert test_file.id is not None
    assert test_file.filename == "test_document.pdf"
    assert test_file.created_at is not None

    # 2. Insert mock chunks
    chunk_1 = ChunkModel(
        file_id=test_file.id,
        chunk_text="This is the first segment of text in the test document.",
        page_number=1,
        chunk_order=1,
        vector_id=101
    )
    chunk_2 = ChunkModel(
        file_id=test_file.id,
        chunk_text="This is the second segment of text in the test document.",
        page_number=2,
        chunk_order=2,
        vector_id=102
    )
    db_session.add_all([chunk_1, chunk_2])
    db_session.commit()

    # Verify chunks relations
    db_session.refresh(test_file)
    assert len(test_file.chunks) == 2
    assert test_file.chunks[0].chunk_text == "This is the first segment of text in the test document."
    assert test_file.chunks[1].page_number == 2
    assert test_file.chunks[0].file.filename == "test_document.pdf"

    # 3. Insert mock tags
    tag_1 = TagModel(file_id=test_file.id, tag="test")
    tag_2 = TagModel(file_id=test_file.id, tag="document")
    db_session.add_all([tag_1, tag_2])
    db_session.commit()

    # Verify tags relations
    db_session.refresh(test_file)
    assert len(test_file.tags) == 2
    assert {t.tag for t in test_file.tags} == {"test", "document"}


def test_cascade_delete(db_session):
    """
    Ensure deleting a file record cascades down to delete related chunks and tags.
    """
    # 1. Setup mock data
    test_file = FileModel(
        filename="to_delete.txt",
        filepath="/path/to/to_delete.txt",
        filetype="txt",
        filesize=1024
    )
    db_session.add(test_file)
    db_session.commit()
    db_session.refresh(test_file)

    chunk = ChunkModel(
        file_id=test_file.id,
        chunk_text="Some content.",
        page_number=1,
        chunk_order=1,
        vector_id=500
    )
    tag = TagModel(file_id=test_file.id, tag="temp")
    db_session.add_all([chunk, tag])
    db_session.commit()

    # Confirm insert matches
    assert db_session.query(FileModel).count() == 1
    assert db_session.query(ChunkModel).count() == 1
    assert db_session.query(TagModel).count() == 1

    # 2. Perform file deletion
    db_session.delete(test_file)
    db_session.commit()

    # 3. Assert cascade delete purged child tables
    assert db_session.query(FileModel).count() == 0
    assert db_session.query(ChunkModel).count() == 0
    assert db_session.query(TagModel).count() == 0
