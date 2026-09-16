import pytest
from datetime import datetime, timezone, timedelta
from backend.repositories.file_repository import FileRepository
from backend.models.models import FileModel, ChunkModel, TagModel


def test_file_repository_crud(db_session):
    """
    Test basic create, retrieve, and delete operations.
    """
    # 1. Test create_file
    filepath = "/data/uploads/sample.txt"
    file_record = FileRepository.create_file(
        db=db_session,
        filename="sample.txt",
        filepath=filepath,
        filetype="txt",
        filesize=512
    )
    assert file_record.id is not None
    assert file_record.filename == "sample.txt"
    assert file_record.filepath == filepath

    # 2. Test get_file and get_file_by_filepath
    fetched_by_id = FileRepository.get_file(db_session, file_record.id)
    assert fetched_by_id is not None
    assert fetched_by_id.filepath == filepath

    fetched_by_path = FileRepository.get_file_by_filepath(db_session, filepath)
    assert fetched_by_path is not None
    assert fetched_by_path.id == file_record.id

    # 3. Test list_files
    files = FileRepository.list_files(db_session)
    assert len(files) == 1
    assert files[0].id == file_record.id

    # 4. Test delete_file
    deleted = FileRepository.delete_file(db_session, file_record.id)
    assert deleted is True
    assert FileRepository.get_file(db_session, file_record.id) is None


def test_list_files_filtering(db_session):
    """
    Test filtering files by type and creation dates.
    """
    now = datetime.now(timezone.utc)
    
    file_pdf = FileRepository.create_file(
        db=db_session, filename="doc.pdf", filepath="/path/doc.pdf", filetype="pdf", filesize=100
    )
    file_txt = FileRepository.create_file(
        db=db_session, filename="doc.txt", filepath="/path/doc.txt", filetype="txt", filesize=200
    )

    # Filter by type
    pdfs = FileRepository.list_files(db_session, filetype="pdf")
    assert len(pdfs) == 1
    assert pdfs[0].id == file_pdf.id

    # Filter by date range
    recent_files = FileRepository.list_files(
        db_session, 
        start_date=now - timedelta(minutes=5), 
        end_date=now + timedelta(minutes=5)
    )
    assert len(recent_files) == 2


def test_bulk_chunks_and_tags_creation(db_session):
    """
    Test bulk chunk insertion and tag attachments.
    """
    file_record = FileRepository.create_file(
        db=db_session, filename="bulk.pdf", filepath="/path/bulk.pdf", filetype="pdf", filesize=123
    )

    # 1. Create chunks bulk
    chunks_data = [
        {"chunk_text": "text chunk one", "page_number": 1, "chunk_order": 1, "vector_id": 10},
        {"chunk_text": "text chunk two", "page_number": 2, "chunk_order": 2, "vector_id": 20}
    ]
    created_chunks = FileRepository.create_chunks(db_session, file_record.id, chunks_data)
    assert len(created_chunks) == 2
    assert created_chunks[0].chunk_text == "text chunk one"
    assert created_chunks[1].vector_id == 20

    # 2. Create tags bulk
    tags_list = ["ai", "machine-learning", "pdf"]
    created_tags = FileRepository.create_tags(db_session, file_record.id, tags_list)
    assert len(created_tags) == 3
    assert {t.tag for t in created_tags} == {"ai", "machine-learning", "pdf"}
