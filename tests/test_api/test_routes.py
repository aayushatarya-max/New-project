import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from backend.main import app
from backend.database.connection import get_db
from backend.services.embedding_service import EmbeddingService
from backend.services.search_service import SearchService

client = TestClient(app)


@pytest.fixture(name="override_db")
def fixture_override_db(db_session):
    """
    Override the get_db dependency to use our testing database session.
    """
    def _get_db():
        try:
            yield db_session
        finally:
            pass
    app.dependency_overrides[get_db] = _get_db
    yield
    app.dependency_overrides.clear()


def test_status_endpoint():
    """
    Verify root status endpoint returns online.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"


@patch.object(EmbeddingService, "get_embeddings")
@patch.object(SearchService, "add_vectors")
def test_upload_text_file(mock_add_vectors, mock_get_embeddings, override_db):
    """
    Verify upload and ingestion pipeline for a plain text file.
    """
    # 1. Mock embedding vectors
    mock_get_embeddings.return_value = [[0.1] * 384]
    mock_add_vectors.return_value = None

    # 2. Upload file
    file_content = b"This is some text content representing a note about personal memory engines."
    response = client.post(
        "/api/upload",
        files={"file": ("notes.txt", file_content, "text/plain")}
    )

    assert response.status_code == 201
    json_data = response.json()
    assert json_data["filename"] == "notes.txt"
    assert json_data["chunks_indexed"] > 0

    # 3. Verify it is listed
    list_response = client.get("/api/files")
    assert list_response.status_code == 200
    files = list_response.json()
    assert len(files) == 1
    assert files[0]["filename"] == "notes.txt"
    assert "notes" in files[0]["tags"]


@patch.object(EmbeddingService, "get_embedding")
@patch.object(SearchService, "semantic_search")
def test_search_memories(mock_semantic, mock_get_emb, override_db):
    """
    Verify search parameter routing and date validations.
    """
    mock_get_emb.return_value = [0.1] * 384
    mock_semantic.return_value = [
        {
            "chunk_id": 1,
            "filename": "notes.txt",
            "filepath": "/uploads/notes.txt",
            "filetype": "txt",
            "page_number": 1,
            "chunk_text": "Sample chunk content.",
            "score": 0.95
        }
    ]

    # Search semantic
    response = client.get("/api/search?query=personal&type=semantic&limit=2")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["filename"] == "notes.txt"

    # Test invalid date exception
    response_invalid_date = client.get("/api/search?query=personal&start_date=invalid-date")
    assert response_invalid_date.status_code == 400
    assert "Invalid date format" in response_invalid_date.json()["detail"]


@patch.object(SearchService, "remove_vectors")
def test_delete_file_endpoint(mock_remove, override_db):
    """
    Verify deletion endpoint cleans up database and triggers FAISS deletion.
    """
    mock_remove.return_value = None

    # First, let's create a file record directly using the upload endpoint
    upload_res = client.post(
        "/api/upload",
        files={"file": ("to_del.txt", b"Content to delete.", "text/plain")}
    )
    assert upload_res.status_code == 201
    file_id = upload_res.json()["file_id"]

    # Delete the file
    del_res = client.delete(f"/api/files/{file_id}")
    assert del_res.status_code == 200
    assert "Successfully deleted" in del_res.json()["message"]

    # Verify listing is empty
    list_res = client.get("/api/files")
    assert len(list_res.json()) == 0
