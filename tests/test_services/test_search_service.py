import os
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
import numpy as np
import faiss
from backend.repositories.file_repository import FileRepository
from backend.services.search_service import SearchService
from backend.services.embedding_service import EmbeddingService


@pytest.fixture(name="setup_search_env")
def fixture_setup_search_env(tmp_path):
    """
    Redirects FAISS index path to a temp folder and resets search service state.
    """
    temp_index_path = os.path.join(tmp_path, "index.faiss")
    
    # Save original values
    orig_path = SearchService._index_path
    
    # Update search service path
    SearchService._index_path = temp_index_path
    SearchService._index = None  # Force re-initialization
    
    yield temp_index_path
    
    # Restore original values
    SearchService._index_path = orig_path
    SearchService._index = None
    
    if os.path.exists(temp_index_path):
        os.remove(temp_index_path)


def test_add_and_remove_vectors(setup_search_env):
    """
    Verify adding vectors writes to disk, and removing deletes them correctly.
    """
    # Initialize empty index
    index = SearchService._get_index()
    assert index.ntotal == 0

    # Mock vectors
    ids = [101, 102]
    vectors = [
        [1.0] + [0.0] * 383,
        [0.0, 1.0] + [0.0] * 382
    ]

    # Add to index
    SearchService.add_vectors(ids, vectors)
    
    # Assert added and saved
    assert os.path.exists(setup_search_env)
    assert SearchService._index.ntotal == 2

    # Remove from index
    SearchService.remove_vectors([101])
    assert SearchService._index.ntotal == 1
    
    # Search remaining vector
    query_np = np.array([[0.0, 1.0] + [0.0] * 382], dtype=np.float32)
    distances, indices = SearchService._index.search(query_np, 1)
    assert indices[0][0] == 102


def test_hybrid_search_flows(setup_search_env, db_session):
    """
    Test Semantic, Keyword, and Hybrid search logic with metadata filtering.
    """
    def mock_get_embedding(q):
        if "cookies" in q:
            return [0.0, 1.0] + [0.0] * 382
        return [1.0] + [0.0] * 383

    with patch.object(EmbeddingService, "get_embedding", side_effect=mock_get_embedding):
        # 1. Seed database with files and chunks
        file1 = FileRepository.create_file(
            db=db_session, filename="machine_learning.txt", filepath="/docs/ml.txt", filetype="txt", filesize=150
        )
        file2 = FileRepository.create_file(
            db=db_session, filename="cooking_recipes.pdf", filepath="/docs/cooking.pdf", filetype="pdf", filesize=300
        )
    
        chunks_f1 = [
            {"chunk_text": "Supervised machine learning algorithms require labeled training data.", "page_number": 1, "chunk_order": 1, "vector_id": 1},
            {"chunk_text": "Neural networks are part of deep learning architectures.", "page_number": 1, "chunk_order": 2, "vector_id": 2}
        ]
        chunks_f2 = [
            {"chunk_text": "To bake chocolate cookies, preheat the oven to 350 degrees.", "page_number": 1, "chunk_order": 1, "vector_id": 3}
        ]
    
        saved_chunks_f1 = FileRepository.create_chunks(db_session, file1.id, chunks_f1)
        saved_chunks_f2 = FileRepository.create_chunks(db_session, file2.id, chunks_f2)
    
        # 2. Add embeddings to FAISS
        # We assign:
        # vector_id 1 (ml chunk) -> direction [1.0, 0.0, ...]
        # vector_id 2 (neural net chunk) -> direction [0.707, 0.707, ...]
        # vector_id 3 (cookie chunk) -> direction [0.0, 1.0, ...]
        vectors = [
            [1.0] + [0.0] * 383,
            [0.707, 0.707] + [0.0] * 382,
            [0.0, 1.0] + [0.0] * 382
        ]
        SearchService.add_vectors([1, 2, 3], vectors)
    
        # 3. Test Keyword Search
        keyword_hits = SearchService.keyword_search(db_session, "learning")
        assert len(keyword_hits) == 2
        assert {h["filename"] for h in keyword_hits} == {"machine_learning.txt"}
    
        # 4. Test Semantic Search with mock embedding queries
        # Mock embedding query: close to ml chunk [1.0, 0.0, ...]
        semantic_hits = SearchService.semantic_search(db_session, "Tell me about ML algorithms.", limit=2)
        assert len(semantic_hits) == 2
        # Highest score should be chunk 1 (score = 1.0)
        assert semantic_hits[0]["chunk_text"].startswith("Supervised machine learning")
        assert semantic_hits[0]["score"] == pytest.approx(1.0)
    
        # Test filtering by filetype
        filtered_hits = SearchService.semantic_search(db_session, "ML", limit=2, filetype="pdf")
        assert len(filtered_hits) == 0  # No pdf contains the ML vectors
    
        # 5. Test Hybrid Search
        # Searching keyword "cookies" should find the cooking recipe
        hybrid_hits = SearchService.hybrid_search(db_session, "cookies", limit=1)
        assert len(hybrid_hits) == 1
        assert hybrid_hits[0]["filename"] == "cooking_recipes.pdf"
