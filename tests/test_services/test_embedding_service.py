import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from backend.services.embedding_service import EmbeddingService


@pytest.fixture(autouse=True)
def reset_embedding_singleton():
    yield


def test_get_embedding_success():
    """
    Verify single string embedding calculation and L2 normalization.
    """
    vector = EmbeddingService.get_embedding("Verify my embedding vector.")
    
    # Assert vector length is 384
    assert len(vector) == 384
    
    # Assert vector is normalized (L2 norm is approximately 1.0)
    norm = np.linalg.norm(vector)
    assert pytest.approx(norm) == 1.0


def test_get_embeddings_batch():
    """
    Verify batch calculation and normalization.
    """
    texts = ["text one", "text two"]
    vectors = EmbeddingService.get_embeddings(texts)
    
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384
    
    # L2 norm of normalized vectors should be 1.0
    assert pytest.approx(np.linalg.norm(vectors[0])) == 1.0
    assert pytest.approx(np.linalg.norm(vectors[1])) == 1.0


def test_empty_string_handling():
    """
    Verify empty or whitespace strings return a zero vector.
    """
    vector = EmbeddingService.get_embedding("   ")
    assert len(vector) == 384
    assert all(val == 0.0 for val in vector)
