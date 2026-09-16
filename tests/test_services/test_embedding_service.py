import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from backend.services.embedding_service import EmbeddingService


@pytest.fixture(autouse=True)
def reset_embedding_singleton():
    """
    Reset singleton state before each test.
    """
    EmbeddingService._model = None
    yield


def test_get_embedding_success():
    """
    Verify single string embedding calculation, singleton load, and L2 normalization.
    """
    with patch("backend.services.embedding_service.SentenceTransformer") as mock_transformer_cls:
        # Create mock model instance
        mock_model = MagicMock()
        # Mock encoding output - an arbitrary 384 dim array
        dummy_vector = np.ones(384) * 2.0  # L2 norm = sqrt(384 * 4)
        mock_model.encode.return_value = dummy_vector
        mock_transformer_cls.return_value = mock_model

        # Run embedding
        vector = EmbeddingService.get_embedding("Verify my embedding vector.")
        
        # Verify calls
        mock_transformer_cls.assert_called_once_with("all-MiniLM-L6-v2")
        mock_model.encode.assert_called_once_with("Verify my embedding vector.", convert_to_numpy=True)
        
        # Assert vector length is 384
        assert len(vector) == 384
        
        # Assert vector is normalized (L2 norm is approximately 1.0)
        norm = np.linalg.norm(vector)
        assert pytest.approx(norm) == 1.0


def test_get_embeddings_batch():
    """
    Verify batch calculation and normalization.
    """
    with patch("backend.services.embedding_service.SentenceTransformer") as mock_transformer_cls:
        mock_model = MagicMock()
        # Mock two vectors
        dummy_vectors = np.array([np.ones(384), np.ones(384) * 3.0])
        mock_model.encode.return_value = dummy_vectors
        mock_transformer_cls.return_value = mock_model

        texts = ["text one", "text two"]
        vectors = EmbeddingService.get_embeddings(texts)
        
        assert len(vectors) == 2
        assert len(vectors[0]) == 384
        
        # L2 norm of normalized vectors should be 1.0
        assert pytest.approx(np.linalg.norm(vectors[0])) == 1.0
        assert pytest.approx(np.linalg.norm(vectors[1])) == 1.0


def test_empty_string_handling():
    """
    Verify empty or whitespace strings return a zero vector without calling the model.
    """
    with patch("backend.services.embedding_service.SentenceTransformer") as mock_transformer_cls:
        vector = EmbeddingService.get_embedding("   ")
        # SentenceTransformer should not be loaded or called
        mock_transformer_cls.assert_not_called()
        assert len(vector) == 384
        assert all(val == 0.0 for val in vector)
