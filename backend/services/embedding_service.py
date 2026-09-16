import os
import logging
import hashlib
import math
from typing import List, Optional, Dict
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Dimension of our pseudo-embedding vectors
_VECTOR_DIM = 384


def _tokenize(text: str) -> List[str]:
    """Lowercases and splits text into word tokens."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def _text_to_vector(text: str) -> List[float]:
    """
    Converts text into a deterministic 384-dim TF-weighted hashed vector.
    Uses the hashing trick: each word maps to a bucket, TF weights are applied.
    No external ML libraries needed — pure Python only.
    """
    tokens = _tokenize(text)
    if not tokens:
        return [0.0] * _VECTOR_DIM

    # Count term frequencies
    tf: Dict[str, int] = {}
    for tok in tokens:
        tf[tok] = tf.get(tok, 0) + 1

    # Place tokens into hashed buckets, weight by TF
    vec = [0.0] * _VECTOR_DIM
    for tok, count in tf.items():
        # Deterministic bucket via SHA-256
        bucket = int(hashlib.sha256(tok.encode()).hexdigest(), 16) % _VECTOR_DIM
        # Sign to reduce collisions
        sign_hash = int(hashlib.md5(tok.encode()).hexdigest(), 16) % 2
        sign = 1 if sign_hash else -1
        vec[bucket] += sign * count

    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class EmbeddingService:
    """
    Service class providing vector embeddings using a pure-Python hashing trick
    (TF + hash projection). No PyTorch, ONNX, or C++ Redistributable required.
    The search quality is keyword/TF-based; semantic synonyms won't match but
    exact-word matches work very well.
    """

    @classmethod
    def get_embedding(cls, text: str) -> List[float]:
        """
        Generate a normalized 384-dim vector embedding for a single text string.

        Args:
            text (str): The input text.

        Returns:
            List[float]: Normalized embedding vector.
        """
        if not text or not text.strip():
            return [0.0] * _VECTOR_DIM
        try:
            return _text_to_vector(text)
        except Exception as e:
            logger.error("Failed to calculate embedding for text snippet: %s", e)
            raise ValueError(f"Embedding generation failed: {e}") from e

    @classmethod
    def get_embeddings(cls, texts: List[str]) -> List[List[float]]:
        """
        Generate normalized 384-dim vector embeddings for a list of texts.

        Args:
            texts (List[str]): List of input text snippets.

        Returns:
            List[List[float]]: List of normalized embedding vectors.
        """
        if not texts:
            return []
        try:
            return [_text_to_vector(t) for t in texts]
        except Exception as e:
            logger.error("Failed to calculate batch embeddings: %s", e)
            raise ValueError(f"Batch embedding generation failed: {e}") from e

