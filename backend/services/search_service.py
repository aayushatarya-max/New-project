import os
import logging
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from sqlalchemy.orm import Session
from backend.models.models import FileModel, ChunkModel
from backend.services.embedding_service import EmbeddingService
from backend.repositories.file_repository import FileRepository

logger = logging.getLogger(__name__)


class SearchService:
    """
    Service class managing the FAISS vector index database and
    providing keyword, semantic, and hybrid search operations.
    """

    _index: Optional[faiss.IndexIDMap] = None
    _index_path: str = os.getenv("FAISS_INDEX_PATH", "vector_store/index.faiss")
    _dimension: int = 384  # Dimensionality of all-MiniLM-L6-v2 embeddings
    _lock = threading.Lock()

    @classmethod
    def _get_index(cls) -> faiss.IndexIDMap:
        """
        Thread-safe method to load the existing FAISS index from disk,
        or create a new empty IndexIDMap if it doesn't exist.
        """
        with cls._lock:
            if cls._index is None:
                # Ensure the containing folder exists
                dir_name = os.path.dirname(cls._index_path)
                if dir_name and not os.path.exists(dir_name):
                    os.makedirs(dir_name, exist_ok=True)
                    logger.info("Created directory for FAISS index: %s", dir_name)

                if os.path.exists(cls._index_path):
                    logger.info("Loading existing FAISS index from: %s", cls._index_path)
                    try:
                        cls._index = faiss.read_index(cls._index_path)
                        logger.info("Successfully loaded FAISS index. Total vectors: %d", cls._index.ntotal)
                    except Exception as e:
                        logger.error("Failed to load FAISS index. Creating a new one instead. Error: %s", e)
                        cls._index = faiss.IndexIDMap(faiss.IndexFlatIP(cls._dimension))
                else:
                    logger.info("FAISS index file not found. Initializing a new IndexIDMap at: %s", cls._index_path)
                    # IndexFlatIP matches cosine similarity when vectors are L2-normalized
                    cls._index = faiss.IndexIDMap(faiss.IndexFlatIP(cls._dimension))
            
            return cls._index

    @classmethod
    def _save_index(cls) -> None:
        """
        Write the current state of the FAISS index to disk.
        """
        if cls._index is not None:
            try:
                faiss.write_index(cls._index, cls._index_path)
                logger.info("Saved FAISS index to disk. Total vectors: %d", cls._index.ntotal)
            except Exception as e:
                logger.error("Failed to write FAISS index to disk: %s", e)
                raise IOError(f"Failed to save FAISS index: {e}") from e

    @classmethod
    def add_vectors(cls, ids: List[int], embeddings: List[List[float]]) -> None:
        """
        Add normalized vectors to the FAISS index and save it to disk.

        Args:
            ids (List[int]): Unique integer IDs mapping to SQLite chunk vector_ids.
            embeddings (List[List[float]]): Normalized vector embeddings.
        """
        if not ids or not embeddings:
            logger.warning("Empty IDs or embeddings provided to add_vectors.")
            return

        index = cls._get_index()
        
        # Format inputs for FAISS
        ids_np = np.array(ids, dtype=np.int64)
        embeddings_np = np.array(embeddings, dtype=np.float32)

        with cls._lock:
            try:
                logger.info("Adding %d vectors to FAISS index...", len(ids))
                index.add_with_ids(embeddings_np, ids_np)
                cls._save_index()
            except Exception as e:
                logger.error("Failed to add vectors to FAISS index: %s", e)
                raise RuntimeError(f"FAISS add_vectors failed: {e}") from e

    @classmethod
    def remove_vectors(cls, ids: List[int]) -> None:
        """
        Remove vectors from the FAISS index by their IDs and save it to disk.

        Args:
            ids (List[int]): List of vector IDs to delete.
        """
        if not ids:
            return

        index = cls._get_index()
        ids_np = np.array(ids, dtype=np.int64)

        with cls._lock:
            try:
                logger.info("Removing %d vectors from FAISS index...", len(ids))
                index.remove_ids(ids_np)
                cls._save_index()
            except Exception as e:
                logger.error("Failed to remove vectors from FAISS index: %s", e)
                raise RuntimeError(f"FAISS remove_vectors failed: {e}") from e

    @classmethod
    def semantic_search(
        cls,
        db: Session,
        query: str,
        limit: int = 5,
        filetype: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Perform a semantic vector similarity search against the FAISS index,
        filtering the results by database metadata.
        """
        index = cls._get_index()
        if index.ntotal == 0:
            logger.warning("FAISS search called but the index is empty.")
            return []

        # 1. Generate query embedding
        query_vector = EmbeddingService.get_embedding(query)
        query_np = np.array([query_vector], dtype=np.float32)

        # 2. Search FAISS index (request double the limit to account for post-query SQL metadata filtering)
        search_limit = max(limit * 3, 50)
        distances, indices = index.search(query_np, search_limit)
        
        # FAISS returns a 2D array, get first row
        hits_ids = indices[0].tolist()
        hits_scores = distances[0].tolist()

        # Filter out invalid indices (-1 indicates no match found in FAISS)
        valid_hits = [(vector_id, score) for vector_id, score in zip(hits_ids, hits_scores) if vector_id != -1]
        if not valid_hits:
            return []

        vector_ids = [hit[0] for hit in valid_hits]
        score_map = {hit[0]: hit[1] for hit in valid_hits}

        # 3. Retrieve chunks from database, joining files for metadata
        query_db = db.query(ChunkModel).join(FileModel).filter(ChunkModel.vector_id.in_(vector_ids))
        
        # Apply metadata filters
        if filetype:
            query_db = query_db.filter(FileModel.filetype == filetype.lower())
        if start_date:
            query_db = query_db.filter(FileModel.created_at >= start_date)
        if end_date:
            query_db = query_db.filter(FileModel.created_at <= end_date)

        chunks = query_db.all()

        # 4. Map SQL rows back to similarity scores and format results
        results = []
        for chunk in chunks:
            score = score_map.get(chunk.vector_id, 0.0)
            # Clip cosine similarity to range [0.0, 1.0] for readability
            score_percentage = float(max(min(score, 1.0), 0.0))
            
            if score_percentage < min_score:
                continue
                
            results.append({
                "chunk_id": chunk.id,
                "filename": chunk.file.filename,
                "filepath": chunk.file.filepath,
                "filetype": chunk.file.filetype,
                "page_number": chunk.page_number,
                "chunk_text": chunk.chunk_text,
                "score": score_percentage,
                "created_at": chunk.file.created_at
            })

        # Sort results descending by similarity score and truncate
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    @classmethod
    def keyword_search(
        cls,
        db: Session,
        query: str,
        limit: int = 5,
        filetype: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform a traditional keyword search using SQL LIKE queries
        on the chunk text.
        """
        if not query or not query.strip():
            return []

        search_pattern = f"%{query.strip()}%"
        query_db = db.query(ChunkModel).join(FileModel).filter(ChunkModel.chunk_text.like(search_pattern))

        # Apply metadata filters
        if filetype:
            query_db = query_db.filter(FileModel.filetype == filetype.lower())
        if start_date:
            query_db = query_db.filter(FileModel.created_at >= start_date)
        if end_date:
            query_db = query_db.filter(FileModel.created_at <= end_date)

        chunks = query_db.limit(limit * 2).all()

        results = []
        for chunk in chunks:
            # For keyword search, similarity score is assigned based on keyword presence
            # Return baseline score of 0.8 for keyword occurrences
            results.append({
                "chunk_id": chunk.id,
                "filename": chunk.file.filename,
                "filepath": chunk.file.filepath,
                "filetype": chunk.file.filetype,
                "page_number": chunk.page_number,
                "chunk_text": chunk.chunk_text,
                "score": 0.8,
                "created_at": chunk.file.created_at
            })
            
        return results[:limit]

    @classmethod
    def hybrid_search(
        cls,
        db: Session,
        query: str,
        limit: int = 5,
        filetype: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Combine Semantic Search and Keyword Search scores to produce a ranked hybrid result list.
        """
        # 1. Fetch semantic results
        semantic_results = cls.semantic_search(
            db=db,
            query=query,
            limit=limit * 2,
            filetype=filetype,
            start_date=start_date,
            end_date=end_date
        )

        # 2. Fetch keyword results
        keyword_results = cls.keyword_search(
            db=db,
            query=query,
            limit=limit * 2,
            filetype=filetype,
            start_date=start_date,
            end_date=end_date
        )

        # 3. Merge results using chunk_id
        merged_map: Dict[int, Dict[str, Any]] = {}
        
        # Process semantic hits (weight: 0.7)
        for idx, item in enumerate(semantic_results):
            chunk_id = item["chunk_id"]
            # Reciprocal rank weighting can also be used, but direct linear score combinations
            # are easier to interpret for user interfaces.
            item["score"] = item["score"] * 0.7
            merged_map[chunk_id] = item

        # Process keyword hits (weight: 0.3)
        for idx, item in enumerate(keyword_results):
            chunk_id = item["chunk_id"]
            if chunk_id in merged_map:
                # Add weighting for hybrid overlap (score + 0.3 * keyword score)
                merged_map[chunk_id]["score"] += item["score"] * 0.3
            else:
                item["score"] = item["score"] * 0.3
                merged_map[chunk_id] = item

        # Sort merged results by final composite score descending
        hybrid_results = list(merged_map.values())
        hybrid_results.sort(key=lambda x: x["score"], reverse=True)

        return hybrid_results[:limit]
