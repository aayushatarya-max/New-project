import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ChunkService:
    """
    Service class handling text chunking logic with sliding window overlaps
    and word-boundary preservation.
    """

    @staticmethod
    def split_text_by_chars(
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ) -> List[str]:
        """
        Split a single string into overlapping chunks, avoiding splitting words
        where possible.

        Args:
            text (str): The raw input text.
            chunk_size (int): Target maximum characters per chunk.
            chunk_overlap (int): Target character overlap between consecutive chunks.

        Returns:
            List[str]: List of text chunks.
        """
        if not text or not text.strip():
            return []

        # Standardize whitespace
        text = " ".join(text.split())
        text_len = len(text)

        if text_len <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < text_len:
            end = start + chunk_size

            # If we are not at the end of the text, look back to find the nearest space
            if end < text_len:
                # Find last space in the range [end - 30, end] to avoid clipping words
                space_idx = text.rfind(" ", end - 30, end)
                if space_idx != -1:
                    end = space_idx

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start pointer forward. Step back by overlap, but ensure we progress
            next_start = end - chunk_overlap
            if next_start <= start:
                next_start = end  # Force progress if overlap size is too large
            start = next_start

        return chunks

    @classmethod
    def chunk_document(
        cls,
        pages: List[Dict[str, Any]],
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Chunk a multi-page document while preserving page context and ordering.

        Args:
            pages (List[Dict[str, Any]]): List of dicts, each containing:
                                          - 'page_number': int
                                          - 'text': str
            chunk_size (int): Maximum character length of each chunk.
            chunk_overlap (int): Overlap character length.

        Returns:
            List[Dict[str, Any]]: List of structured chunks:
                                  [
                                      {
                                          "chunk_text": str,
                                          "page_number": int,
                                          "chunk_order": int
                                      },
                                      ...
                                  ]
        """
        logger.info("Chunking document containing %d pages. Size: %d, Overlap: %d", 
                    len(pages), chunk_size, chunk_overlap)
        
        chunks = []
        chunk_order = 1

        for page in pages:
            page_num = page.get("page_number")
            text = page.get("text", "")
            
            # Split the text on this page
            page_chunks = cls.split_text_by_chars(
                text=text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            for chunk_text in page_chunks:
                chunks.append({
                    "chunk_text": chunk_text,
                    "page_number": page_num,
                    "chunk_order": chunk_order
                })
                chunk_order += 1

        logger.info("Document chunking complete. Generated %d chunks.", len(chunks))
        return chunks
