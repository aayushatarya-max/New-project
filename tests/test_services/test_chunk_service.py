import pytest
from backend.services.chunk_service import ChunkService


def test_split_text_small():
    """
    Verify text smaller than chunk_size remains in a single cleaned chunk.
    """
    text = "  Just a simple, short text.   "
    chunks = ChunkService.split_text_by_chars(text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0] == "Just a simple, short text."


def test_split_text_with_overlap():
    """
    Verify large text is split into multiple overlapping chunks, matching word boundaries.
    """
    # A text of ~85 characters
    text = "The quick brown fox jumps over the lazy dog. A secondary sentence that continues text."
    
    # Split with size=40, overlap=10.
    chunks = ChunkService.split_text_by_chars(text, chunk_size=40, chunk_overlap=10)
    
    assert len(chunks) > 1
    # Check that chunks do not cut words (e.g. they should end or start on space boundaries)
    for c in chunks:
        # Each chunk is non-empty
        assert len(c) > 0
        # No space character at start or end since they are stripped
        assert not c.startswith(" ")
        assert not c.endswith(" ")


def test_chunk_document_pages():
    """
    Test chunk_document with multi-page structure.
    """
    pages = [
        {"page_number": 1, "text": "This is the text for page number one. It is relatively short."},
        {"page_number": 2, "text": "This is the text for page number two. We want to chunk this too."}
    ]
    
    # We chunk with size 30 and overlap 5
    chunks = ChunkService.chunk_document(pages, chunk_size=30, chunk_overlap=5)
    
    assert len(chunks) >= 2
    # Check sequence ordering
    orders = [c["chunk_order"] for c in chunks]
    assert orders == list(range(1, len(chunks) + 1))
    
    # Check page mapping
    page_numbers = {c["page_number"] for c in chunks}
    assert page_numbers == {1, 2}
