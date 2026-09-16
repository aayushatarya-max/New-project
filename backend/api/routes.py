import os
import logging
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.repositories.file_repository import FileRepository
from backend.services.pdf_service import PDFService
from backend.services.image_service import ImageService
from backend.services.speech_service import SpeechService
from backend.services.chunk_service import ChunkService
from backend.services.embedding_service import EmbeddingService
from backend.services.search_service import SearchService
from backend.models.models import ChunkModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads/")
SUPPORTED_EXTENSIONS = {
    # Documents
    ".txt": "txt",
    ".pdf": "pdf",
    # Images
    ".png": "png",
    ".jpg": "jpg",
    ".jpeg": "jpeg",
    # Audio
    ".mp3": "mp3",
    ".wav": "wav",
    ".m4a": "m4a"
}


@router.post("/upload", status_code=status.HTTP_201_CREATED)
def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Ingest a new file, extract text content based on format,
    generate sliding window chunks and embeddings, and index into FAISS.
    """
    filename = file.filename
    _, ext = os.path.splitext(filename.lower())
    
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Supported formats: {list(SUPPORTED_EXTENSIONS.keys())}"
        )

    # 1. Ensure upload directory exists and save file physically
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Check if file already exists in database
    existing = FileRepository.get_file_by_filepath(db, file_path)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{filename}' already exists and is indexed."
        )

    try:
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())
        logger.info("Saved raw file to: %s", file_path)
    except Exception as e:
        logger.error("Failed to save uploaded file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save uploaded file: {e}"
        )

    file_size = os.path.getsize(file_path)
    file_type = SUPPORTED_EXTENSIONS[ext]

    # 2. Extract text page-by-page based on type
    pages = []
    try:
        if file_type == "txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            pages = [{"page_number": 1, "text": content}]
            
        elif file_type == "pdf":
            pages = PDFService.extract_text(file_path)
            
        elif file_type in ["png", "jpg", "jpeg"]:
            text = ImageService.extract_text(file_path)
            if not text or not text.strip():
                logger.info("No OCR text found in image '%s'. Using metadata fallback.", filename)
                clean_name = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
                text = f"Image file: {filename}. Photo / graphic document ({clean_name})."
            pages = [{"page_number": 1, "text": text}]
            
        elif file_type in ["mp3", "wav", "m4a"]:
            text = SpeechService.transcribe(file_path)
            if not text or not text.strip():
                logger.info("No speech transcription found in audio '%s'. Using metadata fallback.", filename)
                clean_name = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ")
                text = f"Audio file: {filename}. Voice / sound recording ({clean_name})."
            pages = [{"page_number": 1, "text": text}]
            
    except Exception as e:
        # Clean up file on extraction failure
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error("Failed to extract text from file %s: %s", filename, e)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process text extraction: {e}"
        )

    # Validate that we extracted some content
    total_text = " ".join([page["text"] for page in pages]).strip()
    if not total_text:
        # Clean up file
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File contains no extractable text content."
        )

    # 3. Create database records
    try:
        db_file = FileRepository.create_file(
            db=db,
            filename=filename,
            filepath=file_path,
            filetype=file_type,
            filesize=file_size
        )

        # Create chunks
        chunks_data = ChunkService.chunk_document(pages)
        if not chunks_data:
            raise ValueError("No text chunks generated.")

        # Save chunks in SQLite first (without vector_ids)
        db_chunks = []
        for idx, chunk in enumerate(chunks_data):
            db_chunk = ChunkModel(
                file_id=db_file.id,
                chunk_text=chunk["chunk_text"],
                page_number=chunk["page_number"],
                chunk_order=chunk["chunk_order"],
                vector_id=None  # Set later
            )
            db_chunks.append(db_chunk)
        db.add_all(db_chunks)
        db.commit()

        # Update vector_id to match chunk database ID
        for chunk in db_chunks:
            chunk.vector_id = chunk.id
        db.commit()

        # 4. Generate embeddings and index in FAISS
        chunk_texts = [c.chunk_text for c in db_chunks]
        embeddings = EmbeddingService.get_embeddings(chunk_texts)
        
        vector_ids = [c.vector_id for c in db_chunks]
        SearchService.add_vectors(vector_ids, embeddings)

        # 5. Create automatic search tags
        tags = [file_type, ext.replace(".", "")]
        if file_type in ["png", "jpg", "jpeg"]:
            tags.append("image")
            tags.append("ocr")
        elif file_type in ["mp3", "wav", "m4a"]:
            tags.append("audio")
            tags.append("voice")
            tags.append("whisper")
        elif file_type == "pdf":
            tags.append("document")
            tags.append("pdf")
        elif file_type == "txt":
            tags.append("note")
            tags.append("text")
        
        # Add tags from file name keywords
        words = [w.strip().lower() for w in filename.replace("_", " ").replace("-", " ").split(".") if w.strip()]
        tags.extend([w for w in words if len(w) > 2 and w not in ["pdf", "txt", "mp3", "wav", "png", "jpg"]])

        FileRepository.create_tags(db, db_file.id, list(set(tags)))

        logger.info("Ingestion completed successfully for file: %s", filename)
        return {
            "message": "File uploaded and indexed successfully",
            "file_id": db_file.id,
            "filename": filename,
            "chunks_indexed": len(db_chunks)
        }

    except Exception as e:
        # DB Rollback and cleanup
        db.rollback()
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error("Failed to index file %s in database/FAISS: %s", filename, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save index metadata: {e}"
        )


@router.get("/search")
def search_memories(
    query: str = Query(..., min_length=1),
    type: str = Query("hybrid", pattern="^(hybrid|semantic|keyword)$"),
    limit: int = Query(5, ge=1, le=50),
    filetype: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Search indexed memory segments using Semantic, Keyword, or Hybrid search.
    """
    # Parse dates if supplied
    parsed_start = None
    parsed_end = None
    
    try:
        if start_date:
            parsed_start = datetime.fromisoformat(start_date)
        if end_date:
            parsed_end = datetime.fromisoformat(end_date)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date format. Use ISO format (YYYY-MM-DD): {e}"
        )

    if type == "semantic":
        results = SearchService.semantic_search(
            db=db, query=query, limit=limit, filetype=filetype, start_date=parsed_start, end_date=parsed_end
        )
    elif type == "keyword":
        results = SearchService.keyword_search(
            db=db, query=query, limit=limit, filetype=filetype, start_date=parsed_start, end_date=parsed_end
        )
    else:
        results = SearchService.hybrid_search(
            db=db, query=query, limit=limit, filetype=filetype, start_date=parsed_start, end_date=parsed_end
        )

    return results


@router.get("/files")
def list_indexed_files(
    filetype: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    List all uploaded files and metadata.
    """
    files = FileRepository.list_files(db, filetype=filetype)
    return [
        {
            "id": f.id,
            "filename": f.filename,
            "filepath": f.filepath,
            "filetype": f.filetype,
            "filesize": f.filesize,
            "created_at": f.created_at,
            "tags": [t.tag for t in f.tags]
        }
        for f in files
    ]


@router.delete("/files/{file_id}")
def delete_indexed_file(
    file_id: int,
    db: Session = Depends(get_db)
):
    """
    De-index and physically delete a file.
    """
    db_file = FileRepository.get_file(db, file_id)
    if not db_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File with ID {file_id} not found."
        )

    # 1. Fetch chunks to get vector_ids
    vector_ids = [chunk.vector_id for chunk in db_file.chunks if chunk.vector_id is not None]

    try:
        # 2. Remove vectors from FAISS index
        SearchService.remove_vectors(vector_ids)

        # 3. Delete metadata records from database (cascades)
        FileRepository.delete_file(db, file_id)

        # 4. Delete file from physical disk storage
        if os.path.exists(db_file.filepath):
            os.remove(db_file.filepath)
            logger.info("Deleted physical file from: %s", db_file.filepath)

        return {"message": f"Successfully deleted and de-indexed file '{db_file.filename}'."}
        
    except Exception as e:
        logger.error("Failed to delete file ID %d: %s", file_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete indexed file: {e}"
        )


@router.get("/files/{file_id}/download")
def download_file(
    file_id: int,
    db: Session = Depends(get_db)
):
    """
    Serve raw file for direct viewing or downloading.
    """
    from fastapi.responses import FileResponse
    db_file = FileRepository.get_file(db, file_id)
    if not db_file or not os.path.exists(db_file.filepath):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File with ID {file_id} not found on disk."
        )
    return FileResponse(
        path=db_file.filepath,
        filename=db_file.filename
    )

