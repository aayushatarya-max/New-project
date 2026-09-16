import logging
from datetime import datetime
from typing import List, Optional
from sqlalchemy.orm import Session
from backend.models.models import FileModel, ChunkModel, TagModel

logger = logging.getLogger(__name__)


class FileRepository:
    """
    Repository class handling SQLite database CRUD operations
    for Files, Chunks, and Tags.
    """

    @staticmethod
    def create_file(
        db: Session,
        filename: str,
        filepath: str,
        filetype: str,
        filesize: int
    ) -> FileModel:
        """
        Create a new file record in the database.
        """
        try:
            db_file = FileModel(
                filename=filename,
                filepath=filepath,
                filetype=filetype,
                filesize=filesize
            )
            db.add(db_file)
            db.commit()
            db.refresh(db_file)
            logger.info("Created file record in database. ID: %d, path: %s", db_file.id, filepath)
            return db_file
        except Exception as e:
            db.rollback()
            logger.error("Failed to create file record for %s: %s", filepath, e)
            raise e

    @staticmethod
    def get_file(db: Session, file_id: int) -> Optional[FileModel]:
        """
        Retrieve a file record by its ID.
        """
        return db.query(FileModel).filter(FileModel.id == file_id).first()

    @staticmethod
    def get_file_by_filepath(db: Session, filepath: str) -> Optional[FileModel]:
        """
        Retrieve a file record by its absolute file path.
        """
        return db.query(FileModel).filter(FileModel.filepath == filepath).first()

    @staticmethod
    def delete_file(db: Session, file_id: int) -> bool:
        """
        Delete a file record. Triggers cascading delete on related chunks and tags.
        """
        try:
            db_file = db.query(FileModel).filter(FileModel.id == file_id).first()
            if db_file:
                db.delete(db_file)
                db.commit()
                logger.info("Deleted file ID %d from database along with its chunks and tags.", file_id)
                return True
            logger.warning("Attempted to delete non-existent file ID %d.", file_id)
            return False
        except Exception as e:
            db.rollback()
            logger.error("Failed to delete file ID %d: %s", file_id, e)
            raise e

    @staticmethod
    def list_files(
        db: Session,
        filetype: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[FileModel]:
        """
        List files with optional filters for file type and creation date range.
        """
        query = db.query(FileModel)
        if filetype:
            query = query.filter(FileModel.filetype == filetype.lower())
        if start_date:
            query = query.filter(FileModel.created_at >= start_date)
        if end_date:
            query = query.filter(FileModel.created_at <= end_date)
        return query.order_by(FileModel.created_at.desc()).all()

    @staticmethod
    def create_chunks(
        db: Session,
        file_id: int,
        chunks_data: List[dict]
    ) -> List[ChunkModel]:
        """
        Bulk insert chunk records for a file.
        
        Args:
            db: SQLAlchemy session
            file_id: The ID of the file these chunks belong to
            chunks_data: List of dicts, each having:
                         - 'chunk_text': str
                         - 'page_number': Optional[int]
                         - 'chunk_order': int
                         - 'vector_id': Optional[int]
        """
        try:
            chunks = []
            for data in chunks_data:
                chunk = ChunkModel(
                    file_id=file_id,
                    chunk_text=data["chunk_text"],
                    page_number=data.get("page_number"),
                    chunk_order=data["chunk_order"],
                    vector_id=data.get("vector_id")
                )
                chunks.append(chunk)
            db.bulk_save_objects(chunks)
            db.commit()
            logger.info("Bulk inserted %d chunks for file ID %d.", len(chunks), file_id)
            
            # Retrieve the created chunks to return them
            return db.query(ChunkModel).filter(ChunkModel.file_id == file_id).order_by(ChunkModel.chunk_order).all()
        except Exception as e:
            db.rollback()
            logger.error("Failed to bulk insert chunks for file ID %d: %s", file_id, e)
            raise e

    @staticmethod
    def create_tags(
        db: Session,
        file_id: int,
        tags: List[str]
    ) -> List[TagModel]:
        """
        Bulk insert tags for a file.
        """
        try:
            tag_objects = [TagModel(file_id=file_id, tag=t.strip().lower()) for t in tags if t.strip()]
            db.bulk_save_objects(tag_objects)
            db.commit()
            logger.info("Inserted %d tags for file ID %d.", len(tag_objects), file_id)
            return db.query(TagModel).filter(TagModel.file_id == file_id).all()
        except Exception as e:
            db.rollback()
            logger.error("Failed to insert tags for file ID %d: %s", file_id, e)
            raise e
