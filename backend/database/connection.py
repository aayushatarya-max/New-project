import os
import logging
from typing import Generator
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Load environment configuration
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///backend/database/memory.db")

# SQLite needs 'check_same_thread=False' for multi-threaded access within FastAPI
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False
    # Extract path and ensure containing folder exists
    db_path = DATABASE_URL.replace("sqlite:///", "")
    dir_name = os.path.dirname(db_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        logger.info("Created database directory: %s", dir_name)

try:
    logger.info("Initializing database engine with URL: %s", DATABASE_URL)
    engine = create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        echo=False  # Set to True to log executed SQL queries for debugging
    )
    
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )
    
    # Declarative Base for models
    Base = declarative_base()
    
    logger.info("Database engine and session factory created successfully.")
except Exception as e:
    logger.exception("Failed to initialize database connection: %s", e)
    raise e


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session and ensures it is
    properly closed after request lifecycle completes.

    Yields:
        Session: The SQLAlchemy database session.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error("Exception encountered in database session context: %s", e)
        db.rollback()
        raise e
    finally:
        db.close()
