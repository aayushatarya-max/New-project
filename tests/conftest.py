import os
import sys
from unittest.mock import MagicMock, mock_open
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.database.connection import Base

# Setup sys.modules mocks if binary DLLs fail to load in this environment
try:
    import fitz
except (ImportError, OSError):
    class MockFileDataError(Exception):
        pass

    mock_fitz = MagicMock()
    mock_fitz.FileDataError = MockFileDataError
    sys.modules['fitz'] = mock_fitz

try:
    import sentence_transformers
except (ImportError, OSError):
    mock_sentrans = MagicMock()
    sys.modules['sentence_transformers'] = mock_sentrans

try:
    import torch
except (ImportError, OSError):
    import importlib.machinery
    mock_torch = MagicMock()
    mock_torch.__spec__ = importlib.machinery.ModuleSpec(name='torch', loader=None)
    sys.modules['torch'] = mock_torch

try:
    import whisper
except (ImportError, OSError):
    mock_whisper = MagicMock()
    sys.modules['whisper'] = mock_whisper


DB_FILE = "test_temp.db"


@pytest.fixture(name="db_session")
def fixture_db_session():
    """
    Fixture providing a clean, isolated file-based SQLite database session
    for each unit test case. Automatically disposes and cleans up the file.
    """
    if os.path.exists(DB_FILE):
        try:
            os.remove(DB_FILE)
        except OSError:
            pass

    engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()  # Release file connection locks
        if os.path.exists(DB_FILE):
            try:
                os.remove(DB_FILE)
            except OSError:
                pass
