import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from backend.database.connection import Base, engine
from backend.api.routes import router

# Set up logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan context manager.
    Triggers database schema creation on server startup.
    """
    logger.info("Starting up FastAPI Personal Memory Engine...")
    try:
        # Create database tables if they do not exist
        logger.info("Initializing SQLite database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.exception("Failed to initialize database tables: %s", e)
        raise e
        
    yield
    
    logger.info("Shutting down FastAPI Personal Memory Engine...")


app = FastAPI(
    title="Personal Memory Search Engine API",
    description="Intelligent desktop semantic search backend powered by FastAPI, SQLite, and FAISS.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware configuration to allow frontend Streamlit requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local desktop deployment, * is safe and convenient
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory if missing
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads/")
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount uploads directory as static directory to serve indexed files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Include api routes
app.include_router(router)


@app.get("/")
def read_root():
    """
    Root status check endpoint.
    """
    return {
        "status": "online",
        "app": "Personal Memory Search Engine API",
        "version": "1.0.0"
    }


@app.exception_handler(Exception)
def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all global exception handler to prevent leak of stacktraces and log details.
    """
    logger.error("Unhandled global exception encountered: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please check logs."}
    )
