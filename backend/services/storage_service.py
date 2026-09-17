import os
import logging
import platform
import subprocess
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "memory-files").strip()
APP_ENV = os.getenv("APP_ENV", "local").strip().lower()

_supabase_client = None


def get_supabase_client():
    """Lazy initializer for Supabase client."""
    global _supabase_client
    if _supabase_client is None and SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        try:
            from supabase import create_client
            _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            logger.info("Initialized Supabase Storage client successfully.")
        except Exception as e:
            logger.warning("Could not initialize Supabase client: %s", e)
            _supabase_client = None
    return _supabase_client


class StorageService:
    """
    Unified Storage Service managing file uploads, local desktop launching,
    and cloud storage via Supabase Storage for Render deployment.
    """

    @classmethod
    def is_live_mode(cls) -> bool:
        """Determines if the application is running in Live Cloud mode."""
        client = get_supabase_client()
        return APP_ENV == "live" or (client is not None)

    @classmethod
    def save_uploaded_file(cls, filename: str, content: bytes, upload_dir: str = "uploads/") -> Dict[str, Any]:
        """
        Saves an uploaded file locally and/or to Supabase Storage based on execution mode.

        Returns:
            Dict containing:
              - 'storage_mode': 'local' or 'live'
              - 'filepath': Local path on disk
              - 'storage_path': Remote object path in bucket
              - 'public_url': Public/signed URL if in live mode
        """
        os.makedirs(upload_dir, exist_ok=True)
        local_filepath = os.path.join(upload_dir, filename)

        # 1. Always save locally first for extraction processing
        with open(local_filepath, "wb") as f:
            f.write(content)

        result = {
            "storage_mode": "local",
            "filepath": local_filepath,
            "storage_path": filename,
            "public_url": None
        }

        # 2. If in Live mode or Supabase is configured, upload to Supabase Storage
        client = get_supabase_client()
        if client:
            try:
                # Ensure bucket exists
                try:
                    client.storage.get_bucket(SUPABASE_BUCKET)
                except Exception:
                    logger.info("Creating bucket '%s' in Supabase Storage...", SUPABASE_BUCKET)
                    client.storage.create_bucket(SUPABASE_BUCKET, options={"public": True})

                storage_path = f"uploads/{filename}"
                logger.info("Uploading %s to Supabase Storage bucket '%s'...", filename, SUPABASE_BUCKET)
                
                # Upload or overwrite file in Supabase
                client.storage.from_(SUPABASE_BUCKET).upload(
                    path=storage_path,
                    file=content,
                    file_options={"upsert": "true"}
                )
                
                # Get public URL
                public_url = client.storage.from_(SUPABASE_BUCKET).get_public_url(storage_path)
                result["storage_mode"] = "live"
                result["storage_path"] = storage_path
                result["public_url"] = public_url
                logger.info("Successfully uploaded %s to Supabase Storage: %s", filename, public_url)

            except Exception as e:
                logger.error("Failed to upload %s to Supabase Storage: %s. Falling back to local mode.", filename, e)

        return result

    @classmethod
    def open_local_file(cls, filepath: str) -> bool:
        """
        Safely opens a file or its containing directory on the local Windows OS.
        Returns True on success, False if file is missing or OS unsupported.
        """
        if not os.path.exists(filepath):
            logger.warning("Local file not found at path: %s", filepath)
            return False

        try:
            if platform.system() == "Windows":
                os.startfile(filepath)
                logger.info("Opened local Windows file: %s", filepath)
                return True
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", filepath])
                return True
            else:  # Linux
                subprocess.Popen(["xdg-open", filepath])
                return True
        except Exception as e:
            logger.error("Failed to open local file %s: %s", filepath, e)
            return False

    @classmethod
    def get_public_url(cls, storage_path: str) -> Optional[str]:
        """Fetches the public or signed URL for a file stored in Supabase Storage."""
        client = get_supabase_client()
        if not client or not storage_path:
            return None
        try:
            return client.storage.from_(SUPABASE_BUCKET).get_public_url(storage_path)
        except Exception as e:
            logger.error("Failed to get public URL for %s: %s", storage_path, e)
            return None

    @classmethod
    def sync_faiss_index_to_cloud(cls, local_index_path: str) -> bool:
        """Uploads the current FAISS index file to Supabase Storage for persistent cloud backup."""
        client = get_supabase_client()
        if not client or not os.path.exists(local_index_path):
            return False
        try:
            with open(local_index_path, "rb") as f:
                index_bytes = f.read()
            logger.info("Syncing FAISS index (%d bytes) to Supabase Storage...", len(index_bytes))
            client.storage.from_(SUPABASE_BUCKET).upload(
                path="faiss_index/index.faiss",
                file=index_bytes,
                file_options={"upsert": "true"}
            )
            logger.info("Successfully synced FAISS index to Supabase Storage.")
            return True
        except Exception as e:
            logger.error("Failed to sync FAISS index to Supabase Storage: %s", e)
            return False

    @classmethod
    def restore_faiss_index_from_cloud(cls, local_index_path: str) -> bool:
        """Downloads persistent FAISS index from Supabase Storage on application startup."""
        client = get_supabase_client()
        if not client:
            return False
        try:
            logger.info("Attempting to restore FAISS index from Supabase Storage...")
            data = client.storage.from_(SUPABASE_BUCKET).download("faiss_index/index.faiss")
            if data:
                dir_name = os.path.dirname(local_index_path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                with open(local_index_path, "wb") as f:
                    f.write(data)
                logger.info("Successfully restored FAISS index from Supabase Storage to %s", local_index_path)
                return True
        except Exception as e:
            logger.info("No cloud FAISS index found to restore (or exception occurred): %s", e)
        return False
