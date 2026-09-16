import os
import logging
from typing import Optional, Any
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class SpeechService:
    """
    Service class handling audio transcription using OpenAI Whisper.
    Uses a singleton/lazy loading pattern for the model to conserve memory.
    """

    _model: Optional[Any] = None
    _model_name: str = os.getenv("WHISPER_MODEL_NAME", "base")

    @classmethod
    def _get_model(cls) -> Any:
        """
        Lazily load the Whisper model from memory or download it if not present.
        Ensures thread-safe access by utilizing class attributes.
        """
        if cls._model is None:
            try:
                import whisper
            except ImportError as e:
                logger.error("Failed to import whisper: %s", e)
                raise RuntimeError("Audio transcription is disabled because 'whisper' or its dependencies (like PyTorch/C++) are not installed properly.") from e

            logger.info("Loading Whisper model '%s' (this may take a few seconds)...", cls._model_name)
            try:
                cls._model = whisper.load_model(cls._model_name)
                logger.info("Whisper model '%s' loaded successfully.", cls._model_name)
            except Exception as e:
                logger.error("Failed to load Whisper model '%s': %s", cls._model_name, e)
                raise RuntimeError(f"Whisper model failed to initialize: {e}") from e
        return cls._model

    @classmethod
    def transcribe(cls, file_path: str) -> str:
        """
        Transcribe an audio file using OpenAI Whisper.

        Args:
            file_path (str): The absolute path to the audio file.

        Returns:
            str: The transcribed text.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            ValueError: If transcription fails.
        """
        if not os.path.exists(file_path):
            logger.error("Audio file not found: %s", file_path)
            raise FileNotFoundError(f"Audio file not found at {file_path}")

        logger.info("Starting transcription for audio file: %s", file_path)

        try:
            # Retrieve model (triggering lazy loading if first call)
            model = cls._get_model()
            
            # Perform transcription
            # fp16=False forces CPU transcription since we're using faiss-cpu and local CPU runs
            result = model.transcribe(file_path, fp16=False)
            
            transcription_text = result.get("text", "").strip()
            logger.info("Successfully transcribed audio file: %s. Output length: %d characters", 
                        file_path, len(transcription_text))
            return transcription_text
            
        except RuntimeError as e:
            raise ValueError(str(e)) from e
        except Exception as e:
            logger.error("An error occurred during transcription of file %s: %s", file_path, e)
            raise ValueError(f"Failed to transcribe audio file: {e}") from e
