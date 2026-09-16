import os
import logging
from PIL import Image
import pytesseract
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Configure pytesseract path if specified in environment variables
tesseract_cmd = os.getenv("TESSERACT_CMD")
if not tesseract_cmd and os.name == "nt":
    tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if tesseract_cmd and os.path.exists(tesseract_cmd):
    logger.info("Configuring Tesseract command path: %s", tesseract_cmd)
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


class ImageService:
    """
    Service class handling text extraction from image files using Tesseract OCR.
    """

    @staticmethod
    def extract_text(file_path: str) -> str:
        """
        Extract text from an image file using Optical Character Recognition (OCR).

        Args:
            file_path (str): The absolute path to the image file.

        Returns:
            str: The extracted text content.

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If Tesseract OCR is not installed/configured or image is corrupted.
        """
        if not os.path.exists(file_path):
            logger.error("Image file not found: %s", file_path)
            raise FileNotFoundError(f"Image file not found at {file_path}")

        logger.info("Starting OCR text extraction from image: %s", file_path)

        try:
            # Load the image
            with Image.open(file_path) as img:
                # Perform OCR extraction
                text = pytesseract.image_to_string(img)
                
                # Basic cleaning of multiple spaces and newlines
                cleaned_text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
                
                logger.info("Successfully extracted %d characters from image: %s", len(cleaned_text), file_path)
                return cleaned_text
                
        except pytesseract.TesseractNotFoundError as e:
            msg = (
                "Tesseract-OCR binary was not found. Please ensure Tesseract is installed "
                f"on your system and that TESSERACT_CMD in your .env file is set to the correct path. "
                f"Current config: TESSERACT_CMD={tesseract_cmd}"
            )
            logger.error(msg)
            raise ValueError(msg) from e
        except Exception as e:
            logger.error("An error occurred during OCR execution on image %s: %s", file_path, e)
            raise ValueError(f"Failed to extract text from image: {e}") from e
