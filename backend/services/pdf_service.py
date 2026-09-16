import os
import logging
from typing import List, Dict, Any
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class PDFService:
    """
    Service class handling text extraction from PDF files using pypdf.
    """

    @staticmethod
    def extract_text(file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text page-by-page from a PDF document.

        Args:
            file_path (str): The absolute path to the PDF file.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing:
                                  - 'page_number': int
                                  - 'text': str (cleaned text of the page)

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is corrupted or not a valid PDF.
        """
        if not os.path.exists(file_path):
            logger.error("PDF file not found: %s", file_path)
            raise FileNotFoundError(f"PDF file not found at {file_path}")

        logger.info("Starting text extraction from PDF: %s", file_path)
        extracted_pages = []

        try:
            # Open PDF document
            reader = PdfReader(file_path)
            logger.info("Successfully opened PDF %s. Total pages: %d", file_path, len(reader.pages))
            
            for i, page in enumerate(reader.pages):
                page_number = i + 1
                # Extract text
                text = page.extract_text() or ""
                
                # Basic whitespace cleaning
                cleaned_text = " ".join(text.split()).strip()
                
                # We store pages even if text is empty (useful for tracking empty pages)
                extracted_pages.append({
                    "page_number": page_number,
                    "text": cleaned_text
                })
                
            logger.info("Successfully extracted text from %d pages in PDF: %s", len(extracted_pages), file_path)
            return extracted_pages
            
        except Exception as e:
            logger.error("An error occurred while processing PDF %s: %s", file_path, e)
            raise ValueError(f"Failed to extract text from PDF: {e}") from e
