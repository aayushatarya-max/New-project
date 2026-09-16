import os
import pytest
from unittest.mock import patch, MagicMock
from backend.services.pdf_service import PDFService


def test_pdf_extraction_success(tmp_path):
    """
    Test extraction of text from a multi-page PDF document.
    """
    pdf_path = os.path.join(tmp_path, "sample_doc.pdf")
    # Create an empty file to pass the os.path.exists check
    with open(pdf_path, "w") as f:
        f.write("%PDF-1.4 dummy contents")

    with patch("fitz.open") as mock_open:
        mock_doc = MagicMock()
        
        # Configure mock page 1
        mock_page1 = MagicMock()
        mock_page1.number = 0
        mock_page1.get_text.return_value = "This is page one text content."
        
        # Configure mock page 2
        mock_page2 = MagicMock()
        mock_page2.number = 1
        mock_page2.get_text.return_value = "This is page two text content."
        
        # Configure doc iterator
        mock_doc.__iter__.return_value = [mock_page1, mock_page2]
        mock_doc.__len__.return_value = 2
        
        # Configure context manager return value
        mock_open.return_value.__enter__.return_value = mock_doc
        
        pages = PDFService.extract_text(pdf_path)
        
        assert len(pages) == 2
        assert pages[0]["page_number"] == 1
        assert pages[0]["text"] == "This is page one text content."
        assert pages[1]["page_number"] == 2
        assert pages[1]["text"] == "This is page two text content."
        
    if os.path.exists(pdf_path):
        os.remove(pdf_path)


def test_pdf_extraction_file_not_found():
    """
    Verify FileNotFoundError is raised for non-existent paths.
    """
    with pytest.raises(FileNotFoundError):
        PDFService.extract_text("non_existent_file.pdf")


def test_pdf_extraction_invalid_file(tmp_path):
    """
    Verify ValueError is raised for corrupted files when fitz throws an exception.
    """
    bad_file = os.path.join(tmp_path, "corrupted.pdf")
    with open(bad_file, "w") as f:
        f.write("Not a pdf file at all")
        
    with patch("fitz.open", side_effect=Exception("Corrupted PDF")):
        with pytest.raises(ValueError):
            PDFService.extract_text(bad_file)
        
    if os.path.exists(bad_file):
        os.remove(bad_file)
