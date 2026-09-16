import os
import pytest
from PIL import Image
from unittest.mock import patch
from backend.services.image_service import ImageService
import pytesseract


@pytest.fixture(name="dummy_image")
def fixture_dummy_image(tmp_path):
    """
    Creates a temporary valid image file for testing.
    """
    image_path = os.path.join(tmp_path, "dummy.png")
    # Create a 100x100 white image
    img = Image.new("RGB", (100, 100), color="white")
    img.save(image_path)
    
    yield image_path
    
    # Clean up
    if os.path.exists(image_path):
        os.remove(image_path)


def test_image_extraction_success(dummy_image):
    """
    Verify successful extraction of text when Tesseract returns a string.
    """
    mock_text = "  OCR text from \n  mock image.  "
    with patch("pytesseract.image_to_string", return_value=mock_text) as mock_tess:
        extracted = ImageService.extract_text(dummy_image)
        
        mock_tess.assert_called_once()
        # Verify text cleaning (spaces trimmed, empty lines ignored)
        assert extracted == "OCR text from\nmock image."


def test_image_extraction_file_not_found():
    """
    Verify FileNotFoundError is raised for non-existent image paths.
    """
    with pytest.raises(FileNotFoundError):
        ImageService.extract_text("non_existent_image.png")


def test_image_extraction_tesseract_missing(dummy_image):
    """
    Verify ValueError is raised if Tesseract-OCR is not installed or configured.
    """
    with patch("pytesseract.image_to_string", side_effect=pytesseract.TesseractNotFoundError()):
        with pytest.raises(ValueError) as exc_info:
            ImageService.extract_text(dummy_image)
        
        assert "Tesseract-OCR binary was not found" in str(exc_info.value)
