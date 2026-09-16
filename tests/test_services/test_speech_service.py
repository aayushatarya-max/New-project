import os
import pytest
from unittest.mock import patch, MagicMock
from backend.services.speech_service import SpeechService


@pytest.fixture(name="dummy_audio")
def fixture_dummy_audio(tmp_path):
    """
    Creates a temporary dummy file to act as an audio file.
    """
    audio_path = os.path.join(tmp_path, "sample_voice.wav")
    with open(audio_path, "wb") as f:
        f.write(b"RIFF....WAVEfmt....data....")  # Dummy audio bytes
        
    yield audio_path
    
    # Clean up
    if os.path.exists(audio_path):
        os.remove(audio_path)


def test_speech_transcription_success(dummy_audio):
    """
    Verify speech service correctly loads the model and calls transcription.
    """
    # Reset singleton model so tests run clean
    SpeechService._model = None
    
    with patch("whisper.load_model") as mock_load:
        # Create a mock Whisper model instance
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  Welcome to the personal memory search engine.  "}
        mock_load.return_value = mock_model
        
        # Run transcription
        transcript = SpeechService.transcribe(dummy_audio)
        
        # Verify calls and output
        mock_load.assert_called_once_with("base")
        mock_model.transcribe.assert_called_once_with(dummy_audio, fp16=False)
        assert transcript == "Welcome to the personal memory search engine."


def test_speech_transcription_singleton_model(dummy_audio):
    """
    Verify that the model is loaded only once (singleton check) across multiple calls.
    """
    SpeechService._model = None
    
    with patch("whisper.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "audio text"}
        mock_load.return_value = mock_model
        
        # Call multiple times
        SpeechService.transcribe(dummy_audio)
        SpeechService.transcribe(dummy_audio)
        
        # Check that load_model was called exactly once
        mock_load.assert_called_once()
        assert mock_model.transcribe.call_count == 2


def test_speech_transcription_file_not_found():
    """
    Verify FileNotFoundError is raised for non-existent audio path files.
    """
    with pytest.raises(FileNotFoundError):
        SpeechService.transcribe("non_existent_audio.wav")
