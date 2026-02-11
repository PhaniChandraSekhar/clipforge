"""Tests for clipforge.transcribe."""
import sys
from unittest.mock import patch, MagicMock

import pytest

from clipforge.errors import TranscriptionError


def _mock_whisper_result():
    """Return a mock Whisper transcription result dict."""
    return {
        "language": "en",
        "segments": [
            {
                "text": " Hello world.",
                "start": 0.0,
                "end": 2.0,
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "probability": 0.95},
                    {"word": "world.", "start": 0.5, "end": 1.0, "probability": 0.90},
                ],
            },
            {
                "text": " This is a test.",
                "start": 2.0,
                "end": 4.0,
                "words": [
                    {"word": "This", "start": 2.0, "end": 2.3, "probability": 0.88},
                    {"word": "is", "start": 2.3, "end": 2.5, "probability": 0.91},
                    {"word": "a", "start": 2.5, "end": 2.6, "probability": 0.85},
                    {"word": "test.", "start": 2.6, "end": 3.0, "probability": 0.93},
                ],
            },
        ],
    }


def test_transcribe_audio_success(tmp_audio, pipeline_config):
    mock_model = MagicMock()
    mock_model.transcribe.return_value = _mock_whisper_result()

    mock_whisper = MagicMock()
    mock_whisper.load_model.return_value = mock_model

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict(sys.modules, {"whisper": mock_whisper, "torch": mock_torch}):
        # Re-import to pick up mocked modules
        from clipforge.transcribe import transcribe_audio

        result = transcribe_audio(tmp_audio, pipeline_config)

        assert result.language == "en"
        assert len(result.segments) == 2
        assert result.segments[0].text == "Hello world."
        assert len(result.segments[0].words) == 2
        assert result.duration == 4.0


def test_transcribe_audio_with_language(tmp_audio, pipeline_config):
    pipeline_config.language = "en"
    mock_model = MagicMock()
    mock_model.transcribe.return_value = _mock_whisper_result()

    mock_whisper = MagicMock()
    mock_whisper.load_model.return_value = mock_model

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict(sys.modules, {"whisper": mock_whisper, "torch": mock_torch}):
        from clipforge.transcribe import transcribe_audio

        result = transcribe_audio(tmp_audio, pipeline_config)
        call_kwargs = mock_model.transcribe.call_args[1]
        assert call_kwargs["language"] == "en"


def test_transcribe_audio_failure(tmp_audio, pipeline_config):
    mock_whisper = MagicMock()
    mock_whisper.load_model.side_effect = RuntimeError("model load failed")

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with patch.dict(sys.modules, {"whisper": mock_whisper, "torch": mock_torch}):
        from clipforge.transcribe import transcribe_audio

        with pytest.raises(TranscriptionError, match="Transcription failed"):
            transcribe_audio(tmp_audio, pipeline_config)
