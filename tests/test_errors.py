"""Tests for clipforge.errors."""
from clipforge.errors import (
    ClipForgeError,
    AudioExtractionError,
    TranscriptionError,
    SegmentationError,
    ValidationError,
    ExtractionError,
    PipelineError,
)


def test_all_errors_inherit_from_base():
    for cls in [
        AudioExtractionError,
        TranscriptionError,
        SegmentationError,
        ValidationError,
        ExtractionError,
        PipelineError,
    ]:
        exc = cls("test message")
        assert isinstance(exc, ClipForgeError)
        assert str(exc) == "test message"


def test_base_error_is_exception():
    assert issubclass(ClipForgeError, Exception)
