"""Custom exception hierarchy for ClipForge."""


class ClipForgeError(Exception):
    """Base exception for all ClipForge errors."""


class AudioExtractionError(ClipForgeError):
    """Raised when audio extraction from video fails."""


class TranscriptionError(ClipForgeError):
    """Raised when audio transcription fails."""


class SegmentationError(ClipForgeError):
    """Raised when topic segmentation fails."""


class ValidationError(ClipForgeError):
    """Raised when clip validation fails."""


class ExtractionError(ClipForgeError):
    """Raised when clip extraction fails."""


class PipelineError(ClipForgeError):
    """Raised when the pipeline orchestration fails."""
