"""Audio transcription using OpenAI Whisper."""
from __future__ import annotations

import logging
from pathlib import Path
from statistics import mean

from clipforge.config import PipelineConfig
from clipforge.errors import TranscriptionError
from clipforge.models import TranscriptionResult, TranscriptionSegment, TranscriptionWord

logger = logging.getLogger("clipforge")


def transcribe_audio(audio_path: Path, config: PipelineConfig) -> TranscriptionResult:
    """Transcribe an audio file using Whisper and return structured results.

    Args:
        audio_path: Path to the audio file to transcribe.
        config: Pipeline configuration with model and device settings.

    Returns:
        A TranscriptionResult containing segments, words, and metadata.

    Raises:
        TranscriptionError: If transcription fails for any reason.
    """
    try:
        import torch
        import whisper

        # Resolve device
        if config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = config.device

        logger.info("Loading Whisper model '%s' on device '%s'", config.whisper_model, device)
        model = whisper.load_model(config.whisper_model, device=device)

        # Build transcribe kwargs
        transcribe_kwargs: dict = {
            "word_timestamps": True,
        }
        if config.language is not None:
            transcribe_kwargs["language"] = config.language

        logger.info("Starting transcription of '%s'", audio_path)
        result = model.transcribe(str(audio_path), **transcribe_kwargs)

        detected_language: str = result["language"]
        logger.info("Detected language: %s", detected_language)

        segments: list[TranscriptionSegment] = []
        for seg in result["segments"]:
            words: list[TranscriptionWord] = []
            for w in seg.get("words", []):
                words.append(
                    TranscriptionWord(
                        word=w["word"],
                        start=w["start"],
                        end=w["end"],
                        confidence=w.get("probability", 0.0),
                    )
                )

            avg_confidence = mean(w.confidence for w in words) if words else 0.0

            segments.append(
                TranscriptionSegment(
                    text=seg["text"].strip(),
                    start=seg["start"],
                    end=seg["end"],
                    words=words,
                    avg_confidence=avg_confidence,
                )
            )

        full_text = " ".join(seg.text for seg in segments)
        duration = segments[-1].end if segments else 0.0

        logger.info(
            "Transcription complete: %d segments, %.1f seconds",
            len(segments),
            duration,
        )

        return TranscriptionResult(
            segments=segments,
            language=detected_language,
            full_text=full_text,
            duration=duration,
        )

    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(f"Transcription failed: {exc}") from exc
