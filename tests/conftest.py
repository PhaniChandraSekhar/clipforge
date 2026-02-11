"""Shared test fixtures for ClipForge."""
from __future__ import annotations

import pytest
from pathlib import Path

from clipforge.config import PipelineConfig
from clipforge.models import (
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
    TopicSegment,
    SegmentationResult,
    ValidatedClip,
)


@pytest.fixture
def tmp_video(tmp_path: Path) -> Path:
    """Create a dummy video file path (does not contain real video data)."""
    video = tmp_path / "test_video.mp4"
    video.write_bytes(b"\x00" * 64)
    return video


@pytest.fixture
def tmp_audio(tmp_path: Path) -> Path:
    """Create a dummy audio file path."""
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"\x00" * 64)
    return audio


@pytest.fixture
def pipeline_config(tmp_path: Path, tmp_video: Path) -> PipelineConfig:
    """Create a minimal PipelineConfig for testing."""
    return PipelineConfig(
        input_video=tmp_video,
        output_dir=tmp_path / "output",
        whisper_model="base",
        ollama_model="llama3.1:8b",
        min_clip_duration=30,
        max_clip_duration=600,
        skip_review=True,
        resume=False,
        device="cpu",
        verbose=False,
    )


@pytest.fixture
def sample_words() -> list[TranscriptionWord]:
    """A sequence of sample transcription words."""
    return [
        TranscriptionWord(word="Hello", start=0.0, end=0.5, confidence=0.95),
        TranscriptionWord(word="world.", start=0.5, end=1.0, confidence=0.90),
        TranscriptionWord(word="This", start=1.0, end=1.3, confidence=0.92),
        TranscriptionWord(word="is", start=1.3, end=1.5, confidence=0.88),
        TranscriptionWord(word="a", start=1.5, end=1.6, confidence=0.91),
        TranscriptionWord(word="test.", start=1.6, end=2.0, confidence=0.93),
        TranscriptionWord(word="Machine", start=30.0, end=30.5, confidence=0.89),
        TranscriptionWord(word="learning", start=30.5, end=31.0, confidence=0.87),
        TranscriptionWord(word="is", start=31.0, end=31.2, confidence=0.92),
        TranscriptionWord(word="great.", start=31.2, end=31.8, confidence=0.95),
        TranscriptionWord(word="Neural", start=60.0, end=60.5, confidence=0.90),
        TranscriptionWord(word="networks", start=60.5, end=61.0, confidence=0.88),
        TranscriptionWord(word="rock.", start=61.0, end=61.5, confidence=0.91),
    ]


@pytest.fixture
def sample_transcription(sample_words: list[TranscriptionWord]) -> TranscriptionResult:
    """A sample transcription result with multiple segments."""
    return TranscriptionResult(
        segments=[
            TranscriptionSegment(
                text="Hello world. This is a test.",
                start=0.0,
                end=2.0,
                words=sample_words[:6],
                avg_confidence=0.915,
            ),
            TranscriptionSegment(
                text="Machine learning is great.",
                start=30.0,
                end=31.8,
                words=sample_words[6:10],
                avg_confidence=0.9075,
            ),
            TranscriptionSegment(
                text="Neural networks rock.",
                start=60.0,
                end=61.5,
                words=sample_words[10:],
                avg_confidence=0.8967,
            ),
        ],
        language="en",
        full_text="Hello world. This is a test. Machine learning is great. Neural networks rock.",
        duration=61.5,
    )


@pytest.fixture
def sample_topic_segments() -> list[TopicSegment]:
    """Sample topic segments from LLM segmentation."""
    return [
        TopicSegment(
            title="Introduction to the Course",
            description="Opening remarks and course overview.",
            start_time=0.0,
            end_time=35.0,
            key_quotes=["Hello world.", "This is a test."],
            confidence=0.9,
        ),
        TopicSegment(
            title="Neural Networks Deep Dive",
            description="Exploring neural network architectures.",
            start_time=35.0,
            end_time=61.5,
            key_quotes=["Neural networks rock."],
            confidence=0.85,
        ),
    ]


@pytest.fixture
def sample_segmentation(sample_topic_segments: list[TopicSegment]) -> SegmentationResult:
    """A sample segmentation result."""
    return SegmentationResult(
        segments=sample_topic_segments,
        model_used="llama3.1:8b",
    )


@pytest.fixture
def sample_validated_clips() -> list[ValidatedClip]:
    """Sample validated clips for testing review and extraction."""
    return [
        ValidatedClip(
            index=1,
            title="Introduction to the Course",
            description="Opening remarks.",
            start_time=0.0,
            end_time=35.0,
            duration=35.0,
            key_quotes=["Hello world."],
            status="pending",
        ),
        ValidatedClip(
            index=2,
            title="Neural Networks Deep Dive",
            description="NN architectures.",
            start_time=35.0,
            end_time=61.5,
            duration=26.5,
            key_quotes=["Neural networks rock."],
            status="pending",
        ),
    ]
