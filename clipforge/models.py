"""Pydantic data models shared across all ClipForge modules."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class TranscriptionWord(BaseModel):
    """A single transcribed word with timing and confidence."""

    word: str
    start: float
    end: float
    confidence: float


class TranscriptionSegment(BaseModel):
    """A segment of transcribed text with word-level detail."""

    text: str
    start: float
    end: float
    words: list[TranscriptionWord]
    avg_confidence: float


class TranscriptionResult(BaseModel):
    """Complete transcription output for a video."""

    segments: list[TranscriptionSegment]
    language: str
    full_text: str
    duration: float


class TopicSegment(BaseModel):
    """A topic-based segment identified by the LLM."""

    title: str
    description: str
    start_time: float
    end_time: float
    key_quotes: list[str]
    confidence: float = 1.0


class SegmentationResult(BaseModel):
    """Result of topic segmentation over a transcription."""

    segments: list[TopicSegment]
    model_used: str


class ValidatedClip(BaseModel):
    """A clip that has been validated against duration constraints."""

    index: int
    title: str
    description: str
    start_time: float
    end_time: float
    duration: float
    key_quotes: list[str]
    status: Literal["approved", "rejected", "pending"] = "pending"


class ClipManifest(BaseModel):
    """Manifest describing all exported clips for a source video."""

    source_video: str
    clips: list[dict]
    settings: dict
    created_at: str


class PipelineCheckpoint(BaseModel):
    """Checkpoint data persisted between pipeline stages."""

    stage: str
    data: dict
    timestamp: str
