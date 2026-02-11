"""Tests for clipforge.segment."""
from unittest.mock import patch, MagicMock

import pytest

from clipforge.segment import (
    _format_transcript,
    _chunk_transcript,
    _segments_overlap,
    _merge_segments,
    segment_topics,
)
from clipforge.models import TopicSegment, TranscriptionResult, TranscriptionSegment, TranscriptionWord
from clipforge.errors import SegmentationError


def test_format_transcript(sample_transcription):
    formatted = _format_transcript(sample_transcription)
    assert "[0s]" in formatted
    assert "[30s]" in formatted
    assert "[60s]" in formatted
    assert "Hello world." in formatted


def test_chunk_transcript_short(sample_transcription):
    formatted = _format_transcript(sample_transcription)
    chunks = _chunk_transcript(formatted, sample_transcription)
    # Short transcript should not be chunked
    assert len(chunks) == 1
    assert chunks[0][1] == 0.0


def test_segments_overlap_true():
    a = TopicSegment(title="A", description="", start_time=0, end_time=60, key_quotes=[])
    b = TopicSegment(title="B", description="", start_time=20, end_time=80, key_quotes=[])
    # overlap = 40s, shorter = 60s, ratio = 0.667 > 0.5
    assert _segments_overlap(a, b) is True


def test_segments_overlap_false():
    a = TopicSegment(title="A", description="", start_time=0, end_time=30, key_quotes=[])
    b = TopicSegment(title="B", description="", start_time=60, end_time=90, key_quotes=[])
    assert _segments_overlap(a, b) is False


def test_segments_overlap_zero_duration():
    a = TopicSegment(title="A", description="", start_time=0, end_time=0, key_quotes=[])
    b = TopicSegment(title="B", description="", start_time=0, end_time=30, key_quotes=[])
    assert _segments_overlap(a, b) is False


def test_merge_segments():
    segs = [
        TopicSegment(title="A", description="", start_time=0, end_time=60, key_quotes=[], confidence=0.9),
        TopicSegment(title="B", description="", start_time=20, end_time=80, key_quotes=[], confidence=0.8),
        TopicSegment(title="C", description="", start_time=100, end_time=150, key_quotes=[], confidence=0.7),
    ]
    merged = _merge_segments(segs)
    assert len(merged) == 2
    # A should be kept (higher confidence than B, and they overlap > 50%)
    assert merged[0].title == "A"
    assert merged[1].title == "C"


def test_merge_segments_empty():
    assert _merge_segments([]) == []


def test_segment_topics_success(sample_transcription, pipeline_config):
    mock_topics = [
        TopicSegment(
            title="Intro",
            description="Opening",
            start_time=0.0,
            end_time=35.0,
            key_quotes=["Hello world."],
            confidence=0.9,
        ),
    ]

    mock_response = MagicMock()
    mock_response.message.content = '{"topics": [{"title": "Intro", "description": "Opening", "start_time": 0.0, "end_time": 35.0, "key_quotes": ["Hello world."], "confidence": 0.9}]}'

    with patch("clipforge.segment.ollama.chat", return_value=mock_response):
        result = segment_topics(sample_transcription, pipeline_config)
        assert len(result.segments) == 1
        assert result.segments[0].title == "Intro"
        assert result.model_used == "llama3.1:8b"


def test_segment_topics_retries_on_failure(sample_transcription, pipeline_config):
    mock_response = MagicMock()
    mock_response.message.content = '{"topics": [{"title": "Intro", "description": "d", "start_time": 0.0, "end_time": 35.0, "key_quotes": [], "confidence": 0.9}]}'

    with patch("clipforge.segment.ollama.chat") as mock_chat:
        mock_chat.side_effect = [
            RuntimeError("connection error"),
            RuntimeError("timeout"),
            mock_response,
        ]
        result = segment_topics(sample_transcription, pipeline_config)
        assert len(result.segments) == 1


def test_segment_topics_all_retries_fail(sample_transcription, pipeline_config):
    with patch("clipforge.segment.ollama.chat", side_effect=RuntimeError("fail")):
        with pytest.raises(SegmentationError, match="Failed to parse"):
            segment_topics(sample_transcription, pipeline_config)
