"""Tests for clipforge.models."""
from clipforge.models import (
    TranscriptionWord,
    TranscriptionSegment,
    TranscriptionResult,
    TopicSegment,
    SegmentationResult,
    ValidatedClip,
    ClipManifest,
    PipelineCheckpoint,
)


def test_transcription_word():
    w = TranscriptionWord(word="hello", start=0.0, end=0.5, confidence=0.9)
    assert w.word == "hello"
    assert w.start == 0.0
    assert w.end == 0.5
    assert w.confidence == 0.9


def test_transcription_segment():
    words = [TranscriptionWord(word="hi", start=0.0, end=0.3, confidence=0.8)]
    seg = TranscriptionSegment(
        text="hi", start=0.0, end=0.3, words=words, avg_confidence=0.8
    )
    assert seg.text == "hi"
    assert len(seg.words) == 1


def test_transcription_result(sample_transcription):
    assert sample_transcription.language == "en"
    assert len(sample_transcription.segments) == 3
    assert sample_transcription.duration == 61.5


def test_topic_segment():
    seg = TopicSegment(
        title="Test",
        description="Desc",
        start_time=0.0,
        end_time=30.0,
        key_quotes=["quote"],
    )
    assert seg.confidence == 1.0  # default


def test_segmentation_result(sample_segmentation):
    assert sample_segmentation.model_used == "llama3.1:8b"
    assert len(sample_segmentation.segments) == 2


def test_validated_clip():
    clip = ValidatedClip(
        index=1,
        title="Test",
        description="Desc",
        start_time=0.0,
        end_time=60.0,
        duration=60.0,
        key_quotes=[],
    )
    assert clip.status == "pending"


def test_clip_manifest():
    m = ClipManifest(
        source_video="test.mp4",
        clips=[],
        settings={},
        created_at="2025-01-01T00:00:00Z",
    )
    assert m.source_video == "test.mp4"


def test_pipeline_checkpoint():
    cp = PipelineCheckpoint(
        stage="audio",
        data={"audio_path": "/tmp/audio.wav"},
        timestamp="2025-01-01T00:00:00Z",
    )
    assert cp.stage == "audio"
