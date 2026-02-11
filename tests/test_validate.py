"""Tests for clipforge.validate."""
import pytest

from clipforge.validate import (
    _build_sentence_boundaries,
    _find_nearest_boundary,
    validate_clips,
)
from clipforge.models import TopicSegment, SegmentationResult


def test_build_sentence_boundaries(sample_transcription):
    boundaries = _build_sentence_boundaries(sample_transcription)
    assert 0.0 in boundaries
    assert sample_transcription.duration in boundaries
    # Words ending in period should create boundaries
    assert 1.0 in boundaries   # "world."
    assert 2.0 in boundaries   # "test."
    assert 31.8 in boundaries  # "great."
    assert 61.5 in boundaries  # "rock."


def test_find_nearest_boundary():
    boundaries = [0.0, 10.0, 20.0, 30.0]
    assert _find_nearest_boundary(5.0, boundaries) == 0.0
    assert _find_nearest_boundary(6.0, boundaries) == 10.0
    assert _find_nearest_boundary(15.0, boundaries) == 10.0
    assert _find_nearest_boundary(16.0, boundaries) == 20.0
    assert _find_nearest_boundary(10.0, boundaries) == 10.0


def test_find_nearest_boundary_edge_cases():
    boundaries = [0.0, 100.0]
    assert _find_nearest_boundary(-5.0, boundaries) == 0.0
    assert _find_nearest_boundary(200.0, boundaries) == 100.0

    assert _find_nearest_boundary(50.0, []) == 50.0


def test_validate_clips_basic(sample_segmentation, sample_transcription, pipeline_config):
    clips = validate_clips(sample_segmentation, sample_transcription, pipeline_config)
    assert len(clips) >= 1
    for clip in clips:
        assert clip.index >= 1
        assert clip.duration >= pipeline_config.min_clip_duration
        assert clip.start_time < clip.end_time


def test_validate_clips_skips_short(sample_transcription, pipeline_config):
    """Segments shorter than min_clip_duration are skipped."""
    short_seg = SegmentationResult(
        segments=[
            TopicSegment(
                title="Too Short",
                description="Very short segment.",
                start_time=0.0,
                end_time=5.0,
                key_quotes=[],
            ),
        ],
        model_used="test",
    )
    clips = validate_clips(short_seg, sample_transcription, pipeline_config)
    assert len(clips) == 0


def test_validate_clips_truncates_long(sample_transcription, pipeline_config):
    """Segments longer than max_clip_duration are truncated."""
    pipeline_config.max_clip_duration = 40
    long_seg = SegmentationResult(
        segments=[
            TopicSegment(
                title="Long Segment",
                description="A very long segment.",
                start_time=0.0,
                end_time=61.5,
                key_quotes=[],
            ),
        ],
        model_used="test",
    )
    clips = validate_clips(long_seg, sample_transcription, pipeline_config)
    if clips:
        assert clips[0].duration <= pipeline_config.max_clip_duration


def test_validate_clips_assigns_sequential_indices(
    sample_segmentation, sample_transcription, pipeline_config
):
    clips = validate_clips(sample_segmentation, sample_transcription, pipeline_config)
    for i, clip in enumerate(clips):
        assert clip.index == i + 1
