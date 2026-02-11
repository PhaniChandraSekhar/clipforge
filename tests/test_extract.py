"""Tests for clipforge.extract."""
from unittest.mock import patch, MagicMock

import pytest

from clipforge.extract import _sanitize_filename, extract_clips
from clipforge.errors import ExtractionError


def test_sanitize_filename_basic():
    assert _sanitize_filename("Hello World") == "Hello World"


def test_sanitize_filename_special_chars():
    assert _sanitize_filename("What's New? (v2.0)") == "What_s New_ _v2_0"


def test_sanitize_filename_length_limit():
    long_title = "A" * 100
    assert len(_sanitize_filename(long_title)) <= 80


def test_sanitize_filename_consecutive_underscores():
    result = _sanitize_filename("a---b___c")
    assert "__" not in result


def test_extract_clips_success(sample_validated_clips, pipeline_config):
    pipeline_config.output_dir.mkdir(parents=True, exist_ok=True)
    # Only approved clips get extracted
    for clip in sample_validated_clips:
        clip.status = "approved"

    mock_result = MagicMock(returncode=0)
    with patch("clipforge.extract.subprocess.run", return_value=mock_result):
        paths = extract_clips(sample_validated_clips, pipeline_config)
        assert len(paths) == 2
        assert all(str(p).endswith(".mp4") for p in paths)


def test_extract_clips_ffmpeg_failure(sample_validated_clips, pipeline_config):
    pipeline_config.output_dir.mkdir(parents=True, exist_ok=True)
    for clip in sample_validated_clips:
        clip.status = "approved"

    mock_result = MagicMock(returncode=1, stderr="encoding error")
    with patch("clipforge.extract.subprocess.run", return_value=mock_result):
        with pytest.raises(ExtractionError, match="FFmpeg failed"):
            extract_clips(sample_validated_clips, pipeline_config)
