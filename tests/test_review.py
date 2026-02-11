"""Tests for clipforge.review."""
from unittest.mock import patch

import pytest

from clipforge.review import _format_time, _parse_time, review_clips


def test_format_time():
    assert _format_time(0) == "00:00"
    assert _format_time(65) == "01:05"
    assert _format_time(3600) == "60:00"
    assert _format_time(125.7) == "02:05"


def test_parse_time_valid():
    assert _parse_time("00:00") == 0.0
    assert _parse_time("01:30") == 90.0
    assert _parse_time("10:05") == 605.0


def test_parse_time_invalid():
    with pytest.raises(ValueError, match="Invalid time format"):
        _parse_time("abc")

    with pytest.raises(ValueError, match="Invalid time format"):
        _parse_time("1:2:3")

    with pytest.raises(ValueError, match="Invalid time format"):
        _parse_time("00:60")

    with pytest.raises(ValueError, match="Invalid time format"):
        _parse_time("-1:00")


def test_review_clips_skip_review(sample_validated_clips, pipeline_config):
    pipeline_config.skip_review = True
    result = review_clips(sample_validated_clips, pipeline_config)
    assert all(c.status == "approved" for c in result)


def test_review_clips_approve_all(sample_validated_clips, pipeline_config):
    pipeline_config.skip_review = False
    inputs = ["a", "a", "y"]  # approve both clips, then confirm
    with patch("builtins.input", side_effect=inputs):
        result = review_clips(sample_validated_clips, pipeline_config)
        assert all(c.status == "approved" for c in result)


def test_review_clips_reject_one(sample_validated_clips, pipeline_config):
    pipeline_config.skip_review = False
    inputs = ["a", "r", "y"]  # approve first, reject second, confirm
    with patch("builtins.input", side_effect=inputs):
        result = review_clips(sample_validated_clips, pipeline_config)
        approved = [c for c in result if c.status == "approved"]
        assert len(approved) == 1


def test_review_clips_edit_title(sample_validated_clips, pipeline_config):
    pipeline_config.skip_review = False
    inputs = ["t", "New Title", "a", "a", "y"]
    with patch("builtins.input", side_effect=inputs):
        result = review_clips(sample_validated_clips, pipeline_config)
        assert result[0].title == "New Title"


def test_review_clips_cancel(sample_validated_clips, pipeline_config):
    pipeline_config.skip_review = False
    inputs = ["a", "a", "n"]  # approve both, then cancel
    with patch("builtins.input", side_effect=inputs):
        with pytest.raises(SystemExit, match="cancelled"):
            review_clips(sample_validated_clips, pipeline_config)
