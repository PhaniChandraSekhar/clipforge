"""Tests for clipforge.pipeline."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from clipforge.pipeline import Pipeline, Stage


def test_stage_values():
    assert Stage.AUDIO.value == "audio"
    assert Stage.TRANSCRIBE.value == "transcribe"
    assert Stage.SEGMENT.value == "segment"
    assert Stage.VALIDATE.value == "validate"
    assert Stage.REVIEW.value == "review"
    assert Stage.EXTRACT.value == "extract"
    assert Stage.MANIFEST.value == "manifest"


def test_pipeline_creates_output_dir(pipeline_config):
    pipeline = Pipeline(pipeline_config)
    assert not pipeline_config.output_dir.exists()

    with patch.object(pipeline, "_run_audio_stage") as mock_audio, \
         patch.object(pipeline, "_run_transcribe_stage") as mock_trans, \
         patch.object(pipeline, "_run_segment_stage") as mock_seg, \
         patch.object(pipeline, "_run_validate_stage", return_value=[]) as mock_val, \
         patch.object(pipeline, "_run_review_stage", return_value=[]) as mock_rev, \
         patch.object(pipeline, "_run_extract_stage", return_value=[]) as mock_ext, \
         patch.object(pipeline, "_run_manifest_stage") as mock_man:

        mock_audio.return_value = Path("audio.wav")
        mock_trans.return_value = MagicMock()
        mock_seg.return_value = MagicMock()

        result = pipeline.run()

        assert pipeline_config.output_dir.exists()
        assert result == pipeline_config.output_dir


def test_checkpoint_save_and_load(pipeline_config):
    pipeline_config.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(pipeline_config)

    pipeline._save_checkpoint(Stage.AUDIO, {"audio_path": "/tmp/audio.wav"})
    loaded = pipeline._load_checkpoint(Stage.AUDIO)
    assert loaded["audio_path"] == "/tmp/audio.wav"


def test_checkpoint_load_missing(pipeline_config):
    pipeline_config.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(pipeline_config)
    assert pipeline._load_checkpoint(Stage.AUDIO) is None


def test_should_skip_without_resume(pipeline_config):
    pipeline_config.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_config.resume = False
    pipeline = Pipeline(pipeline_config)

    pipeline._save_checkpoint(Stage.AUDIO, {"audio_path": "/tmp/audio.wav"})
    assert pipeline._should_skip(Stage.AUDIO) is False


def test_should_skip_with_resume(pipeline_config):
    pipeline_config.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_config.resume = True
    pipeline = Pipeline(pipeline_config)

    pipeline._save_checkpoint(Stage.AUDIO, {"audio_path": "/tmp/audio.wav"})
    assert pipeline._should_skip(Stage.AUDIO) is True


def test_should_skip_no_checkpoint(pipeline_config):
    pipeline_config.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_config.resume = True
    pipeline = Pipeline(pipeline_config)
    assert pipeline._should_skip(Stage.AUDIO) is False
