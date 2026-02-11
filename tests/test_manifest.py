"""Tests for clipforge.manifest."""
import json
from pathlib import Path

from clipforge.manifest import generate_manifest


def test_generate_manifest(sample_validated_clips, pipeline_config):
    pipeline_config.output_dir.mkdir(parents=True, exist_ok=True)
    for clip in sample_validated_clips:
        clip.status = "approved"

    fake_paths = [
        pipeline_config.output_dir / "01_intro.mp4",
        pipeline_config.output_dir / "02_neural.mp4",
    ]

    manifest_path = generate_manifest(sample_validated_clips, fake_paths, pipeline_config)

    assert manifest_path.exists()
    assert manifest_path.name == "manifest.json"

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["source_video"] == pipeline_config.input_video.name
    assert len(data["clips"]) == 2
    assert data["clips"][0]["title"] == "Introduction to the Course"
    assert data["settings"]["whisper_model"] == "base"
    assert "created_at" in data


def test_generate_manifest_empty(pipeline_config):
    pipeline_config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = generate_manifest([], [], pipeline_config)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["clips"] == []
