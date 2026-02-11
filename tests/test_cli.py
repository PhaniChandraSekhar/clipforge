"""Tests for clipforge.cli."""
import pytest
from pathlib import Path

from clipforge.cli import parse_args, build_config


def test_parse_args_minimal(tmp_video):
    args = parse_args([str(tmp_video)])
    assert args.input_video == tmp_video
    assert args.output_dir is None
    assert args.whisper_model == "base"
    assert args.llm_provider == "ollama"


def test_parse_args_all_flags(tmp_video, tmp_path):
    out = tmp_path / "out"
    args = parse_args([
        str(tmp_video),
        "-o", str(out),
        "-w", "large",
        "--provider", "anthropic",
        "--ollama-model", "mistral:7b",
        "--anthropic-model", "claude-sonnet-4-5-20250929",
        "--min-clip", "15",
        "--max-clip", "120",
        "--language", "en",
        "--skip-review",
        "--resume",
        "--device", "cuda",
        "--verbose",
    ])
    assert args.output_dir == out
    assert args.whisper_model == "large"
    assert args.llm_provider == "anthropic"
    assert args.ollama_model == "mistral:7b"
    assert args.anthropic_model == "claude-sonnet-4-5-20250929"
    assert args.min_clip_duration == 15
    assert args.max_clip_duration == 120
    assert args.language == "en"
    assert args.skip_review is True
    assert args.resume is True
    assert args.device == "cuda"
    assert args.verbose is True


def test_build_config_default_output_dir(tmp_video):
    args = parse_args([str(tmp_video)])
    config = build_config(args)
    assert "clips_test_video" in str(config.output_dir)


def test_build_config_custom_output_dir(tmp_video, tmp_path):
    out = tmp_path / "custom"
    args = parse_args([str(tmp_video), "-o", str(out)])
    config = build_config(args)
    assert config.output_dir == out


def test_parse_args_missing_input():
    with pytest.raises(SystemExit):
        parse_args([])
