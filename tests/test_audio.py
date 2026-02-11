"""Tests for clipforge.audio."""
import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from clipforge.audio import probe_video, extract_audio
from clipforge.errors import AudioExtractionError


class TestProbeVideo:
    def test_file_not_found(self, tmp_path):
        missing = tmp_path / "missing.mp4"
        with pytest.raises(AudioExtractionError, match="not found"):
            probe_video(missing)

    def test_ffprobe_not_installed(self, tmp_video):
        with patch("clipforge.audio.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(AudioExtractionError, match="ffprobe not found"):
                probe_video(tmp_video)

    def test_ffprobe_failure(self, tmp_video):
        error = subprocess.CalledProcessError(1, "ffprobe", stderr="error")
        with patch("clipforge.audio.subprocess.run", side_effect=error):
            with pytest.raises(AudioExtractionError, match="ffprobe failed"):
                probe_video(tmp_video)

    def test_no_audio_stream(self, tmp_video):
        probe_output = json.dumps({
            "streams": [{"codec_type": "video"}],
            "format": {"duration": "120.0"},
        })
        mock_result = MagicMock()
        mock_result.stdout = probe_output
        with patch("clipforge.audio.subprocess.run", return_value=mock_result):
            with pytest.raises(AudioExtractionError, match="No audio stream"):
                probe_video(tmp_video)

    def test_successful_probe(self, tmp_video):
        probe_output = json.dumps({
            "streams": [
                {"codec_type": "audio", "codec_name": "aac"},
            ],
            "format": {"duration": "120.0"},
        })
        mock_result = MagicMock()
        mock_result.stdout = probe_output
        with patch("clipforge.audio.subprocess.run", return_value=mock_result):
            result = probe_video(tmp_video)
            assert result["duration"] == 120.0
            assert result["audio_stream"]["codec_name"] == "aac"

    def test_invalid_json(self, tmp_video):
        mock_result = MagicMock()
        mock_result.stdout = "not json"
        with patch("clipforge.audio.subprocess.run", return_value=mock_result):
            with pytest.raises(AudioExtractionError, match="parse ffprobe"):
                probe_video(tmp_video)


class TestExtractAudio:
    def test_video_not_found(self, tmp_path):
        missing = tmp_path / "missing.mp4"
        out = tmp_path / "audio.wav"
        with pytest.raises(AudioExtractionError, match="not found"):
            extract_audio(missing, out)

    def test_ffmpeg_not_installed(self, tmp_video, tmp_path):
        out = tmp_path / "audio.wav"
        with patch("clipforge.audio.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(AudioExtractionError, match="ffmpeg not found"):
                extract_audio(tmp_video, out)

    def test_ffmpeg_failure(self, tmp_video, tmp_path):
        out = tmp_path / "audio.wav"
        error = subprocess.CalledProcessError(1, "ffmpeg", stderr="encoding failed")
        with patch("clipforge.audio.subprocess.run", side_effect=error):
            with pytest.raises(AudioExtractionError, match="extraction failed"):
                extract_audio(tmp_video, out)

    def test_successful_extraction(self, tmp_video, tmp_path):
        out = tmp_path / "audio.wav"
        mock_result = MagicMock(returncode=0)

        def create_output(*args, **kwargs):
            out.write_bytes(b"\x00" * 16)
            return mock_result

        with patch("clipforge.audio.subprocess.run", side_effect=create_output):
            result = extract_audio(tmp_video, out)
            assert result == out
            assert out.exists()

    def test_no_output_file(self, tmp_video, tmp_path):
        out = tmp_path / "audio.wav"
        mock_result = MagicMock(returncode=0)
        with patch("clipforge.audio.subprocess.run", return_value=mock_result):
            with pytest.raises(AudioExtractionError, match="no output"):
                extract_audio(tmp_video, out)
