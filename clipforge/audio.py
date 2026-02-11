"""FFmpeg-based audio extraction utilities."""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from clipforge.errors import AudioExtractionError

logger = logging.getLogger("clipforge")


def probe_video(path: Path) -> dict:
    """Probe a video file to verify it has an audio stream and get metadata.

    Args:
        path: Path to the video file.

    Returns:
        Dictionary with keys 'duration' (float) and 'audio_stream' (dict).

    Raises:
        AudioExtractionError: If ffprobe fails or no audio stream is found.
    """
    if not path.exists():
        raise AudioExtractionError(f"Video file not found: {path}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise AudioExtractionError(
            "ffprobe not found. Please install FFmpeg and ensure it is on PATH."
        )
    except subprocess.CalledProcessError as exc:
        raise AudioExtractionError(
            f"ffprobe failed for {path}: {exc.stderr.strip()}"
        )

    try:
        probe_data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AudioExtractionError(f"Failed to parse ffprobe output: {exc}")

    audio_streams = [
        s for s in probe_data.get("streams", [])
        if s.get("codec_type") == "audio"
    ]

    if not audio_streams:
        raise AudioExtractionError(f"No audio stream found in {path}")

    duration = float(probe_data.get("format", {}).get("duration", 0.0))

    logger.debug("Probed %s: duration=%.2fs, audio_codec=%s",
                 path, duration, audio_streams[0].get("codec_name"))

    return {
        "duration": duration,
        "audio_stream": audio_streams[0],
    }


def extract_audio(video_path: Path, output_path: Path) -> Path:
    """Extract audio from a video file as 16 kHz mono WAV.

    Args:
        video_path: Path to the source video.
        output_path: Destination path for the WAV file.

    Returns:
        The output_path on success.

    Raises:
        AudioExtractionError: If ffmpeg fails or the video file is missing.
    """
    if not video_path.exists():
        raise AudioExtractionError(f"Video file not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ]

    logger.info("Extracting audio from %s -> %s", video_path, output_path)

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        raise AudioExtractionError(
            "ffmpeg not found. Please install FFmpeg and ensure it is on PATH."
        )
    except subprocess.CalledProcessError as exc:
        raise AudioExtractionError(
            f"ffmpeg audio extraction failed: {exc.stderr.strip()}"
        )

    if not output_path.exists():
        raise AudioExtractionError(
            f"Audio extraction produced no output file at {output_path}"
        )

    logger.info("Audio extracted successfully: %s", output_path)
    return output_path
