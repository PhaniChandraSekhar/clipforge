"""FFmpeg-based clip extraction."""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from clipforge.errors import ExtractionError

if TYPE_CHECKING:
    from clipforge.config import PipelineConfig
    from clipforge.models import ValidatedClip

logger = logging.getLogger("clipforge")


def _sanitize_filename(title: str) -> str:
    """Return a filesystem-safe version of *title*.

    Non-alphanumeric characters (except spaces) are replaced with underscores.
    Multiple consecutive underscores are collapsed, leading/trailing
    underscores are stripped, and the result is limited to 80 characters.
    """
    name = re.sub(r"[^a-zA-Z0-9 ]", "_", title)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name[:80]


def extract_clips(
    clips: list[ValidatedClip], config: PipelineConfig
) -> list[Path]:
    """Extract each *clip* from the source video using FFmpeg.

    Returns a list of output file paths corresponding to the extracted clips.
    Raises :class:`ExtractionError` if FFmpeg returns a non-zero exit code.
    """
    output_paths: list[Path] = []

    for clip in clips:
        filename = f"{clip.index:02d}_{_sanitize_filename(clip.title)}.mp4"
        output_path = config.output_dir / filename

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(config.input_video),
            "-ss", str(clip.start_time),
            "-to", str(clip.end_time),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise ExtractionError(
                f"FFmpeg failed for clip {clip.index} ({filename}): {result.stderr}"
            )

        logger.info("Extracted clip %d: %s", clip.index, filename)
        output_paths.append(output_path)

    return output_paths
