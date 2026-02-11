"""JSON manifest generation for extracted clips."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from clipforge.models import ClipManifest

if TYPE_CHECKING:
    from clipforge.config import PipelineConfig
    from clipforge.models import ValidatedClip

logger = logging.getLogger("clipforge")


def generate_manifest(
    clips: list[ValidatedClip],
    extracted_paths: list[Path],
    config: PipelineConfig,
) -> Path:
    """Write a JSON manifest describing all extracted clips.

    Returns the path to the written manifest file.
    """
    clip_entries: list[dict] = []
    for clip, path in zip(clips, extracted_paths):
        clip_entries.append(
            {
                "title": clip.title,
                "description": clip.description,
                "start_time": clip.start_time,
                "end_time": clip.end_time,
                "duration": clip.duration,
                "key_quotes": clip.key_quotes,
                "filename": path.name,
                "status": clip.status,
            }
        )

    settings = {
        "whisper_model": config.whisper_model,
        "ollama_model": config.ollama_model,
        "min_clip_duration": config.min_clip_duration,
        "max_clip_duration": config.max_clip_duration,
        "language": config.language,
        "device": config.device,
    }

    manifest = ClipManifest(
        source_video=config.input_video.name,
        clips=clip_entries,
        settings=settings,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")

    logger.info("Manifest written to %s", manifest_path)
    return manifest_path
