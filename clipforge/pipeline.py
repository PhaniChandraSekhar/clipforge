"""Pipeline orchestrator for ClipForge."""
from __future__ import annotations

import enum
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clipforge.config import PipelineConfig
from clipforge.models import (
    SegmentationResult,
    TranscriptionResult,
    ValidatedClip,
)

logger = logging.getLogger("clipforge")


class Stage(enum.Enum):
    """Pipeline processing stages in execution order."""

    AUDIO = "audio"
    TRANSCRIBE = "transcribe"
    SEGMENT = "segment"
    VALIDATE = "validate"
    REVIEW = "review"
    EXTRACT = "extract"
    MANIFEST = "manifest"


class Pipeline:
    """Orchestrates the full ClipForge processing pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(self) -> Path:
        """Run all pipeline stages in order.

        Returns:
            Path to the output directory containing extracted clips.
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Audio extraction
        audio_path = self._run_audio_stage()

        # Stage 2: Transcription
        transcription = self._run_transcribe_stage(audio_path)

        # Stage 3: Topic segmentation
        segmentation = self._run_segment_stage(transcription)

        # Stage 4: Validation
        clips = self._run_validate_stage(segmentation, transcription)

        # Stage 5: Review
        clips = self._run_review_stage(clips)

        # Stage 6: Extraction
        extracted_paths = self._run_extract_stage(clips)

        # Stage 7: Manifest
        self._run_manifest_stage(clips, extracted_paths)

        return self.config.output_dir

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _run_audio_stage(self) -> Path:
        stage = Stage.AUDIO

        if self._should_skip(stage):
            checkpoint = self._load_checkpoint(stage)
            logger.info("Resuming from %s checkpoint", stage.value)
            return Path(checkpoint["audio_path"])

        logger.info("Stage: %s - Extracting audio", stage.value)
        from clipforge.audio import extract_audio, probe_video

        probe_video(self.config.input_video)
        audio_path = self.config.output_dir / "audio.wav"
        extract_audio(self.config.input_video, audio_path)

        self._save_checkpoint(stage, {"audio_path": str(audio_path)})
        return audio_path

    def _run_transcribe_stage(self, audio_path: Path) -> TranscriptionResult:
        stage = Stage.TRANSCRIBE

        if self._should_skip(stage):
            checkpoint = self._load_checkpoint(stage)
            logger.info("Resuming from %s checkpoint", stage.value)
            return TranscriptionResult(**checkpoint["transcription"])

        logger.info("Stage: %s - Transcribing audio", stage.value)
        from clipforge.transcribe import transcribe_audio

        transcription = transcribe_audio(audio_path, self.config)

        self._save_checkpoint(stage, {
            "transcription": transcription.model_dump(),
        })
        return transcription

    def _run_segment_stage(
        self, transcription: TranscriptionResult
    ) -> SegmentationResult:
        stage = Stage.SEGMENT

        if self._should_skip(stage):
            checkpoint = self._load_checkpoint(stage)
            logger.info("Resuming from %s checkpoint", stage.value)
            return SegmentationResult(**checkpoint["segmentation"])

        logger.info("Stage: %s - Segmenting topics", stage.value)
        from clipforge.segment import segment_topics

        segmentation = segment_topics(transcription, self.config)

        self._save_checkpoint(stage, {
            "segmentation": segmentation.model_dump(),
        })
        return segmentation

    def _run_validate_stage(
        self,
        segmentation: SegmentationResult,
        transcription: TranscriptionResult,
    ) -> list[ValidatedClip]:
        stage = Stage.VALIDATE

        if self._should_skip(stage):
            checkpoint = self._load_checkpoint(stage)
            logger.info("Resuming from %s checkpoint", stage.value)
            return [ValidatedClip(**c) for c in checkpoint["clips"]]

        logger.info("Stage: %s - Validating clips", stage.value)
        from clipforge.validate import validate_clips

        clips = validate_clips(segmentation, transcription, self.config)

        self._save_checkpoint(stage, {
            "clips": [c.model_dump() for c in clips],
        })
        return clips

    def _run_review_stage(
        self, clips: list[ValidatedClip]
    ) -> list[ValidatedClip]:
        stage = Stage.REVIEW

        if self._should_skip(stage):
            checkpoint = self._load_checkpoint(stage)
            logger.info("Resuming from %s checkpoint", stage.value)
            return [ValidatedClip(**c) for c in checkpoint["clips"]]

        logger.info("Stage: %s - Reviewing clips", stage.value)
        from clipforge.review import review_clips

        clips = review_clips(clips, self.config)

        self._save_checkpoint(stage, {
            "clips": [c.model_dump() for c in clips],
        })
        return clips

    def _run_extract_stage(self, clips: list[ValidatedClip]) -> list[Path]:
        stage = Stage.EXTRACT

        if self._should_skip(stage):
            checkpoint = self._load_checkpoint(stage)
            logger.info("Resuming from %s checkpoint", stage.value)
            return [Path(p) for p in checkpoint.get("paths", [])]

        logger.info("Stage: %s - Extracting clips", stage.value)

        approved = [c for c in clips if c.status == "approved"]
        if not approved:
            logger.warning("No approved clips to extract.")
            self._save_checkpoint(stage, {"extracted": 0, "paths": []})
            return []

        from clipforge.extract import extract_clips

        paths = extract_clips(approved, self.config)

        self._save_checkpoint(stage, {
            "extracted": len(approved),
            "paths": [str(p) for p in paths],
        })
        return paths

    def _run_manifest_stage(
        self,
        clips: list[ValidatedClip],
        extracted_paths: list[Path],
    ) -> None:
        stage = Stage.MANIFEST

        if self._should_skip(stage):
            logger.info("Resuming from %s checkpoint", stage.value)
            return

        logger.info("Stage: %s - Writing manifest", stage.value)
        from clipforge.manifest import generate_manifest

        approved = [c for c in clips if c.status == "approved"]
        manifest_path = generate_manifest(approved, extracted_paths, self.config)

        self._save_checkpoint(stage, {"manifest_path": str(manifest_path)})

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _checkpoint_path(self, stage: Stage) -> Path:
        return self.config.output_dir / f".checkpoint_{stage.value}.json"

    def _save_checkpoint(self, stage: Stage, data: dict[str, Any]) -> None:
        """Persist checkpoint data for a completed stage."""
        from clipforge.models import PipelineCheckpoint

        checkpoint = PipelineCheckpoint(
            stage=stage.value,
            data=data,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        path = self._checkpoint_path(stage)
        path.write_text(
            json.dumps(checkpoint.model_dump(), indent=2, default=str),
            encoding="utf-8",
        )
        logger.debug("Checkpoint saved: %s", path)

    def _load_checkpoint(self, stage: Stage) -> dict[str, Any] | None:
        """Load checkpoint data for a stage, if it exists."""
        path = self._checkpoint_path(stage)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw.get("data")

    def _should_skip(self, stage: Stage) -> bool:
        """Return True if the stage can be skipped via a valid checkpoint."""
        if not self.config.resume:
            return False
        return self._load_checkpoint(stage) is not None
