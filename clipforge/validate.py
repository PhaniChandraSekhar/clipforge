"""Clip validation: duration enforcement, boundary snapping, and overlap resolution."""
from __future__ import annotations

import logging
from bisect import bisect_left
from typing import TYPE_CHECKING

from clipforge.errors import ValidationError
from clipforge.models import ValidatedClip

if TYPE_CHECKING:
    from clipforge.config import PipelineConfig
    from clipforge.models import SegmentationResult, TranscriptionResult

logger = logging.getLogger("clipforge")

_SENTENCE_ENDINGS = frozenset(".!?")


def _build_sentence_boundaries(transcription: TranscriptionResult) -> list[float]:
    """Extract sentence boundary timestamps from word-level transcription data.

    A sentence boundary is placed at the end time of any word whose stripped
    text ends with '.', '!', or '?'.  The list always includes 0.0 and the
    total duration.
    """
    boundaries: set[float] = {0.0, transcription.duration}

    for segment in transcription.segments:
        for word in segment.words:
            stripped = word.word.strip()
            if stripped and stripped[-1] in _SENTENCE_ENDINGS:
                boundaries.add(word.end)

    return sorted(boundaries)


def _find_nearest_boundary(time: float, boundaries: list[float]) -> float:
    """Return the boundary value closest to *time*.

    Uses binary search for efficiency.
    """
    if not boundaries:
        return time

    idx = bisect_left(boundaries, time)

    if idx == 0:
        return boundaries[0]
    if idx == len(boundaries):
        return boundaries[-1]

    before = boundaries[idx - 1]
    after = boundaries[idx]
    return before if (time - before) <= (after - time) else after


def validate_clips(
    segments: SegmentationResult,
    transcription: TranscriptionResult,
    config: PipelineConfig,
) -> list[ValidatedClip]:
    """Validate and adjust topic segments into well-bounded clips.

    Snaps clip boundaries to sentence edges, enforces duration constraints,
    and resolves overlaps between adjacent clips.

    Args:
        segments: The raw topic segments from the LLM.
        transcription: The full transcription (used for sentence boundaries).
        config: Pipeline configuration with duration constraints.

    Returns:
        A list of ValidatedClip objects sorted by start_time.

    Raises:
        ValidationError: If validation fails unexpectedly.
    """
    try:
        boundaries = _build_sentence_boundaries(transcription)
        logger.info("Built %d sentence boundaries for clip snapping", len(boundaries))

        clips: list[ValidatedClip] = []
        skipped = 0

        for seg in segments.segments:
            start = _find_nearest_boundary(seg.start_time, boundaries)
            end = _find_nearest_boundary(seg.end_time, boundaries)

            # Ensure start < end after snapping
            if end <= start:
                logger.warning(
                    "Skipping segment '%s': invalid range after snapping (%.1f-%.1f)",
                    seg.title,
                    start,
                    end,
                )
                skipped += 1
                continue

            duration = end - start

            # Too short -> skip
            if duration < config.min_clip_duration:
                logger.warning(
                    "Skipping segment '%s': duration %.1fs < minimum %ds",
                    seg.title,
                    duration,
                    config.min_clip_duration,
                )
                skipped += 1
                continue

            # Too long -> truncate and re-snap
            if duration > config.max_clip_duration:
                raw_end = start + config.max_clip_duration
                end = _find_nearest_boundary(raw_end, boundaries)
                # If snapping pushed end beyond max, try the boundary before
                if end - start > config.max_clip_duration:
                    idx = bisect_left(boundaries, raw_end)
                    if idx > 0:
                        end = boundaries[idx - 1]
                duration = end - start
                logger.info(
                    "Truncated segment '%s' to %.1fs (max %ds)",
                    seg.title,
                    duration,
                    config.max_clip_duration,
                )

            clips.append(
                ValidatedClip(
                    index=0,  # placeholder, assigned after sorting
                    title=seg.title,
                    description=seg.description,
                    start_time=start,
                    end_time=end,
                    duration=duration,
                    key_quotes=seg.key_quotes,
                    status="pending",
                )
            )

        # Sort by start_time
        clips.sort(key=lambda c: c.start_time)

        # Resolve overlaps between adjacent clips
        for i in range(len(clips) - 1):
            current = clips[i]
            next_clip = clips[i + 1]
            if current.end_time > next_clip.start_time:
                midpoint = (current.end_time + next_clip.start_time) / 2.0
                new_current_end = _find_nearest_boundary(midpoint, boundaries)
                new_next_start = _find_nearest_boundary(midpoint, boundaries)

                # Try to find distinct boundaries on each side of the midpoint
                mid_idx = bisect_left(boundaries, midpoint)
                if mid_idx > 0 and mid_idx < len(boundaries):
                    new_current_end = boundaries[mid_idx - 1]
                    new_next_start = boundaries[mid_idx]
                elif mid_idx == 0:
                    new_current_end = boundaries[0]
                    new_next_start = boundaries[0]
                else:
                    new_current_end = boundaries[-1]
                    new_next_start = boundaries[-1]

                # Only adjust if the result is valid
                if new_current_end > current.start_time and new_next_start < next_clip.end_time:
                    clips[i] = current.model_copy(
                        update={
                            "end_time": new_current_end,
                            "duration": new_current_end - current.start_time,
                        }
                    )
                    clips[i + 1] = next_clip.model_copy(
                        update={
                            "start_time": new_next_start,
                            "duration": next_clip.end_time - new_next_start,
                        }
                    )
                    logger.info(
                        "Resolved overlap between '%s' and '%s' at boundary %.1fs",
                        current.title,
                        next_clip.title,
                        midpoint,
                    )

        # Assign sequential indices starting at 1
        for i, clip in enumerate(clips):
            clips[i] = clip.model_copy(update={"index": i + 1})

        if skipped:
            logger.info("Skipped %d segments that did not meet duration requirements", skipped)
        logger.info("Validation complete: %d clips approved", len(clips))

        return clips

    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(f"Clip validation failed: {exc}") from exc
