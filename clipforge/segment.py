"""Topic segmentation of transcripts using an Ollama-hosted LLM."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import ollama
from pydantic import BaseModel

from clipforge.errors import SegmentationError
from clipforge.models import SegmentationResult, TopicSegment

if TYPE_CHECKING:
    from clipforge.config import PipelineConfig
    from clipforge.models import TranscriptionResult

logger = logging.getLogger("clipforge")

MAX_RETRIES = 3
CHARS_PER_TOKEN = 4
MAX_CONTEXT_CHARS = 24000
CHUNK_SIZE = 20000
CHUNK_OVERLAP = 2000

SYSTEM_PROMPT = """\
You are an expert video editor specializing in AI/ML training content. Your task \
is to analyze a timestamped transcript and identify distinct, self-contained topic \
segments suitable for YouTube Shorts or standalone clips (30 seconds to 10 minutes).

Each line in the transcript is prefixed with the timestamp in TOTAL SECONDS, like:
  [120s] Some spoken text...
This means the line starts at 120 seconds into the video.

For each topic segment, provide:
- title: A catchy, descriptive title that would work as a YouTube video title.
- description: A concise summary of the topic covered in the segment.
- start_time: The start time as a float in TOTAL SECONDS from the beginning of \
the video. Use the [Xs] timestamps shown in the transcript directly. For example, \
if a topic starts at [360s], set start_time to 360.0.
- end_time: The end time as a float in TOTAL SECONDS from the beginning of the \
video. Must be greater than start_time by at least 30 seconds.
- key_quotes: A list of 1-3 notable verbatim quotes from the segment.
- confidence: A float between 0 and 1 indicating how confident you are that \
this is a distinct, self-contained topic.

Guidelines:
- Each segment should cover a single, coherent topic.
- Prefer natural topic boundaries; do not split mid-sentence or mid-thought.
- Titles should be engaging and descriptive, suitable for social media.
- Include the most impactful or informative quotes.
- Segments must not overlap.
- IMPORTANT: start_time and end_time MUST be in total seconds, matching the [Xs] \
timestamps in the transcript. Do NOT use minutes or MM:SS notation.
- Return your answer as a JSON object with a single key "topics" containing the \
list of topic segments.
"""


class TopicList(BaseModel):
    """Wrapper model for structured LLM output."""

    topics: list[TopicSegment]


def _format_transcript(transcription: TranscriptionResult) -> str:
    """Format a transcription into a timestamped string for the LLM.

    Uses total seconds (e.g. ``[360s]``) instead of MM:SS to avoid ambiguity
    when the LLM returns start_time / end_time values.
    """
    lines: list[str] = []
    for seg in transcription.segments:
        lines.append(f"[{int(seg.start)}s] {seg.text}")
    return "\n".join(lines)


def _chunk_transcript(
    formatted: str,
    transcription: TranscriptionResult,
) -> list[tuple[str, float]]:
    """Split a long transcript into overlapping chunks with timestamp offsets.

    Returns a list of (chunk_text, timestamp_offset) tuples.
    """
    if len(formatted) <= MAX_CONTEXT_CHARS:
        return [(formatted, 0.0)]

    lines = formatted.split("\n")

    # Build a mapping from each line's character offset to its segment start time
    line_char_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_char_offsets.append(offset)
        offset += len(line) + 1  # +1 for the newline

    # Also map line index -> segment start time by parsing the timestamp prefix
    def _parse_line_time(line: str) -> float:
        """Extract seconds from a '[Xs] ...' formatted line."""
        try:
            ts = line.split("]")[0].lstrip("[").rstrip("s")
            return float(ts)
        except (IndexError, ValueError):
            return 0.0

    chunks: list[tuple[str, float]] = []
    start_idx = 0

    while start_idx < len(lines):
        # Accumulate lines until we reach CHUNK_SIZE characters
        char_count = 0
        end_idx = start_idx
        while end_idx < len(lines) and char_count + len(lines[end_idx]) + 1 <= CHUNK_SIZE:
            char_count += len(lines[end_idx]) + 1
            end_idx += 1

        # If we haven't consumed any lines, force at least one
        if end_idx == start_idx:
            end_idx = start_idx + 1

        chunk_text = "\n".join(lines[start_idx:end_idx])
        chunk_offset = _parse_line_time(lines[start_idx])
        chunks.append((chunk_text, chunk_offset))

        # Advance, but step back by enough lines to create overlap
        overlap_chars = 0
        overlap_start = end_idx
        while overlap_start > start_idx and overlap_chars < CHUNK_OVERLAP:
            overlap_start -= 1
            overlap_chars += len(lines[overlap_start]) + 1

        start_idx = max(overlap_start, start_idx + 1)

    logger.info("Transcript split into %d chunks for processing", len(chunks))
    return chunks


def _segments_overlap(a: TopicSegment, b: TopicSegment) -> bool:
    """Return True if two segments overlap by more than 50% of the shorter one."""
    overlap_start = max(a.start_time, b.start_time)
    overlap_end = min(a.end_time, b.end_time)
    overlap_duration = max(0.0, overlap_end - overlap_start)
    shorter_duration = min(a.end_time - a.start_time, b.end_time - b.start_time)
    if shorter_duration <= 0:
        return False
    return overlap_duration / shorter_duration > 0.5


def _merge_segments(all_segments: list[TopicSegment]) -> list[TopicSegment]:
    """Remove duplicate/overlapping segments, keeping the higher-confidence one."""
    if not all_segments:
        return []

    # Sort by start_time
    sorted_segs = sorted(all_segments, key=lambda s: s.start_time)
    merged: list[TopicSegment] = [sorted_segs[0]]

    for seg in sorted_segs[1:]:
        if _segments_overlap(merged[-1], seg):
            # Keep the one with higher confidence
            if seg.confidence > merged[-1].confidence:
                merged[-1] = seg
        else:
            merged.append(seg)

    return merged


def _call_ollama(chunk: str, config: PipelineConfig) -> list[TopicSegment]:
    """Send a transcript chunk to Ollama and parse the structured response."""
    user_message = (
        "Analyze the following transcript and identify distinct topic segments.\n\n"
        f"{chunk}"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=config.ollama_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                format=TopicList.model_json_schema(),
                options={"temperature": 0.1},
            )

            parsed = TopicList.model_validate_json(response.message.content)
            return parsed.topics

        except Exception as exc:
            logger.warning(
                "LLM parse attempt %d/%d failed: %s",
                attempt,
                MAX_RETRIES,
                exc,
            )
            if attempt == MAX_RETRIES:
                raise SegmentationError(
                    f"Failed to parse LLM response after {MAX_RETRIES} attempts: {exc}"
                ) from exc

    # Unreachable, but satisfies type checkers
    raise SegmentationError("Exhausted retries without result")  # pragma: no cover


def segment_topics(
    transcription: TranscriptionResult,
    config: PipelineConfig,
) -> SegmentationResult:
    """Identify topic segments in a transcription using an LLM.

    Args:
        transcription: The full transcription result to segment.
        config: Pipeline configuration with model settings.

    Returns:
        A SegmentationResult containing identified topic segments.

    Raises:
        SegmentationError: If segmentation fails after retries.
    """
    try:
        formatted = _format_transcript(transcription)
        logger.info(
            "Transcript formatted: %d characters (~%d tokens)",
            len(formatted),
            len(formatted) // CHARS_PER_TOKEN,
        )

        chunks = _chunk_transcript(formatted, transcription)

        all_segments: list[TopicSegment] = []
        for i, (chunk_text, offset) in enumerate(chunks):
            logger.info("Processing chunk %d/%d (offset=%.1fs)", i + 1, len(chunks), offset)
            segments = _call_ollama(chunk_text, config)
            logger.info("Chunk %d/%d returned %d segments", i + 1, len(chunks), len(segments))
            all_segments.extend(segments)

        # Merge duplicates from overlapping chunks
        if len(chunks) > 1:
            all_segments = _merge_segments(all_segments)
            logger.info("After merging: %d unique segments", len(all_segments))

        logger.info("Segmentation complete: %d topic segments identified", len(all_segments))

        return SegmentationResult(
            segments=all_segments,
            model_used=config.ollama_model,
        )

    except SegmentationError:
        raise
    except Exception as exc:
        raise SegmentationError(f"Topic segmentation failed: {exc}") from exc
