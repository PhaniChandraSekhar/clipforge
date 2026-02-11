"""Interactive terminal review for validated clips."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clipforge.config import PipelineConfig
    from clipforge.models import ValidatedClip

logger = logging.getLogger("clipforge")


def _format_time(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def _parse_time(time_str: str) -> float:
    """Parse MM:SS or M:SS back to seconds.

    Raises ``ValueError`` on invalid format.
    """
    parts = time_str.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time format: {time_str!r} (expected MM:SS)")
    try:
        minutes = int(parts[0])
        seconds = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid time format: {time_str!r} (expected MM:SS)")
    if minutes < 0 or seconds < 0 or seconds >= 60:
        raise ValueError(f"Invalid time format: {time_str!r} (expected MM:SS)")
    return minutes * 60.0 + seconds


def review_clips(
    clips: list[ValidatedClip], config: PipelineConfig
) -> list[ValidatedClip]:
    """Interactively review clips in the terminal.

    When *config.skip_review* is ``True``, every clip is auto-approved and
    returned immediately.  Otherwise the user is presented with each clip and
    may approve, reject, edit the title, or adjust start/end times.
    """
    if config.skip_review:
        for clip in clips:
            clip.status = "approved"
        logger.info("Review skipped — all %d clips auto-approved", len(clips))
        return clips

    total = len(clips)

    for clip in clips:
        while True:
            print("=" * 60)
            print(f"Clip {clip.index}/{total}")
            print(f"Title: {clip.title}")
            print(
                f"Time: {_format_time(clip.start_time)} - "
                f"{_format_time(clip.end_time)} ({clip.duration:.0f}s)"
            )
            print(f"Description: {clip.description}")
            print("Key Quotes:")
            for quote in clip.key_quotes:
                print(f"  - {quote}")
            print()
            print("[a]pprove  [r]eject  [t]edit title  [s]edit start  [e]edit end  [d]one")

            choice = input("> ").strip().lower()

            if choice == "a":
                clip.status = "approved"
                logger.info("Clip %d approved", clip.index)
                break
            elif choice == "r":
                clip.status = "rejected"
                logger.info("Clip %d rejected", clip.index)
                break
            elif choice == "t":
                new_title = input("New title: ").strip()
                if new_title:
                    clip.title = new_title
                # loop back to show updated clip
            elif choice == "s":
                raw = input("New start (MM:SS): ").strip()
                try:
                    new_start = _parse_time(raw)
                    clip.start_time = new_start
                    clip.duration = clip.end_time - clip.start_time
                except ValueError as exc:
                    print(str(exc))
                # loop back to show updated clip
            elif choice == "e":
                raw = input("New end (MM:SS): ").strip()
                try:
                    new_end = _parse_time(raw)
                    clip.end_time = new_end
                    clip.duration = clip.end_time - clip.start_time
                except ValueError as exc:
                    print(str(exc))
                # loop back to show updated clip
            elif choice == "d":
                break
            else:
                print("Invalid option")

        # 'd' breaks the inner loop and the outer for-loop should also stop
        if choice == "d":
            break

    # Summary
    approved = sum(1 for c in clips if c.status == "approved")
    rejected = sum(1 for c in clips if c.status == "rejected")
    pending = sum(1 for c in clips if c.status == "pending")
    print()
    print(f"Approved: {approved}, Rejected: {rejected}, Pending: {pending}")

    confirm = input("Proceed with extraction? (y/n): ").strip().lower()
    if confirm == "n":
        raise SystemExit("Extraction cancelled by user")

    # Auto-approve any remaining pending clips
    for clip in clips:
        if clip.status == "pending":
            clip.status = "approved"
            logger.info("Clip %d auto-approved (was pending)", clip.index)

    return [clip for clip in clips if clip.status == "approved"]
