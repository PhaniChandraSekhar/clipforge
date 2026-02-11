"""Command-line interface for ClipForge."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clipforge.config import PipelineConfig
from clipforge.errors import ClipForgeError
from clipforge.logger import setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list to parse; defaults to sys.argv[1:].

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        prog="clipforge",
        description="ClipForge: AIML Training Video to YouTube Shorts Pipeline",
    )

    parser.add_argument(
        "input_video",
        type=Path,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory for clips. Defaults to clips_<video_stem>/.",
    )
    parser.add_argument(
        "-w", "--whisper-model",
        type=str,
        default="base",
        help="Whisper model size (default: base).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="ollama",
        choices=["ollama", "anthropic"],
        dest="llm_provider",
        help="LLM provider for topic segmentation (default: ollama).",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="llama3.1:8b",
        help="Ollama model for topic segmentation (default: llama3.1:8b).",
    )
    parser.add_argument(
        "--anthropic-model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Anthropic model for topic segmentation (default: claude-sonnet-4-5-20250929).",
    )
    parser.add_argument(
        "--min-clip",
        type=int,
        default=30,
        dest="min_clip_duration",
        help="Minimum clip duration in seconds (default: 30).",
    )
    parser.add_argument(
        "--max-clip",
        type=int,
        default=600,
        dest="max_clip_duration",
        help="Maximum clip duration in seconds (default: 600).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code for transcription (auto-detected if omitted).",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        default=False,
        help="Skip interactive clip review and approve all clips.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from last checkpoint if available.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device: cpu, cuda, or auto (default: auto).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG) logging.",
    )

    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Build a PipelineConfig from parsed CLI arguments.

    If output_dir was not specified, it defaults to
    ``clips_<video_stem>`` in the current working directory.

    Args:
        args: Parsed argument namespace from parse_args().

    Returns:
        Populated PipelineConfig instance.
    """
    output_dir = args.output_dir
    if output_dir is None:
        video_stem = args.input_video.stem
        output_dir = Path.cwd() / f"clips_{video_stem}"

    return PipelineConfig(
        input_video=args.input_video,
        output_dir=output_dir,
        whisper_model=args.whisper_model,
        llm_provider=args.llm_provider,
        ollama_model=args.ollama_model,
        anthropic_model=args.anthropic_model,
        min_clip_duration=args.min_clip_duration,
        max_clip_duration=args.max_clip_duration,
        language=args.language,
        skip_review=args.skip_review,
        resume=args.resume,
        device=args.device,
        verbose=args.verbose,
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ClipForge CLI."""
    args = parse_args(argv)
    config = build_config(args)
    logger = setup_logging(verbose=config.verbose)

    logger.info("ClipForge starting")
    logger.debug("Config: %s", config.model_dump())

    try:
        from clipforge.pipeline import Pipeline

        pipeline = Pipeline(config)
        output = pipeline.run()
        logger.info("Pipeline complete. Output directory: %s", output)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(130)
    except ClipForgeError as exc:
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)
