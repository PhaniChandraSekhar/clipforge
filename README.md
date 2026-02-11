# ClipForge

ClipForge is a CLI tool that automatically converts long AI/ML training videos into short, topic-based clips suitable for YouTube Shorts.

It uses **OpenAI Whisper** for speech-to-text transcription and a local **Ollama LLM** for intelligent topic segmentation.

## Prerequisites

- **Python** 3.10+
- **FFmpeg** installed and available on PATH
- **Ollama** running locally with a pulled model (default: `llama3.1:8b`)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/clipforge.git
cd clipforge

# Install in development mode
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Usage

```bash
clipforge input_video.mp4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output-dir` | `clips_<video_stem>/` | Output directory for clips |
| `-w, --whisper-model` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) |
| `--ollama-model` | `llama3.1:8b` | Ollama model for topic segmentation |
| `--min-clip` | `30` | Minimum clip duration in seconds |
| `--max-clip` | `600` | Maximum clip duration in seconds |
| `--language` | auto-detect | Language code for transcription |
| `--skip-review` | `false` | Skip interactive review; approve all clips |
| `--resume` | `false` | Resume from last checkpoint |
| `--device` | `auto` | Compute device: `cpu`, `cuda`, or `auto` |
| `--verbose` | `false` | Enable debug logging |

### Examples

```bash
# Basic usage
clipforge lecture.mp4

# Specify output directory and use a larger Whisper model
clipforge lecture.mp4 -o my_clips -w medium

# Skip interactive review and use GPU
clipforge lecture.mp4 --skip-review --device cuda

# Resume a previously interrupted run
clipforge lecture.mp4 --resume
```

## Pipeline

ClipForge processes videos through a 7-stage pipeline:

1. **Audio Extraction** - Extracts 16 kHz mono WAV audio from the video using FFmpeg.
2. **Transcription** - Runs Whisper to produce word-level timestamped text.
3. **Topic Segmentation** - Sends the transcript to an Ollama LLM to identify self-contained topics.
4. **Validation** - Enforces duration constraints, snaps boundaries to sentence edges, resolves overlaps.
5. **Review** - Interactive terminal UI to approve, reject, or edit each clip.
6. **Extraction** - Cuts approved clips from the source video with H.264/AAC encoding.
7. **Manifest** - Writes a JSON manifest describing all exported clips.

Each stage saves a checkpoint so long-running pipelines can be resumed with `--resume`.

## Project Structure

```
clipforge/
  __init__.py      # Package metadata
  __main__.py      # python -m clipforge entry point
  cli.py           # Argument parsing and CLI entry point
  config.py        # Pydantic-based pipeline configuration
  models.py        # Shared data models
  errors.py        # Custom exception hierarchy
  logger.py        # Logging setup
  pipeline.py      # Pipeline orchestrator
  audio.py         # FFmpeg audio extraction
  transcribe.py    # Whisper transcription
  segment.py       # Ollama-based topic segmentation
  validate.py      # Clip validation and boundary snapping
  review.py        # Interactive terminal review
  extract.py       # FFmpeg clip extraction
  manifest.py      # JSON manifest generation
```

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=clipforge
```

## License

MIT
