"""Logging setup for ClipForge."""
from __future__ import annotations

import logging


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return the clipforge logger.

    Args:
        verbose: If True, set log level to DEBUG; otherwise INFO.

    Returns:
        Configured logger instance named "clipforge".
    """
    logger = logging.getLogger("clipforge")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
