"""Tests for clipforge.logger."""
import logging

from clipforge.logger import setup_logging


def test_setup_logging_default():
    logger = setup_logging(verbose=False)
    assert logger.name == "clipforge"
    assert logger.level == logging.INFO


def test_setup_logging_verbose():
    logger = setup_logging(verbose=True)
    assert logger.level == logging.DEBUG


def test_setup_logging_has_handler():
    logger = setup_logging()
    assert len(logger.handlers) >= 1
