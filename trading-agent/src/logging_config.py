"""Centralized logging configuration for trading-agent modules
"""
import logging
from typing import Optional


def configure_logging(level: int = logging.INFO, fmt: Optional[str] = None):
    """Configure root logger for CLI/CI runs.

    Call this early in entrypoint scripts.
    """
    if fmt is None:
        fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
