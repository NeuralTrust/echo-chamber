"""Module for logging related code."""

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """Define project logger.

    Args:
        name (str): name of the file where the logger is set.

    Returns:
        (logging.Logger)
    """
    logger = logging.getLogger(name)

    # Get log level from environment variable, default to INFO
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, log_level, logging.INFO))

    formatter = logging.Formatter(
        "[%(asctime)s]-[%(name)s]-[%(levelname)s]: %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
