# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.

"""Centralized logging configuration for the ACT package."""

import logging
import os

# Flag to track if logging has been configured by this module
_logging_configured = False


def get_logger(name: str) -> logging.Logger:
    """Get a logger with auto-configured environment variable support.

    This function automatically configures logging based on LOGLEVEL environment
    variable on first call, then returns a logger for the specified name.

    Args:
        name: Usually __name__ from the calling module

    Returns:
        logging.Logger: Configured logger for the module

    Example:
        logger = get_logger(__name__)
        logger.debug("This respects LOGLEVEL environment variable")
    """
    # Configure logging on first call
    global _logging_configured
    if not _logging_configured:
        _setup_logging()
        _logging_configured = True

    return logging.getLogger(name)


def _setup_logging():
    """Internal function to configure logging based on LOGLEVEL environment variable."""
    log_level = os.getenv("LOGLEVEL", "INFO").upper()

    # Map string levels to logging constants
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    level = level_map.get(log_level, logging.INFO)

    # Configure root logger - force=True ensures we override any existing config
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Override any existing configuration
    )

    # Ensure the root logger level is also set correctly
    print("*** Setting loglevel to", level)
    logging.getLogger().setLevel(level)
