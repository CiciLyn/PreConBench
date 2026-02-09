"""
Logging configuration module for the medical consultation simulation system.

This module provides functionality for:
1. Setting up logging configuration
2. Creating log file handlers
3. Configuring log format and levels
4. Managing console and file output
"""

import os
import sys
import time
import logging
from typing import NoReturn
from pathlib import Path


def setup_logger() -> logging.Logger:
    """Set up and configure the main logger.

    Returns:
        logging.Logger: Configured logger instance
    """
    # Initialize logger
    logger = logging.getLogger("main_run")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate logging
    logger.handlers.clear()
    logger.propagate = False

    # Create log directory if it doesn't exist
    log_dir = Path(__file__).parent.parent
    log_path = log_dir / "main_run.log"

    # Initialize log file with timestamp
    _initialize_log_file(log_path)

    # Configure handlers
    file_handler = _create_file_handler(log_path)
    console_handler = _create_console_handler()

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def _initialize_log_file(log_path: Path) -> NoReturn:
    """Initialize the log file with a timestamp.

    Args:
        log_path: Path to the log file
    """
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"日志开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception as e:
        print(f"Failed to initialize log file: {str(e)}")
        sys.exit(1)


def _create_file_handler(log_path: Path) -> logging.FileHandler:
    """Create and configure a file handler for logging.

    Args:
        log_path: Path to the log file

    Returns:
        logging.FileHandler: Configured file handler
    """
    handler = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    handler.setFormatter(_create_formatter())
    return handler


def _create_console_handler() -> logging.StreamHandler:
    """Create and configure a console handler for logging.

    Returns:
        logging.StreamHandler: Configured console handler
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_create_formatter())
    return handler


def _create_formatter() -> logging.Formatter:
    """Create a logging formatter.

    Returns:
        logging.Formatter: Configured formatter
    """
    return logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# Initialize logger
logger = setup_logger()
