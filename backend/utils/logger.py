"""
backend/utils/logger.py
───────────────────────
Structured logging using structlog.
Outputs JSON in production, pretty console output in development.
"""

import logging
import sys
from pathlib import Path

import structlog


def setup_logging(log_level: str = "INFO", log_format: str = "console", log_file: str | None = None) -> None:
    """
    Configure structlog for the entire application.

    Args:
        log_level: Logging level (DEBUG / INFO / WARNING / ERROR)
        log_format: "json" for structured JSON, "console" for human-readable
        log_file: Optional path to write logs to file
    """
    # Build processor chain
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if log_format == "json":
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(log_level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging (used by uvicorn, fastapi, etc.)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(log_level),
    )

    # Optional: write to file
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.getLevelName(log_level))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a named logger.

    Usage:
        logger = get_logger(__name__)
        logger.info("Processing document", doc_id="abc123", chunk_count=42)
    """
    return structlog.get_logger(name)
