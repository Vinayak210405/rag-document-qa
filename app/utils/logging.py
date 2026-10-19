"""Logging helpers: one-line, key=value structured events that are easy to grep."""

from __future__ import annotations

import json
import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once (idempotent)."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def log_event(logger: logging.Logger, event: str, **fields: object) -> None:
    """Log ``event=<name> key=value ...`` (strings JSON-quoted so spaces stay unambiguous)."""
    rendered = " ".join(
        f"{k}={json.dumps(v) if isinstance(v, str) else v}" for k, v in fields.items()
    )
    logger.info("event=%s %s", event, rendered)
