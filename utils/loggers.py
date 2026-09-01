"""
SAFEBAND AI - Centralized Logging Utility

Provides application-wide logging for:

    - Sensor activity
    - AI processing
    - Safety events
    - Emergency events
    - Cellular communication
    - Cloud synchronization
    - Application errors

The logging interface is intentionally centralized so individual
modules do not need to configure their own handlers.

The same API can be used during simulation, hardware integration,
local development, and future deployment.
"""


import logging
from pathlib import Path
from typing import Optional


from config.settings import (
    BASE_DIR,
    LOG_CONFIG,
)


# ============================================================
# CONFIGURATION
# ============================================================

LOG_DIR = (
    BASE_DIR
    / "logs"
)

LOG_FILE = (
    LOG_DIR
    / "safeband_ai.log"
)

LOGGER_NAME = "SAFEBAND_AI"

DEFAULT_LOG_LEVEL = "INFO"

DEFAULT_MAX_HISTORY = 100


# ============================================================
# INTERNAL LOGGER REGISTRY
# ============================================================

_LOGGERS: dict[str, logging.Logger] = {}


# ============================================================
# LOG DIRECTORY
# ============================================================

def _ensure_log_directory() -> None:
    """Create the log directory when logging is enabled."""

    if not LOG_CONFIG.get(
        "enabled",
        True,
    ):
        return

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# LOG LEVEL
# ============================================================

def _get_log_level() -> int:
    """Return the configured Python logging level."""

    level_name = str(
        LOG_CONFIG.get(
            "log_level",
            DEFAULT_LOG_LEVEL,
        )
    ).upper()

    return getattr(
        logging,
        level_name,
        logging.INFO,
    )


# ============================================================
# FORMATTER
# ============================================================

def _create_formatter() -> logging.Formatter:
    """Create the standard SAFEBAND log formatter."""

    return logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ============================================================
# LOGGER FACTORY
# ============================================================

def get_logger(
    name: str = LOGGER_NAME,
) -> logging.Logger:
    """
    Return a configured SAFEBAND logger.

    Reuses existing logger instances to prevent duplicate
    handlers and duplicate log messages.
    """

    normalized_name = str(
        name
    ).strip()

    if not normalized_name:
        normalized_name = LOGGER_NAME

    if normalized_name in _LOGGERS:
        return _LOGGERS[
            normalized_name
        ]

    logger = logging.getLogger(
        normalized_name
    )

    level = _get_log_level()

    logger.setLevel(
        level
    )

    logger.propagate = False

    # --------------------------------------------------------
    # Logging disabled
    # --------------------------------------------------------

    if not LOG_CONFIG.get(
        "enabled",
        True,
    ):

        logger.disabled = True

        _LOGGERS[
            normalized_name
        ] = logger

        return logger

    logger.disabled = False

    # --------------------------------------------------------
    # Prevent duplicate handlers
    # --------------------------------------------------------

    if logger.handlers:

        _LOGGERS[
            normalized_name
        ] = logger

        return logger

    _ensure_log_directory()

    formatter = _create_formatter()

    # --------------------------------------------------------
    # File handler
    # --------------------------------------------------------

    try:

        file_handler = logging.FileHandler(
            LOG_FILE,
            encoding="utf-8",
        )

        file_handler.setLevel(
            level
        )

        file_handler.setFormatter(
            formatter
        )

        logger.addHandler(
            file_handler
        )

    except OSError:
        # The application should still be usable when a local
        # log file cannot be created.
        pass

    # --------------------------------------------------------
    # Console handler
    # --------------------------------------------------------

    console_handler = (
        logging.StreamHandler()
    )

    console_handler.setLevel(
        level
    )

    console_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        console_handler
    )

    _LOGGERS[
        normalized_name
    ] = logger

    return logger


# ============================================================
# DEFAULT LOGGER
# ============================================================

logger = get_logger(
    LOGGER_NAME
)


# ============================================================
# BASIC LOGGING
# ============================================================

def log_debug(
    message: str,
) -> None:
    """Log a debug message."""

    logger.debug(
        str(message)
    )


def log_info(
    message: str,
) -> None:
    """Log an informational message."""

    logger.info(
        str(message)
    )


def log_warning(
    message: str,
) -> None:
    """Log a warning message."""

    logger.warning(
        str(message)
    )


def log_error(
    message: str,
) -> None:
    """Log an error message."""

    logger.error(
        str(message)
    )


def log_critical(
    message: str,
) -> None:
    """Log a critical/emergency message."""

    logger.critical(
        str(message)
    )


# ============================================================
# DOMAIN-SPECIFIC LOGGING
# ============================================================

def log_sensor_event(
    sensor_name: str,
    message: str,
) -> None:
    """Log a sensor-related event."""

    logger.info(
        "[SENSOR:%s] %s",
        str(sensor_name).upper(),
        message,
    )


def log_ai_event(
    module: str,
    message: str,
) -> None:
    """Log an AI-processing event."""

    logger.info(
        "[AI:%s] %s",
        str(module),
        message,
    )


def log_safety_event(
    activity: str,
    risk_score: float,
    message: str,
) -> None:
    """Log a safety/risk event."""

    logger.warning(
        "[SAFETY] "
        "Activity=%s | "
        "Risk=%.1f | "
        "%s",
        str(activity).upper(),
        float(risk_score),
        message,
    )


def log_emergency_event(
    activity: str,
    risk_score: float,
    message: str,
) -> None:
    """Log an emergency event."""

    logger.critical(
        "[EMERGENCY] "
        "Activity=%s | "
        "Risk=%.1f | "
        "%s",
        str(activity).upper(),
        float(risk_score),
        message,
    )


def log_communication_event(
    module: str,
    message: str,
) -> None:
    """Log a communication-related event."""

    logger.info(
        "[COMMUNICATION:%s] %s",
        str(module),
        message,
    )


def log_cloud_event(
    message: str,
) -> None:
    """Log a cloud synchronization event."""

    logger.info(
        "[CLOUD] %s",
        message,
    )


# ============================================================
# EXCEPTION LOGGING
# ============================================================

def log_exception(
    message: str,
    exception: Optional[Exception] = None,
) -> None:
    """
    Log an exception with traceback information.

    If an exception is supplied, its message is included.
    """

    if exception is not None:

        logger.error(
            "%s: %s",
            message,
            exception,
            exc_info=True,
        )

    else:

        logger.error(
            message,
            exc_info=True,
        )


# ============================================================
# LOG FILE ACCESS
# ============================================================

def get_log_file() -> Path:
    """Return the SAFEBAND application log-file path."""

    return LOG_FILE


def get_log_directory() -> Path:
    """Return the SAFEBAND log directory."""

    return LOG_DIR


def clear_log_file() -> bool:
    """
    Clear the current log file.

    Returns
    -------
    bool
        True when the file was cleared successfully.
    """

    try:

        _ensure_log_directory()

        LOG_FILE.write_text(
            "",
            encoding="utf-8",
        )

        logger.info(
            "Log file cleared."
        )

        return True

    except OSError as error:

        logger.error(
            "Unable to clear log file: %s",
            error,
        )

        return False


# ============================================================
# LOGGER STATUS
# ============================================================

def is_logging_enabled() -> bool:
    """Return whether application logging is enabled."""

    return bool(
        LOG_CONFIG.get(
            "enabled",
            True,
        )
    )


def get_logger_status() -> dict:
    """Return basic logging-system status."""

    return {
        "enabled": is_logging_enabled(),
        "log_directory": str(
            LOG_DIR
        ),
        "log_file": str(
            LOG_FILE
        ),
        "log_level": str(
            LOG_CONFIG.get(
                "log_level",
                DEFAULT_LOG_LEVEL,
            )
        ).upper(),
        "logger_count": len(
            _LOGGERS
        ),
    }


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "get_logger",
    "logger",

    "log_debug",
    "log_info",
    "log_warning",
    "log_error",
    "log_critical",

    "log_sensor_event",
    "log_ai_event",
    "log_safety_event",
    "log_emergency_event",
    "log_communication_event",
    "log_cloud_event",

    "log_exception",

    "get_log_file",
    "get_log_directory",
    "clear_log_file",

    "is_logging_enabled",
    "get_logger_status",
]