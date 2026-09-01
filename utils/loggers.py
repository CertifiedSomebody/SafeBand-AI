"""
SAFEBAND AI - Logging Utility

Provides centralized application logging for the SAFEBAND AI
prototype.

Logs can be used for:
- Sensor activity
- AI processing
- Safety events
- Emergency alerts
- Cellular communication
- Cloud synchronization
- Application errors
"""

import logging
from pathlib import Path
from typing import Optional

from config.settings import BASE_DIR, LOG_CONFIG


# ============================================================
# LOG DIRECTORY
# ============================================================

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_FILE = LOG_DIR / "safeband_ai.log"


# ============================================================
# LOGGER CONFIGURATION
# ============================================================

_LOGGERS = {}


def get_logger(
    name: str = "SAFEBAND_AI"
) -> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
        Configured application logger.
    """

    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)

    if logger.handlers:
        _LOGGERS[name] = logger
        return logger

    level_name = str(
        LOG_CONFIG.get(
            "log_level",
            "INFO"
        )
    ).upper()

    level = getattr(
        logging,
        level_name,
        logging.INFO
    )

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --------------------------------------------------------
    # File Handler
    # --------------------------------------------------------

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )

    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(
        file_handler
    )

    # --------------------------------------------------------
    # Console Handler
    # --------------------------------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(
        console_handler
    )

    logger.propagate = False

    _LOGGERS[name] = logger

    return logger


# ============================================================
# DEFAULT LOGGER
# ============================================================

logger = get_logger(
    "SAFEBAND_AI"
)


# ============================================================
# LOGGING FUNCTIONS
# ============================================================

def log_debug(
    message: str
) -> None:
    """Log a debug message."""

    logger.debug(
        message
    )


def log_info(
    message: str
) -> None:
    """Log an informational message."""

    logger.info(
        message
    )


def log_warning(
    message: str
) -> None:
    """Log a warning message."""

    logger.warning(
        message
    )


def log_error(
    message: str
) -> None:
    """Log an error message."""

    logger.error(
        message
    )


def log_critical(
    message: str
) -> None:
    """Log a critical/emergency message."""

    logger.critical(
        message
    )


# ============================================================
# DOMAIN-SPECIFIC LOGGING
# ============================================================

def log_sensor_event(
    sensor_name: str,
    message: str
) -> None:
    """Log a sensor-related event."""

    logger.info(
        f"[SENSOR:{sensor_name}] {message}"
    )


def log_ai_event(
    module: str,
    message: str
) -> None:
    """Log an AI-processing event."""

    logger.info(
        f"[AI:{module}] {message}"
    )


def log_safety_event(
    activity: str,
    risk_score: float,
    message: str
) -> None:
    """Log a safety/risk event."""

    logger.warning(
        "[SAFETY] "
        f"Activity={activity} | "
        f"Risk={risk_score:.1f} | "
        f"{message}"
    )


def log_emergency_event(
    activity: str,
    risk_score: float,
    message: str
) -> None:
    """Log an emergency event."""

    logger.critical(
        "[EMERGENCY] "
        f"Activity={activity} | "
        f"Risk={risk_score:.1f} | "
        f"{message}"
    )


def log_communication_event(
    module: str,
    message: str
) -> None:
    """Log a communication event."""

    logger.info(
        f"[COMMUNICATION:{module}] {message}"
    )


def log_cloud_event(
    message: str
) -> None:
    """Log a cloud synchronization event."""

    logger.info(
        f"[CLOUD] {message}"
    )


# ============================================================
# EXCEPTION LOGGING
# ============================================================

def log_exception(
    message: str,
    exception: Optional[Exception] = None
) -> None:
    """
    Log an application exception.

    Parameters
    ----------
    message : str
        Description of the error.

    exception : Exception, optional
        Exception object associated with the error.
    """

    if exception is not None:
        logger.exception(
            f"{message}: {exception}"
        )
    else:
        logger.exception(
            message
        )


# ============================================================
# LOG FILE ACCESS
# ============================================================

def get_log_file() -> Path:
    """Return the SAFEBAND AI log-file path."""

    return LOG_FILE


def clear_log_file() -> None:
    """Clear the current log file."""

    try:
        LOG_FILE.write_text(
            "",
            encoding="utf-8"
        )

        logger.info(
            "Log file cleared."
        )

    except OSError as error:
        logger.error(
            f"Unable to clear log file: {error}"
        )