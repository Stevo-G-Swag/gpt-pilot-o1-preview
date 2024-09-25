import os
from logging import FileHandler, Formatter, Logger, StreamHandler, getLogger

from core.config import LogConfig


def setup(config: LogConfig, force: bool = False):
    """
    Set up logging based on the current configuration.

    The method is idempotent unless `force` is set to True,
    in which case it will reconfigure the logging.
    """

    root = getLogger()
    logger = getLogger("core")
    # Only clear/remove existing log handlers if we're forcing a new setup
    if not force and (root.handlers or logger.handlers):
        return

    while force and root.handlers:
        root.removeHandler(root.handlers[0])

    while force and logger.handlers:
        logger.removeHandler(logger.handlers[0])

    level = config.level
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if config.output:
        # Ensure the log directory exists
        log_dir = os.path.dirname(config.output)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handler = FileHandler(config.output, encoding="utf-8")
    else:
        handler = StreamHandler()

    handler.setFormatter(formatter)
    handler.setLevel(level)

    logger.setLevel(level)
    logger.addHandler(handler)


def get_logger(name) -> Logger:
    """
    Get logger instance for a given (module) name.

    :return: Logger instance
    """
    return getLogger(name)


__all__ = ["setup", "get_logger"]
