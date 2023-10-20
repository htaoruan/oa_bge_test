import logging
import os
from logging import Logger

from configs.kb_configs import LOG_LEVEL
from configs.kb_configs import LOG_ROOT_PATH
from configs.kb_configs import LOG_DEFAULT_FILE


def get_log_level_from_str(log_level_str: str = LOG_LEVEL) -> int:
    log_level_dict = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }

    return log_level_dict.get(log_level_str.upper(), logging.INFO)


def setup_logger(
    name: str = __name__, 
    log_level: int = get_log_level_from_str(),
    log_file_name: str = LOG_DEFAULT_FILE
    ) -> Logger:

    logger = logging.getLogger(name)

    # If the logger already has handlers, assume it was already configured and return it.
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    formatter = logging.Formatter(
        '%(asctime)s - %(filename)s:%(lineno)d - %(message)s', 
        datefmt='%Y-%m-%d %I:%M:%S %p'
    )

    log_file = os.path.join(LOG_ROOT_PATH, log_file_name)
    handler = logging.FileHandler(log_file)
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
