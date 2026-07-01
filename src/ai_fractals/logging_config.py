import logging
import sys
from pathlib import Path


def get_logger(name: str, redirect_path: Path = None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # remove old handlers so we can control level
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if redirect_path:
        file_handler = logging.FileHandler(redirect_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
