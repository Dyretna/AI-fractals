# logging_setup.py

import logging
import sys
from pathlib import Path


def setup_logging(level=logging.INFO, redirect_path: Path = None):
    root = logging.getLogger()
    root.setLevel(level)

    # rensa gamla handlers
    root.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(level)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if redirect_path:
        file_handler = logging.FileHandler(redirect_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
