#!/usr/bin/env python3
# read version from installed package
from importlib.metadata import version

__version__ = version("ptarcade")
"""Get version from pyproject.toml metadata."""

from rich.console import Console

console = Console()


# Set up logging
import logging

from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=console,
            show_time=False,
            show_path=False,
        ),
    ],
)
