#!/usr/bin/env python3

# read version from installed package
from importlib.metadata import version
__version__ = version("ptarcade")
"""Get version from pyproject.toml metadata."""
