"""Compatibility namespace for older imports referencing tema_template.*
This module re-exports the new src.tema package where possible so callers can
import from tema_template without changing code immediately.
"""
from tema import *  # re-export public API from src.tema

__all__ = getattr(__import__("tema"), "__all__", [])
