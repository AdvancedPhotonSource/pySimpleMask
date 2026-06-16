"""Qt-free core: domain model, readers, and compute utilities."""

from .file_handler import get_handler
from .model import SimpleMaskModel

__all__ = ["SimpleMaskModel", "get_handler"]
