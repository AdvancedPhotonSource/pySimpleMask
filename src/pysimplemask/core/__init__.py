# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Qt-free core: domain model, readers, and compute utilities."""

from .file_handler import get_handler
from .model import SimpleMaskModel

_REPORT_NAMES = ("generate_report", "generate_report_from_qmap", "report_from_qmap")

__all__ = [
    "SimpleMaskModel",
    "get_handler",
    "generate_report",
    "generate_report_from_qmap",
    "report_from_qmap",
]



def __getattr__(name):
    # report.py pulls in matplotlib; load it only when a report is requested
    if name in _REPORT_NAMES:
        from . import report

        return getattr(report, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
