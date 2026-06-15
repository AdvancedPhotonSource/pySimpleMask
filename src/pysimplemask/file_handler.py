import logging

from .reader import get_reader

logger = logging.getLogger(__name__)


def get_handler(beamline, fname, **kwargs):
    """Return a reader for the given beamline/file, or ``None`` on failure.

    Thin delegator over :func:`pysimplemask.reader.get_reader`; failures are
    logged and surfaced as ``None`` so the kernel can degrade gracefully.
    """
    try:
        return get_reader(beamline, fname, **kwargs)
    except Exception:
        logger.error(
            "failed to create a reader for beamline=%s file=%s",
            beamline,
            fname,
            exc_info=True,
        )
        return None
