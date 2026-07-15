# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Flask + Dash application and module-level model singleton."""

from __future__ import annotations

import argparse

import dash
import flask

from pysimplemask.core.model import SimpleMaskModel

# Module-level singletons — single-user, single-process.
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, title="pySimpleMask")
model = SimpleMaskModel()


def run_web(
    host: str = "127.0.0.1",
    port: int = 8050,
    path: str | None = None,
    debug: bool = False,
) -> None:
    """Start the Dash web server.

    Called by ``main_web()`` and by the ``pysimplemask web`` subcommand.
    """
    from pysimplemask.web import layout as _layout

    try:
        from pysimplemask.web import callbacks as _callbacks  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pysimplemask.web.callbacks not found. "
            "Ensure the web package is fully installed."
        ) from exc
    try:
        from pysimplemask.web import mask_callbacks as _mask_callbacks  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pysimplemask.web.mask_callbacks not found. "
            "Ensure the web package is fully installed."
        ) from exc
    try:
        from pysimplemask.web import partition_callbacks as _partition_callbacks  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pysimplemask.web.partition_callbacks not found. "
            "Ensure the web package is fully installed."
        ) from exc

    app.layout = _layout.build_layout(initial_path=path or "")
    print(f"pySimpleMask web interface running at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


def main_web() -> None:
    """Console-script entry point: ``pysimplemask-web``."""
    parser = argparse.ArgumentParser(
        prog="pysimplemask-web",
        description="Launch the pySimpleMask web interface.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8050,
        help="Port number (default: 8050)",
    )
    parser.add_argument(
        "--path", default=None,
        help="Pre-populate the file-path field with this path",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable Dash debug/hot-reload mode",
    )
    args = parser.parse_args()
    run_web(host=args.host, port=args.port, path=args.path, debug=args.debug)
