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

    from pysimplemask.web import layout as _layout  # noqa: F401
    from pysimplemask.web import callbacks as _callbacks  # noqa: F401

    app.layout = _layout.build_layout(initial_path=args.path or "")

    print(f"pySimpleMask web interface running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
