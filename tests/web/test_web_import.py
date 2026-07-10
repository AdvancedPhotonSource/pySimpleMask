"""Smoke tests — verify web package imports without error."""

import pytest

dash = pytest.importorskip("dash")
import flask  # noqa: E402
import dash as dash_mod  # noqa: E402


def test_server_is_flask_app():
    from pysimplemask.web.server import server
    assert isinstance(server, flask.Flask)


def test_app_is_dash_app():
    from pysimplemask.web.server import app
    assert isinstance(app, dash_mod.Dash)


def test_model_is_simplemaskmodel():
    from pysimplemask.web.server import model
    from pysimplemask.core.model import SimpleMaskModel
    assert isinstance(model, SimpleMaskModel)


def test_main_web_is_callable():
    from pysimplemask.web.server import main_web
    assert callable(main_web)


def test_build_layout_returns_html_div():
    from pysimplemask.web.layout import build_layout
    from dash import html
    layout = build_layout(initial_path="/test/path.h5")
    assert isinstance(layout, html.Div)


def test_build_layout_file_path_pre_populated():
    from pysimplemask.web.layout import build_layout
    layout = build_layout(initial_path="/data/scan.h5")
    # Recursively search for the file-path Input component
    def find_value(component, target_id):
        if hasattr(component, "id") and component.id == target_id:
            return component.value
        for child in getattr(component, "children", []) or []:
            if isinstance(child, list):
                for c in child:
                    result = find_value(c, target_id)
                    if result is not None:
                        return result
            result = find_value(child, target_id)
            if result is not None:
                return result
        return None

    value = find_value(layout, "file-path")
    assert value == "/data/scan.h5"


def test_callbacks_module_imports():
    # Importing callbacks registers all @callback decorators as a side effect.
    # This test ensures no NameError or import cycle at import time.
    import pysimplemask.web.callbacks  # noqa: F401


def test_mask_callbacks_imports():
    import pysimplemask.web.mask_callbacks  # noqa: F401
