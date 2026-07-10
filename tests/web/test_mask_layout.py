"""Smoke tests for mask_layout — verify component IDs and structure."""

import pytest

dash = pytest.importorskip("dash")
from dash import dcc, html  # noqa: E402

from pysimplemask.web.mask_layout import build_mask_section  # noqa: E402

REQUIRED_IDS = {
    "mask-reset-btn", "mask-status", "mask-tabs",
    # blemish/files
    "blemish-path", "blemish-key", "blemish-eval-btn", "blemish-apply-btn",
    "maskfile-path", "maskfile-key", "maskfile-eval-btn", "maskfile-apply-btn",
    # threshold
    "thresh-low-enable", "thresh-low", "thresh-high-enable", "thresh-high",
    "morph-erode", "morph-dilate", "morph-open", "morph-close",
    "thresh-eval-btn", "thresh-apply-btn",
    # draw
    "draw-shape", "draw-mode", "draw-activate-btn", "draw-clear-btn",
    "draw-eval-btn", "draw-apply-btn",
    # manual
    "manual-pixels", "manual-upload", "manual-eval-btn", "manual-apply-btn",
    # outlier
    "outlier-target", "outlier-method", "outlier-cutoff", "outlier-param",
    "outlier-param-label", "outlier-eval-btn", "outlier-apply-btn",
    # parameterize
    "param-add-btn", "param-remove-btn", "param-rows",
    "param-eval-btn", "param-apply-btn",
}


def _collect_ids(component: object, found: set) -> None:
    if hasattr(component, "id") and component.id is not None:
        found.add(component.id)
    children = getattr(component, "children", None)
    if children is None:
        return
    items = children if isinstance(children, list) else [children]
    for child in items:
        _collect_ids(child, found)


def test_build_mask_section_returns_html_div():
    assert isinstance(build_mask_section(), html.Div)


def test_all_required_ids_present():
    section = build_mask_section()
    found: set = set()
    _collect_ids(section, found)
    missing = REQUIRED_IDS - found
    assert not missing, f"Missing IDs: {missing}"


def test_six_tabs_present():
    section = build_mask_section()

    def find_tabs(comp):
        if isinstance(comp, dcc.Tabs):
            return comp
        for child in getattr(comp, "children", None) or []:
            result = find_tabs(child)
            if result is not None:
                return result
        return None

    tabs = find_tabs(section)
    assert tabs is not None, "dcc.Tabs not found"
    values = {t.value for t in tabs.children if hasattr(t, "value")}
    assert values == {"blemish", "threshold", "draw", "manual", "outlier", "parameterize"}
