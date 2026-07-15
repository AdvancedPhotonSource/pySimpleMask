# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Smoke tests for partition_layout — verify component IDs and structure."""

import pytest

dash = pytest.importorskip("dash")
from dash import dcc, html  # noqa: E402

from pysimplemask.web.partition_layout import build_partition_section  # noqa: E402

REQUIRED_IDS = {
    "partition-tabs", "partition-compute-btn", "partition-status",
    "save-path", "save-partition-btn", "save-mask-btn", "save-status",
    "download-partition-btn", "download-partition-data",
    # q-phi
    "qphi-dq-num", "qphi-dp-num", "qphi-sq-num", "qphi-sp-num",
    "qphi-style", "qphi-phi-offset", "qphi-symmetry-fold",
    # xy-mesh
    "xy-dq-num", "xy-dp-num", "xy-sq-num", "xy-sp-num",
    # general
    "gen-map0", "gen-dq-num", "gen-sq-num", "gen-style",
    "gen-map1", "gen-dp-num", "gen-sp-num",
    # eq-ephi
    "eqephi-dq-num", "eqephi-dp-num", "eqephi-sq-num", "eqephi-sp-num",
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


def test_build_partition_section_returns_html_div():
    assert isinstance(build_partition_section(), html.Div)


def test_all_required_ids_present():
    section = build_partition_section()
    found: set = set()
    _collect_ids(section, found)
    missing = REQUIRED_IDS - found
    assert not missing, f"Missing IDs: {missing}"


def test_four_tabs_present():
    section = build_partition_section()

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
    assert values == {"q-phi", "xy-mesh", "general", "eq-ephi"}
