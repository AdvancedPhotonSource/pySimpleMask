"""Tests for pysimplemask subcommand dispatcher."""


def test_run_web_importable():
    from pysimplemask.web.server import run_web  # noqa: F401
    assert callable(run_web)
