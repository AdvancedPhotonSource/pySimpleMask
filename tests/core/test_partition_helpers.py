import pytest

from pysimplemask.core.partition import least_multiple


@pytest.mark.parametrize(
    "a,b,expected",
    [(10, 100, 100), (10, 95, 100), (36, 360, 360), (36, 350, 360), (7, 1, 7)],
)
def test_least_multiple(a, b, expected):
    assert least_multiple(a, b) == expected
