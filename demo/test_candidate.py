"""Test suite exercising the candidate module."""

import pytest

from candidate import mean, median, normalize, percentile, stdev, summarize, variance


def test_mean():
    assert mean([1, 2, 3, 4]) == 2.5


def test_median_odd():
    assert median([3, 1, 2]) == 2


def test_median_even():
    assert median([1, 2, 3, 4]) == 2.5


def test_variance_sample():
    assert variance([1, 2, 3, 4]) == pytest.approx(1.6666666, rel=1e-4)


def test_stdev():
    assert stdev([1, 2, 3, 4]) == pytest.approx(1.2909944, rel=1e-4)


def test_normalize_range():
    assert normalize([0, 5, 10]) == [0.0, 0.5, 1.0]


def test_normalize_constant():
    assert normalize([4, 4, 4]) == [0.0, 0.0, 0.0]


def test_percentile_max():
    assert percentile([1, 2, 3, 4, 5], 100) == 5


def test_summarize():
    out = summarize([1, 2, 3, 4])
    assert out["mean"] == 2.5
    assert out["median"] == 2.5
