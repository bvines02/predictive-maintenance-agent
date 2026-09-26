"""Tests for src/nasa_score.py (PHM08 / C-MAPSS asymmetric scoring function)."""

import numpy as np
import pytest

from src.nasa_score import nasa_cmapss_score, nasa_cmapss_scores


def test_zero_error_scores_zero():
    assert nasa_cmapss_scores([0.0])[0] == pytest.approx(0.0)


def test_known_positive_error_value():
    # d = 10 (late/optimistic): exp(10/10) - 1 = e - 1
    assert nasa_cmapss_scores([10.0])[0] == pytest.approx(np.e - 1)


def test_known_negative_error_value():
    # d = -13 (early/conservative): exp(13/13) - 1 = e - 1
    assert nasa_cmapss_scores([-13.0])[0] == pytest.approx(np.e - 1)


def test_late_error_penalised_more_than_early_error_of_same_magnitude():
    late = nasa_cmapss_scores([13.0])[0]  # over-prediction
    early = nasa_cmapss_scores([-13.0])[0]  # under-prediction, same magnitude

    assert late > early


def test_score_increases_with_error_magnitude():
    small = nasa_cmapss_scores([5.0])[0]
    large = nasa_cmapss_scores([20.0])[0]

    assert large > small


def test_total_score_sums_per_observation_scores():
    errors = [0.0, 10.0, -13.0]

    total = nasa_cmapss_score(errors)

    assert total == pytest.approx(sum(nasa_cmapss_scores(errors)))


def test_never_negative():
    scores = nasa_cmapss_scores([-50.0, -5.0, 0.0, 5.0, 50.0])

    assert (scores >= 0).all()
