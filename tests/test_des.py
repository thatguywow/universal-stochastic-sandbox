"""Jump sampling must reproduce Poisson-process statistics exactly."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from uss import des


def test_event_count_over_horizon_is_poisson() -> None:
    rng = np.random.default_rng(31)
    rate, horizon = 2.0, 50.0
    counts = [des.jump_sample(rate, horizon, rng).count for _ in range(1500)]
    expected = rate * horizon
    assert np.mean(counts) == pytest.approx(expected, rel=0.03)
    # Poisson variance equals its mean.
    assert np.var(counts) == pytest.approx(expected, rel=0.15)


def test_timestamps_are_sorted_and_within_horizon() -> None:
    stream = des.jump_sample(5.0, 20.0, np.random.default_rng(32))
    assert np.all(np.diff(stream.timestamps) > 0)
    assert np.all(stream.timestamps < 20.0)
    assert np.all(stream.timestamps >= 0)


def test_inter_arrival_times_are_exponential() -> None:
    rng = np.random.default_rng(33)
    rate = 3.0
    gaps = des.next_event_times(rate, 200_000, rng)
    assert gaps.mean() == pytest.approx(1.0 / rate, rel=0.01)
    ks = stats.kstest(gaps, "expon", args=(0.0, 1.0 / rate))
    assert ks.pvalue > 0.01


def test_jump_sampling_skips_empty_time_cheaply() -> None:
    """A 200,000-unit horizon at rate 5e-6 costs one small array, not 200k steps."""
    stream = des.jump_sample(5e-6, 200_000.0, np.random.default_rng(34))
    assert stream.count < 20  # ~1 expected event
    assert stream.horizon == 200_000.0


def test_zero_rate_yields_no_events() -> None:
    stream = des.jump_sample(0.0, 100.0, np.random.default_rng(35))
    assert stream.count == 0
    assert stream.empirical_rate == 0.0


def test_thinning_reproduces_non_homogeneous_intensity() -> None:
    """lambda(t) = 1 + sin(t) over [0, 20pi] integrates to 20pi."""
    rng = np.random.default_rng(36)
    horizon = 20 * np.pi

    def rate_fn(t: np.ndarray) -> np.ndarray:
        return 1.0 + np.sin(t)

    counts = [
        des.thinned_sample(rate_fn, 2.0, horizon, rng).count for _ in range(400)
    ]
    assert np.mean(counts) == pytest.approx(horizon, rel=0.05)


def test_thinning_rejects_rate_exceeding_dominating_bound() -> None:
    rng = np.random.default_rng(37)
    with pytest.raises(ValueError, match="exceeded rate_max"):
        des.thinned_sample(lambda t: np.full_like(t, 5.0), 1.0, 10.0, rng)


def test_thinning_rejects_negative_intensity() -> None:
    rng = np.random.default_rng(38)
    with pytest.raises(ValueError, match="negative intensity"):
        des.thinned_sample(lambda t: -np.ones_like(t), 1.0, 10.0, rng)


def test_time_to_first_event_matches_analytic_mean() -> None:
    waits = des.time_to_first_event(0.25, 200_000, np.random.default_rng(39))
    assert waits.mean() == pytest.approx(4.0, rel=0.02)


def test_invalid_parameters_raise() -> None:
    rng = np.random.default_rng(40)
    with pytest.raises(ValueError, match="horizon must be positive"):
        des.jump_sample(1.0, 0.0, rng)
    with pytest.raises(ValueError, match="rate must be non-negative"):
        des.jump_sample(-1.0, 10.0, rng)
