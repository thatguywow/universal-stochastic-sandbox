"""Adaptive sample sizing."""

from __future__ import annotations

import numpy as np
import pytest

from uss import UniversalStochasticSandbox


def test_stops_once_target_precision_is_reached() -> None:
    sandbox = UniversalStochasticSandbox(seed=71)
    target = 5e-4
    result = sandbox.run_until_precision(
        "gaussian",
        {"mean": 20.0, "std_dev": 3.0},
        target_standard_error=target,
        batch_size=500_000,
    )
    assert result.report.monte_carlo_error <= target
    # sigma/sqrt(n) = 5e-4 implies n ~ 3.6e7; it must not have run to the cap.
    assert result.sample_size < 200_000_000
    assert result.report.point_estimate == pytest.approx(20.0, abs=0.01)


def test_easy_target_stops_after_one_batch() -> None:
    sandbox = UniversalStochasticSandbox(seed=72)
    result = sandbox.run_until_precision(
        "gaussian",
        {"mean": 0.0, "std_dev": 1.0},
        target_standard_error=0.01,
        batch_size=100_000,
    )
    assert result.sample_size == 100_000


def test_unreachable_target_stops_at_cap_and_says_so() -> None:
    sandbox = UniversalStochasticSandbox(seed=73)
    result = sandbox.run_until_precision(
        "gaussian",
        {"mean": 0.0, "std_dev": 1.0},
        target_standard_error=1e-9,
        max_samples=200_000,
        batch_size=100_000,
    )
    assert result.sample_size == 200_000
    assert any("above the target" in c for c in result.report.caveats)


def test_running_moments_match_a_full_rescan() -> None:
    """The incremental stopping statistic must agree with the final report."""
    sandbox = UniversalStochasticSandbox(seed=74)
    result = sandbox.run_until_precision(
        "gaussian",
        {"mean": 5.0, "std_dev": 2.0},
        target_standard_error=2e-3,
        batch_size=250_000,
    )
    expected_se = np.sqrt(result.report.variance / result.sample_size)
    assert result.report.monte_carlo_error == pytest.approx(expected_se, rel=1e-9)


def test_invalid_arguments_rejected() -> None:
    sandbox = UniversalStochasticSandbox(seed=75)
    with pytest.raises(ValueError, match="target_standard_error must be positive"):
        sandbox.run_until_precision("gaussian", {}, target_standard_error=0.0)
    with pytest.raises(ValueError, match="batch_size must be at least 2"):
        sandbox.run_until_precision("gaussian", {}, target_standard_error=0.1, batch_size=1)
