"""Variance reduction must actually reduce variance -- and report it correctly."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from uss import variance


def test_antithetic_reduces_variance_on_monotone_functional() -> None:
    """E[exp(0.3 X)] for X ~ N(0,1) has closed form exp(0.045)."""
    truth = np.exp(0.5 * 0.3**2)
    rng = np.random.default_rng(11)
    n = 400_000

    plain = variance.plain(np.exp(0.3 * stats.norm.ppf(rng.random(n))))
    anti = variance.antithetic(
        lambda u: np.exp(0.3 * stats.norm.ppf(u)), rng.random(n // 2)
    )

    assert anti.value == pytest.approx(truth, abs=4 * anti.standard_error)
    # Same number of function evaluations, materially smaller error.
    assert anti.standard_error < plain.standard_error / 2


def test_antithetic_standard_error_is_pair_based_not_pooled() -> None:
    """Pooling dependent halves and calling stats.sem overstates the error.

    This is the specific bug the estimator exists to prevent: the naive pooled
    SEM ignores the negative correlation between antithetic partners.
    """
    rng = np.random.default_rng(12)
    u = rng.random(200_000)
    sampler = lambda uu: np.exp(0.3 * stats.norm.ppf(uu))

    est = variance.antithetic(sampler, u)
    pooled = np.concatenate([sampler(u), sampler(1.0 - u)])
    naive_se = float(stats.sem(pooled))

    assert naive_se > est.standard_error * 2
    assert est.effective_sample_size > est.n_evaluations


def test_antithetic_is_exact_for_symmetric_target() -> None:
    """norm.ppf(1-u) == -norm.ppf(u), so every pair mean is exactly zero."""
    u = np.random.default_rng(13).random(10_000)
    est = variance.antithetic(lambda uu: stats.norm.ppf(uu), u)
    assert est.value == pytest.approx(0.0, abs=1e-12)
    assert est.standard_error == pytest.approx(0.0, abs=1e-12)


def test_importance_sampling_reaches_probabilities_plain_mc_cannot() -> None:
    """P(X > 6) for standard normal is 9.87e-10; plain MC finds zero hits."""
    truth = float(stats.norm.sf(6.0))
    rng = np.random.default_rng(14)
    n = 400_000

    plain_hits = int((rng.standard_normal(n) > 6.0).sum())
    assert plain_hits == 0  # the motivating failure

    est = variance.importance_sample(
        indicator=lambda x: (x > 6.0).astype(np.float64),
        log_target_pdf=lambda x: -0.5 * x**2,
        log_proposal_pdf=lambda x: -0.5 * (x - 6.0) ** 2,
        proposal_draws=rng.standard_normal(n) + 6.0,
    )

    assert est.value == pytest.approx(truth, rel=0.05)
    assert est.standard_error > 0
    assert est.effective_sample_size > 0


def test_importance_sampling_handles_extreme_tail_without_underflow() -> None:
    """P(X > 30) ~ 5e-198 underflows any plain-ratio weighting."""
    truth = float(stats.norm.sf(30.0))
    rng = np.random.default_rng(15)
    est = variance.importance_sample(
        indicator=lambda x: (x > 30.0).astype(np.float64),
        log_target_pdf=lambda x: -0.5 * x**2,
        log_proposal_pdf=lambda x: -0.5 * (x - 30.0) ** 2,
        proposal_draws=rng.standard_normal(200_000) + 30.0,
    )
    assert est.value > 0.0
    assert est.value == pytest.approx(truth, rel=0.1)


def test_importance_sampling_returns_zero_when_no_draw_contributes() -> None:
    rng = np.random.default_rng(16)
    est = variance.importance_sample(
        indicator=lambda x: np.zeros_like(x),
        log_target_pdf=lambda x: -0.5 * x**2,
        log_proposal_pdf=lambda x: -0.5 * x**2,
        proposal_draws=rng.standard_normal(1000),
    )
    assert est.value == 0.0
    assert est.effective_sample_size == 0.0


def test_control_variate_reduces_error_against_known_mean() -> None:
    rng = np.random.default_rng(17)
    n = 200_000
    x = rng.standard_normal(n)
    y = 2.0 * x + rng.standard_normal(n) * 0.1  # strongly correlated with x

    plain = variance.plain(y)
    cv = variance.control_variate(y, x, control_mean=0.0)

    assert cv.standard_error < plain.standard_error / 5
    assert cv.value == pytest.approx(0.0, abs=4 * cv.standard_error)


def test_empty_sample_raises() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        variance.plain(np.array([]))


def test_control_variate_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="identical shape"):
        variance.control_variate(np.ones(10), np.ones(5), 0.0)
