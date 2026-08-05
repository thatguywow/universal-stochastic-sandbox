"""Regression tests for the mathematical flaws found in the end-to-end audit.

Each test here reproduces a specific defect that shipped in an earlier revision
and would be visible to a reader checking the maths rather than the code.
"""

from __future__ import annotations

import numpy as np
import pytest

from uss import (
    UniversalStochasticSandbox,
    calibration,
    combine_uncertainty,
    sobol_indices,
    summarize,
)
from uss.estimators import convergence_trace


# --------------------------------------------------------------------------
# A. Catastrophic cancellation in running variance
# --------------------------------------------------------------------------
@pytest.mark.parametrize("mean", [0.0, 1e3, 1e6, 1e8])
def test_convergence_trace_variance_survives_large_means(mean: float) -> None:
    """E[X^2]-E[X]^2 returns 2.0 for unit variance at mean 1e8. Centred form must not."""
    x = np.random.default_rng(1).normal(mean, 1.0, 200_000)
    trace = convergence_trace(x)
    _, _, final_sem = trace[-1]
    expected_sem = x.std(ddof=1) / np.sqrt(x.size)
    assert final_sem == pytest.approx(expected_sem, rel=1e-6)


def test_convergence_trace_mean_survives_large_offset() -> None:
    x = np.random.default_rng(2).normal(1e8, 1.0, 100_000)
    n_final, mean_final, _ = convergence_trace(x)[-1]
    assert n_final == x.size
    assert mean_final == pytest.approx(float(x.mean()), rel=1e-12)


def test_adaptive_stopping_variance_survives_large_means() -> None:
    """run_until_precision accumulates moments incrementally; check the pivot."""
    sandbox = UniversalStochasticSandbox(seed=3)
    result = sandbox.run_until_precision(
        "gaussian",
        {"mean": 1e8, "std_dev": 1.0},
        target_standard_error=5e-3,
        batch_size=100_000,
        max_samples=2_000_000,
    )
    # sigma/sqrt(n) = 5e-3 needs n = 40,000 -- so it must stop in the first batch.
    assert result.sample_size == 100_000
    assert result.report.variance == pytest.approx(1.0, rel=0.02)
    assert result.report.monte_carlo_error == pytest.approx(1.0 / np.sqrt(100_000), rel=0.02)


# --------------------------------------------------------------------------
# B. Parameter uncertainty conflated with inner Monte Carlo noise
# --------------------------------------------------------------------------
def test_inner_mc_noise_is_subtracted_from_parameter_uncertainty() -> None:
    """Var(draw) = Var_param + sigma^2/n_inner; the second term is not parameter spread."""
    rng = np.random.default_rng(4)
    true_param_sd, inner_n, sigma, n_draws = 0.01, 1_000, 0.5, 4_000

    params = rng.normal(0.3, true_param_sd, n_draws)
    draws = params + rng.normal(0.0, sigma / np.sqrt(inner_n), n_draws)

    base = summarize(rng.normal(0.3, sigma, 50_000))
    naive = combine_uncertainty(base, draws)
    naive_sd = naive.parameter_uncertainty

    base2 = summarize(rng.normal(0.3, sigma, 50_000))
    corrected = combine_uncertainty(
        base2, draws, inner_mc_variance=sigma**2 / inner_n
    )

    assert naive_sd > true_param_sd * 1.5          # the defect
    assert corrected.parameter_uncertainty == pytest.approx(true_param_sd, rel=0.15)


def test_parameter_uncertainty_floors_at_zero_when_noise_explains_all() -> None:
    rng = np.random.default_rng(5)
    draws = rng.normal(0.3, 0.01, 2_000)  # pure noise, no parameter variation
    report = summarize(rng.normal(0.3, 1.0, 10_000))
    out = combine_uncertainty(report, draws, inner_mc_variance=1.0)
    assert out.parameter_uncertainty == 0.0
    assert any("not resolvable" in c for c in out.caveats)


def test_negative_inner_variance_rejected() -> None:
    report = summarize(np.random.default_rng(6).normal(size=1000))
    with pytest.raises(ValueError, match="must be non-negative"):
        combine_uncertainty(report, np.random.default_rng(7).normal(size=100), inner_mc_variance=-1.0)


def test_engine_reports_corrected_parameter_uncertainty() -> None:
    """End-to-end: a tight posterior must not be inflated by inner noise."""
    from uss import update_bernoulli

    sandbox = UniversalStochasticSandbox(sample_size=200_000, seed=8)
    result = sandbox.execute_with_priors(
        "bernoulli",
        {"probability": update_bernoulli(30_000, 100_000)},  # p pinned to ~0.0014 sd
        n_parameter_draws=256,
        inner_sample_size=20_000,
    )
    # Posterior sd of Beta(30001, 70001) is ~0.00145. Inner noise at n=20k is
    # ~0.0032, i.e. larger -- so the uncorrected figure would be dominated by it.
    assert result.report.parameter_uncertainty < 0.0025


# --------------------------------------------------------------------------
# D. Antithetic runs must report the marginal distribution, not pair means
# --------------------------------------------------------------------------
def test_antithetic_quantiles_describe_the_distribution_not_pair_means() -> None:
    """Pair averaging pulls both tails inward; p01/p99 must not inherit that."""
    plain = UniversalStochasticSandbox(sample_size=400_000, seed=9).execute_query(
        "gaussian", {"mean": 0.0, "std_dev": 1.0}
    )
    anti = UniversalStochasticSandbox(sample_size=400_000, seed=9).execute_query(
        "gaussian", {"mean": 0.0, "std_dev": 1.0}, antithetic=True
    )

    # Standard normal: p99 = 2.326, p01 = -2.326.
    assert anti.quantiles["p99"] == pytest.approx(2.326, abs=0.05)
    assert anti.quantiles["p01"] == pytest.approx(-2.326, abs=0.05)
    assert anti.quantiles["p99"] == pytest.approx(plain.quantiles["p99"], abs=0.05)
    assert anti.report.variance == pytest.approx(1.0, rel=0.02)
    assert anti.max_observed > 3.0


def test_antithetic_still_reports_reduced_standard_error() -> None:
    """Fixing the quantiles must not undo the variance reduction."""
    plain = UniversalStochasticSandbox(sample_size=200_000, seed=10).execute_query(
        "lognormal", {"mean": 0.0, "std_dev": 0.6}
    )
    anti = UniversalStochasticSandbox(sample_size=200_000, seed=10).execute_query(
        "lognormal", {"mean": 0.0, "std_dev": 0.6}, antithetic=True
    )
    assert anti.report.monte_carlo_error < plain.report.monte_carlo_error


# --------------------------------------------------------------------------
# E. PIT must not falsely reject calibrated discrete forecasts
# --------------------------------------------------------------------------
def test_plain_pit_falsely_rejects_a_perfect_poisson_forecast() -> None:
    """Documents the defect: the standard transform is not uniform for counts."""
    rng = np.random.default_rng(11)
    truth = rng.poisson(3.0, 20_000)
    preds = rng.poisson(3.0, (20_000, 400))
    res = calibration.pit_report(preds, truth, discrete=False)
    assert res.ks_pvalue < 1e-6  # false alarm on a correctly-specified model


def test_randomised_pit_accepts_a_perfect_poisson_forecast() -> None:
    rng = np.random.default_rng(12)
    truth = rng.poisson(3.0, 20_000)
    preds = rng.poisson(3.0, (20_000, 400))
    res = calibration.pit_report(preds, truth, discrete=True, rng=rng)
    assert res.verdict == "calibrated"
    assert res.ks_pvalue > 0.05
    assert res.randomised


def test_pit_autodetects_discrete_forecasts() -> None:
    rng = np.random.default_rng(13)
    truth = rng.poisson(4.0, 10_000)
    preds = rng.poisson(4.0, (10_000, 300))
    res = calibration.pit_report(preds, truth, rng=rng)
    assert res.randomised
    assert res.ks_pvalue > 0.05


def test_pit_autodetects_continuous_forecasts() -> None:
    rng = np.random.default_rng(14)
    truth = rng.normal(0.0, 1.0, 5_000)
    preds = rng.normal(0.0, 1.0, (5_000, 300))
    res = calibration.pit_report(preds, truth)
    assert not res.randomised
    assert res.ks_pvalue > 0.05


def test_randomised_pit_still_detects_a_genuinely_wrong_discrete_forecast() -> None:
    """The fix must not blunt the diagnostic."""
    rng = np.random.default_rng(15)
    truth = rng.poisson(8.0, 10_000)
    preds = rng.poisson(3.0, (10_000, 300))  # badly mis-specified
    res = calibration.pit_report(preds, truth, discrete=True, rng=rng)
    assert res.ks_pvalue < 1e-6


# --------------------------------------------------------------------------
# F. Sobol must report unexplained variance from a stochastic model
# --------------------------------------------------------------------------
def test_deterministic_model_has_no_unexplained_variance() -> None:
    res = sobol_indices(
        lambda u: 3.0 * u[:, 0] + u[:, 1], 2, 60_000, np.random.default_rng(16)
    )
    assert res.unexplained_variance < 0.05
    assert res.warnings == ()


def test_independent_model_noise_is_flagged() -> None:
    """Fresh randomness per call inflates total effects above 1. Must be caught."""

    def noisy(u: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng()  # fresh each call: genuine model noise
        return 3.0 * u[:, 0] + u[:, 1] + rng.normal(0.0, 3.0, u.shape[0])

    res = sobol_indices(noisy, 2, 40_000, np.random.default_rng(17))
    assert res.total_effect.sum() > 1.2  # inflated, not deflated
    assert any("not deterministic" in w for w in res.warnings)
    assert "!" in res.summary()


def test_shared_model_noise_is_flagged_as_unexplained_variance() -> None:
    """Noise common to all Saltelli matrices deflates indices instead."""

    def shared_noise(u: np.ndarray) -> np.ndarray:
        # Same draw every call, so it cancels in differences but not in Var(Y).
        noise = np.random.default_rng(0).normal(0.0, 3.0, u.shape[0])
        return 3.0 * u[:, 0] + u[:, 1] + noise

    res = sobol_indices(shared_noise, 2, 40_000, np.random.default_rng(170))
    assert res.unexplained_variance > 0.2
    assert "unexplained variance" in res.summary()


def test_seeding_the_model_removes_the_warning() -> None:
    """Common random numbers restore determinism given the inputs."""

    def seeded(u: np.ndarray) -> np.ndarray:
        # Noise driven by the inputs themselves, so identical rows agree.
        jitter = np.sin(1e4 * u[:, 0]) * 0.05
        return 3.0 * u[:, 0] + u[:, 1] + jitter

    res = sobol_indices(seeded, 2, 60_000, np.random.default_rng(18))
    assert res.unexplained_variance < 0.05
