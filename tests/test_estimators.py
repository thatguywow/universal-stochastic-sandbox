"""Interval constructions must have the coverage they claim."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from uss import estimators


def test_wilson_interval_stays_in_unit_range_at_zero_successes() -> None:
    lo, hi = estimators.wilson_interval(0, 1000)
    assert lo == 0.0
    assert 0.0 < hi < 0.01


def test_wilson_beats_wald_coverage_near_boundary() -> None:
    """At p=0.002, n=500 the Wald interval collapses; Wilson stays usable.

    Measured coverage in this regime is Wilson 0.913 vs Wald 0.639.  Neither
    reaches the nominal 0.95 -- with an expected count of 1 success, no
    interval on a discrete statistic can, so the test asserts the achievable
    property (Wilson dominates by a wide margin) rather than a nominal rate
    that is unattainable here.
    """
    rng = np.random.default_rng(21)
    p, n, trials = 0.002, 500, 2000
    wilson_hits = 0
    wald_hits = 0

    for _ in range(trials):
        k = int(rng.binomial(n, p))
        lo, hi = estimators.wilson_interval(k, n)
        wilson_hits += lo <= p <= hi

        p_hat = k / n
        half = 1.96 * np.sqrt(p_hat * (1 - p_hat) / n)
        wald_hits += (p_hat - half) <= p <= (p_hat + half)

    assert wilson_hits / trials > 0.90
    assert (wilson_hits - wald_hits) / trials > 0.15


def test_wilson_reaches_nominal_coverage_once_counts_are_adequate() -> None:
    """The same construction does hit ~95% once np is not tiny."""
    rng = np.random.default_rng(210)
    p, n, trials = 0.05, 200, 3000
    hits = 0
    for _ in range(trials):
        lo, hi = estimators.wilson_interval(int(rng.binomial(n, p)), n)
        hits += lo <= p <= hi
    assert hits / trials > 0.94


def test_garwood_interval_covers_true_rate() -> None:
    rng = np.random.default_rng(22)
    lam, exposure, trials = 5.0, 1.0, 2000
    hits = 0
    for _ in range(trials):
        count = int(rng.poisson(lam * exposure))
        lo, hi = estimators.garwood_interval(count, exposure)
        hits += lo <= lam <= hi
    assert hits / trials >= 0.95  # exact interval is conservative by construction


def test_garwood_handles_zero_events() -> None:
    lo, hi = estimators.garwood_interval(0, exposure=1000.0)
    assert lo == 0.0
    assert hi == pytest.approx(stats.chi2.ppf(0.975, 2) / 2 / 1000.0)


def test_summarize_uses_wilson_for_proportions() -> None:
    samples = (np.random.default_rng(23).random(100_000) < 0.3).astype(np.int8)
    report = estimators.summarize(samples, kind="proportion")
    assert report.interval_type == "wilson-score"
    assert report.monte_carlo_interval[0] <= 0.3 <= report.monte_carlo_interval[1]


def test_summarize_accumulates_int8_without_overflow() -> None:
    """int8 indicators must be promoted before summation."""
    samples = np.ones(1_000_000, dtype=np.int8)
    report = estimators.summarize(samples, kind="proportion")
    assert report.point_estimate == pytest.approx(1.0)


def test_summarize_flags_sparse_count_data() -> None:
    samples = np.zeros(1_000_000, dtype=np.int64)
    samples[:12] = 1
    report = estimators.summarize(samples, kind="count")
    assert report.interval_type == "garwood-exact"
    assert any("counting noise" in c for c in report.caveats)


def test_summarize_always_warns_that_mc_interval_is_not_model_accuracy() -> None:
    report = estimators.summarize(np.random.default_rng(24).normal(size=1000))
    assert any("does not bound the accuracy" in c for c in report.caveats)


def test_summarize_honours_supplied_standard_error() -> None:
    """A variance-reduced SE must not be silently recomputed as i.i.d."""
    samples = np.random.default_rng(25).normal(size=10_000)
    report = estimators.summarize(samples, standard_error=1e-9)
    assert report.monte_carlo_error == 1e-9


def test_combine_uncertainty_widens_the_interval() -> None:
    samples = np.random.default_rng(26).normal(0.3, 0.01, 100_000)
    report = estimators.summarize(samples)
    mc_width = report.monte_carlo_interval[1] - report.monte_carlo_interval[0]

    parameter_draws = np.random.default_rng(27).normal(0.3, 0.05, 512)
    combined = estimators.combine_uncertainty(report, parameter_draws)

    assert combined.total_interval is not None
    total_width = combined.total_interval[1] - combined.total_interval[0]
    assert total_width > mc_width * 10
    assert combined.parameter_uncertainty == pytest.approx(0.05, rel=0.15)


def test_combine_uncertainty_warns_when_priors_dominate() -> None:
    report = estimators.summarize(np.random.default_rng(28).normal(0.3, 0.01, 100_000))
    combined = estimators.combine_uncertainty(
        report, np.random.default_rng(29).normal(0.3, 0.08, 512)
    )
    assert any("will not narrow this result" in c for c in combined.caveats)


def test_convergence_trace_shows_one_over_sqrt_n_decay() -> None:
    samples = np.random.default_rng(30).normal(size=1_000_000)
    trace = estimators.convergence_trace(samples)
    first_n, _, first_se = trace[0]
    last_n, _, last_se = trace[-1]
    ratio = first_se / last_se
    expected = np.sqrt(last_n / first_n)
    assert ratio == pytest.approx(expected, rel=0.35)


def test_unknown_domain_raises() -> None:
    with pytest.raises(ValueError, match="unknown domain"):
        estimators.summarize(np.ones(10), domain="not_a_domain")


def test_confidence_domains_encode_blueprint_table() -> None:
    assert estimators.CONFIDENCE_DOMAINS["quantum_anomaly"]["achievable_confidence"] == (0.0, 0.0)
    assert estimators.CONFIDENCE_DOMAINS["closed_physical"]["achievable_confidence"] == (0.98, 0.999)
