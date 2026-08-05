"""End-to-end engine behaviour, including the blueprint's own Part V queries."""

from __future__ import annotations

import numpy as np
import pytest

from uss import UniversalStochasticSandbox, update_bernoulli, update_poisson


@pytest.fixture
def sandbox() -> UniversalStochasticSandbox:
    return UniversalStochasticSandbox(sample_size=200_000, seed=42)


def test_blueprint_tank_top_query_runs(sandbox) -> None:
    result = sandbox.execute_query("bernoulli", {"probability": 0.322})
    assert result.report.point_estimate == pytest.approx(0.322, abs=0.005)
    assert result.report.interval_type == "wilson-score"


def test_blueprint_balloon_query_uses_exact_interval(sandbox) -> None:
    """lambda passes through the alias, and sparse counts get Garwood."""
    result = sandbox.execute_query("poisson", {"lambda": 0.000005})
    assert result.parameters == {"lam": 0.000005}
    assert result.report.interval_type == "garwood-exact"
    lo, hi = result.report.monte_carlo_interval
    assert lo >= 0.0
    assert hi > lo


def test_bias_multiplier_is_rejected_with_guidance(sandbox) -> None:
    with pytest.raises(ValueError, match="bias_multiplier is not supported"):
        sandbox.execute_query(
            "bernoulli", {"probability": 0.28, "bias_multiplier": 1.15}
        )


def test_parameter_aliases_normalise(sandbox) -> None:
    result = sandbox.execute_query("gaussian", {"mu": 5.0, "sigma": 2.0})
    assert result.parameters == {"mean": 5.0, "std_dev": 2.0}
    assert result.report.point_estimate == pytest.approx(5.0, abs=0.05)


def test_same_seed_reproduces_identical_results() -> None:
    a = UniversalStochasticSandbox(sample_size=50_000, seed=7)
    b = UniversalStochasticSandbox(sample_size=50_000, seed=7)
    ra = a.execute_query("gaussian", {"mean": 0.0, "std_dev": 1.0})
    rb = b.execute_query("gaussian", {"mean": 0.0, "std_dev": 1.0})
    assert ra.report.point_estimate == rb.report.point_estimate


def test_reset_restores_the_stream(sandbox) -> None:
    first = sandbox.execute_query("gaussian", {"mean": 0.0}).report.point_estimate
    sandbox.reset()
    second = sandbox.execute_query("gaussian", {"mean": 0.0}).report.point_estimate
    assert first == second


def test_antithetic_path_reduces_reported_error() -> None:
    a = UniversalStochasticSandbox(sample_size=200_000, seed=11)
    b = UniversalStochasticSandbox(sample_size=200_000, seed=11)
    plain = a.execute_query("lognormal", {"mean": 0.0, "std_dev": 0.6})
    anti = b.execute_query("lognormal", {"mean": 0.0, "std_dev": 0.6}, antithetic=True)
    assert anti.report.monte_carlo_error < plain.report.monte_carlo_error


def test_result_dict_keeps_blueprint_keys(sandbox) -> None:
    """Downstream consumers of the Part V dict shape must keep working."""
    payload = sandbox.execute_query("gaussian", {"mean": 1.0}).as_dict()
    for key in [
        "query_class",
        "sample_size",
        "mean_point_estimate",
        "variance",
        "standard_error",
        "confidence_level",
        "confidence_interval",
        "min_observed",
        "max_observed",
    ]:
        assert key in payload


def test_summary_reports_mc_and_total_intervals_separately() -> None:
    sandbox = UniversalStochasticSandbox(sample_size=100_000, seed=13)
    result = sandbox.execute_with_priors(
        "bernoulli",
        {"probability": update_bernoulli(3, 10)},
        n_parameter_draws=128,
        inner_sample_size=5_000,
    )
    text = result.summary()
    assert "monte carlo error" in text
    assert "TOTAL interval" in text
    assert result.report.total_interval is not None


def test_priors_widen_the_interval_beyond_monte_carlo_error() -> None:
    """10 observations cannot be rescued by 10^5 simulated draws."""
    sandbox = UniversalStochasticSandbox(sample_size=100_000, seed=14)
    result = sandbox.execute_with_priors(
        "bernoulli",
        {"probability": update_bernoulli(3, 10)},
        n_parameter_draws=256,
        inner_sample_size=5_000,
    )
    mc_lo, mc_hi = result.report.monte_carlo_interval
    total_lo, total_hi = result.report.total_interval
    assert (total_hi - total_lo) > (mc_hi - mc_lo) * 5
    assert result.report.parameter_uncertainty > result.report.monte_carlo_error


def test_poisson_priors_propagate(sandbox) -> None:
    """Moderate rate, so the inner simulation resolves lambda without 10^9 draws."""
    result = sandbox.execute_with_priors(
        "poisson",
        {"lam": update_poisson(500, 100.0)},  # lambda ~ 5.0
        n_parameter_draws=64,
        inner_sample_size=50_000,
    )
    assert result.report.total_interval is not None
    assert result.report.parameter_uncertainty > 0
    assert not any("quantised by the inner" in c for c in result.report.caveats)


def test_undersized_inner_sample_is_flagged_as_quantised(sandbox) -> None:
    """A 5e-6 rate simulated at n=100k yields grid points, not a posterior."""
    result = sandbox.execute_with_priors(
        "poisson",
        {"lam": update_poisson(50, 10_000_000.0)},
        n_parameter_draws=128,
        inner_sample_size=100_000,
    )
    assert any("quantised by the inner" in c for c in result.report.caveats)


def test_adequate_inner_sample_is_not_flagged() -> None:
    sandbox = UniversalStochasticSandbox(sample_size=100_000, seed=15)
    result = sandbox.execute_with_priors(
        "bernoulli",
        {"probability": update_bernoulli(300, 1000)},
        n_parameter_draws=128,
        inner_sample_size=200_000,
    )
    assert not any("quantised by the inner" in c for c in result.report.caveats)


def test_domain_ceiling_is_attached(sandbox) -> None:
    result = sandbox.execute_query(
        "bernoulli", {"probability": 0.3}, domain="macro_behavioral"
    )
    payload = result.as_dict()
    assert payload["domain"] == "macro_behavioral"
    assert payload["domain_ceiling"]["achievable_confidence"] == (0.90, 0.95)


def test_rare_event_probability_via_importance_sampling(sandbox) -> None:
    from scipy import stats

    est = sandbox.rare_event_probability(
        indicator=lambda x: (x > 6.0).astype(np.float64),
        log_target_pdf=lambda x: -0.5 * x**2,
        log_proposal_pdf=lambda x: -0.5 * (x - 6.0) ** 2,
        proposal_sampler=lambda n, rng: rng.standard_normal(n) + 6.0,
        sample_size=200_000,
    )
    assert est.value == pytest.approx(float(stats.norm.sf(6.0)), rel=0.08)


def test_quantiles_are_reported(sandbox) -> None:
    result = sandbox.execute_query("gaussian", {"mean": 0.0, "std_dev": 1.0})
    q = result.quantiles
    assert q["p50"] == pytest.approx(0.0, abs=0.02)
    assert q["p01"] < q["p25"] < q["p50"] < q["p75"] < q["p99"]


def test_invalid_sample_size_raises() -> None:
    with pytest.raises(ValueError, match="sample_size must be at least 2"):
        UniversalStochasticSandbox(sample_size=1)


def test_unknown_query_class_raises(sandbox) -> None:
    with pytest.raises(ValueError, match="Unknown query class"):
        sandbox.execute_query("teleportation", {})
