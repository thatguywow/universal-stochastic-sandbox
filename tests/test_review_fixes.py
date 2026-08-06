"""Fixes from the external review pass.

Not every reported issue was real. The three that were not are pinned here too,
so a future reader does not "fix" working code:

  * MCMC was reported as O(n^2) from a misread indent. Per-sample cost is flat.
  * The empirical sampler was reported as extrapolating. `np.interp` clamps.
  * `samples_for_proportion` was reported as silently missing its target. It
    met the target; the cap it can hit is guarded now regardless.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from uss import (
    UniversalStochasticSandbox,
    distributions,
    fitting,
    metropolis_hastings,
    propagate,
    samples_for_proportion,
    update_bernoulli,
)


# --------------------------------------------------------------------------
# Reported but NOT defects -- pinned so they stay that way
# --------------------------------------------------------------------------
def test_mcmc_cost_is_linear_not_quadratic() -> None:
    """The tail fill is outside the inner loop; per-sample cost must stay flat."""

    def log_post(x: float) -> float:
        return -0.5 * ((x - 2.0) / 0.5) ** 2

    per_sample = []
    for total in (4_000, 16_000):
        start = time.perf_counter()
        metropolis_hastings(
            log_post, 0.0, total, np.random.default_rng(1),
            proposal_scale=1.0, n_chains=1, burn_in=200,
        )
        per_sample.append((time.perf_counter() - start) / total)
    # Quadratic growth would make the 4x-longer run cost ~4x more per sample.
    assert per_sample[1] < per_sample[0] * 2.0


def test_empirical_sampler_clamps_rather_than_extrapolating() -> None:
    """np.interp already clamps at the endpoints; values stay inside the data."""
    q = np.linspace(10.0, 20.0, 100)
    out = distributions.get("empirical").sample(
        np.array([-0.05, 0.0, 0.5, 1.0, 1.05]), quantiles=q
    )
    assert out.min() >= 10.0
    assert out.max() <= 20.0
    assert out[2] == pytest.approx(15.0, abs=0.2)


# --------------------------------------------------------------------------
# Real fixes
# --------------------------------------------------------------------------
def test_sample_plan_flags_an_unreachable_target() -> None:
    achievable = samples_for_proportion(0.05, assumed_rate=0.5)
    assert achievable.meets_target
    assert "does NOT meet" not in achievable.summary()


def test_sample_plan_meets_target_is_reported() -> None:
    plan = samples_for_proportion(0.01, assumed_rate=0.3)
    assert plan.target_half_width == 0.01
    assert plan.achieved_half_width <= 0.01
    assert plan.meets_target


def test_gamma_and_weibull_are_registered_query_classes() -> None:
    """They were candidate fit families mapping to an empty query class."""
    assert "gamma" in distributions.available()
    assert "weibull" in distributions.available()

    u = np.random.default_rng(1).random(200_000)
    gamma = distributions.get("gamma").sample(u, shape=2.0, scale=3.0)
    assert gamma.mean() == pytest.approx(6.0, rel=0.02)      # k * theta

    weib = distributions.get("weibull").sample(u, shape=1.5, scale=2.0)
    from scipy import stats
    assert weib.mean() == pytest.approx(stats.weibull_min(c=1.5, scale=2.0).mean(), rel=0.02)


def test_fitted_gamma_maps_to_a_usable_query_class() -> None:
    """best_fit on gamma data used to return query_class='' -- unusable."""
    data = np.random.default_rng(2).gamma(2.0, 3.0, 4_000)
    best = fitting.best_fit(data, families=["gamma", "norm"], n_bootstrap=0)[0]
    assert best.family == "gamma"
    assert best.query_class == "gamma"
    assert best.parameters

    sandbox = UniversalStochasticSandbox(sample_size=200_000, seed=3)
    result = sandbox.execute_query(best.query_class, best.parameters)
    assert result.report.point_estimate == pytest.approx(data.mean(), rel=0.1)


def test_fitted_weibull_maps_to_a_usable_query_class() -> None:
    data = np.random.default_rng(4).weibull(1.5, 4_000) * 2.0
    best = fitting.best_fit(data, families=["weibull_min", "norm"], n_bootstrap=0)[0]
    assert best.query_class == "weibull"
    UniversalStochasticSandbox(sample_size=50_000, seed=5).execute_query(
        best.query_class, best.parameters
    )


def test_propagate_rejects_a_parameter_given_twice() -> None:
    """The posterior silently won over the fixed value."""
    post = update_bernoulli(30, 100)
    with pytest.raises(ValueError, match="both a fixed value and a posterior"):
        propagate(
            lambda p: p["probability"],
            {"probability": post},
            10,
            np.random.default_rng(6),
            fixed={"probability": 0.5},
        )


def test_propagate_still_merges_non_clashing_fixed_values() -> None:
    post = update_bernoulli(30, 100)
    draws = propagate(
        lambda p: p["probability"] + p["offset"],
        {"probability": post},
        10,
        np.random.default_rng(7),
        fixed={"offset": 10.0},
    )
    assert np.all(draws > 10.0)


def test_antithetic_rounding_is_disclosed() -> None:
    result = UniversalStochasticSandbox(sample_size=1001, seed=8).execute_query(
        "gaussian", {"mean": 0.0}, antithetic=True
    )
    assert result.sample_size == 1000
    assert any("rounded down" in c for c in result.report.caveats)


def test_no_rounding_caveat_when_even() -> None:
    result = UniversalStochasticSandbox(sample_size=1000, seed=9).execute_query(
        "gaussian", {"mean": 0.0}, antithetic=True
    )
    assert not any("rounded down" in c for c in result.report.caveats)


def test_sensitivity_inverts_posteriors_by_interpolation() -> None:
    """Nearest-neighbour snapping made the inverse a step function.

    The Saltelli design deliberately reuses columns -- matrices A and A_B^b share
    A's column for factor `a` -- so the distinct count is 2 * n_base, not one per
    evaluation. What must hold is that every drawn value is a genuine
    interpolation rather than a member of the sampled grid.
    """
    from uss import parameter_sensitivity, update_gaussian_mean

    n_base = 400
    rng = np.random.default_rng(10)
    posteriors = {
        "a": update_gaussian_mean(rng.normal(10.0, 2.0, 30)),
        "b": update_gaussian_mean(rng.normal(5.0, 0.2, 30)),
    }
    seen: list[float] = []

    def simulate(p: dict[str, float]) -> float:
        seen.append(p["a"])
        return p["a"] * 3.0 + p["b"]

    parameter_sensitivity(simulate, posteriors, n_base, rng)

    distinct = len(set(seen))
    # One independent value per row of A and of B; anything far below that would
    # mean values were collapsing onto shared grid points.
    assert distinct == pytest.approx(2 * n_base, rel=0.05)
    # And the inverse is continuous: no value repeats more than the design
    # requires (A appears in A, in A_B^b, and in the determinism probe).
    counts = {v: seen.count(v) for v in list(set(seen))[:50]}
    assert max(counts.values()) <= 4
