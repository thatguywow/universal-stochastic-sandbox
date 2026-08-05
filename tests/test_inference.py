"""Posterior updating and the parameter-uncertainty bridge."""

from __future__ import annotations

import numpy as np
import pytest

from uss import inference


def test_beta_binomial_recovers_true_probability() -> None:
    rng = np.random.default_rng(51)
    true_p, n = 0.28, 5000
    successes = int(rng.binomial(n, true_p))
    post = inference.update_bernoulli(successes, n)

    assert post.mean == pytest.approx(true_p, abs=0.02)
    lo, hi = post.interval()
    assert lo <= true_p <= hi


def test_beta_binomial_interval_narrows_with_data() -> None:
    wide = inference.update_bernoulli(3, 10)
    narrow = inference.update_bernoulli(300, 1000)
    assert (wide.interval()[1] - wide.interval()[0]) > (
        narrow.interval()[1] - narrow.interval()[0]
    ) * 5


def test_uniform_prior_with_no_data_is_flat() -> None:
    post = inference.update_bernoulli(0, 0)
    lo, hi = post.interval()
    assert lo == pytest.approx(0.025, abs=0.01)
    assert hi == pytest.approx(0.975, abs=0.01)


def test_gamma_poisson_recovers_rate() -> None:
    rng = np.random.default_rng(52)
    true_lam, exposure = 5e-6, 10_000_000
    count = int(rng.poisson(true_lam * exposure))
    post = inference.update_poisson(count, exposure)

    assert post.mean == pytest.approx(true_lam, rel=0.3)
    lo, hi = post.interval()
    assert lo <= true_lam <= hi


def test_gamma_poisson_rejects_improper_posterior_at_zero_events() -> None:
    with pytest.raises(ValueError, match="improper posterior"):
        inference.update_poisson(0, 1000.0, prior_shape=0.0, prior_rate=0.0)


def test_normal_update_recovers_mean() -> None:
    rng = np.random.default_rng(53)
    obs = rng.normal(20.0, 3.0, 2000)
    post = inference.update_gaussian_mean(obs)
    assert post.mean == pytest.approx(20.0, abs=0.25)
    lo, hi = post.interval()
    assert lo <= 20.0 <= hi


def test_posterior_sampling_matches_its_own_moments() -> None:
    rng = np.random.default_rng(54)
    post = inference.update_bernoulli(280, 1000)
    draws = post.sample(200_000, rng)
    assert draws.mean() == pytest.approx(post.mean, rel=0.01)
    assert np.all((draws >= 0) & (draws <= 1))


def test_metropolis_hastings_recovers_a_known_posterior() -> None:
    """Target: Normal(2, 0.5). MCMC should match its moments."""
    rng = np.random.default_rng(55)

    def log_post(x: float) -> float:
        return -0.5 * ((x - 2.0) / 0.5) ** 2

    post = inference.metropolis_hastings(log_post, 0.0, 40_000, rng, proposal_scale=1.0)
    draws = np.asarray(post.params["draws"])

    assert draws.mean() == pytest.approx(2.0, abs=0.05)
    assert draws.std() == pytest.approx(0.5, rel=0.1)
    assert 0.1 < post.params["acceptance_rate"] < 0.9


def test_metropolis_hastings_rejects_infeasible_start() -> None:
    with pytest.raises(ValueError, match="not finite at the initial value"):
        inference.metropolis_hastings(
            lambda x: -np.inf, 0.0, 100, np.random.default_rng(56)
        )


def test_propagate_spread_reflects_posterior_width() -> None:
    """A wide posterior must produce a wide spread of simulated estimates."""
    rng = np.random.default_rng(57)
    tight = inference.update_bernoulli(2800, 10_000)
    loose = inference.update_bernoulli(3, 10)

    def simulate(params: dict) -> float:
        return float(params["probability"])

    tight_draws = inference.propagate(simulate, {"probability": tight}, 500, rng)
    loose_draws = inference.propagate(simulate, {"probability": loose}, 500, rng)

    assert loose_draws.std() > tight_draws.std() * 5


def test_propagate_requires_at_least_two_draws() -> None:
    post = inference.update_bernoulli(5, 10)
    with pytest.raises(ValueError, match="at least 2 parameter draws"):
        inference.propagate(lambda p: 1.0, {"probability": post}, 1, np.random.default_rng(58))


def test_propagate_merges_fixed_parameters() -> None:
    rng = np.random.default_rng(59)
    post = inference.update_bernoulli(50, 100)
    seen: list[dict] = []

    def simulate(params: dict) -> float:
        seen.append(dict(params))
        return params["probability"] + params["offset"]

    inference.propagate(simulate, {"probability": post}, 5, rng, fixed={"offset": 10.0})
    assert all(s["offset"] == 10.0 for s in seen)
    assert len(seen) == 5


def test_posterior_predictive_beats_point_estimate_calibration() -> None:
    """The engine's central claim: propagated intervals cover, point ones don't.

    With only 10 observations the true p is poorly known.  An interval built
    from Monte Carlo error alone (around the point estimate) misses the truth
    most of the time; the posterior interval covers it at the nominal rate.
    """
    rng = np.random.default_rng(60)
    true_p, n_obs, trials = 0.28, 10, 600
    posterior_hits = 0
    mc_only_hits = 0

    for _ in range(trials):
        k = int(rng.binomial(n_obs, true_p))
        post = inference.update_bernoulli(k, n_obs)
        lo, hi = post.interval(0.95)
        posterior_hits += lo <= true_p <= hi

        # What the blueprint's Part V reports: CI around the plugged-in value,
        # width set by simulation size only.
        p_hat = k / n_obs
        mc_half = 1.96 * np.sqrt(max(p_hat * (1 - p_hat), 1e-12) / 10_000_000)
        mc_only_hits += (p_hat - mc_half) <= true_p <= (p_hat + mc_half)

    assert posterior_hits / trials > 0.90
    assert mc_only_hits / trials < 0.20
