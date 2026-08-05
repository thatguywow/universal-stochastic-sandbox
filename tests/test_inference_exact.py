"""Posterior exactness: the t-marginal for an unknown variance, and dispersion checks."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from uss import summarize, update_gaussian_mean


def test_unknown_sigma_gives_student_t_posterior() -> None:
    obs = np.array([118, 131, 105, 127, 142, 109, 122, 136, 115, 128, 133, 111, 124, 139], float)
    post = update_gaussian_mean(obs)
    assert post.family == "student_t"
    assert post.params["df"] == len(obs) - 1
    assert post.params["loc"] == pytest.approx(obs.mean())
    assert post.params["scale"] == pytest.approx(obs.std(ddof=1) / np.sqrt(len(obs)))


def test_t_posterior_is_wider_than_the_normal_approximation() -> None:
    """Treating an estimated sigma as known understates the interval at small n."""
    obs = np.random.default_rng(1).normal(100.0, 10.0, 14)
    t_post = update_gaussian_mean(obs)
    normal_post = update_gaussian_mean(obs, known_sigma=float(obs.std(ddof=1)))

    t_lo, t_hi = t_post.interval(0.95)
    n_lo, n_hi = normal_post.interval(0.95)
    assert (t_hi - t_lo) > (n_hi - n_lo) * 1.10


def test_t_posterior_matches_the_classical_interval() -> None:
    obs = np.random.default_rng(2).normal(50.0, 4.0, 25)
    lo, hi = update_gaussian_mean(obs).interval(0.95)
    crit = stats.t.ppf(0.975, df=obs.size - 1)
    half = crit * obs.std(ddof=1) / np.sqrt(obs.size)
    assert lo == pytest.approx(obs.mean() - half)
    assert hi == pytest.approx(obs.mean() + half)


def test_t_and_normal_converge_at_large_n() -> None:
    obs = np.random.default_rng(3).normal(0.0, 1.0, 5000)
    t_lo, t_hi = update_gaussian_mean(obs).interval(0.95)
    n_lo, n_hi = update_gaussian_mean(obs, known_sigma=float(obs.std(ddof=1))).interval(0.95)
    assert (t_hi - t_lo) == pytest.approx(n_hi - n_lo, rel=0.01)


def test_t_posterior_covers_at_the_nominal_rate() -> None:
    """The real test: repeated trials must cover the true mean 95% of the time."""
    rng = np.random.default_rng(4)
    true_mu, n, trials = 7.0, 8, 3000
    t_hits = normal_hits = 0
    for _ in range(trials):
        obs = rng.normal(true_mu, 2.0, n)
        lo, hi = update_gaussian_mean(obs).interval(0.95)
        t_hits += lo <= true_mu <= hi
        lo2, hi2 = update_gaussian_mean(obs, known_sigma=float(obs.std(ddof=1))).interval(0.95)
        normal_hits += lo2 <= true_mu <= hi2

    assert t_hits / trials == pytest.approx(0.95, abs=0.02)
    assert normal_hits / trials < 0.93  # the understatement, quantified


def test_t_posterior_sampling_matches_its_interval() -> None:
    obs = np.random.default_rng(5).normal(20.0, 3.0, 12)
    post = update_gaussian_mean(obs)
    draws = post.sample(200_000, np.random.default_rng(6))
    lo, hi = post.interval(0.95)
    assert np.mean((draws >= lo) & (draws <= hi)) == pytest.approx(0.95, abs=0.01)


def test_single_observation_requires_known_sigma() -> None:
    with pytest.raises(ValueError, match="at least 2 observations"):
        update_gaussian_mean(np.array([5.0]))
    post = update_gaussian_mean(np.array([5.0]), known_sigma=1.0)
    assert post.family == "normal"


def test_overdispersed_counts_are_flagged() -> None:
    """Garwood assumes a Poisson sum; negative-binomial counts must warn."""
    rng = np.random.default_rng(7)
    # Negative binomial with mean 5, variance ~15.
    counts = rng.negative_binomial(2.5, 2.5 / (2.5 + 5.0), 20_000)
    report = summarize(counts, kind="count")
    assert any("overdispersed" in c for c in report.caveats)


def test_genuine_poisson_counts_are_not_flagged() -> None:
    counts = np.random.default_rng(8).poisson(5.0, 20_000)
    report = summarize(counts, kind="count")
    assert not any("overdispersed" in c for c in report.caveats)
