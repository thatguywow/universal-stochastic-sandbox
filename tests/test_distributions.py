"""Samplers are checked against analytic truth, not merely against not-crashing."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from uss import distributions


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260804)


@pytest.mark.parametrize("mu", [1e-6, 5e-6, 0.5, 3.0, 25.0, 500.0])
def test_poisson_table_matches_scipy_ppf_exactly(mu: float, rng) -> None:
    """The searchsorted fast path must be bit-identical to scipy's ppf."""
    u = rng.random(200_000)
    fast = distributions.get("poisson").sample(u, lam=mu)
    reference = stats.poisson.ppf(u, mu=mu)
    assert np.array_equal(fast.astype(np.float64), reference)


def test_poisson_recovers_lambda() -> None:
    u = np.random.default_rng(1).random(2_000_000)
    lam = 4.0
    samples = distributions.get("poisson").sample(u, lam=lam)
    # Mean and variance of a Poisson both equal lambda.
    assert samples.mean() == pytest.approx(lam, rel=2e-3)
    assert samples.var() == pytest.approx(lam, rel=5e-3)


def test_poisson_zero_lambda_is_all_zeros() -> None:
    u = np.linspace(0.0, 0.999, 1000)
    assert np.all(distributions.get("poisson").sample(u, lam=0.0) == 0)


def test_bernoulli_recovers_p() -> None:
    u = np.random.default_rng(2).random(2_000_000)
    samples = distributions.get("bernoulli").sample(u, probability=0.322)
    assert samples.mean() == pytest.approx(0.322, abs=1e-3)


def test_bernoulli_rejects_out_of_range_probability() -> None:
    u = np.array([0.1, 0.5])
    with pytest.raises(ValueError, match="probability must lie"):
        distributions.get("bernoulli").sample(u, probability=1.4)


def test_gaussian_matches_moments() -> None:
    u = np.random.default_rng(3).random(1_000_000)
    samples = distributions.get("gaussian").sample(u, mean=20.0, std_dev=3.0)
    assert samples.mean() == pytest.approx(20.0, abs=0.02)
    assert samples.std() == pytest.approx(3.0, rel=1e-3)


def test_gumbel_matches_analytic_mean() -> None:
    """Gumbel mean is loc + scale * Euler-Mascheroni."""
    u = np.random.default_rng(4).random(1_000_000)
    loc, scale = 2.0, 1.5
    samples = distributions.get("extreme_value").sample(u, loc=loc, scale=scale)
    expected = loc + scale * np.euler_gamma
    assert samples.mean() == pytest.approx(expected, abs=0.02)


def test_genextreme_shape_matches_scipy_convention() -> None:
    """shape=xi must map onto scipy's c=-xi."""
    u = np.random.default_rng(5).random(50_000)
    xi = 0.3
    ours = distributions.get("extreme_value").sample(u, loc=0.0, scale=1.0, shape=xi)
    reference = stats.genextreme.ppf(u, c=-xi, loc=0.0, scale=1.0)
    assert np.allclose(ours, reference)


def test_exponential_matches_analytic_mean() -> None:
    u = np.random.default_rng(6).random(1_000_000)
    rate = 2.5
    samples = distributions.get("exponential").sample(u, rate=rate)
    assert samples.mean() == pytest.approx(1.0 / rate, rel=3e-3)
    assert np.all(samples >= 0)


def test_exponential_uses_log1p_for_tail_accuracy() -> None:
    """-log1p(-u) must stay finite at u close to 1, where -log(1-u) loses bits."""
    u = np.array([1.0 - 1e-16, 1.0 - 1e-15])
    samples = distributions.get("exponential").sample(u, rate=1.0)
    assert np.all(np.isfinite(samples))


def test_lognormal_matches_analytic_mean() -> None:
    u = np.random.default_rng(7).random(2_000_000)
    mu, sigma = 0.5, 0.4
    samples = distributions.get("lognormal").sample(u, mean=mu, std_dev=sigma)
    assert samples.mean() == pytest.approx(np.exp(mu + sigma**2 / 2), rel=5e-3)


def test_empirical_reproduces_source_distribution() -> None:
    source = np.random.default_rng(8).normal(10.0, 2.0, 500_000)
    quantiles = np.quantile(source, np.linspace(0, 1, 4096))
    u = np.random.default_rng(9).random(500_000)
    resampled = distributions.get("empirical").sample(u, quantiles=quantiles)
    assert resampled.mean() == pytest.approx(source.mean(), abs=0.02)
    assert resampled.std() == pytest.approx(source.std(), rel=5e-3)


def test_inverse_transform_is_monotone() -> None:
    """F^-1 must be non-decreasing in U for every registered class."""
    u = np.linspace(1e-9, 1 - 1e-9, 10_000)
    for name in distributions.available():
        qc = distributions.get(name)
        kwargs = {"quantiles": np.linspace(0, 1, 100)} if name == "empirical" else {}
        samples = qc.sample(u, **kwargs).astype(np.float64)
        assert np.all(np.diff(samples) >= 0), f"{name} is not monotone in U"


def test_samplers_accept_per_draw_vector_parameters() -> None:
    """Vector parameters are what make composition graphs possible."""
    n = 50_000
    u = np.random.default_rng(80).random(n)
    p_vec = np.full(n, 0.4)
    mu_vec = np.full(n, 7.0)

    assert distributions.get("bernoulli").sample(u, probability=p_vec).mean() == pytest.approx(0.4, abs=0.01)
    assert distributions.get("gaussian").sample(u, mean=mu_vec, std_dev=1.0).mean() == pytest.approx(7.0, abs=0.02)
    assert distributions.get("lognormal").sample(u, mean=mu_vec * 0, std_dev=np.full(n, 0.5)).size == n
    assert distributions.get("exponential").sample(u, rate=np.full(n, 2.0)).mean() == pytest.approx(0.5, rel=0.02)
    assert distributions.get("extreme_value").sample(u, loc=mu_vec, scale=1.0).size == n


def test_vector_parameters_are_validated_elementwise() -> None:
    u = np.random.default_rng(81).random(100)
    bad_p = np.full(100, 0.5)
    bad_p[7] = 1.5
    with pytest.raises(ValueError, match="probability must lie"):
        distributions.get("bernoulli").sample(u, probability=bad_p)

    bad_sd = np.full(100, 1.0)
    bad_sd[3] = -1.0
    with pytest.raises(ValueError, match="std_dev must be positive"):
        distributions.get("gaussian").sample(u, mean=0.0, std_dev=bad_sd)


def test_poisson_rejects_vector_lambda_with_guidance() -> None:
    u = np.random.default_rng(82).random(100)
    with pytest.raises(ValueError, match="requires a scalar lambda"):
        distributions.get("poisson").sample(u, lam=np.full(100, 3.0))


def test_registry_rejects_duplicate_without_overwrite() -> None:
    qc = distributions.QueryClass("gaussian", lambda u: u, "continuous", "dup")
    with pytest.raises(ValueError, match="already registered"):
        distributions.register(qc)


def test_registry_accepts_custom_class() -> None:
    """Extensibility is the 'non-case-limited' claim; verify it holds."""

    def triangular(u, low=0.0, high=1.0, **_):
        return low + (high - low) * np.sqrt(u)

    distributions.register(
        distributions.QueryClass("triangular_test", triangular, "continuous", "test"),
        overwrite=True,
    )
    assert "triangular_test" in distributions.available()
    out = distributions.get("triangular_test").sample(np.array([0.0, 1.0]))
    assert out[0] == pytest.approx(0.0) and out[1] == pytest.approx(1.0)


def test_unknown_class_raises_with_suggestions() -> None:
    with pytest.raises(ValueError, match="Unknown query class"):
        distributions.get("does_not_exist")
