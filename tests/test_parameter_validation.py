"""Unrecognised parameters must be refused, never silently ignored.

Samplers take `**kwargs` so a shared parameter dict can be passed around. That
made unknown names vanish without trace, and the result still looked confident:

  * `gaussian` given `men=20` returned -0.003 instead of 20 -- the typo fell
    through to the default of 0.0.
  * A second posterior on a one-parameter distribution was drawn from, listed in
    the results table, counted in the "2 posterior(s)" caveat, and then discarded
    by the sampler without changing the answer.

Both are wrong answers delivered with a tight interval, which is the single
failure mode this engine exists to prevent.
"""

from __future__ import annotations

import numpy as np
import pytest

from uss import UniversalStochasticSandbox, distributions, update_bernoulli


@pytest.fixture
def u() -> np.ndarray:
    return np.random.default_rng(0).random(10_000)


# --------------------------------------------------------------------------
# The sampler layer
# --------------------------------------------------------------------------
def test_unknown_parameter_is_rejected(u) -> None:
    with pytest.raises(ValueError, match="does not accept"):
        distributions.get("bernoulli").sample(u, probability=0.3, total_nonsense=42.0)


def test_rejection_names_the_offender_and_the_valid_set(u) -> None:
    with pytest.raises(ValueError) as exc:
        distributions.get("gaussian").sample(u, mean=1.0, wobble=2.0)
    message = str(exc.value)
    assert "'wobble'" in message
    assert "It takes:" in message
    assert "std_dev" in message and "mean" in message


def test_near_miss_gets_a_suggestion(u) -> None:
    with pytest.raises(ValueError, match=r"did you mean 'mean'"):
        distributions.get("gaussian").sample(u, men=20.0)


def test_the_reported_case_is_rejected(u) -> None:
    """'probability 2' on a Bernoulli -- the exact input from the bug report."""
    with pytest.raises(ValueError, match=r"probability 2"):
        distributions.get("bernoulli").sample(u, **{"probability": 0.3, "probability 2": 0.2})


def test_valid_parameters_still_work(u) -> None:
    assert distributions.get("bernoulli").sample(u, probability=0.3).mean() == pytest.approx(0.3, abs=0.02)
    assert distributions.get("gaussian").sample(u, mean=5.0, std_dev=2.0).mean() == pytest.approx(5.0, abs=0.1)
    assert distributions.get("extreme_value").sample(u, loc=0.0, scale=1.0, shape=0.2).size == u.size


def test_accepted_parameters_matches_each_signature() -> None:
    assert distributions.get("bernoulli").accepted_parameters() == {"probability"}
    assert distributions.get("gaussian").accepted_parameters() == {"mean", "std_dev"}
    assert distributions.get("poisson").accepted_parameters() == {"lam"}
    assert distributions.get("empirical").accepted_parameters() == {"quantiles"}
    assert distributions.get("extreme_value").accepted_parameters() == {"loc", "scale", "shape"}


def test_custom_registered_class_is_validated_too() -> None:
    """Extensibility must not reopen the hole."""

    def triangular(u, low=0.0, high=1.0, **_):
        return low + (high - low) * np.sqrt(u)

    distributions.register(
        distributions.QueryClass("tri_validate", triangular, "continuous", "test"),
        overwrite=True,
    )
    qc = distributions.get("tri_validate")
    assert qc.accepted_parameters() == {"low", "high"}
    with pytest.raises(ValueError, match="does not accept"):
        qc.sample(np.array([0.5]), low=0.0, hihg=1.0)


# --------------------------------------------------------------------------
# The engine layer
# --------------------------------------------------------------------------
def test_typo_no_longer_returns_a_confident_wrong_answer() -> None:
    """`men=20` silently produced -0.003 with a tight interval around it."""
    sandbox = UniversalStochasticSandbox(sample_size=50_000, seed=1)
    good = sandbox.execute_query("gaussian", {"mean": 20.0, "std_dev": 3.0})
    assert good.report.point_estimate == pytest.approx(20.0, abs=0.1)

    with pytest.raises(ValueError, match=r"did you mean 'mean'"):
        sandbox.execute_query("gaussian", {"men": 20.0, "std_dev": 3.0})


def test_extra_posterior_on_a_one_parameter_class_is_rejected() -> None:
    """It used to be sampled, reported, and thrown away."""
    sandbox = UniversalStochasticSandbox(sample_size=50_000, seed=2)
    with pytest.raises(ValueError, match=r"probability 2"):
        sandbox.execute_with_priors(
            "bernoulli",
            {
                "probability": update_bernoulli(32, 152),
                "probability 2": update_bernoulli(28, 140),
            },
            n_parameter_draws=20,
            inner_sample_size=5_000,
        )


def test_multi_parameter_class_accepts_multiple_posteriors() -> None:
    """The legitimate case must keep working."""
    from uss import update_gaussian_mean

    sandbox = UniversalStochasticSandbox(sample_size=50_000, seed=3)
    result = sandbox.execute_with_priors(
        "gaussian",
        {
            "mean": update_gaussian_mean(np.array([10.0, 12.0, 9.0, 11.0, 13.0])),
            "std_dev": update_gaussian_mean(np.array([2.0, 2.2, 1.9, 2.1])),
        },
        n_parameter_draws=64,
        inner_sample_size=20_000,
    )
    assert result.report.total_interval is not None
    assert result.report.parameter_uncertainty > 0


def test_graph_nodes_are_validated() -> None:
    from uss import ScenarioGraph

    g = ScenarioGraph(seed=4)
    g.add("a", "gaussian", mean=0.0, stdev=1.0)  # 'stdev' is not 'std_dev'
    with pytest.raises(ValueError, match="does not accept"):
        g.run(1_000)


def test_alias_normalisation_still_works() -> None:
    """Blueprint-era names must survive the stricter check."""
    sandbox = UniversalStochasticSandbox(sample_size=50_000, seed=5)
    r = sandbox.execute_query("poisson", {"lambda": 3.0})
    assert r.parameters == {"lam": 3.0}
    assert r.report.point_estimate == pytest.approx(3.0, rel=0.05)

    r2 = sandbox.execute_query("gaussian", {"mu": 7.0, "sigma": 2.0})
    assert r2.report.point_estimate == pytest.approx(7.0, abs=0.1)
