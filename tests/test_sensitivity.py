"""Sobol indices are checked against functions with analytic decompositions."""

from __future__ import annotations

import numpy as np
import pytest

from uss import parameter_sensitivity, sobol_indices, update_bernoulli
from uss.sensitivity import one_at_a_time


def test_ishigami_matches_analytic_indices() -> None:
    """The standard Sobol benchmark, with known closed-form indices.

    f = sin(x1) + a sin^2(x2) + b x3^4 sin(x1),  xi ~ U(-pi, pi), a=7, b=0.1
    S1 = 0.3139, S2 = 0.4424, S3 = 0.0, and S_T3 = 0.2437 (pure interaction).
    """
    a, b = 7.0, 0.1

    def model(u: np.ndarray) -> np.ndarray:
        x = -np.pi + 2 * np.pi * u
        return (
            np.sin(x[:, 0])
            + a * np.sin(x[:, 1]) ** 2
            + b * x[:, 2] ** 4 * np.sin(x[:, 0])
        )

    res = sobol_indices(
        model, 3, 200_000, np.random.default_rng(31), names=("x1", "x2", "x3")
    )

    assert res.first_order[0] == pytest.approx(0.3139, abs=0.03)
    assert res.first_order[1] == pytest.approx(0.4424, abs=0.03)
    assert res.first_order[2] == pytest.approx(0.0, abs=0.03)
    assert res.total_effect[2] == pytest.approx(0.2437, abs=0.04)


def test_ishigami_detects_interaction() -> None:
    """x3 has zero first-order effect but a large total effect."""
    a, b = 7.0, 0.1

    def model(u: np.ndarray) -> np.ndarray:
        x = -np.pi + 2 * np.pi * u
        return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])

    res = sobol_indices(model, 3, 100_000, np.random.default_rng(32))
    assert res.interaction_strength > 0.15


def test_additive_model_has_no_interaction() -> None:
    """A purely additive model: sum(S_i) ~ 1 and sum(S_Ti) ~ 1."""

    def model(u: np.ndarray) -> np.ndarray:
        return 3.0 * u[:, 0] + 1.0 * u[:, 1]

    res = sobol_indices(model, 2, 100_000, np.random.default_rng(33))
    assert res.first_order.sum() == pytest.approx(1.0, abs=0.03)
    assert abs(res.interaction_strength) < 0.03
    # Variance contribution scales with the square of the coefficient: 9 vs 1.
    assert res.first_order[0] == pytest.approx(0.9, abs=0.03)


def test_irrelevant_input_scores_zero() -> None:
    def model(u: np.ndarray) -> np.ndarray:
        return u[:, 0] * 5.0  # u[:,1] unused

    res = sobol_indices(model, 2, 50_000, np.random.default_rng(34))
    assert res.total_effect[1] == pytest.approx(0.0, abs=0.01)
    assert res.first_order[0] == pytest.approx(1.0, abs=0.02)


def test_ranked_orders_by_total_effect() -> None:
    def model(u: np.ndarray) -> np.ndarray:
        return 0.5 * u[:, 0] + 4.0 * u[:, 1]

    res = sobol_indices(model, 2, 50_000, np.random.default_rng(35), names=("small", "big"))
    assert res.ranked()[0][0] == "big"
    assert "big drives" in res.summary()


def test_constant_model_returns_zero_indices() -> None:
    res = sobol_indices(
        lambda u: np.ones(u.shape[0]), 2, 5_000, np.random.default_rng(36)
    )
    assert res.output_variance == 0.0
    assert np.all(res.total_effect == 0.0)


def test_evaluation_count_matches_saltelli_design() -> None:
    res = sobol_indices(lambda u: u[:, 0], 4, 1000, np.random.default_rng(37))
    assert res.n_model_evaluations == 1000 * (4 + 2)


def test_names_length_validated() -> None:
    with pytest.raises(ValueError, match="names has"):
        sobol_indices(lambda u: u[:, 0], 2, 100, np.random.default_rng(38), names=("a",))


def test_parameter_sensitivity_ranks_the_dominant_posterior() -> None:
    """A parameter known from 10 observations should outrank one known from 10,000."""
    well_known = update_bernoulli(3000, 10_000)
    barely_known = update_bernoulli(3, 10)

    res = parameter_sensitivity(
        lambda p: p["vague"] + p["precise"],
        {"precise": well_known, "vague": barely_known},
        4_000,
        np.random.default_rng(39),
    )
    ranked = res.ranked()
    assert ranked[0][0] == "vague"
    assert res.total_effect[res.names.index("vague")] > 0.9


def test_parameter_sensitivity_requires_a_posterior() -> None:
    with pytest.raises(ValueError, match="at least one posterior"):
        parameter_sensitivity(lambda p: 1.0, {}, 100, np.random.default_rng(40))


def test_one_at_a_time_measures_swing() -> None:
    swings = one_at_a_time(
        lambda p: 3.0 * p["a"] + 1.0 * p["b"],
        baseline={"a": 0.5, "b": 0.5},
        perturbations={"a": (0.0, 1.0), "b": (0.0, 1.0)},
    )
    assert swings["a"] == pytest.approx(3.0)
    assert swings["b"] == pytest.approx(1.0)
    assert swings["__baseline__"] == pytest.approx(2.0)


def test_one_at_a_time_rejects_unknown_parameter() -> None:
    with pytest.raises(ValueError, match="not in the baseline"):
        one_at_a_time(lambda p: 1.0, {"a": 1.0}, {"zzz": (0.0, 1.0)})
