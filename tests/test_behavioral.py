"""Behavioral operators, including the clipping failure they replace."""

from __future__ import annotations

import numpy as np
import pytest

from uss import behavioral


def test_logistic_link_matches_closed_form() -> None:
    p = behavioral.logistic_link(0.0, [1.0], [0.0])
    assert float(p) == pytest.approx(0.5)
    p = behavioral.logistic_link(-1.0, [2.0], [1.5])
    assert float(p) == pytest.approx(1.0 / (1.0 + np.exp(-2.0)))


def test_logistic_link_is_overflow_safe() -> None:
    """A bare 1/(1+exp(-z)) overflows here; expit must not."""
    assert float(behavioral.logistic_link(0.0, [1.0], [1000.0])) == pytest.approx(1.0)
    assert float(behavioral.logistic_link(0.0, [1.0], [-1000.0])) == pytest.approx(0.0)


def test_logistic_link_handles_batched_covariates() -> None:
    x = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
    p = behavioral.logistic_link(0.0, [1.0, 1.0], x)
    assert p.shape == (3,)
    assert p[0] == pytest.approx(0.5)
    assert p[1] > p[0] > p[2]


def test_logistic_link_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match=r"incompatible|features"):
        behavioral.logistic_link(0.0, [1.0, 2.0], [1.0])


def test_odds_ratio_shift_never_leaves_unit_interval() -> None:
    """This is the fix for clip(p * multiplier, 0, 1)."""
    for p in [0.01, 0.28, 0.5, 0.9, 0.99]:
        for ratio in [0.1, 1.15, 5.0, 1000.0]:
            out = behavioral.odds_ratio_shift(p, ratio)
            assert 0.0 < out < 1.0


def test_odds_ratio_shift_preserves_ordering_where_clipping_destroys_it() -> None:
    """p=0.9 under multipliers 1.2 and 5.0: clipping collapses both to 1.0."""
    a = behavioral.odds_ratio_shift(0.9, 1.2)
    b = behavioral.odds_ratio_shift(0.9, 5.0)
    assert a < b < 1.0
    # The blueprint's original path loses the distinction entirely.
    assert np.clip(0.9 * 1.2, 0, 1) == np.clip(0.9 * 5.0, 0, 1) == 1.0


def test_odds_ratio_of_one_is_identity() -> None:
    assert behavioral.odds_ratio_shift(0.28, 1.0) == pytest.approx(0.28)


def test_loss_aversion_scales_only_losses() -> None:
    out = behavioral.apply_loss_aversion(np.array([-2.0, 0.0, 3.0]))
    assert out[0] == pytest.approx(-4.5)  # -2 * 2.25
    assert out[1] == 0.0
    assert out[2] == 3.0


def test_prospect_value_is_steeper_for_losses() -> None:
    gain = behavioral.prospect_value(np.array([10.0]))[0]
    loss = behavioral.prospect_value(np.array([-10.0]))[0]
    assert abs(loss) > abs(gain)
    assert abs(loss / gain) == pytest.approx(behavioral.LOSS_AVERSION_LAMBDA, rel=1e-9)


def test_social_proof_increases_adoption() -> None:
    rng = np.random.default_rng(41)
    traj = behavioral.social_proof_cascade(0.1, alpha=0.5, steps=10, rng=rng, population=200_000)
    assert traj.size == 11
    assert traj[-1] > traj[0]
    assert np.all((traj >= 0) & (traj <= 1))


def test_social_proof_with_zero_alpha_is_flat() -> None:
    rng = np.random.default_rng(42)
    traj = behavioral.social_proof_cascade(0.3, alpha=0.0, steps=8, rng=rng, population=200_000)
    assert np.allclose(traj, 0.3, atol=0.01)


def test_social_proof_requires_population_for_scalar_input() -> None:
    with pytest.raises(ValueError, match="population is required"):
        behavioral.social_proof_cascade(0.3, 0.5, 5, np.random.default_rng(43))


def test_hyperbolic_discount_decays_with_delay() -> None:
    v = behavioral.hyperbolic_discount(100.0, np.array([0.0, 1.0, 10.0]), k=0.5)
    assert v[0] == pytest.approx(100.0)
    assert v[1] == pytest.approx(100.0 / 1.5)
    assert v[2] == pytest.approx(100.0 / 6.0)
    assert v[0] > v[1] > v[2]


def test_hyperbolic_discount_rejects_negative_delay() -> None:
    with pytest.raises(ValueError, match="delay must be non-negative"):
        behavioral.hyperbolic_discount(10.0, np.array([-1.0]))


def test_multinomial_logit_normalises_and_is_overflow_safe() -> None:
    probs = behavioral.multinomial_logit(np.array([1000.0, 999.0, 998.0]))
    assert probs.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(probs))
    assert probs[0] > probs[1] > probs[2]


def test_multinomial_logit_batched() -> None:
    probs = behavioral.multinomial_logit(np.array([[1.0, 2.0], [3.0, 1.0]]))
    assert np.allclose(probs.sum(axis=1), 1.0)
