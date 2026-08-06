"""Comparing two uncertain quantities."""

from __future__ import annotations

import numpy as np
import pytest

from uss import compare, rank, update_bernoulli, update_gaussian_mean


def test_clear_winner_is_resolved() -> None:
    """Large, well-measured gap: the interval must exclude zero."""
    res = compare(
        update_bernoulli(50, 1000),      # 5%
        update_bernoulli(150, 1000),     # 15%
        rng=np.random.default_rng(1),
        names=("control", "variant"),
    )
    assert res.resolved
    assert res.better == "variant"
    assert res.difference == pytest.approx(0.10, abs=0.01)
    assert res.difference_interval[0] > 0
    assert res.probability_b_beats_a > 0.999
    assert "is better" in res.summary()


def test_high_win_probability_with_an_interval_spanning_zero_is_flagged() -> None:
    """The headline trap: 96% to win, effect size not established."""
    res = compare(
        update_bernoulli(30, 500),
        update_bernoulli(45, 500),
        rng=np.random.default_rng(2),
        names=("control", "variant"),
    )
    assert res.probability_b_beats_a > 0.90
    assert not res.resolved
    assert res.better is None
    assert any("includes zero" in c for c in res.caveats)
    assert "NOT RESOLVED" in res.summary()


def test_identical_arms_are_not_resolved() -> None:
    res = compare(
        update_bernoulli(100, 1000),
        update_bernoulli(100, 1000),
        rng=np.random.default_rng(3),
    )
    assert not res.resolved
    assert res.probability_b_beats_a == pytest.approx(0.5, abs=0.05)
    assert res.difference == pytest.approx(0.0, abs=0.01)


def test_direction_is_detected_both_ways() -> None:
    worse = compare(
        update_bernoulli(150, 1000),
        update_bernoulli(50, 1000),
        rng=np.random.default_rng(4),
        names=("control", "variant"),
    )
    assert worse.resolved
    assert worse.better == "control"
    assert worse.difference < 0
    assert worse.probability_b_beats_a < 0.001


def test_difference_interval_carries_both_uncertainties() -> None:
    """Widening either side must widen the comparison."""
    tight = compare(
        update_bernoulli(500, 10_000), update_bernoulli(600, 10_000),
        rng=np.random.default_rng(5),
    )
    loose = compare(
        update_bernoulli(5, 100), update_bernoulli(6, 100),
        rng=np.random.default_rng(6),
    )
    tight_w = tight.difference_interval[1] - tight.difference_interval[0]
    loose_w = loose.difference_interval[1] - loose.difference_interval[0]
    assert loose_w > tight_w * 5


def test_relative_lift_is_reported_with_an_interval() -> None:
    res = compare(
        update_bernoulli(100, 1000), update_bernoulli(150, 1000),
        rng=np.random.default_rng(7),
    )
    assert res.relative_lift == pytest.approx(0.5, abs=0.1)
    lo, hi = res.relative_lift_interval
    assert lo < res.relative_lift < hi


def test_small_samples_are_flagged() -> None:
    res = compare(
        update_bernoulli(3, 10), update_bernoulli(5, 10),
        rng=np.random.default_rng(8),
    )
    assert any("observations" in c for c in res.caveats)


def test_works_with_gaussian_posteriors() -> None:
    rng = np.random.default_rng(9)
    res = compare(
        update_gaussian_mean(rng.normal(10.0, 1.0, 50)),
        update_gaussian_mean(rng.normal(14.0, 1.0, 50)),
        rng=rng,
        names=("before", "after"),
    )
    assert res.resolved
    assert res.better == "after"
    assert res.difference == pytest.approx(4.0, abs=0.5)


def test_confidence_level_widens_the_interval() -> None:
    a, b = update_bernoulli(30, 500), update_bernoulli(45, 500)
    narrow = compare(a, b, rng=np.random.default_rng(10), confidence_level=0.80)
    wide = compare(a, b, rng=np.random.default_rng(10), confidence_level=0.99)
    assert (wide.difference_interval[1] - wide.difference_interval[0]) > (
        narrow.difference_interval[1] - narrow.difference_interval[0]
    )


def test_invalid_arguments_rejected() -> None:
    a, b = update_bernoulli(5, 10), update_bernoulli(6, 10)
    with pytest.raises(ValueError, match="n_draws must be at least 2"):
        compare(a, b, n_draws=1)
    with pytest.raises(ValueError, match="confidence_level must lie"):
        compare(a, b, confidence_level=1.5)


# --------------------------------------------------------------------------
# Ranking several options
# --------------------------------------------------------------------------
def test_rank_identifies_the_best_option() -> None:
    ranked = rank(
        {
            "a": update_bernoulli(50, 1000),
            "b": update_bernoulli(150, 1000),
            "c": update_bernoulli(90, 1000),
        },
        rng=np.random.default_rng(11),
    )
    assert ranked[0][0] == "b"
    assert ranked[0][1] > 0.99


def test_rank_probabilities_sum_to_one() -> None:
    ranked = rank(
        {n: update_bernoulli(100, 1000) for n in ("a", "b", "c", "d")},
        rng=np.random.default_rng(12),
    )
    assert sum(p for _, p in ranked) == pytest.approx(1.0, abs=1e-9)


def test_indistinguishable_options_split_evenly() -> None:
    """A flat ranking is the honest answer when nothing separates them."""
    ranked = rank(
        {n: update_bernoulli(100, 1000) for n in ("a", "b", "c")},
        rng=np.random.default_rng(13),
    )
    for _, p in ranked:
        assert p == pytest.approx(1 / 3, abs=0.05)


def test_rank_requires_two_options() -> None:
    with pytest.raises(ValueError, match="at least two options"):
        rank({"only": update_bernoulli(5, 10)})
