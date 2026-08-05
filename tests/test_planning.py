"""Study planning: sample sizes, exposure, and breakeven analysis."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from uss import (
    breakeven,
    exposure_for_rate,
    proportion_half_width,
    proportion_tradeoff,
    rate_upper_bound,
    samples_for_proportion,
    update_bernoulli,
    wilson_interval,
)


def test_planned_width_matches_what_you_actually_get() -> None:
    """The plan must agree with the interval the engine later reports."""
    plan = samples_for_proportion(0.05, assumed_rate=0.4)
    successes = int(round(0.4 * plan.n_required))
    lo, hi = wilson_interval(successes, plan.n_required)
    assert (hi - lo) / 2 == pytest.approx(plan.achieved_half_width, rel=1e-9)
    assert (hi - lo) / 2 <= 0.05


def test_plan_agrees_with_the_posterior_it_will_produce() -> None:
    """Sanity: the Beta posterior from that many counts is about as wide."""
    plan = samples_for_proportion(0.05, assumed_rate=0.4)
    post = update_bernoulli(int(round(0.4 * plan.n_required)), plan.n_required)
    lo, hi = post.interval(0.95)
    assert (hi - lo) / 2 == pytest.approx(0.05, abs=0.01)


def test_n_required_is_minimal() -> None:
    plan = samples_for_proportion(0.05, assumed_rate=0.5)
    assert proportion_half_width(plan.n_required, 0.5) <= 0.05
    assert proportion_half_width(plan.n_required - 1, 0.5) > 0.05


def test_quartering_the_width_roughly_quadruples_the_count() -> None:
    """Precision costs quadratically -- the key planning fact."""
    wide = samples_for_proportion(0.10, assumed_rate=0.5).n_required
    narrow = samples_for_proportion(0.05, assumed_rate=0.5).n_required
    assert narrow / wide == pytest.approx(4.0, rel=0.15)


def test_rates_near_the_boundary_need_fewer_samples() -> None:
    assert (
        samples_for_proportion(0.05, assumed_rate=0.05).n_required
        < samples_for_proportion(0.05, assumed_rate=0.5).n_required
    )


def test_half_width_shrinks_with_n() -> None:
    widths = [w for _, w in proportion_tradeoff(assumed_rate=0.3)]
    assert all(a > b for a, b in itertools.pairwise(widths))


def test_half_width_validates_inputs() -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        proportion_half_width(0)
    with pytest.raises(ValueError, match="assumed_rate must lie"):
        proportion_half_width(100, assumed_rate=1.5)
    with pytest.raises(ValueError, match="target_half_width must lie"):
        samples_for_proportion(0.0)


def test_exposure_for_rate_delivers_the_needed_event_count() -> None:
    """Relative precision on a Poisson rate is set by the event count."""
    rate = 5e-6
    exposure = exposure_for_rate(0.10, assumed_rate=rate)
    expected_events = rate * exposure
    assert expected_events == pytest.approx((1.959963985 / 0.10) ** 2, rel=1e-6)
    # ~384 events, independent of how rare the process is.
    assert 380 < expected_events < 390


def test_exposure_scales_inversely_with_rate() -> None:
    a = exposure_for_rate(0.1, assumed_rate=1e-6)
    b = exposure_for_rate(0.1, assumed_rate=1e-9)
    assert b / a == pytest.approx(1000.0, rel=1e-6)


def test_rate_upper_bound_matches_garwood_at_zero_events() -> None:
    bound = rate_upper_bound(1.262e11)
    assert bound == pytest.approx(2.922e-11, rel=1e-3)


def test_rate_upper_bound_tightens_with_exposure() -> None:
    assert rate_upper_bound(1e12) < rate_upper_bound(1e9)


# --------------------------------------------------------------------------
# Breakeven
# --------------------------------------------------------------------------
def test_breakeven_finds_the_crossing() -> None:
    # Profit crosses zero when the conversion rate hits 0.02.
    res = breakeven(lambda p: p * 5000.0 - 100.0, threshold=0.0, low=0.0, high=1.0)
    assert res.decisive
    assert res.value == pytest.approx(0.02, abs=1e-6)


def test_breakeven_reports_when_no_crossing_exists() -> None:
    """If the decision never flips, the input is not worth measuring."""
    res = breakeven(lambda p: p * 10.0 + 1000.0, threshold=0.0, low=0.0, high=1.0)
    assert not res.decisive
    assert res.value is None
    assert "does not depend on this input" in res.summary()
    assert "stop measuring it" in res.summary()


def test_breakeven_handles_a_decreasing_function() -> None:
    res = breakeven(lambda p: 100.0 - p * 200.0, threshold=0.0, low=0.0, high=1.0)
    assert res.value == pytest.approx(0.5, abs=1e-6)


def test_breakeven_endpoint_exactly_on_threshold() -> None:
    res = breakeven(lambda p: p, threshold=0.0, low=0.0, high=1.0)
    assert res.value == pytest.approx(0.0)


def test_breakeven_on_a_simulated_outcome() -> None:
    """Realistic use: what stockout rate flips a restocking decision?"""

    def p_stockout(oat_fraction: float) -> float:
        rng = np.random.default_rng(7)  # seeded: must be deterministic
        orders = rng.poisson(124.0 * 3.0 * oat_fraction, size=40_000)
        return float(np.mean(orders * 150.0 > 18_000.0))

    res = breakeven(p_stockout, threshold=0.10, low=0.05, high=0.60)
    assert res.decisive
    assert 0.20 < res.value < 0.40
    assert "decision flips" in res.summary("oat fraction")


def test_breakeven_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="high must exceed low"):
        breakeven(lambda x: x, 0.0, 1.0, 0.0)


def test_sample_plan_summary_is_readable() -> None:
    text = samples_for_proportion(0.05, assumed_rate=0.4).summary()
    assert "count" in text and "95% confidence" in text
