"""Study planning: how much evidence do you need before the answer is useful?

`update_bernoulli(yes, total)` needs two counts you have to go and collect. This
module answers the questions that come immediately before and after that:

    "How many do I have to count to get an interval this narrow?"
    "I can only count 200 -- how good will the answer be?"
    "I have no data at all. What would the value have to be to change my mind?"

The last one is the most useful when data is expensive: a breakeven value needs
no observations, and often ends the question outright ("we'd need a 60% rate for
this to be worth doing, and it is obviously nowhere near that").

Sample sizes are computed against the Wilson interval, the same construction
`uss.estimators` reports, so the planned width is the width you will actually
get rather than a normal-approximation estimate of it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .estimators import garwood_interval, wilson_interval

_MAX_N = 10_000_000


@dataclass(frozen=True)
class SamplePlan:
    """How much counting a target precision requires."""

    n_required: int
    achieved_half_width: float
    assumed_rate: float
    confidence_level: float

    def summary(self) -> str:
        return (
            f"count {self.n_required:,} to reach +/-{self.achieved_half_width:.1%} "
            f"at {self.confidence_level:.0%} confidence "
            f"(assuming a rate near {self.assumed_rate:.0%})"
        )


def proportion_half_width(n: int, assumed_rate: float = 0.5, confidence_level: float = 0.95) -> float:
    """Half-width of the Wilson interval you would get from `n` observations."""
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 <= assumed_rate <= 1.0:
        raise ValueError(f"assumed_rate must lie in [0, 1], got {assumed_rate}")
    successes = int(round(assumed_rate * n))
    lo, hi = wilson_interval(successes, n, confidence_level)
    return (hi - lo) / 2.0


def samples_for_proportion(
    target_half_width: float,
    assumed_rate: float = 0.5,
    confidence_level: float = 0.95,
) -> SamplePlan:
    """Smallest n whose Wilson interval is at most +/- `target_half_width`.

    `assumed_rate` is your prior guess at the answer; 0.5 is the worst case and
    the safe default when you have no idea. Guessing wrong is not fatal -- it
    only means the realised interval is narrower or wider than planned.
    """
    if not 0.0 < target_half_width < 1.0:
        raise ValueError(
            f"target_half_width must lie in (0, 1), got {target_half_width}"
        )

    # Normal approximation gives a starting point, then step up until the exact
    # Wilson width actually clears the target.
    z = float(stats.norm.ppf((1.0 + confidence_level) / 2.0))
    p = min(max(assumed_rate, 1e-6), 1 - 1e-6)
    n = max(2, int((z / target_half_width) ** 2 * p * (1 - p)))

    while n < _MAX_N and proportion_half_width(n, assumed_rate, confidence_level) > target_half_width:
        n = int(n * 1.15) + 1
    # Walk back down to the true minimum.
    while n > 2 and proportion_half_width(n - 1, assumed_rate, confidence_level) <= target_half_width:
        n -= 1

    return SamplePlan(
        n_required=n,
        achieved_half_width=proportion_half_width(n, assumed_rate, confidence_level),
        assumed_rate=assumed_rate,
        confidence_level=confidence_level,
    )


def proportion_tradeoff(
    assumed_rate: float = 0.5,
    confidence_level: float = 0.95,
    counts: tuple[int, ...] = (10, 30, 50, 100, 200, 400, 1000, 2500),
) -> list[tuple[int, float]]:
    """(n, half-width) pairs, for deciding how much effort is worth it."""
    return [
        (n, proportion_half_width(n, assumed_rate, confidence_level)) for n in counts
    ]


def exposure_for_rate(
    target_relative_precision: float,
    assumed_rate: float,
    confidence_level: float = 0.95,
) -> float:
    """Observation exposure needed to pin a Poisson rate to within +/- r (relative).

    Rare events are governed by the *event count*, not the exposure: you need
    roughly (z/r)^2 events regardless of how long that takes. Returns the
    exposure that delivers them at the assumed rate.
    """
    if not 0.0 < target_relative_precision < 1.0:
        raise ValueError("target_relative_precision must lie in (0, 1)")
    if assumed_rate <= 0:
        raise ValueError(f"assumed_rate must be positive, got {assumed_rate}")
    z = float(stats.norm.ppf((1.0 + confidence_level) / 2.0))
    events_needed = (z / target_relative_precision) ** 2
    return float(events_needed / assumed_rate)


def rate_upper_bound(exposure: float, confidence_level: float = 0.95) -> float:
    """Upper bound on a rate given `exposure` of observation and zero events.

    The honest output when something has never been seen. The lower bound is
    always exactly zero -- see `examples/case2_balloon.py`.
    """
    return garwood_interval(0, exposure, confidence_level)[1]


@dataclass(frozen=True)
class Breakeven:
    """The input value at which a decision flips."""

    value: float | None
    threshold: float
    low: float
    high: float
    outcome_at_low: float
    outcome_at_high: float

    @property
    def decisive(self) -> bool:
        return self.value is not None

    def summary(self, name: str = "the rate") -> str:
        if not self.decisive:
            return (
                f"no breakeven in [{self.low:g}, {self.high:g}]: the outcome runs "
                f"from {self.outcome_at_low:.4g} to {self.outcome_at_high:.4g}, "
                f"never crossing {self.threshold:.4g}. The decision does not "
                "depend on this input -- stop measuring it."
            )
        return (
            f"decision flips when {name} = {self.value:.4g}. "
            f"Below that the outcome stays under {self.threshold:.4g}; above it, over. "
            "Judge whether reality is plausibly on one side before collecting data."
        )


def breakeven(
    outcome: Callable[[float], float],
    threshold: float,
    low: float,
    high: float,
    *,
    tolerance: float = 1e-9,
    max_iterations: int = 200,
) -> Breakeven:
    """Find the input value where `outcome` crosses `threshold`.

    Requires no observations at all. When the crossing sits far outside anything
    plausible, you have answered the question without collecting data; when it
    sits right in the middle of the plausible range, you have learned that the
    measurement is worth paying for.

    `outcome` must be monotone over [low, high]; a bisection is used.
    """
    if high <= low:
        raise ValueError(f"high must exceed low, got [{low}, {high}]")

    f_low = float(outcome(low)) - threshold
    f_high = float(outcome(high)) - threshold

    if f_low == 0.0:
        return Breakeven(low, threshold, low, high, f_low + threshold, f_high + threshold)
    if f_high == 0.0:
        return Breakeven(high, threshold, low, high, f_low + threshold, f_high + threshold)
    if np.sign(f_low) == np.sign(f_high):
        return Breakeven(None, threshold, low, high, f_low + threshold, f_high + threshold)

    a, b = low, high
    for _ in range(max_iterations):
        mid = 0.5 * (a + b)
        f_mid = float(outcome(mid)) - threshold
        if abs(f_mid) < tolerance or (b - a) < tolerance:
            break
        if np.sign(f_mid) == np.sign(f_low):
            a, f_low = mid, f_mid
        else:
            b = mid
    return Breakeven(
        0.5 * (a + b), threshold, low, high, float(outcome(low)), float(outcome(high))
    )
