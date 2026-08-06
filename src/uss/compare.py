"""Comparing two uncertain quantities -- the A/B question.

`execute_with_priors` handles one distribution whose *parameters* are uncertain.
Comparing two separate things -- control against variant, this month against
last -- is a different shape: two posteriors, and a derived quantity (their
difference, their ratio) that carries the uncertainty of both.

The reason this deserves its own module rather than a snippet in the docs is the
failure it prevents. Two common summaries of the same comparison routinely point
opposite ways:

    P(variant beats control) = 96.3%      sounds decisive
    95% interval on the difference: [-0.003, +0.063]   includes zero

Both are correct. The first is the probability the sign is positive; the second
is the range of plausible effect sizes, and it still contains "no difference".
Reporting only the first is how a 52.8% lift that might be nothing gets shipped
as a win. `ComparisonResult` reports both and says plainly when they disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .inference import Posterior


@dataclass
class ComparisonResult:
    """Two posteriors, their difference, and whether the sign is resolved."""

    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    interval_a: tuple[float, float]
    interval_b: tuple[float, float]

    difference: float
    difference_interval: tuple[float, float]
    relative_lift: float
    relative_lift_interval: tuple[float, float]
    probability_b_beats_a: float

    confidence_level: float
    n_draws: int
    caveats: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        """Whether the interval on the difference excludes zero."""
        lo, hi = self.difference_interval
        return lo > 0.0 or hi < 0.0

    @property
    def better(self) -> str | None:
        """Which side wins, or None when the comparison is unresolved."""
        if not self.resolved:
            return None
        return self.name_b if self.difference > 0 else self.name_a

    def summary(self) -> str:
        pct = int(self.confidence_level * 100)
        lines = [
            f"{self.name_a:<12} {self.mean_a:>10.4f}   "
            f"[{self.interval_a[0]:.4f}, {self.interval_a[1]:.4f}]",
            f"{self.name_b:<12} {self.mean_b:>10.4f}   "
            f"[{self.interval_b[0]:.4f}, {self.interval_b[1]:.4f}]",
            "",
            f"difference   {self.difference:>+10.4f}   "
            f"{pct}% [{self.difference_interval[0]:+.4f}, {self.difference_interval[1]:+.4f}]",
            f"relative     {self.relative_lift:>+10.1%}   "
            f"{pct}% [{self.relative_lift_interval[0]:+.1%}, {self.relative_lift_interval[1]:+.1%}]",
            f"P({self.name_b} > {self.name_a}) = {self.probability_b_beats_a:.1%}",
            "",
        ]
        if self.resolved:
            lines.append(
                f"-> {self.better} is better. The interval on the difference "
                "excludes zero."
            )
        else:
            lines.append(
                "-> NOT RESOLVED. The interval on the difference includes zero, "
                "so 'no difference' remains consistent with this data."
            )
        for caveat in self.caveats:
            lines.append(f"  ! {caveat}")
        return "\n".join(lines)


def compare(
    a: Posterior,
    b: Posterior,
    *,
    rng: np.random.Generator | None = None,
    n_draws: int = 200_000,
    confidence_level: float = 0.95,
    names: tuple[str, str] = ("A", "B"),
) -> ComparisonResult:
    """Compare two posteriors by sampling their difference.

    Both are sampled independently and subtracted draw by draw, so the reported
    interval carries the uncertainty of *both* sides rather than treating either
    as fixed.

    Independence is the assumption: it holds for separately-collected samples
    (two arms of a test, two locations, two periods). It does not hold when the
    two estimates share observations, and this function has no way to detect
    that -- see `uss.graph.gaussian_copula` for coupling inputs deliberately.
    """
    if n_draws < 2:
        raise ValueError(f"n_draws must be at least 2, got {n_draws}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must lie in (0, 1), got {confidence_level}"
        )

    generator = rng if rng is not None else np.random.default_rng()
    draws_a = np.asarray(a.sample(n_draws, generator), dtype=np.float64)
    draws_b = np.asarray(b.sample(n_draws, generator), dtype=np.float64)

    diff = draws_b - draws_a
    alpha = 1.0 - confidence_level
    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0

    # Relative lift is undefined where the baseline is zero, and explodes near
    # it; drop those draws rather than reporting an infinity.
    usable = draws_a != 0.0
    if np.any(usable):
        lift = draws_b[usable] / draws_a[usable] - 1.0
        lift_mean = float(np.mean(lift))
        lift_interval = (
            float(np.quantile(lift, lo_q)),
            float(np.quantile(lift, hi_q)),
        )
    else:
        lift_mean = float("nan")
        lift_interval = (float("nan"), float("nan"))

    caveats: list[str] = []
    p_better = float(np.mean(diff > 0.0))
    difference_interval = (
        float(np.quantile(diff, lo_q)),
        float(np.quantile(diff, hi_q)),
    )
    unresolved = difference_interval[0] <= 0.0 <= difference_interval[1]

    # The headline trap: a high win probability sitting on an interval that
    # still contains zero. Both numbers are right; quoting only the first is not.
    if unresolved and max(p_better, 1.0 - p_better) > 0.90:
        caveats.append(
            f"P({names[1]} > {names[0]}) is {p_better:.1%}, but the "
            f"{int(confidence_level * 100)}% interval on the difference still "
            "includes zero. The direction is likely; the effect size is not "
            "established. Collect more before acting on the magnitude."
        )
    if not np.any(usable):
        caveats.append(
            "baseline draws include zero throughout, so relative lift is undefined"
        )
    elif np.mean(usable) < 0.99:
        caveats.append(
            f"{1 - np.mean(usable):.1%} of baseline draws were zero and were "
            "excluded from the relative-lift figure"
        )
    if min(a.n_observations, b.n_observations) < 30:
        caveats.append(
            f"one side rests on {min(a.n_observations, b.n_observations)} "
            "observations; the comparison is dominated by that, not by the "
            "number of simulation draws"
        )

    return ComparisonResult(
        name_a=names[0],
        name_b=names[1],
        mean_a=float(a.mean),
        mean_b=float(b.mean),
        interval_a=a.interval(confidence_level),
        interval_b=b.interval(confidence_level),
        difference=float(np.mean(diff)),
        difference_interval=difference_interval,
        relative_lift=lift_mean,
        relative_lift_interval=lift_interval,
        probability_b_beats_a=p_better,
        confidence_level=confidence_level,
        n_draws=n_draws,
        caveats=caveats,
    )


def rank(
    posteriors: dict[str, Posterior],
    *,
    rng: np.random.Generator | None = None,
    n_draws: int = 200_000,
) -> list[tuple[str, float]]:
    """Probability that each option is the best of the set.

    With several arms, pairwise comparisons multiply and mislead. This gives the
    quantity that actually answers "which should I pick": the share of draws in
    which each option comes out on top. The values sum to 1.

    A flat result -- every option near 1/k -- means the data cannot separate
    them, however different the point estimates look.
    """
    if len(posteriors) < 2:
        raise ValueError("ranking needs at least two options")
    if n_draws < 2:
        raise ValueError(f"n_draws must be at least 2, got {n_draws}")

    generator = rng if rng is not None else np.random.default_rng()
    names = list(posteriors)
    matrix = np.column_stack(
        [np.asarray(posteriors[n].sample(n_draws, generator), dtype=np.float64) for n in names]
    )
    winners = np.argmax(matrix, axis=1)
    counts = np.bincount(winners, minlength=len(names))
    return sorted(
        ((names[i], float(counts[i] / n_draws)) for i in range(len(names))),
        key=lambda row: row[1],
        reverse=True,
    )
