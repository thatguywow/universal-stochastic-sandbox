"""Does the engine's stated confidence actually hold?

The blueprint mandates statistical integrity but provides no way to check it.
An engine that reports 95% intervals is making a falsifiable claim: run it
against known outcomes 100 times and roughly 95 of the intervals should contain
the truth.  This module measures that.

Two diagnostics:

    coverage_curve   for interval claims -- empirical vs nominal across levels.
                     Below the diagonal means overconfident (intervals too
                     narrow), above means conservative.

    pit_histogram    for full predictive distributions.  The probability
                     integral transform of well-calibrated forecasts is
                     Uniform(0,1).  A U-shape means under-dispersed forecasts;
                     a hump means over-dispersed; a slope means bias.

Use these before trusting a model on a domain where you cannot check the
answer -- validate on a domain where you can.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class CoverageResult:
    """Empirical coverage of interval claims across nominal levels."""

    nominal_levels: np.ndarray
    empirical_coverage: np.ndarray
    n_trials: int

    @property
    def calibration_error(self) -> float:
        """Mean absolute deviation from the diagonal."""
        return float(np.mean(np.abs(self.empirical_coverage - self.nominal_levels)))

    @property
    def verdict(self) -> str:
        signed = float(np.mean(self.empirical_coverage - self.nominal_levels))
        if abs(signed) < 0.02:
            return "calibrated"
        return "overconfident" if signed < 0 else "conservative"

    def summary(self) -> str:
        lines = [
            f"Coverage over {self.n_trials:,} trials  [{self.verdict}]",
            f"  {'nominal':>9}{'empirical':>12}{'error':>10}",
        ]
        for nom, emp in zip(
            self.nominal_levels, self.empirical_coverage, strict=True
        ):
            lines.append(f"  {nom:>9.2f}{emp:>12.3f}{emp - nom:>+10.3f}")
        lines.append(f"  mean absolute calibration error: {self.calibration_error:.4f}")
        return "\n".join(lines)


@dataclass(frozen=True)
class PITResult:
    """Probability-integral-transform diagnostics for predictive distributions."""

    values: np.ndarray
    ks_statistic: float
    ks_pvalue: float
    n_observations: int
    randomised: bool = False

    @property
    def verdict(self) -> str:
        if self.ks_pvalue > 0.05:
            return "calibrated"
        # Variance of U(0,1) is 1/12; more spread than that means the forecast
        # distribution was too narrow.
        return (
            "under-dispersed (forecasts too confident)"
            if float(np.var(self.values)) > 1.0 / 12.0
            else "over-dispersed (forecasts too vague)"
        )

    def histogram(self, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
        counts, edges = np.histogram(self.values, bins=bins, range=(0.0, 1.0))
        return counts, edges

    def summary(self, bins: int = 10) -> str:
        counts, _ = self.histogram(bins)
        expected = self.n_observations / bins
        transform = "randomised (discrete)" if self.randomised else "standard"
        lines = [
            f"PIT over {self.n_observations:,} observations  [{self.verdict}]",
            f"  transform: {transform}",
            f"  KS statistic {self.ks_statistic:.4f}  p={self.ks_pvalue:.4g}"
            f"  (uniform PIT => calibrated)",
        ]
        peak = max(int(counts.max()), 1)
        for i, count in enumerate(counts):
            bar = "#" * int(30 * count / peak)
            lines.append(f"  [{i / bins:.1f},{(i + 1) / bins:.1f})  {bar} {count}")
        lines.append(f"  uniform would put ~{expected:,.0f} in each bin")
        return "\n".join(lines)


def coverage_curve(
    trial: Callable[[float, np.random.Generator], tuple[tuple[float, float], float]],
    n_trials: int,
    rng: np.random.Generator,
    *,
    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95, 0.99),
) -> CoverageResult:
    """Measure empirical coverage of an interval procedure.

    `trial(level, rng)` must run one independent replication and return
    `((lower, upper), truth)` -- the interval the engine would report at that
    confidence level, and the true value it is trying to capture.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be positive, got {n_trials}")
    nominal = np.asarray(levels, dtype=np.float64)
    if np.any((nominal <= 0) | (nominal >= 1)):
        raise ValueError("levels must lie strictly in (0, 1)")

    empirical = np.empty(nominal.size, dtype=np.float64)
    for i, level in enumerate(nominal):
        hits = 0
        for _ in range(n_trials):
            (lo, hi), truth = trial(float(level), rng)
            hits += lo <= truth <= hi
        empirical[i] = hits / n_trials

    return CoverageResult(nominal, empirical, n_trials)


def pit_values(
    predictive_samples: np.ndarray,
    observations: np.ndarray,
    *,
    discrete: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """PIT of each observation against its predictive sample set.

    `predictive_samples` is (n_observations, n_draws); entry i is the engine's
    predictive distribution for observation i.  Returns F_i(y_i) per observation.

    For a **discrete** predictive distribution the plain transform F(y) is not
    uniform even when the forecast is perfect -- the CDF jumps, so PIT mass
    piles up at the jump points.  Tested on a correctly-specified Poisson
    forecast the plain form returns KS p = 0, a false alarm.  Set
    `discrete=True` to use the randomised PIT of Czado et al.,

        u * F(y-1) + (1 - u) * F(y),   u ~ Uniform(0,1)

    which restores uniformity under correct specification.
    """
    preds = np.asarray(predictive_samples, dtype=np.float64)
    obs = np.asarray(observations, dtype=np.float64).ravel()
    if preds.ndim != 2:
        raise ValueError("predictive_samples must be 2-D (n_observations, n_draws)")
    if preds.shape[0] != obs.size:
        raise ValueError(
            f"got {preds.shape[0]} predictive rows for {obs.size} observations"
        )

    upper = np.mean(preds <= obs[:, None], axis=1)
    if not discrete:
        return upper

    generator = rng if rng is not None else np.random.default_rng()
    lower = np.mean(preds <= (obs[:, None] - 1.0), axis=1)
    u = generator.random(obs.size)
    return lower + u * (upper - lower)


def _looks_discrete(values: np.ndarray) -> bool:
    """Heuristic: integer-valued predictive draws imply a discrete forecast."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return bool(np.all(finite == np.round(finite)))


def pit_report(
    predictive_samples: np.ndarray,
    observations: np.ndarray,
    *,
    discrete: bool | None = None,
    rng: np.random.Generator | None = None,
) -> PITResult:
    """PIT values plus a uniformity test.

    `discrete` defaults to auto-detection from the predictive draws; integer
    valued draws select the randomised transform.  Pass it explicitly to
    override.
    """
    preds = np.asarray(predictive_samples, dtype=np.float64)
    is_discrete = _looks_discrete(preds) if discrete is None else bool(discrete)
    values = pit_values(preds, observations, discrete=is_discrete, rng=rng)
    ks = stats.kstest(values, "uniform")
    return PITResult(
        values=values,
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        n_observations=values.size,
        randomised=is_discrete,
    )


def interval_score(
    lower: np.ndarray, upper: np.ndarray, truth: np.ndarray, alpha: float = 0.05
) -> float:
    """Winkler interval score -- lower is better.

    Rewards narrow intervals but penalises misses in proportion to how far
    outside they fall, so it cannot be gamed by widening intervals the way raw
    coverage can.
    """
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if not (lo.shape == hi.shape == y.shape):
        raise ValueError("lower, upper and truth must share a shape")
    if np.any(hi < lo):
        raise ValueError("upper bound is below lower bound")

    width = hi - lo
    below = (2.0 / alpha) * np.clip(lo - y, 0.0, None)
    above = (2.0 / alpha) * np.clip(y - hi, 0.0, None)
    return float(np.mean(width + below + above))


def sharpness(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean interval width. Report alongside coverage, never instead of it."""
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))
