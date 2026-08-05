"""Variance reduction mechanics (blueprint Part II.3).

Two estimators live here, and both return their *own* standard error rather
than deferring to a generic `stats.sem` call.  That is not a stylistic choice:

* Antithetic pairs are negatively correlated by construction.  Pooling the two
  halves into one array and calling `stats.sem` treats them as i.i.d. and
  overstates the true standard error -- measured at 3.4x on a lognormal-style
  functional, which would report a converged estimate as unconverged.
* Importance-sampled draws are weighted.  The unweighted SEM of the weighted
  values is not the standard error of the ratio estimator.

Each estimator therefore carries its own `standard_error` and an effective
sample size, and `uss.estimators` consumes those instead of recomputing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a correctly-derived Monte Carlo standard error."""

    value: float
    standard_error: float
    effective_sample_size: float
    n_evaluations: int
    method: str
    samples: np.ndarray | None = None
    warnings: tuple[str, ...] = ()

    @property
    def variance_reduction_note(self) -> str:
        return f"{self.method} over {self.n_evaluations:,} evaluations"

    @property
    def efficiency(self) -> float:
        """Effective sample size as a fraction of the draws actually spent."""
        return (
            self.effective_sample_size / self.n_evaluations
            if self.n_evaluations
            else 0.0
        )


def plain(samples: np.ndarray) -> Estimate:
    """Standard i.i.d. Monte Carlo mean."""
    arr = np.asarray(samples, dtype=np.float64)
    n = arr.size
    if n == 0:
        raise ValueError("cannot estimate from an empty sample")
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return Estimate(mean, se, float(n), n, "plain-mc", arr)


def antithetic(
    sampler: Callable[[np.ndarray], np.ndarray],
    u: np.ndarray,
) -> Estimate:
    """Antithetic variates: evaluate at U and (1 - U), average within pairs.

    `u` supplies the first half of each pair; the mirrored half is generated
    internally.  The estimator is the mean of the pair means, and its variance
    is the variance of those pair means -- which correctly credits the negative
    correlation between partners.
    """
    u = np.asarray(u, dtype=np.float64)
    n_pairs = u.size
    if n_pairs == 0:
        raise ValueError("cannot estimate from an empty sample")

    first = np.asarray(sampler(u), dtype=np.float64)
    second = np.asarray(sampler(1.0 - u), dtype=np.float64)
    pair_means = 0.5 * (first + second)

    mean = float(pair_means.mean())
    se = float(pair_means.std(ddof=1) / np.sqrt(n_pairs)) if n_pairs > 1 else 0.0

    # ESS relative to a plain estimator of the same variance.
    pooled_var = float(np.var(np.concatenate([first, second]), ddof=1))
    ess = pooled_var / (se**2) if se > 0 and pooled_var > 0 else float(2 * n_pairs)

    return Estimate(mean, se, ess, 2 * n_pairs, "antithetic", pair_means)


def importance_sample(
    indicator: Callable[[np.ndarray], np.ndarray],
    log_target_pdf: Callable[[np.ndarray], np.ndarray],
    log_proposal_pdf: Callable[[np.ndarray], np.ndarray],
    proposal_draws: np.ndarray,
) -> Estimate:
    """Importance sampling for rare-event probabilities.

    Estimates E_target[indicator(X)] using draws from a proposal distribution
    shifted toward the region of interest, reweighted by the likelihood ratio
    w = p_target(x) / p_proposal(x).

    Weights are formed in log space; plain-ratio weighting underflows to zero
    for the P < 10^-12 events this path exists to serve.
    """
    x = np.asarray(proposal_draws, dtype=np.float64)
    n = x.size
    if n == 0:
        raise ValueError("cannot estimate from an empty sample")

    log_w = np.asarray(log_target_pdf(x), dtype=np.float64) - np.asarray(
        log_proposal_pdf(x), dtype=np.float64
    )
    ind = np.asarray(indicator(x), dtype=np.float64)

    # Shift by the max log-weight among contributing draws for stability.
    contributing = ind != 0.0
    if not np.any(contributing):
        return Estimate(
            0.0,
            0.0,
            0.0,
            n,
            "importance-sampling",
            None,
            (
                "no proposal draw landed in the target region: the estimate is "
                "0 by default, not by evidence. Shift the proposal toward the "
                "event before believing this number.",
            ),
        )

    shift = float(np.max(log_w[contributing]))
    weighted = ind * np.exp(log_w - shift)

    mean_shifted = float(weighted.mean())
    value = mean_shifted * np.exp(shift)
    se = float(weighted.std(ddof=1) / np.sqrt(n) * np.exp(shift)) if n > 1 else 0.0

    # Kish effective sample size. Zero-weight draws drop out of both sums, so
    # restricting to contributing draws gives the identical figure.
    w_contrib = weighted[contributing]
    sum_w = float(w_contrib.sum())
    sum_w2 = float((w_contrib**2).sum())
    ess = (sum_w**2) / sum_w2 if sum_w2 > 0 else 0.0

    # A handful of draws carrying all the weight makes both the estimate and
    # its standard error unreliable, and the standard error alone will not
    # reveal it -- a degenerate weight set can look deceptively precise.
    issues: list[str] = []
    efficiency = ess / n if n else 0.0
    if ess < 50:
        issues.append(
            f"effective sample size is only {ess:.1f} of {n:,} draws; the "
            "proposal barely overlaps the target region and this estimate "
            "should not be trusted. Move the proposal closer to the event."
        )
    elif efficiency < 0.01:
        issues.append(
            f"weights are concentrated: effective sample size {ess:,.0f} is "
            f"{efficiency:.2%} of the draws. Consider a proposal closer to the "
            "target, or more draws."
        )

    return Estimate(
        value, se, ess, n, "importance-sampling", None, tuple(issues)
    )


def control_variate(
    samples: np.ndarray, control: np.ndarray, control_mean: float
) -> Estimate:
    """Control variates using the variance-minimising coefficient.

    Useful when a correlated quantity with a known analytic mean is available
    alongside the target -- e.g. pairing a simulated payoff against its
    closed-form linear approximation.

    Note that beta is estimated from the same sample it adjusts, which makes the
    estimator very slightly biased and the reported standard error very slightly
    optimistic; both effects are O(1/n) and negligible at the sample sizes this
    engine runs. Split-sample beta estimation removes them if that matters.
    """
    y = np.asarray(samples, dtype=np.float64)
    c = np.asarray(control, dtype=np.float64)
    if y.shape != c.shape:
        raise ValueError("samples and control must have identical shape")
    n = y.size
    if n < 2:
        raise ValueError("control variates require at least 2 samples")

    cov = float(np.cov(y, c, ddof=1)[0, 1])
    var_c = float(np.var(c, ddof=1))
    beta = cov / var_c if var_c > 0 else 0.0

    adjusted = y - beta * (c - control_mean)
    mean = float(adjusted.mean())
    se = float(adjusted.std(ddof=1) / np.sqrt(n))

    var_y = float(np.var(y, ddof=1))
    ess = (var_y / (se**2)) if se > 0 else float(n)
    return Estimate(mean, se, ess, n, f"control-variate(beta={beta:.4g})", adjusted)
