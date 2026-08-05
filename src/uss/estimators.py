"""Confidence bounds and the honest accounting of what they mean.

Blueprint Part IV mandates SEM, variance and exact 95% CIs on every run.  This
module provides them, and draws a distinction the Part V script did not:

    monte_carlo_error   how precisely the engine evaluated its own integral.
                        Shrinks as 1/sqrt(N).  Says nothing about reality.

    parameter_uncertainty  how uncertain the *inputs* are.  Does not shrink
                        with N.  This is what actually bounds a real-world claim.

A run of 10^7 draws at p = 0.322 reports a +/-0.0003 interval, but that interval
describes the RNG, not the world: the engine recovered the number it was handed.
Reporting the two components separately is what keeps the output from
overstating its own authority.  `total_interval` combines them when a prior is
supplied; without a prior, `parameter_uncertainty` is None and the result is
explicitly labelled as simulation-precision only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

#: Blueprint Part IV, encoded so the engine can attach a realistic ceiling to
#: any result rather than implying uniform precision across domains.
CONFIDENCE_DOMAINS: dict[str, dict[str, Any]] = {
    "closed_physical": {
        "label": "Closed Physical Systems (orbital, fluid)",
        "achievable_confidence": (0.98, 0.999),
        "governing": "Deterministic physics + Gaussian noise",
        "bottleneck": "Floating-point & initial-condition noise",
    },
    "macro_behavioral": {
        "label": "Aggregated Macro Behavior (demographics)",
        "achievable_confidence": (0.90, 0.95),
        "governing": "Multinomial logit + behavioral multipliers",
        "bottleneck": "Exogenous shift events (black-swan shifts)",
    },
    "complex_network": {
        "label": "Complex Non-Linear Networks (markets)",
        "achievable_confidence": (0.50, 0.80),
        "governing": "Stochastic differential equations",
        "bottleneck": "Chaotic divergence (e^(lambda t) error scaling)",
    },
    "quantum_anomaly": {
        "label": "Quantum Anomalies / Spontaneous Creation",
        "achievable_confidence": (0.0, 0.0),
        "governing": "Poisson point process / QFT",
        "bottleneck": "Heisenberg uncertainty & absence of empirical priors",
    },
}


@dataclass
class UncertaintyReport:
    """Point estimate plus a decomposed account of what is uncertain."""

    point_estimate: float
    variance: float
    monte_carlo_error: float
    monte_carlo_interval: tuple[float, float]
    confidence_level: float
    n_evaluations: int
    effective_sample_size: float
    method: str
    interval_type: str
    parameter_uncertainty: float | None = None
    total_interval: tuple[float, float] | None = None
    domain: str | None = None
    caveats: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "point_estimate": self.point_estimate,
            "variance": self.variance,
            "monte_carlo_error": self.monte_carlo_error,
            "monte_carlo_interval": self.monte_carlo_interval,
            "confidence_level": self.confidence_level,
            "n_evaluations": self.n_evaluations,
            "effective_sample_size": self.effective_sample_size,
            "method": self.method,
            "interval_type": self.interval_type,
            "parameter_uncertainty": self.parameter_uncertainty,
            "total_interval": self.total_interval,
            "caveats": list(self.caveats),
        }
        if self.domain:
            payload["domain"] = self.domain
            payload["domain_ceiling"] = CONFIDENCE_DOMAINS[self.domain]
        return payload


def _critical_value(confidence_level: float, df: float) -> float:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            f"confidence_level must lie in (0, 1), got {confidence_level}"
        )
    return float(stats.t.ppf((1.0 + confidence_level) / 2.0, df=df))


def wilson_interval(
    successes: int, n: int, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the Wald/normal interval because it stays inside [0, 1] and
    retains coverage when p is near 0 or 1 -- exactly the regime the engine hits
    on rare-behaviour queries.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes must lie in [0, {n}], got {successes}")

    z = float(stats.norm.ppf((1.0 + confidence_level) / 2.0))
    p_hat = successes / n
    denom = 1.0 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom

    lower = max(0.0, centre - half)
    upper = min(1.0, centre + half)
    # At the boundaries centre and half cancel exactly in real arithmetic, but
    # the sqrt leaves ~1e-19 of residue. Snap so that "zero events observed"
    # reports a lower bound of exactly zero.
    if successes == 0:
        lower = 0.0
    if successes == n:
        upper = 1.0
    return (lower, upper)


def garwood_interval(
    count: int, exposure: float = 1.0, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Exact (Garwood) Poisson interval for a rate.

    The normal-approximation interval is invalid when only a handful of events
    land -- the balloon-materialisation query produces ~50 events in 10^7 draws.
    This chi-squared construction is exact for any count including zero.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    if exposure <= 0:
        raise ValueError(f"exposure must be positive, got {exposure}")

    alpha = 1.0 - confidence_level
    lower = stats.chi2.ppf(alpha / 2.0, 2 * count) / 2.0 if count > 0 else 0.0
    upper = stats.chi2.ppf(1.0 - alpha / 2.0, 2 * (count + 1)) / 2.0
    return (float(lower) / exposure, float(upper) / exposure)


def summarize(
    samples: np.ndarray,
    *,
    kind: str = "continuous",
    confidence_level: float = 0.95,
    standard_error: float | None = None,
    effective_sample_size: float | None = None,
    method: str = "plain-mc",
    domain: str | None = None,
) -> UncertaintyReport:
    """Build an uncertainty report from a sample vector.

    `standard_error` may be supplied by a variance-reduction estimator, whose
    dependent draws make a naive recomputation wrong; when omitted the i.i.d.
    SEM is used.
    """
    arr = np.asarray(samples)
    n = int(arr.size)
    if n == 0:
        raise ValueError("cannot summarise an empty sample")
    if domain is not None and domain not in CONFIDENCE_DOMAINS:
        raise ValueError(
            f"unknown domain {domain!r}; expected one of {sorted(CONFIDENCE_DOMAINS)}"
        )

    # Accumulate in float64 regardless of the sample dtype: int8 Bernoulli
    # indicators would otherwise overflow their own sum well before 10^7.
    arr64 = arr.astype(np.float64, copy=False)
    point = float(arr64.mean())
    variance = float(arr64.var(ddof=1)) if n > 1 else 0.0

    if standard_error is None:
        standard_error = float(np.sqrt(variance / n)) if n > 1 else 0.0
    ess = float(effective_sample_size) if effective_sample_size is not None else float(n)

    caveats: list[str] = []

    # Choose the interval construction that is actually valid for this quantity.
    if kind == "proportion":
        successes = int(np.count_nonzero(arr))
        interval = wilson_interval(successes, n, confidence_level)
        interval_type = "wilson-score"
    elif kind == "count":
        total = float(arr64.sum())
        # Garwood is exact for a Poisson sum. If the counts are overdispersed
        # (variance materially above the mean) that assumption fails and the
        # interval is too narrow, so check the dispersion index before relying
        # on it. Under Poisson, (n-1)*s^2/mean ~ chi^2_{n-1}.
        if n > 1 and point > 0:
            dispersion = variance / point
            chi2_stat = (n - 1) * dispersion
            over_p = float(stats.chi2.sf(chi2_stat, n - 1))
            if over_p < 0.001 and dispersion > 1.5:
                caveats.append(
                    f"counts are overdispersed (variance/mean = {dispersion:.2f}, "
                    "expected 1.0 under Poisson); the exact Poisson interval "
                    "below is too narrow for this data"
                )
        if total < 1000:
            lo, hi = garwood_interval(int(round(total)), float(n), confidence_level)
            interval = (lo, hi)
            interval_type = "garwood-exact"
            if total < 30:
                caveats.append(
                    f"only {int(total)} events observed; the rate estimate is "
                    "dominated by counting noise -- consider importance sampling"
                )
        else:
            t_crit = _critical_value(confidence_level, n - 1)
            half = standard_error * t_crit
            interval = (point - half, point + half)
            interval_type = "student-t"
    else:
        t_crit = _critical_value(confidence_level, max(n - 1, 1))
        half = standard_error * t_crit
        interval = (point - half, point + half)
        interval_type = "student-t"

    caveats.append(
        "monte_carlo_interval bounds simulation precision only; it does not "
        "bound the accuracy of the input parameters"
    )

    return UncertaintyReport(
        point_estimate=point,
        variance=variance,
        monte_carlo_error=float(standard_error),
        monte_carlo_interval=(float(interval[0]), float(interval[1])),
        confidence_level=confidence_level,
        n_evaluations=n,
        effective_sample_size=ess,
        method=method,
        interval_type=interval_type,
        domain=domain,
        caveats=caveats,
    )


def combine_uncertainty(
    report: UncertaintyReport,
    parameter_draws: np.ndarray,
    *,
    inner_mc_variance: float | None = None,
) -> UncertaintyReport:
    """Fold parameter uncertainty into an existing report.

    `parameter_draws` are point estimates produced under different sampled
    parameter values (see `uss.inference.propagate`).

    Each draw is itself the mean of a finite inner simulation, so its spread is
    a sum of two components:

        Var(draw) = Var_parameter + sigma_inner^2 / n_inner

    Reporting the raw spread as parameter uncertainty therefore double-counts
    the simulation noise -- measured at 1.9x overstatement when the inner run is
    small. Supplying `inner_mc_variance` (= sigma_inner^2 / n_inner) subtracts
    that component so the reported figure is the parameter contribution alone.

    The returned `total_interval` is a quantile interval over the draws, which
    legitimately contains both components; only the reported
    `parameter_uncertainty` scalar is variance-corrected.
    """
    draws = np.asarray(parameter_draws, dtype=np.float64)
    if draws.size < 2:
        raise ValueError("parameter uncertainty requires at least 2 draws")

    total_var = float(draws.var(ddof=1))
    if inner_mc_variance is not None:
        if inner_mc_variance < 0:
            raise ValueError("inner_mc_variance must be non-negative")
        param_var = total_var - float(inner_mc_variance)
        if param_var < 0:
            # Inner noise explains the whole spread; parameters add nothing
            # detectable at this resolution.
            param_var = 0.0
            report.caveats.append(
                "observed spread is within inner simulation noise; parameter "
                "uncertainty is not resolvable at this inner_sample_size"
            )
    else:
        param_var = total_var

    param_sd = float(np.sqrt(param_var))
    total_se = float(np.sqrt(report.monte_carlo_error**2 + param_sd**2))

    alpha = 1.0 - report.confidence_level
    lo = float(np.quantile(draws, alpha / 2.0))
    hi = float(np.quantile(draws, 1.0 - alpha / 2.0))

    report.parameter_uncertainty = param_sd
    report.total_interval = (lo, hi)

    if param_sd > 10 * report.monte_carlo_error and report.monte_carlo_error > 0:
        report.caveats.append(
            f"parameter uncertainty ({param_sd:.3g}) exceeds Monte Carlo error "
            f"({report.monte_carlo_error:.3g}) by {param_sd / report.monte_carlo_error:.0f}x; "
            "increasing sample_size will not narrow this result"
        )
    report.caveats.append(f"total standard error including priors: {total_se:.6g}")
    return report


def convergence_trace(
    samples: np.ndarray, checkpoints: int = 20
) -> list[tuple[int, float, float]]:
    """Running (n, estimate, sem) at logarithmically spaced checkpoints.

    Lets a caller verify the 1/sqrt(N) decay actually held rather than assuming
    the final SEM is trustworthy.
    """
    arr = np.asarray(samples, dtype=np.float64)
    n = arr.size
    if n < 2:
        raise ValueError("convergence trace requires at least 2 samples")

    marks = np.unique(np.geomspace(max(2, n // 10**4), n, checkpoints).astype(int))

    # Centre before accumulating. Cumulative raw second moments combined as
    # E[X^2] - E[X]^2 lose all significance when the mean dominates the spread
    # (at mean 1e8, unit variance, that form returns 2.0 instead of 1.0).
    pivot = float(arr.mean())
    centred = arr - pivot
    csum = np.cumsum(centred)
    csum2 = np.cumsum(centred**2)

    trace: list[tuple[int, float, float]] = []
    for m in marks:
        mean_shift = csum[m - 1] / m
        var_m = max((csum2[m - 1] - m * mean_shift**2) / max(m - 1, 1), 0.0)
        trace.append((int(m), float(pivot + mean_shift), float(np.sqrt(var_m / m))))
    return trace
