"""The Universal Stochastic Sandbox engine.

Pipeline, unchanged from the blueprint's Part II:

    U ~ Uniform(0, 1)  ->  F^-1(U)  ->  vectorised statistics  ->  bounded report

What differs from the Part V script is the accounting at the end.  The engine
reports Monte Carlo precision and parameter uncertainty as separate quantities,
because they answer different questions and only one of them shrinks when you
raise `sample_size`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import distributions, estimators, variance
from .estimators import UncertaintyReport
from .inference import Posterior, propagate

#: Blueprint Part V used the key "lambda", which cannot be a Python identifier.
_PARAMETER_ALIASES = {"lambda": "lam", "rate_lambda": "lam", "sigma": "std_dev", "mu": "mean"}

#: Above this many draws the engine chunks its work to bound peak memory.
_CHUNK_THRESHOLD = 50_000_000


@dataclass
class SimulationResult:
    """Full result of one query, including provenance."""

    query_class: str
    sample_size: int
    parameters: dict[str, Any]
    report: UncertaintyReport
    seed: int | None
    min_observed: float
    max_observed: float
    quantiles: dict[str, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "query_class": self.query_class,
            "sample_size": self.sample_size,
            "parameters": self.parameters,
            "seed": self.seed,
            "mean_point_estimate": self.report.point_estimate,
            "variance": self.report.variance,
            "standard_error": self.report.monte_carlo_error,
            "confidence_level": self.report.confidence_level,
            "confidence_interval": self.report.monte_carlo_interval,
            "min_observed": self.min_observed,
            "max_observed": self.max_observed,
            "quantiles": self.quantiles,
            "elapsed_seconds": self.elapsed_seconds,
        }
        payload.update(self.report.as_dict())
        return payload

    def summary(self) -> str:
        r = self.report
        lines = [
            f"{self.query_class}  n={self.sample_size:,}  ({self.elapsed_seconds:.3f}s)",
            f"  point estimate      : {r.point_estimate:.10g}",
            f"  variance            : {r.variance:.6g}",
            f"  monte carlo error   : {r.monte_carlo_error:.6g}",
            f"  {int(r.confidence_level * 100)}% MC interval    : "
            f"[{r.monte_carlo_interval[0]:.10g}, {r.monte_carlo_interval[1]:.10g}]"
            f"  ({r.interval_type})",
        ]
        if r.parameter_uncertainty is not None:
            lines.append(f"  parameter sd        : {r.parameter_uncertainty:.6g}")
            if r.total_interval:
                lines.append(
                    f"  {int(r.confidence_level * 100)}% TOTAL interval : "
                    f"[{r.total_interval[0]:.10g}, {r.total_interval[1]:.10g}]"
                )
        for caveat in r.caveats:
            lines.append(f"  ! {caveat}")
        return "\n".join(lines)


def _normalise(parameters: dict[str, Any]) -> dict[str, Any]:
    """Map blueprint-era parameter names onto engine identifiers."""
    out: dict[str, Any] = {}
    for key, value in parameters.items():
        out[_PARAMETER_ALIASES.get(key, key)] = value
    if "bias_multiplier" in out:
        raise ValueError(
            "bias_multiplier is not supported: clipping p_base * multiplier at 1.0 "
            "discards the covariate signal. Use uss.behavioral.odds_ratio_shift() "
            "for a multiplicative effect, or logistic_link() for the Part I logit form."
        )
    return out


class UniversalStochasticSandbox:
    """Core production engine for universal statistical sandboxing."""

    def __init__(
        self, sample_size: int = 5_000_000, seed: int | None = 42
    ) -> None:
        if sample_size < 2:
            raise ValueError(f"sample_size must be at least 2, got {sample_size}")
        self.sample_size = int(sample_size)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # -- core query path ---------------------------------------------------
    def execute_query(
        self,
        query_class: str,
        parameters: dict[str, Any] | None = None,
        confidence_level: float = 0.95,
        *,
        sample_size: int | None = None,
        antithetic: bool = False,
        domain: str | None = None,
    ) -> SimulationResult:
        """Map a query onto its probability class and sample it.

        Set `antithetic=True` to evaluate at U and (1 - U) and average within
        pairs; the reported standard error is then computed across pair means,
        which is the only correct construction for dependent draws.
        """
        import time

        params = _normalise(parameters or {})
        qc = distributions.get(query_class)
        n = int(sample_size or self.sample_size)
        t0 = time.perf_counter()

        if antithetic:
            # An odd n cannot be split into pairs; report what actually ran
            # rather than the requested figure, so sample_size never overstates
            # the work done.
            n = (n // 2) * 2
            if n < 2:
                raise ValueError(
                    "antithetic sampling needs a sample_size of at least 2"
                )
            u = self.rng.random(n // 2)
            estimate = variance.antithetic(lambda uu: qc.sample(uu, **params), u)
            pair_means = estimate.samples
            assert pair_means is not None
            report = estimators.summarize(
                pair_means,
                kind="continuous",  # pair means are continuous even for indicators
                confidence_level=confidence_level,
                standard_error=estimate.standard_error,
                effective_sample_size=estimate.effective_sample_size,
                method=estimate.method,
                domain=domain,
            )
            # Quantiles, min and max must describe the sampled *distribution*,
            # not the pair means. Averaging antithetic partners pulls both tails
            # toward the centre, so reporting pair-mean quantiles as p01/p99
            # would understate the spread the caller asked about.
            marginal = np.concatenate(
                [qc.sample(u, **params), qc.sample(1.0 - u, **params)]
            )
            raw_min, raw_max = float(marginal.min()), float(marginal.max())
            quantiles = self._quantiles(marginal)
            report.variance = float(marginal.astype(np.float64, copy=False).var(ddof=1))
        else:
            samples = self._draw(qc, params, n)
            report = estimators.summarize(
                samples,
                kind=qc.kind,
                confidence_level=confidence_level,
                method="plain-mc",
                domain=domain,
            )
            raw_min, raw_max = float(samples.min()), float(samples.max())
            quantiles = self._quantiles(samples)

        elapsed = time.perf_counter() - t0

        return SimulationResult(
            query_class=query_class,
            sample_size=n,
            parameters=params,
            report=report,
            seed=self.seed,
            min_observed=raw_min,
            max_observed=raw_max,
            quantiles=quantiles,
            elapsed_seconds=elapsed,
        )

    def _draw(
        self, qc: distributions.QueryClass, params: dict[str, Any], n: int
    ) -> np.ndarray:
        """Generate U and apply the inverse CDF, chunking very large runs."""
        if n <= _CHUNK_THRESHOLD:
            return qc.sample(self.rng.random(n), **params)

        # Beyond ~50M draws a single float64 vector plus its transform exceeds
        # typical consumer RAM; accumulate in chunks instead.
        chunks = []
        remaining = n
        while remaining > 0:
            take = min(_CHUNK_THRESHOLD, remaining)
            chunks.append(qc.sample(self.rng.random(take), **params))
            remaining -= take
        return np.concatenate(chunks)

    @staticmethod
    def _quantiles(samples: np.ndarray) -> dict[str, float]:
        if samples.size == 0:
            return {}
        qs = np.quantile(samples.astype(np.float64, copy=False), [0.01, 0.25, 0.5, 0.75, 0.99])
        return {
            "p01": float(qs[0]),
            "p25": float(qs[1]),
            "p50": float(qs[2]),
            "p75": float(qs[3]),
            "p99": float(qs[4]),
        }

    # -- uncertainty-aware path -------------------------------------------
    def execute_with_priors(
        self,
        query_class: str,
        posteriors: dict[str, Posterior],
        *,
        fixed: dict[str, Any] | None = None,
        n_parameter_draws: int = 256,
        inner_sample_size: int | None = None,
        confidence_level: float = 0.95,
        domain: str | None = None,
    ) -> SimulationResult:
        """Run the query under parameter uncertainty drawn from posteriors.

        This is the form that produces a defensible real-world interval: the
        reported total interval widens with genuine ignorance about the inputs
        and does not collapse just because `sample_size` was raised.
        """
        inner_n = int(inner_sample_size or max(10_000, self.sample_size // 100))

        base = self.execute_query(
            query_class,
            {**(fixed or {}), **{k: p.mean for k, p in posteriors.items()}},
            confidence_level=confidence_level,
            sample_size=self.sample_size,
            domain=domain,
        )

        def simulate(params: dict[str, Any]) -> float:
            qc = distributions.get(query_class)
            u = self.rng.random(inner_n)
            return float(qc.sample(u, **_normalise(params)).astype(np.float64).mean())

        draws = propagate(
            simulate, posteriors, n_parameter_draws, self.rng, fixed=fixed
        )
        # Each draw is a mean over `inner_n` samples, so it carries inner MC
        # noise of variance sigma^2/inner_n on top of the parameter variation.
        # Pass it through so the reported parameter_uncertainty subtracts it.
        base.report = estimators.combine_uncertainty(
            base.report, draws, inner_mc_variance=base.report.variance / inner_n
        )
        base.report.caveats.insert(
            0,
            f"parameters drawn from {len(posteriors)} posterior(s) over "
            f"{n_parameter_draws} replications",
        )

        # A rare-event query run at a small inner_sample_size produces inner
        # means that are multiples of 1/inner_n, so the "total interval" ends up
        # reporting grid points rather than the posterior. Detect and say so.
        distinct = int(np.unique(draws).size)
        if distinct < min(20, n_parameter_draws // 4):
            base.report.caveats.append(
                f"inner_sample_size={inner_n:,} resolves only {distinct} distinct "
                "outcome level(s); the total interval is quantised by the inner "
                "simulation grid, not by the posterior. Raise inner_sample_size "
                "until the expected event count per replication exceeds ~30."
            )
        return base

    # -- convenience -------------------------------------------------------
    def rare_event_probability(
        self,
        indicator: Callable[[np.ndarray], np.ndarray],
        log_target_pdf: Callable[[np.ndarray], np.ndarray],
        log_proposal_pdf: Callable[[np.ndarray], np.ndarray],
        proposal_sampler: Callable[[int, np.random.Generator], np.ndarray],
        *,
        sample_size: int | None = None,
        confidence_level: float = 0.95,
    ) -> variance.Estimate:
        """Importance-sampled tail probability for events plain MC cannot reach."""
        n = int(sample_size or self.sample_size)
        draws = proposal_sampler(n, self.rng)
        return variance.importance_sample(
            indicator, log_target_pdf, log_proposal_pdf, draws
        )

    def run_until_precision(
        self,
        query_class: str,
        parameters: dict[str, Any] | None = None,
        *,
        target_standard_error: float,
        max_samples: int = 200_000_000,
        batch_size: int = 1_000_000,
        confidence_level: float = 0.95,
        domain: str | None = None,
    ) -> SimulationResult:
        """Sample in batches until the Monte Carlo error meets a target.

        Picking `sample_size` up front is guesswork: too small and the answer is
        noise, too large and most of the compute bought nothing. This grows the
        run until the requested precision is reached, then stops.

        Note this targets *simulation* precision only. If the parameters carry
        real uncertainty, no target here will make the answer more accurate --
        use `execute_with_priors` for that.
        """
        import time

        if target_standard_error <= 0:
            raise ValueError(
                f"target_standard_error must be positive, got {target_standard_error}"
            )
        if batch_size < 2:
            raise ValueError(f"batch_size must be at least 2, got {batch_size}")

        params = _normalise(parameters or {})
        qc = distributions.get(query_class)
        t0 = time.perf_counter()

        collected: list[np.ndarray] = []
        total = 0
        # Running moments about a fixed pivot, so the stopping check never
        # re-scans the history. Accumulating raw sum-of-squares and forming
        # E[X^2] - E[X]^2 instead would cancel catastrophically: at mean 1e8 and
        # unit variance that expression returns 2.0 for a true variance of 1.0.
        # Shifting by a pivot near the mean keeps the summands O(sigma).
        pivot: float | None = None
        shifted_sum = 0.0
        shifted_sq = 0.0

        while total < max_samples:
            take = min(batch_size, max_samples - total)
            batch = qc.sample(self.rng.random(take), **params)
            collected.append(batch)
            b64 = batch.astype(np.float64, copy=False)

            if pivot is None:
                pivot = float(b64.mean())
            centred = b64 - pivot
            shifted_sum += float(centred.sum())
            shifted_sq += float((centred**2).sum())
            total += take

            if total < 2:
                continue
            mean_shift = shifted_sum / total
            var = (shifted_sq - total * mean_shift**2) / (total - 1)
            var = max(var, 0.0)
            if np.sqrt(var / total) <= target_standard_error:
                break

        samples = np.concatenate(collected) if len(collected) > 1 else collected[0]
        report = estimators.summarize(
            samples,
            kind=qc.kind,
            confidence_level=confidence_level,
            method="adaptive-mc",
            domain=domain,
        )
        if report.monte_carlo_error > target_standard_error:
            report.caveats.append(
                f"stopped at max_samples={max_samples:,} with standard error "
                f"{report.monte_carlo_error:.3g}, above the target "
                f"{target_standard_error:.3g}"
            )

        return SimulationResult(
            query_class=query_class,
            sample_size=total,
            parameters=params,
            report=report,
            seed=self.seed,
            min_observed=float(samples.min()),
            max_observed=float(samples.max()),
            quantiles=self._quantiles(samples),
            elapsed_seconds=time.perf_counter() - t0,
        )

    def reset(self, seed: int | None = None) -> None:
        """Restore the generator, so a run can be reproduced exactly."""
        self.seed = self.seed if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
