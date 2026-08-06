"""Variance decomposition: which assumption is actually driving the answer.

The uncertainty layer tells you *how wide* the answer is.  This tells you
*what is making it wide*, which is the actionable half -- it converts "my
estimate is +/- 0.24" into "80% of that spread comes from the baseline
probability, so go measure that and ignore the rest".

Sobol indices decompose output variance across independent inputs:

    S_i   first-order: the share of variance removed by learning X_i exactly.
    S_Ti  total effect: the share remaining if everything *except* X_i is known.
          S_Ti > S_i means X_i matters through interactions with other inputs.

Estimated with the Saltelli two-matrix design using the Jansen estimators,
which are the low-bias choice for total effects at practical sample sizes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .inference import Posterior


@dataclass(frozen=True)
class SensitivityResult:
    """First-order and total-effect indices for each named input."""

    names: tuple[str, ...]
    first_order: np.ndarray
    total_effect: np.ndarray
    output_variance: float
    n_base_samples: int
    n_model_evaluations: int
    warnings: tuple[str, ...] = ()
    first_order_interval: np.ndarray | None = None
    total_effect_interval: np.ndarray | None = None
    confidence_level: float = 0.95

    @property
    def unexplained_variance(self) -> float:
        """Share of output variance no input accounts for.

        For any deterministic f, sum of total effects is >= 1, so this is 0.
        A positive value means variance is going somewhere the inputs do not
        explain -- in practice, model noise that is *shared* across the Saltelli
        matrices (it enters the denominator but cancels in the differences).
        Measured: an additive model with shared noise sd 1.0 reports
        sum(S_T) = 0.45, i.e. 55% unexplained.

        Note that *independent* per-call noise fails the other way, inflating
        total effects above 1 -- which is why `warnings` tests determinism
        directly rather than relying on this number alone.
        """
        return max(0.0, 1.0 - float(self.total_effect.sum()))

    def ranked(self) -> list[tuple[str, float, float]]:
        """Inputs sorted by total effect, descending."""
        rows = [
            (self.names[i], float(self.first_order[i]), float(self.total_effect[i]))
            for i in range(len(self.names))
        ]
        return sorted(rows, key=lambda r: r[2], reverse=True)

    @property
    def interaction_strength(self) -> float:
        """sum(S_Ti) - sum(S_i); ~0 means the model is effectively additive."""
        return float(self.total_effect.sum() - self.first_order.sum())

    def summary(self) -> str:
        lines = [
            f"Sobol indices  ({self.n_model_evaluations:,} model evaluations, "
            f"output variance {self.output_variance:.6g})",
        ]
        if self.total_effect_interval is not None:
            pct = int(self.confidence_level * 100)
            lines.append(
                f"  {'input':<20}{'first-order':>13}{'total':>10}"
                f"{f'{pct}% CI on total':>22}"
            )
            for name, s1, st in self.ranked():
                i = self.names.index(name)
                lo, hi = self.total_effect_interval[i]
                lines.append(
                    f"  {name:<20}{s1:>13.4f}{st:>10.4f}{f'[{lo:.3f}, {hi:.3f}]':>22}"
                )
            lines.append(f"  interaction strength: {self.interaction_strength:.4f}")
            return self._append_notes(lines)

        lines.append(f"  {'input':<24}{'first-order':>13}{'total':>10}")
        for name, s1, st in self.ranked():
            lines.append(f"  {name:<24}{s1:>13.4f}{st:>10.4f}")
        lines.append(f"  interaction strength: {self.interaction_strength:.4f}")
        return self._append_notes(lines)

    def _append_notes(self, lines: list[str]) -> str:
        if self.unexplained_variance > 0.01:
            lines.append(f"  unexplained variance: {self.unexplained_variance:.4f}")
        dominant = self.ranked()[0]
        lines.append(
            f"  -> {dominant[0]} drives {dominant[2]:.0%} of output variance; "
            "measuring it first gives the largest reduction"
        )
        if self.separates_top_two():
            lines.append("     (this ranking is resolved at the stated confidence)")
        elif self.total_effect_interval is not None and len(self.names) > 1:
            lines.append(
                "     ! the top two inputs overlap within their confidence "
                "intervals -- this ranking is not resolved; raise n_base"
            )
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)

    def separates_top_two(self) -> bool:
        """Whether the leading input is distinguishable from the runner-up.

        A ranking whose top two intervals overlap is not evidence for acting on
        one over the other, however different the point estimates look.
        """
        if self.total_effect_interval is None or len(self.names) < 2:
            return False
        order = self.ranked()
        first, second = self.names.index(order[0][0]), self.names.index(order[1][0])
        return bool(
            self.total_effect_interval[first][0] > self.total_effect_interval[second][1]
        )


def sobol_indices(
    model: Callable[[np.ndarray], np.ndarray],
    n_factors: int,
    n_base: int,
    rng: np.random.Generator,
    *,
    names: Sequence[str] | None = None,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
) -> SensitivityResult:
    """Saltelli/Jansen Sobol estimation on the unit hypercube.

    `model` maps an (m, n_factors) array of U(0,1) rows to an (m,) output.
    Express each input's distribution as its inverse CDF of the corresponding
    column, so all factors enter as independent uniforms.

    Costs n_base * (n_factors + 2) model evaluations.

    Indices are themselves Monte Carlo estimates and move around: at n_base
    2,000 a first-order index on an additive model varied between 0.83 and 0.98
    across seeds.  Reporting a bare point estimate would repeat exactly the
    mistake this engine exists to prevent, so bootstrap intervals are computed
    by resampling the evaluated triples -- no extra model calls.  Set
    `n_bootstrap=0` to skip.
    """
    if n_factors < 1:
        raise ValueError(f"n_factors must be positive, got {n_factors}")
    if n_base < 2:
        raise ValueError(f"n_base must be at least 2, got {n_base}")

    labels = tuple(names) if names is not None else tuple(
        f"x{i}" for i in range(n_factors)
    )
    if len(labels) != n_factors:
        raise ValueError(
            f"names has {len(labels)} entries but n_factors is {n_factors}"
        )

    a = rng.random((n_base, n_factors))
    b = rng.random((n_base, n_factors))

    y_a = np.asarray(model(a), dtype=np.float64).ravel()
    y_b = np.asarray(model(b), dtype=np.float64).ravel()
    if y_a.size != n_base or y_b.size != n_base:
        raise ValueError("model must return one output per input row")

    var_y = float(np.var(np.concatenate([y_a, y_b]), ddof=1))
    if var_y <= 0:
        return SensitivityResult(
            labels,
            np.zeros(n_factors),
            np.zeros(n_factors),
            0.0,
            n_base,
            2 * n_base,
        )

    first = np.empty(n_factors)
    total = np.empty(n_factors)
    y_ab_all = np.empty((n_factors, n_base))

    for i in range(n_factors):
        # A_B^i: column i taken from B, everything else from A.
        ab = a.copy()
        ab[:, i] = b[:, i]
        y_ab = np.asarray(model(ab), dtype=np.float64).ravel()
        y_ab_all[i] = y_ab

        # Saltelli et al. (2010) recommend pairing these two specific
        # estimators: Sobol'-Mauntz for first order, Jansen for total effect.
        # The Jansen first-order form biases S_i high enough at practical n to
        # push sum(S_i) above sum(S_Ti), which reads as negative interaction.
        first[i] = float(np.mean(y_b * (y_ab - y_a))) / var_y
        total[i] = 0.5 * float(np.mean((y_a - y_ab) ** 2)) / var_y

    # Bootstrap over the evaluated triples: resampling rows re-estimates every
    # index without calling the model again.
    first_ci: np.ndarray | None = None
    total_ci: np.ndarray | None = None
    if n_bootstrap > 0 and n_base >= 8:
        alpha = 1.0 - confidence_level
        boot_first = np.empty((n_bootstrap, n_factors))
        boot_total = np.empty((n_bootstrap, n_factors))
        for b_ix in range(n_bootstrap):
            idx = rng.integers(0, n_base, n_base)
            ya_r, yb_r = y_a[idx], y_b[idx]
            var_r = float(np.var(np.concatenate([ya_r, yb_r]), ddof=1))
            if var_r <= 0:
                boot_first[b_ix] = first
                boot_total[b_ix] = total
                continue
            for i in range(n_factors):
                yab_r = y_ab_all[i][idx]
                boot_first[b_ix, i] = float(np.mean(yb_r * (yab_r - ya_r))) / var_r
                boot_total[b_ix, i] = 0.5 * float(np.mean((ya_r - yab_r) ** 2)) / var_r
        first_ci = np.clip(
            np.quantile(boot_first, [alpha / 2, 1 - alpha / 2], axis=0).T, 0.0, 1.0
        )
        total_ci = np.clip(
            np.quantile(boot_total, [alpha / 2, 1 - alpha / 2], axis=0).T, 0.0, 1.0
        )

    # Indices are variance shares; clip the estimator noise that lands outside.
    np.clip(first, 0.0, 1.0, out=first)
    np.clip(total, 0.0, 1.0, out=total)

    issues: list[str] = []

    # Sobol decomposition is defined for a deterministic f. Rather than infer
    # non-determinism from the index sums -- which is ambiguous, since shared
    # model noise deflates the indices while independent per-call noise inflates
    # them -- test it directly: re-evaluate a subsample and check agreement.
    probe_rows = min(512, n_base)
    repeat = np.asarray(model(a[:probe_rows]), dtype=np.float64).ravel()
    drift = float(np.mean(np.abs(repeat - y_a[:probe_rows])))
    scale = float(np.sqrt(var_y))
    if scale > 0 and drift > 1e-8 * scale:
        issues.append(
            f"model is not deterministic: re-evaluating identical inputs moved "
            f"the output by {drift / scale:.1%} of its standard deviation. Sobol "
            "indices assume a deterministic f -- seed the model's randomness so "
            "repeated calls agree, or the indices measure that noise too."
        )

    if float(total.sum()) > 1.0 + max(0.05, 2.0 / np.sqrt(n_base)) and float(
        first.sum()
    ) < 0.5:
        issues.append(
            "total effects sum well above 1 while first-order effects are near "
            "zero -- the signature of output noise being attributed to the "
            "inputs. Check model determinism and raise n_base."
        )
    if float(first.sum()) > float(total.sum()) + 0.05:
        issues.append(
            "sum(first-order) exceeds sum(total-effect), which is impossible "
            "for a deterministic model; treat these indices as noise-dominated "
            "and raise n_base."
        )

    return SensitivityResult(
        names=labels,
        first_order=first,
        total_effect=total,
        output_variance=var_y,
        n_base_samples=n_base,
        n_model_evaluations=n_base * (n_factors + 2),
        warnings=tuple(issues),
        first_order_interval=first_ci,
        total_effect_interval=total_ci,
        confidence_level=confidence_level,
    )


def parameter_sensitivity(
    simulate: Callable[[dict[str, float]], float],
    posteriors: dict[str, Posterior],
    n_base: int,
    rng: np.random.Generator,
    *,
    fixed: dict[str, float] | None = None,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
) -> SensitivityResult:
    """Which uncertain *parameter* contributes most to the spread of the answer.

    Each posterior is mapped through its own quantile function, so the factors
    entering the Sobol design are independent uniforms even though the
    parameters themselves are not uniformly distributed.

    This is the data-collection triage tool: the parameter with the largest
    total effect is the one worth gathering evidence on.
    """
    if not posteriors:
        raise ValueError("parameter_sensitivity requires at least one posterior")

    names = tuple(posteriors)
    quantile_grids = {}
    # Build an empirical quantile grid per posterior once, then invert by
    # interpolation -- works uniformly across conjugate and MCMC posteriors.
    for name, post in posteriors.items():
        draws = np.sort(post.sample(max(20_000, n_base), rng))
        quantile_grids[name] = draws

    # Precompute the grid positions once; np.interp gives the linear inverse.
    _positions = {name: np.linspace(0.0, 1.0, g.size) for name, g in quantile_grids.items()}

    def model(unit_rows: np.ndarray) -> np.ndarray:
        out = np.empty(unit_rows.shape[0], dtype=np.float64)
        for row in range(unit_rows.shape[0]):
            params = dict(fixed or {})
            for j, name in enumerate(names):
                # Interpolate between grid points rather than snapping to the
                # nearest. Nearest-neighbour turns a smooth posterior into a step
                # function -- measured up to 1.7e-3 away from the true inverse --
                # and Sobol decomposition assumes the inputs vary smoothly.
                params[name] = float(
                    np.interp(
                        unit_rows[row, j], _positions[name], quantile_grids[name]
                    )
                )
            out[row] = float(simulate(params))
        return out

    return sobol_indices(
        model,
        len(names),
        n_base,
        rng,
        names=names,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


def one_at_a_time(
    simulate: Callable[[dict[str, float]], float],
    baseline: dict[str, float],
    perturbations: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Cheap local sensitivity: swing each input low/high, hold the rest fixed.

    Useful as a first pass, but it cannot see interactions and it explores only
    one point of the input space.  Prefer `parameter_sensitivity` before acting
    on the result.
    """
    base_value = float(simulate(dict(baseline)))
    swings: dict[str, float] = {}
    for name, (low, high) in perturbations.items():
        if name not in baseline:
            raise ValueError(f"{name!r} is not in the baseline parameter set")
        lo = float(simulate({**baseline, name: low}))
        hi = float(simulate({**baseline, name: high}))
        swings[name] = abs(hi - lo)
    swings["__baseline__"] = base_value
    return swings
