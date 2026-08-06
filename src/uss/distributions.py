"""Inverse-CDF sampling registry.

Every query class is a pure function mapping a uniform vector U ~ Uniform(0, 1)
onto a target distribution via its percent point function F^-1(U).  Keeping the
mapping pure -- U in, samples out -- is what lets the variance-reduction layer
(`uss.variance`) drive antithetic and importance-sampled variants of the exact
same code path.

Blueprint Part I defines four families; they are registered here.  Additional
families are registered with `register`, which is what makes the engine
non-case-limited without touching `uss.core`.
"""

from __future__ import annotations

import difflib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

# scipy's generic discrete .ppf() is a Python-level bisection search and costs
# ~13 s for 10^7 draws.  We instead build an exact CDF lookup table and use
# np.searchsorted, which is bit-identical and ~120x faster.  The table is
# truncated where the CDF saturates in float64.
_CDF_TABLE_TAIL = 1e-15
_MAX_TABLE_SIZE = 1 << 22  # 4M entries (32 MB); beyond this fall back to scipy


@dataclass(frozen=True)
class ParamSpec:
    """Declares one tunable parameter so interfaces can build inputs for it."""

    name: str
    default: float
    label: str
    minimum: float | None = None
    maximum: float | None = None
    step: float = 0.1

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "default": self.default,
            "label": self.label,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
        }


@dataclass(frozen=True)
class QueryClass:
    """A named inverse-CDF mapping plus metadata used by the estimator layer."""

    name: str
    sampler: Callable[..., np.ndarray]
    kind: str  # "proportion" | "count" | "continuous"
    describe: str
    params: tuple[ParamSpec, ...] = ()

    def accepted_parameters(self) -> frozenset[str]:
        """Names this sampler actually reads, taken from its signature.

        Derived rather than declared, so custom registered classes are covered
        automatically and the list cannot drift from the implementation.
        """
        sig = inspect.signature(self.sampler)
        return frozenset(
            name
            for name, param in sig.parameters.items()
            if name != "u"
            and param.kind
            not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL)
        )

    def sample(self, u: np.ndarray, **parameters: Any) -> np.ndarray:
        # Samplers accept **kwargs so a shared parameter dict can be passed
        # around, which means an unrecognised name would otherwise vanish in
        # silence. That produced confident wrong answers: `gaussian` given
        # `men=20` quietly used the default 0.0, and a second posterior on a
        # one-parameter distribution was sampled, reported in the results table,
        # and then discarded without affecting anything.
        accepted = self.accepted_parameters()
        unknown = sorted(set(parameters) - accepted)
        if unknown:
            hints = []
            for name in unknown:
                close = difflib.get_close_matches(name, accepted, n=1, cutoff=0.6)
                hints.append(
                    repr(name) + (f" (did you mean {close[0]!r}?)" if close else "")
                )
            raise ValueError(
                f"query class {self.name!r} does not accept "
                + ", ".join(hints)
                + f". It takes: {', '.join(sorted(accepted)) or 'no parameters'}."
            )
        return self.sampler(u, **parameters)

    def defaults(self) -> dict[str, float]:
        return {p.name: p.default for p in self.params}


_REGISTRY: dict[str, QueryClass] = {}


def _param(value: Any, name: str) -> Any:
    """Accept a scalar or a per-draw vector.

    Vector parameters are what let `uss.graph` feed one node's samples in as
    another node's parameter, so validation has to hold elementwise rather than
    assuming a single float.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "fiub":
            raise ValueError(f"{name} array must be numeric, got dtype {value.dtype}")
        return value.astype(np.float64, copy=False)
    return float(value)


def _require(condition: Any, message: str) -> None:
    """Raise unless the condition holds for every element."""
    if not bool(np.all(condition)):
        raise ValueError(message)


def register(query_class: QueryClass, *, overwrite: bool = False) -> QueryClass:
    """Add a query class to the registry."""
    if query_class.name in _REGISTRY and not overwrite:
        raise ValueError(
            f"query class {query_class.name!r} already registered; "
            "pass overwrite=True to replace it"
        )
    _REGISTRY[query_class.name] = query_class
    return query_class


def get(name: str) -> QueryClass:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown query class: {name!r}. Available: {sorted(_REGISTRY)}"
        ) from None


def available() -> list[str]:
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------
# Family 1: Bernoulli / Binomial (discrete choice)
# --------------------------------------------------------------------------
def _bernoulli(u: np.ndarray, probability: float = 0.5, **_: Any) -> np.ndarray:
    """True inverse CDF of Bernoulli(p): 0 while u <= 1-p, then 1.

    Blueprint Part V used `u < p`, which yields the same marginal distribution
    but is *decreasing* in u, so it is not F^-1.  The distinction is invisible
    for a single query and matters as soon as draws are coupled: antithetic
    partners, common random numbers across correlated queries, and any
    monotone-coupling argument all require a non-decreasing transform.  Using
    `u >= 1-p` keeps P(X=1) = p exactly while restoring monotonicity.

    The effective probability is expected to be computed upstream -- see
    `uss.behavioral.logistic_link` for the blueprint's Part I logit form.
    """
    p = _param(probability, "probability")
    _require((p >= 0.0) & (p <= 1.0), f"probability must lie in [0, 1], got {p}")
    return (u >= 1.0 - p).astype(np.int8)


def _poisson(u: np.ndarray, lam: float = 1.0, **_: Any) -> np.ndarray:
    """Exact Poisson inverse CDF via a saturating lookup table."""
    if isinstance(lam, np.ndarray):
        # The CDF lookup table is built per scalar lambda. `uss.graph` routes
        # per-draw lambda to scipy's elementwise ppf instead; reaching here
        # directly means that route was bypassed.
        raise ValueError(
            "poisson requires a scalar lambda; for a per-draw lambda use a "
            "ScenarioGraph node, which falls back to scipy's elementwise ppf"
        )
    mu = float(lam)
    if mu < 0:
        raise ValueError(f"lambda must be non-negative, got {mu}")
    if mu == 0:
        return np.zeros_like(u, dtype=np.int64)

    k_max = int(stats.poisson.isf(_CDF_TABLE_TAIL, mu)) + 1
    if k_max + 1 > _MAX_TABLE_SIZE:
        return stats.poisson.ppf(u, mu=mu).astype(np.int64)

    cdf_table = stats.poisson.cdf(np.arange(k_max + 1), mu)
    # side="left" reproduces scipy's ppf convention: F^-1(u) = min{k : F(k) >= u}
    out = np.searchsorted(cdf_table, u, side="left")
    # The table is truncated where the survival function falls below 1e-15, so a
    # u above that point clips to k_max instead of continuing up the tail. The
    # induced bias is bounded by that tail mass: at 1e7 draws the expected number
    # of affected samples is 1e-8. Raise _CDF_TABLE_TAIL if the extreme upper
    # tail of a count distribution is itself the quantity of interest.
    np.clip(out, 0, k_max, out=out)
    return out.astype(np.int64)


def _gaussian(
    u: np.ndarray, mean: float = 0.0, std_dev: float = 1.0, **_: Any
) -> np.ndarray:
    loc = _param(mean, "mean")
    scale = _param(std_dev, "std_dev")
    _require(scale > 0, f"std_dev must be positive, got {std_dev}")
    return stats.norm.ppf(u, loc=loc, scale=scale)


def _extreme_value(
    u: np.ndarray, loc: float = 0.0, scale: float = 1.0, shape: float = 0.0, **_: Any
) -> np.ndarray:
    """Generalised extreme value.

    shape (xi) == 0 recovers the Gumbel case the blueprint names explicitly;
    non-zero xi gives the Frechet (xi > 0) and Weibull (xi < 0) families needed
    for the full EVT form G(x) = exp(-[1 + xi (x-mu)/sigma]^(-1/xi)).
    """
    location = _param(loc, "loc")
    spread = _param(scale, "scale")
    xi = _param(shape, "shape")
    _require(spread > 0, f"scale must be positive, got {scale}")
    if np.all(xi == 0.0):
        return stats.gumbel_r.ppf(u, loc=location, scale=spread)
    # scipy's genextreme uses c = -xi relative to the standard EVT convention.
    return stats.genextreme.ppf(u, c=-xi, loc=location, scale=spread)


def _exponential(u: np.ndarray, rate: float = 1.0, **_: Any) -> np.ndarray:
    """Inter-arrival times, the continuous half of the DES layer."""
    lam = _param(rate, "rate")
    _require(lam > 0, f"rate must be positive, got {rate}")
    return -np.log1p(-u) / lam


def _lognormal(
    u: np.ndarray, mean: float = 0.0, std_dev: float = 1.0, **_: Any
) -> np.ndarray:
    mu = _param(mean, "mean")
    sigma = _param(std_dev, "std_dev")
    _require(sigma > 0, f"std_dev must be positive, got {std_dev}")
    return stats.lognorm.ppf(u, s=sigma, scale=np.exp(mu))


def _gamma(u: np.ndarray, shape: float = 2.0, scale: float = 1.0, **_: Any) -> np.ndarray:
    """Waiting time for several events; the workhorse for positive skewed data."""
    k = _param(shape, "shape")
    theta = _param(scale, "scale")
    _require(k > 0, f"shape must be positive, got {shape}")
    _require(theta > 0, f"scale must be positive, got {scale}")
    return stats.gamma.ppf(u, a=k, scale=theta)


def _weibull(u: np.ndarray, shape: float = 1.5, scale: float = 1.0, **_: Any) -> np.ndarray:
    """Time to failure with a wear-in or wear-out trend."""
    k = _param(shape, "shape")
    lam = _param(scale, "scale")
    _require(k > 0, f"shape must be positive, got {shape}")
    _require(lam > 0, f"scale must be positive, got {scale}")
    return lam * np.power(-np.log1p(-u), 1.0 / k)


def _empirical(u: np.ndarray, quantiles: np.ndarray, **_: Any) -> np.ndarray:
    """Non-parametric sampling from a fitted empirical quantile function."""
    q = np.asarray(quantiles, dtype=np.float64)
    if q.ndim != 1 or q.size < 2:
        raise ValueError("quantiles must be a 1-D array with at least 2 entries")
    grid = np.linspace(0.0, 1.0, q.size)
    return np.interp(u, grid, q)


register(
    QueryClass(
        "bernoulli",
        _bernoulli,
        "proportion",
        "Yes/no outcome with a fixed chance",
        (ParamSpec("probability", 0.3, "Chance of yes", 0.0, 1.0, 0.01),),
    )
)
register(
    QueryClass(
        "poisson",
        _poisson,
        "count",
        "How many events land in a window",
        (ParamSpec("lam", 5.0, "Average events per window", 0.0, None, 0.1),),
    )
)
register(
    QueryClass(
        "gaussian",
        _gaussian,
        "continuous",
        "A measurement that clusters around a typical value",
        (
            ParamSpec("mean", 0.0, "Typical value", None, None, 0.1),
            ParamSpec("std_dev", 1.0, "Spread (std dev)", 1e-12, None, 0.1),
        ),
    )
)
register(
    QueryClass(
        "extreme_value",
        _extreme_value,
        "continuous",
        "Worst case / record maximum (EVT)",
        (
            ParamSpec("loc", 0.0, "Location", None, None, 0.1),
            ParamSpec("scale", 1.0, "Scale", 1e-12, None, 0.1),
            ParamSpec("shape", 0.0, "Tail shape (0 = Gumbel)", None, None, 0.05),
        ),
    )
)
register(
    QueryClass(
        "exponential",
        _exponential,
        "continuous",
        "Waiting time until the next event",
        (ParamSpec("rate", 1.0, "Events per unit time", 1e-12, None, 0.1),),
    )
)
register(
    QueryClass(
        "lognormal",
        _lognormal,
        "continuous",
        "Durations and multiplicative growth (right-skewed)",
        (
            ParamSpec("mean", 0.0, "Log-scale mean", None, None, 0.1),
            ParamSpec("std_dev", 1.0, "Log-scale spread", 1e-12, None, 0.1),
        ),
    )
)
register(
    QueryClass(
        "gamma",
        _gamma,
        "continuous",
        "Waiting time for several events (positive, skewed)",
        (
            ParamSpec("shape", 2.0, "Shape (k)", 1e-12, None, 0.1),
            ParamSpec("scale", 1.0, "Scale", 1e-12, None, 0.1),
        ),
    )
)
register(
    QueryClass(
        "weibull",
        _weibull,
        "continuous",
        "Time to failure with wear-in or wear-out",
        (
            ParamSpec("shape", 1.5, "Shape (k)", 1e-12, None, 0.1),
            ParamSpec("scale", 1.0, "Scale", 1e-12, None, 0.1),
        ),
    )
)
register(
    QueryClass(
        "empirical",
        _empirical,
        "continuous",
        "Resample a real dataset directly",
        (),
    )
)
