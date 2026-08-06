"""Universal Stochastic Sandbox Engine.

A non-case-limited, statistically bounded simulator built on vectorized
inverse-transform sampling.

    >>> from uss import UniversalStochasticSandbox
    >>> sandbox = UniversalStochasticSandbox(sample_size=1_000_000)
    >>> result = sandbox.execute_query("gaussian", {"mean": 20.0, "std_dev": 3.0})
    >>> print(result.summary())
"""

from .calibration import (
    CoverageResult,
    PITResult,
    coverage_curve,
    interval_score,
    pit_report,
    pit_values,
    sharpness,
)
from .compare import ComparisonResult, compare, rank
from .core import SimulationResult, UniversalStochasticSandbox
from .distributions import ParamSpec, QueryClass, available, get, register
from .estimators import (
    CONFIDENCE_DOMAINS,
    UncertaintyReport,
    combine_uncertainty,
    garwood_interval,
    summarize,
    wilson_interval,
)
from .graph import GraphResult, Node, ScenarioGraph, gaussian_copula
from .inference import (
    Posterior,
    metropolis_hastings,
    propagate,
    update_bernoulli,
    update_gaussian_mean,
    update_poisson,
)
from .planning import (
    Breakeven,
    SamplePlan,
    breakeven,
    exposure_for_rate,
    proportion_half_width,
    proportion_tradeoff,
    rate_upper_bound,
    samples_for_proportion,
)
from .sensitivity import (
    SensitivityResult,
    one_at_a_time,
    parameter_sensitivity,
    sobol_indices,
)

try:
    # Single source of truth: the version declared in pyproject.toml. Hard-coding
    # it here as well guarantees the two drift apart, which they already had.
    from importlib.metadata import version as _version

    __version__ = _version("universal-stochastic-sandbox")
except Exception:
    __version__ = "0.0.0+unknown"

__all__ = [
    "CONFIDENCE_DOMAINS",
    "Breakeven",
    "ComparisonResult",
    "CoverageResult",
    "GraphResult",
    "Node",
    "PITResult",
    "ParamSpec",
    "Posterior",
    "QueryClass",
    "SamplePlan",
    "ScenarioGraph",
    "SensitivityResult",
    "SimulationResult",
    "UncertaintyReport",
    "UniversalStochasticSandbox",
    "available",
    "breakeven",
    "combine_uncertainty",
    "compare",
    "coverage_curve",
    "exposure_for_rate",
    "garwood_interval",
    "gaussian_copula",
    "get",
    "interval_score",
    "metropolis_hastings",
    "one_at_a_time",
    "parameter_sensitivity",
    "pit_report",
    "pit_values",
    "propagate",
    "proportion_half_width",
    "proportion_tradeoff",
    "rank",
    "rate_upper_bound",
    "register",
    "samples_for_proportion",
    "sharpness",
    "sobol_indices",
    "summarize",
    "update_bernoulli",
    "update_gaussian_mean",
    "update_poisson",
    "wilson_interval",
]
