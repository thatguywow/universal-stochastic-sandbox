# Contributing

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"     # .venv\Scripts\pip on Windows
pytest -q
```

## The one rule

**Every statistical claim must be checked against something external to the
code** — an analytic result, a measured coverage rate, or a reference
implementation. A test that only asserts the code runs, or that it agrees with
itself, does not count.

Existing tests show the pattern:

| Claim | How it is checked |
|---|---|
| Poisson sampler is exact | bit-identical to `scipy.stats.poisson.ppf` across λ ∈ [1e-6, 500] |
| Gumbel sampler is correct | mean matches `loc + scale·γ` analytically |
| Wilson beats Wald near p=0 | coverage measured over 2,000 replications |
| Antithetic reduces variance | 11.6× reduction measured at equal cost |
| Sobol indices are right | recovers the Ishigami function's analytic indices |
| Bootstrap KS p-value is valid | uniform under the null over 120 replications |
| MCMC diagnostics work | a deliberately mistuned chain must be flagged |

If you add a sampler, estimator, or interval, add the corresponding check.

## Adding a query class

Any non-decreasing `F⁻¹(U)` works:

```python
from uss import register, QueryClass, ParamSpec
import numpy as np

def weibull(u, shape=1.5, scale=1.0, **_):
    return scale * (-np.log1p(-u)) ** (1.0 / shape)

register(QueryClass(
    "weibull", weibull, "continuous", "Failure times",
    (ParamSpec("shape", 1.5, "Shape", 1e-12, None, 0.1),
     ParamSpec("scale", 1.0, "Scale", 1e-12, None, 0.1)),
))
```

- `kind` selects the interval construction: `proportion` → Wilson,
  `count` → Garwood, `continuous` → Student-t. Getting this wrong produces
  intervals that are silently invalid, so pick deliberately.
- Accept **vector** parameters (validate elementwise, don't call `float()`) or
  the class cannot be used inside a `ScenarioGraph`.
- Declaring `ParamSpec`s makes the class appear in the web interface with
  proper inputs — no UI code needed.
- Add a monotonicity test; `test_inverse_transform_is_monotone` covers every
  registered class automatically.

## Things that are deliberate

Please don't "fix" these without discussion:

- **`bias_multiplier` is rejected.** `clip(p * m, 0, 1)` destroys covariate
  signal. Use `odds_ratio_shift` or `logistic_link`.
- **Bernoulli uses `u >= 1-p`, not `u < p`.** The latter is decreasing in `u`
  and therefore not an inverse CDF; it breaks antithetic pairing and common
  random numbers.
- **Antithetic standard errors are computed across pairs.** Pooling the halves
  and calling `stats.sem` overstates error ~3.4×.
- **Running variances accumulate about a pivot.** `E[X²] − E[X]²` returns 2.0
  for unit variance at mean 1e8.
- **Sobol requires a deterministic model.** The estimator tests this directly
  and warns; don't remove the check.

## Style

- Explain *why* in comments, not *what*. Prefer noting the failure a line
  prevents over restating the line.
- Warnings are part of the output contract. If a result could mislead, say so
  in `caveats` or `warnings` rather than assuming the reader will infer it.
- Run `pytest -q` before opening a PR; CI runs Linux and Windows on Python
  3.10–3.13.
