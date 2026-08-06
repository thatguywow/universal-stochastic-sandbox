# Changelog

## 1.3.0

### Added — comparing two options

Adding more uncertain parameters was never a comparison: they fill different
slots of a single distribution. Contrasting two separate quantities is a
different shape, and it was the most common real question the engine could not
answer directly.

`uss.compare` samples both posteriors and subtracts draw by draw, so the
reported interval carries the uncertainty of both sides. It exists to stop one
specific mistake:

```
P(variant > control) = 96.3%          sounds decisive
95% interval on the difference: [-0.003, +0.063]    includes zero
```

Both are correct. The first says the direction is probably right; the second
says the effect size is not established. `resolved` is True only when the
interval excludes zero, `better` returns None until then, and the disagreement
is raised as a caveat.

`rank` extends this to several options, giving each one's probability of being
the best — pairwise tests multiply and mislead across many arms.

A **Compare** tab in the web interface covers both: two options give a
difference, three or more give a ranking.

## 1.2.2

External review pass. Of eighteen reported issues, three were not defects and
are now pinned by tests so nobody "fixes" working code: the MCMC loop is O(n)
(the tail fill sits outside the inner loop — measured 2.00 / 1.83 / 1.86 µs per
sample as the chain grows), the empirical sampler clamps rather than
extrapolating (`np.interp` does not extrapolate), and `samples_for_proportion`
met its target in the reported case.

### Fixed

- **Server errors left no trace.** HTTP logging is suppressed to keep the
  console readable, and the 500 handler returned the message without printing
  the traceback anywhere — a bug in the interface was undiagnosable. Unexpected
  exceptions now print a traceback to stderr; `ValueError` still returns a clean
  400, since there the message is the whole diagnosis.
- **`confidence_level` was ignored by two endpoints.** `/api/priors` and
  `/api/sensitivity` always used 95% regardless of the request. Verified end to
  end: 80% now yields a width of 0.114 against 0.176 at 95%.
- **No ceiling on replication counts.** `parameter_draws` and `n_base` were
  uncapped, so one request could ask for trillions of inner samples and hang the
  process. Capped at 5,000 and 20,000.
- **Sobol inverted posteriors by nearest-neighbour snapping**, turning a smooth
  posterior into a step function up to 1.7e-3 away from the true inverse. Now
  linear interpolation, which the decomposition's smoothness assumption needs.
- **`gamma` and `weibull_min` were fit candidates mapping to an empty query
  class** — winning the AIC ranking produced a result that could not be sampled.
  Both are now registered query classes with parameter translation.
- **`propagate` let a posterior silently override a fixed value** of the same
  name. Now rejected.
- **Antithetic rounding is disclosed** when an odd `sample_size` is floored.
- Test count in the interface no longer disagrees with the README.

## 1.2.1

### Fixed — unrecognised parameters were silently discarded

Samplers accept `**kwargs` so a shared parameter dict can be passed between
layers. Any name a sampler did not recognise therefore vanished without trace,
and the engine still reported a confident interval around the result:

- `execute_query("gaussian", {"men": 20})` returned **-0.003** instead of 20 —
  the typo fell through to the default of 0.0.
- On the Uncertainty tab, a second uncertain input added to a one-parameter
  distribution was drawn from its posterior, listed in the results table, and
  counted in the "2 posterior(s)" caveat — then dropped by the sampler without
  affecting the answer at all.

`QueryClass.sample` now derives its accepted names from the sampler signature
and rejects anything else, with a near-miss suggestion (`did you mean 'mean'?`).
Custom registered classes are covered automatically. The web interface offers
the parameter name as a dropdown of what the chosen distribution actually reads,
and refuses to add more inputs than it has parameters, pointing at
`ScenarioGraph` for genuinely multi-quantity models.

## 1.2.0

Statistical rigour pass. Each item was a defect that produced plausible numbers,
found by testing against something external to the code.

### Fixed — statistics

- **KS goodness-of-fit p-values were invalid.** Estimating parameters from the
  same data pulls the fitted distribution toward the sample and inflates the
  p-value. Measured on 400 genuinely-normal samples the test rejected at the 5%
  level **0.0%** of the time (mean p-value 0.79 against an expected 0.5), so it
  could never detect a bad fit. Replaced with a parametric bootstrap; results
  now carry `pvalue_is_valid`.
- **Sobol indices were reported without uncertainty.** Across seeds at
  `n_base=2000` a first-order index ranged 0.83–0.98. Added bootstrap
  confidence intervals (no extra model evaluations) and `separates_top_two()`,
  which states whether a ranking is actually resolved.
- **MCMC exposed no convergence diagnostics.** A chain with
  `proposal_scale=0.005` returned a mean of 0.19 against a truth of 2.0 while
  reporting 5,000 clean "samples". Now runs multiple dispersed chains and
  reports split R-hat plus autocorrelation-adjusted effective sample size,
  warning when either fails.
- **Importance sampling was silent when degenerate.** A proposal that misses the
  target returned exactly 0 with no indication. Now warns on low effective
  sample size.
- **Posterior for a mean assumed σ was known.** Understated the interval at
  small n (92% coverage against a nominal 95% at n=8). Now returns the exact
  Student-t marginal under the reference prior.
- **Garwood intervals assumed Poisson counts.** Added an overdispersion check;
  negative-binomial data is flagged rather than given a too-narrow interval.

### Fixed — numerics

- **Catastrophic cancellation in running variance.** `E[X²] − E[X]²` returns
  2.0 for unit variance at mean 1e8. `run_until_precision` and
  `convergence_trace` now accumulate about a pivot.
- **Parameter uncertainty double-counted simulation noise.** Each propagated
  draw carries `σ²/n_inner` on top of parameter variation; the raw spread
  overstated parameter uncertainty by 1.9×. `combine_uncertainty` now accepts
  `inner_mc_variance`.
- **`ScenarioGraph.run()` was not reproducible.** A stored `SeedSequence`
  advanced its spawn counter, so repeated runs on one graph silently returned
  different draws. Now deterministic; use `replicate=1,2,…` for independent runs.
- **Antithetic runs reported pair-mean quantiles** as distribution quantiles,
  and overstated `sample_size` for odd n.

### Fixed — interface

- Tab panels stayed visible because `.grid{display:grid}` outranks the user
  agent's `[hidden]{display:none}`; every tab rendered stacked.
- Evidence forms displayed defaults that were never submitted — a form reading
  `events=50, exposure=10000` sent `events=0, exposure=1`.
- Input is now validated and refused rather than coerced to zero.
- Unknown POST routes responded without draining the request body, aborting the
  connection on Windows.
- Histograms mis-binned continuous data whose maximum happened to be integral.

### Added

- **Info tab** ("Start here") with live-computed reference tables.
- `uss.planning` — sample-size targets, exposure planning, and breakeven
  analysis for when data is unobtainable.
- Security warning when `uss gui` binds beyond loopback.
- `py.typed` marker, ruff configuration, CI across Linux/Windows on Python
  3.10–3.13.

## 1.1.0

- `uss.graph` — scenario composition with per-draw parameter coupling and a
  Gaussian copula for correlated inputs.
- `uss.sensitivity` — Sobol first-order and total-effect indices.
- `uss.calibration` — coverage curves, PIT diagnostics, interval score.
- `run_until_precision` — adaptive sample sizing against a target standard error.

## 1.0.0

Initial implementation of the blueprint: vectorized inverse-transform sampling,
four probability families, DES jump sampling, variance reduction, behavioural
operators, exact interval constructions, empirical fitting, Bayesian prior
updating, CLI and local web interface.
