# Changelog

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
