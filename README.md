# Universal Stochastic Sandbox

A vectorized Monte Carlo engine that separates **how precisely it computed** from
**how much you actually know** — and refuses to let the first masquerade as the
second.

[![tests](https://github.com/thatguywow/universal-stochastic-sandbox/actions/workflows/tests.yml/badge.svg)](https://github.com/thatguywow/universal-stochastic-sandbox/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

```
U ~ Uniform(0,1)  →  F⁻¹(U)  →  vectorized statistics  →  bounded report
```

---

## The problem it exists to solve

Run ten million draws at `p = 0.322`:

```python
sandbox.execute_query("bernoulli", {"probability": 0.322})
# point estimate 0.3221706, 95% interval [0.3219, 0.3224]
```

That interval is real, and it is **about the random number generator**. The
engine was handed `p` and recovered `p`. It would be exactly that narrow if the
0.322 came from a survey of three people.

So every result separates two quantities:

| | shrinks with sample size | shrinks with more data | quote it? |
|---|---|---|---|
| `monte_carlo_error` | **yes**, as 1/√N | no | no |
| `parameter_uncertainty` | no | **yes** | **yes** |

Given the same `p` estimated from 10 observations, the honest interval is
**830× wider**:

```
95% MC interval    : [0.3329, 0.3335]     width 0.00058
95% TOTAL interval : [0.1127, 0.5974]     width 0.485
! parameter uncertainty exceeds Monte Carlo error by 869x;
  increasing sample_size will not narrow this result
```

Over 600 trials with 10 observations, posterior intervals cover the truth 90%+
of the time; simulation-only intervals cover it under 20%.

---

## Install

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"      # .venv\Scripts\pip on Windows
```

The `uss` command installs into the venv. Activate it, or call
`.venv/bin/uss` directly.

`.` alone installs the tool; `.[dev]` adds pytest and ruff. If `pytest` reports
*no module named pytest*, you installed without the extra.

Verified from a clean clone: 301 tests, lint, CLI, examples and the web
interface all pass on a fresh install.

## Use

```bash
uss gui
```

A local interface — no network calls, nothing leaves the machine. Four tabs:
**Start here** (the concepts, with live-computed reference tables), **Simulate**,
**Uncertainty** (the one that produces a defensible number), and
**What to measure**.

```python
from uss import UniversalStochasticSandbox, update_bernoulli

sandbox = UniversalStochasticSandbox(sample_size=1_000_000, seed=42)

# Parameters you trust:
print(sandbox.execute_query("gaussian", {"mean": 20.0, "std_dev": 3.0}).summary())

# Parameters you don't:
print(sandbox.execute_with_priors(
    "bernoulli", {"probability": update_bernoulli(successes=61, trials=152)}
).summary())
```

**[GUIDE.md](GUIDE.md)** — task-oriented walkthrough, start here.
**[USE_CASES.md](USE_CASES.md)** — two worked examples: one answerable, one not.
**[CONTRIBUTING.md](CONTRIBUTING.md)** — how the tests are expected to prove things.
**[CHANGELOG.md](CHANGELOG.md)** — what changed and why.

> The interface binds to `127.0.0.1` and has no authentication. `--host` will
> bind elsewhere and warns loudly when you do; don't expose it on an untrusted
> network.

---

## What's in it

| Module | Provides |
|---|---|
| `core` | query engine, prior propagation, adaptive sample sizing |
| `distributions` | extensible inverse-CDF registry (9 families built in) |
| `estimators` | Wilson, Garwood, Student-t; uncertainty decomposition |
| `inference` | conjugate posteriors, MCMC with R-hat and ESS |
| `variance` | antithetic, importance sampling, control variates |
| `graph` | scenario composition, Gaussian copula |
| `sensitivity` | Sobol indices with bootstrap intervals |
| `calibration` | coverage curves, PIT, interval score |
| `planning` | sample-size and breakeven analysis |
| `des` | Δt = −ln(U)/λ jump sampling, thinning |
| `fitting` | CSV/Parquet → fitted CDFs |
| `behavioral` | logit link, loss aversion, social proof, discounting |

### Beyond simulating

- **Composition** — one node's samples become another's parameter, so a
  multi-stage question propagates uncertainty and correlation in one coupled
  pass instead of multiplying point estimates.
- **Sensitivity** — turns "±0.24" into "70% of that is one input", with
  bootstrap intervals and an explicit verdict on whether the ranking is resolved.
- **Calibration** — makes the engine's own confidence claims falsifiable.
  Validate on a problem where you know the answer, then trust it where you don't.
- **Planning** — how many observations you need before starting, or what value
  would change your decision when data is unobtainable.

---

## Performance

10⁷ draws, single core, consumer hardware:

| Path | Throughput |
|---|---|
| bernoulli | 499M draws/s |
| poisson (λ=5e-6) | 96M draws/s |
| gaussian | 20M draws/s |
| full pipeline incl. statistics | 15–65M draws/s |

Poisson uses a saturating CDF table with `searchsorted` instead of
`scipy.stats.poisson.ppf` — **117× faster at λ=5e-6**, verified bit-identical
across λ ∈ [1e-6, 500]. (The original spec's own code ran at 820k draws/s, 12×
below its stated target.)

---

## Statistical choices, and why

Every one of these was a bug first, found by testing against something external.

| Choice | Because |
|---|---|
| Wilson / Garwood, not normal approximations | at p=0.002, n=500: Wilson covers 91.3%, Wald 63.9% |
| Bernoulli sampler uses `u ≥ 1−p` | `u < p` is *decreasing* in u, so not an inverse CDF; breaks antithetic pairing |
| Antithetic SEs computed across pairs | pooling dependent halves overstates error 3.4× |
| Running variance accumulates about a pivot | `E[X²]−E[X]²` returns 2.0 for unit variance at mean 1e8 |
| Inner MC noise subtracted from parameter spread | `Var(draw) = Var_param + σ²/n_inner`; ignoring it overstated by 1.9× |
| Randomised PIT for discrete forecasts | the standard transform rejected a *correct* Poisson forecast at p=0 |
| Student-t posterior for a mean | treating an estimated σ as known covers 92%, not 95%, at n=8 |
| Bootstrap KS p-values | the textbook test with fitted parameters rejected true models **0.0%** of the time |
| Bootstrap CIs on Sobol indices | indices ranged 0.83–0.98 across seeds; a bare point estimate repeats the mistake this tool exists to prevent |
| Sobol tests model determinism directly | shared noise deflates indices, independent noise inflates them — index sums can't distinguish the two |
| MCMC reports R-hat and ESS | a mistuned chain returned mean 0.19 against a truth of 2.0 while looking healthy |
| Odds-ratio shifts, not `p × multiplier` | at p=0.9, multipliers 1.2 and 5.0 both clip to exactly 1.0 |
| Unknown parameter names are refused | `**kwargs` swallowed them: `gaussian` with `men=20` returned **-0.003**, and an extra posterior was sampled, reported, then discarded |

**301 tests.** Samplers are checked against analytic truth (Poisson
mean/variance, Gumbel's `loc + scale·γ`, Ishigami's Sobol indices); intervals
against measured coverage over repeated trials; diagnostics against deliberately
broken inputs.

---

## Limits

The engine bounds sampling error. It cannot tell you whether you sampled the
right population, and it will not invent evidence you don't have.

`uss domains` prints the reachable confidence per domain:

| Domain | Reachable | Limited by |
|---|---|---|
| Closed physical systems | 98–99.9% | floating point, initial conditions |
| Aggregated macro behaviour | 90–95% | exogenous shift events |
| Complex non-linear networks | 50–80% | chaotic divergence |
| Spontaneous / unobserved events | ~0% | no empirical priors exist |

For that last row the honest output is a bound, not a probability — see
[USE_CASES.md](USE_CASES.md) case 2, where the answer is `rate < 2.4e-11` with a
floor of exactly zero, and no amount of computation improves it.

**Not implemented:** GPU/JAX acceleration (the spec gates it at N > 10⁸; CPU
already clears the 10⁷/s target, and an unverified accelerator path is worse
than none). MCMC is scalar-only — joint posteriors want NUTS or an ensemble
sampler.

## License

MIT — see [LICENSE](LICENSE).
