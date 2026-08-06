# Using the Universal Stochastic Sandbox

A practical guide. For what was built and why, see [README.md](README.md).

---

## 0. Setup

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
```

The `[dev]` extra adds pytest and ruff. Installing plain `-e .` gives you a
fully working tool but no test runner — if `pytest` reports *no module named
pytest*, that is why.

Verify:

```bash
.venv\Scripts\python -m pytest -q
```

Expect `289 passed`.

### Running commands

The `uss` command installs **inside the venv**, not on your system PATH. A bare
`uss gui` gives `The term 'uss' is not recognized`. Either activate the venv
first:

```bash
.venv\Scripts\Activate.ps1
```

after which `uss gui`, `uss query ...` and `python` all resolve to the venv for
that terminal session — this guide assumes you have done this. Or skip
activation and call it directly every time:

```bash
.venv\Scripts\uss.exe gui
```

Both are equivalent. If PowerShell blocks the activation script with a script
execution error, use the direct form, or allow local scripts for your user
account with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

---

## 0.5 The interface

```bash
uss gui
```

Opens `http://127.0.0.1:8765` in your browser. Local only — no network calls,
no accounts, nothing leaves the machine. Stop it with Ctrl+C.

| Tab | Answers |
|---|---|
| **Start here** | the concepts, with reference tables computed live by the engine |
| **Simulate** | "What does this distribution look like, given parameters I trust?" |
| **Uncertainty** | "…and how wrong could my parameters be?" — the tab that produces a defensible number |
| **What to measure** | "Which unknown should I go collect data on first?" |

On the Uncertainty tab, the parameter name is a dropdown of what the chosen
distribution actually reads. One uncertain input per parameter — a Bernoulli
takes only `probability`, so there is nothing to add a second input *for*. To
model several uncertain quantities feeding one outcome, use a `ScenarioGraph`
(§5); that is the construct built for it.

The Uncertainty tab deliberately labels the two intervals **ignore this one**
and **quote this one**. That is the single most common misreading of this
engine's output, so the interface makes it hard to get wrong.

Forms are generated from the query-class registry, so a class you register
yourself (§13) appears with proper inputs and no UI code.

Flags: `--port`, `--host`, `--no-browser`.

---

## 1. Which tool for which question

This is the most important table in the guide. Picking the wrong entry point is
the main way to get a confident wrong answer.

| Your question | Use | Section |
|---|---|---|
| "I'd rather click than write code" | `uss gui` | [0.5](#05-the-interface) |
| "Where do I get the numbers to put in?" | count, query, cite, or bound | [3.5](#35-where-do-how-many-yes-and-out-of-how-many-come-from) |
| "How many do I need to count?" | `samples_for_proportion` | [3.5](#how-many-do-i-need-to-count) |
| "I have no data and need a decision" | `breakeven` | [3.5](#running-it-backwards-what-would-change-your-mind) |
| "What's the distribution of X, given parameters I trust?" | `execute_query` | [2](#2-a-single-query) |
| "…and how wrong could my parameters be?" | `execute_with_priors` | [4](#4-when-you-dont-trust-your-inputs) |
| "X depends on Y which depends on Z" | `ScenarioGraph` | [5](#5-multi-stage-scenarios) |
| "Which assumption should I go measure?" | `parameter_sensitivity` | [6](#6-where-to-spend-your-next-hour) |
| "Should I believe these intervals at all?" | `coverage_curve` / `pit_report` | [7](#7-checking-the-engine-is-honest) |
| "How often does this ~never happen?" | `rare_event_probability` | [8](#8-events-too-rare-to-simulate) |
| "When does the next event occur?" | `uss.des.jump_sample` | [8](#8-events-too-rare-to-simulate) |
| "I have a CSV of real observations" | `uss.fitting` | [9](#9-starting-from-real-data) |
| "How many samples do I need?" | `run_until_precision` | [10](#10-choosing-sample-size) |

---

## 2. A single query

```python
from uss import UniversalStochasticSandbox

sandbox = UniversalStochasticSandbox(sample_size=1_000_000, seed=42)
result = sandbox.execute_query("gaussian", {"mean": 20.0, "std_dev": 3.0})
print(result.summary())
```

```
gaussian  n=1,000,000  (0.063s)
  point estimate      : 19.99472592
  variance            : 8.99328
  monte carlo error   : 0.00424106
  95% MC interval     : [19.98641358, 20.00303825]  (student-t)
  ! monte_carlo_interval bounds simulation precision only; it does not
    bound the accuracy of the input parameters
```

**Available query classes** (`uss classes`):

| Class | Kind | Parameters | Use for |
|---|---|---|---|
| `bernoulli` | proportion | `probability` | binary choice, yes/no outcomes |
| `poisson` | count | `lambda` (or `lam`) | events in a region of space/time |
| `gaussian` | continuous | `mean`, `std_dev` | measurements, aggregates |
| `extreme_value` | continuous | `loc`, `scale`, `shape` | tail risk, records, maxima |
| `exponential` | continuous | `rate` | waiting times |
| `lognormal` | continuous | `mean`, `std_dev` | multiplicative growth, durations |
| `empirical` | continuous | `quantiles` | a fitted real dataset |

`shape` on `extreme_value` is the EVT ξ: `0` is Gumbel, `>0` Fréchet, `<0` Weibull.

Aliases are accepted so blueprint-era code keeps working: `lambda`→`lam`,
`mu`→`mean`, `sigma`→`std_dev`.

### Reading the output

| Field | Means | Shrinks with `sample_size`? |
|---|---|---|
| `point_estimate` | the answer | — |
| `variance` | spread of the sampled quantity itself | no |
| `monte_carlo_error` | precision of the integration | **yes**, as 1/√N |
| `monte_carlo_interval` | where the *simulation* has pinned the mean | yes |
| `parameter_uncertainty` | spread from uncertain inputs | **no** |
| `total_interval` | the interval to actually quote | no |
| `interval_type` | which construction was used | — |
| `caveats` | things that would otherwise mislead you | — |

**Always read `caveats`.** They are where the engine tells you the answer is
weaker than the numbers suggest.

---

## 3. The one thing to internalise

```python
sandbox.execute_query("bernoulli", {"probability": 0.322})
# point estimate: 0.3221706, 95% interval [0.3219, 0.3224]
```

That interval is real, and it is **about the random number generator**. You
handed the engine `0.322` and it gave back `0.322`. Ten million draws bought
precision about your own assumption, not knowledge about the world.

Simulation precision ≠ empirical accuracy. If your parameters came from a
guess, the answer is a guess with a very tight error bar on it. Section 4 is
how you fix that.

---

## 3.5 Where do "how many yes" and "out of how many" come from?

The engine cannot supply these. They are the only real information in the whole
pipeline, and you have to get them from one of four places.

### A. Count them

The default. Define the population **first**, then count both numbers.

| | |
|---|---|
| **out of how many** | every case you looked at |
| **how many yes** | the subset with the property |

The trap is not arithmetic, it is the sampling frame. "61 of 152 people on
Tuesday afternoon on this street" is a real measurement of *that* population. It
becomes wrong the moment you call it "the city". Write down what you actually
sampled and quote the answer for that.

### B. Pull them from data you already have

Usually one query. Both numbers come out together:

```sql
SELECT count(*) FILTER (WHERE churned) AS yes, count(*) AS total FROM users;
```

Same for a spreadsheet: `COUNTIF(range,"yes")` and `COUNTA(range)`.

### C. Convert a published statistic

A study reporting "23% of respondents, n = 500" gives `yes = 115, total = 500`.
If the source does **not** report its sample size, you cannot use it as
evidence — it is an assumption, and Section 4's warning applies.

### D. You have nothing

Do not invent counts. Two honest moves instead:

- The event has never been observed → report a bound, not a probability.
  See [USE_CASES.md](USE_CASES.md) case 2 and `rate_upper_bound()`.
- You need a decision, not a number → run it backwards (below).

### How many do I need to count?

```python
from uss import proportion_tradeoff, samples_for_proportion

samples_for_proportion(0.05, assumed_rate=0.3).summary()
# 'count 320 to reach +/-5.0% at 95% confidence (assuming a rate near 30%)'
```

Assuming the answer is near 30%:

| count this many | answer good to | |
|---|---|---|
| 10 | ±24.8% | basically useless |
| 30 | ±15.6% | |
| 100 | ±8.8% | usual sweet spot |
| 400 | ±4.5% | |
| 2,500 | ±1.8% | rarely worth it |

**Precision costs quadratically** — halving the interval costs 4× the counting.
`assumed_rate` is only a planning guess; being wrong makes the realised interval
somewhat narrower or wider, nothing worse. Use 0.5 if you have no idea (worst
case).

For rare events the governing quantity is the **event count**, not the sample
size — you need ~384 events for ±10% however rare they are:

```python
exposure_for_rate(0.10, assumed_rate=1e-6)   # 384,145,882 units of exposure
```

### Running it backwards: what would change your mind?

Often the best move when data is expensive. Instead of "what is p?", ask "what
would p have to be for this to matter?" — which needs **no observations at all**:

```python
from uss import breakeven

# A promo costs 100 to run and returns 5000 x conversion rate.
res = breakeven(lambda p: p * 5000 - 100, threshold=0.0, low=0.0, high=1.0)
res.summary("conversion rate")
# 'decision flips when conversion rate = 0.02. ...'
```

If 2% is obviously unreachable, you are finished without measuring anything. If
it is plausible, you now know exactly what to measure and how precisely. When
the crossing lies outside the plausible range entirely, `decisive` is `False`
and the answer is *stop measuring this input*.

---

## 4. When you don't trust your inputs

Represent what you actually know as a posterior, then propagate it.

```python
from uss import UniversalStochasticSandbox, update_bernoulli

sandbox = UniversalStochasticSandbox(sample_size=1_000_000, seed=42)

# You surveyed 10 people; 3 wore tank tops.
prior = update_bernoulli(successes=3, trials=10)

result = sandbox.execute_with_priors(
    "bernoulli",
    {"probability": prior},
    n_parameter_draws=512,
    inner_sample_size=100_000,
)
print(result.summary())
```

```
  95% MC interval    : [0.3329, 0.3335]     width 0.00058
  95% TOTAL interval : [0.1127, 0.5974]     width 0.485    (830x wider)
  ! parameter uncertainty exceeds Monte Carlo error by 869x;
    increasing sample_size will not narrow this result
```

**Quote the TOTAL interval.** The MC interval answers a question nobody asked.

### Building posteriors

| Function | For | Conjugate form |
|---|---|---|
| `update_bernoulli(successes, trials)` | a probability | Beta-Binomial |
| `update_poisson(event_count, exposure)` | a rate λ | Gamma-Poisson |
| `update_gaussian_mean(observations)` | a mean | Normal-Normal |
| `metropolis_hastings(log_post, x0, n, rng)` | anything else | MCMC |

Defaults are deliberately uninformative — `update_bernoulli(0, 0)` gives a flat
Beta(1,1) over [0,1], the honest starting point for a probability nobody has
measured.

> **Sizing warning.** For rare-event queries, `inner_sample_size` must be large
> enough that each replication sees ~30+ events, or the total interval reports
> the simulation's discretisation grid rather than your posterior. The engine
> detects this and adds a `quantised by the inner simulation grid` caveat — if
> you see it, raise `inner_sample_size`.

---

## 5. Multi-stage scenarios

Real questions are chains. Temperature drives comfort, which drives a
probability, which drives a count. A `ScenarioGraph` couples them so each draw
flows through every stage together.

```python
import numpy as np
from scipy.special import expit
from uss import ScenarioGraph

g = ScenarioGraph(seed=42)

temperature = g.add("temperature", "gaussian", mean=31.0, std_dev=4.0)
humidity    = g.add("humidity",    "gaussian", mean=55.0, std_dev=12.0)

discomfort  = g.derive("discomfort", lambda t, h: t + 0.05 * h, temperature, humidity)
p_tanktop   = g.derive("p", lambda d: expit(-8.0 + 0.22 * d), discomfort)

g.add("wears_tanktop", "bernoulli", probability=p_tanktop)   # parameter is a node

res = g.run(500_000)
print(res.summary())
print(res.correlation("temperature", "wears_tanktop"))   # +0.366
```

- `add(...)` creates a stochastic node; any parameter may be another node.
- `derive(...)` creates a deterministic transform of parent draws.
- Cycles are detected and rejected.
- Each node gets an independent seed stream, so **appending a node never
  changes existing nodes' draws**.

Why this matters: estimating each stage separately and multiplying point
estimates discards the driver's variance entirely. The coupled run keeps it.

```python
res.report("wears_tanktop")    # full uncertainty report for one node
res.to_frame()                 # polars DataFrame of all nodes
print(g.to_mermaid())          # diagram of the graph
```

### Correlated inputs

Independent roots are the default. To couple them:

```python
from uss import gaussian_copula
import numpy as np

corr = np.array([[1.0, 0.8], [0.8, 1.0]])
u = gaussian_copula(corr, 100_000, np.random.default_rng(1))

res = g.run(100_000, uniform_overrides={
    "temperature": u[:, 0],
    "humidity":    u[:, 1],
})
```

> **Poisson caveat.** A per-draw `lambda` falls back to scipy's elementwise
> `ppf`, ~69× slower than the scalar path. The engine warns. For large λ,
> a Gaussian approximation is usually the better trade.

---

## 6. Where to spend your next hour

Sensitivity analysis converts "my answer is ±0.24" into "80% of that spread is
one input — go measure that one."

```python
import numpy as np
from uss import parameter_sensitivity, update_bernoulli, update_gaussian_mean

posteriors = {
    "baseline_rate": update_bernoulli(28, 100),
    "temp_mean":     update_gaussian_mean(recorded_temps),
    "slope":         update_gaussian_mean(study_estimates),
}

def predict(p: dict) -> float:
    ...   # your model, returning one number

sens = parameter_sensitivity(predict, posteriors, 20_000, np.random.default_rng(3))
print(sens.summary())
```

```
  input                     first-order     total
  baseline_rate                  0.6604    0.7017
  temp_mean                      0.3053    0.2881
  slope                          0.0056    0.0051
  interaction strength: 0.0237
  -> baseline_rate drives 70% of output variance
```

- **first-order** — variance removed by learning that input exactly.
- **total** — variance remaining if you learned *everything else*.
- `total > first` means the input matters through interactions.
- `interaction_strength ≈ 0` means the model is effectively additive.

Cost is `n_base × (n_factors + 2)` model evaluations. Use `one_at_a_time` for a
rough first pass, but don't act on it — it can't see interactions.

---

## 7. Checking the engine is honest

An engine claiming 95% intervals makes a falsifiable claim. Test it on a
problem where you know the answer, then trust it on one where you don't.

```python
import numpy as np
from uss import calibration, wilson_interval

def trial(level, rng):
    true_p = 0.31
    observed = int(rng.binomial(400, true_p))
    return wilson_interval(observed, 400, level), true_p

cov = calibration.coverage_curve(trial, 800, np.random.default_rng(4))
print(cov.summary())     # verdict: calibrated / overconfident / conservative
```

For full predictive distributions, use PIT — calibrated forecasts give a
uniform histogram:

```python
res = calibration.pit_report(predictive_samples, observations)
print(res.summary())
# U-shaped  -> forecasts too confident
# hump      -> forecasts too vague
# sloped    -> biased
```

`interval_score` scores sharpness and coverage together, so it can't be gamed
by just widening intervals the way raw coverage can.

---

## 8. Events too rare to simulate

Plain Monte Carlo at N=10⁶ finds exactly zero hits for a 10⁻¹² event. Shift the
sampling distribution and reweight:

```python
import numpy as np
from uss import UniversalStochasticSandbox

sandbox = UniversalStochasticSandbox(seed=42)
est = sandbox.rare_event_probability(
    indicator       = lambda x: (x > 7.0).astype(float),
    log_target_pdf  = lambda x: -0.5 * x**2,             # what you want
    log_proposal_pdf= lambda x: -0.5 * (x - 7.0)**2,     # where you sample
    proposal_sampler= lambda n, rng: rng.standard_normal(n) + 7.0,
    sample_size     = 1_000_000,
)
# 1.279586e-12 +/- 7.13e-15   (truth 1.279813e-12, 0.018% error)
```

Weights are formed in log space; a plain ratio underflows to zero here.

**For timing rather than probability**, jump between events instead of ticking a
clock:

```python
from uss import des
stream = des.jump_sample(rate=5e-6, horizon=10_000_000.0, rng=rng)
stream.count            # 50
stream.timestamps[:5]   # when they occurred
```

`des.thinned_sample(rate_fn, rate_max, horizon, rng)` handles time-varying
intensity — diurnal traffic, seasonal demand.

---

## 9. Starting from real data

```bash
uss fit observations.parquet duration_s
```

```
family                   AIC     KS stat        KS p  parameters
lognorm            103442.52     0.00259      0.9875  mean=1.00592, std_dev=0.496133
genextreme         103562.71     0.00900     0.01541  loc=2.30649, scale=1.07697, shape=0.127314
norm               114529.20     0.09859  2.929e-254  mean=3.07884, std_dev=1.63201

best by AIC: lognorm -> query_class='lognormal'
```

Then feed the parameters straight back:

```python
from uss import fitting
ranked, ecdf = fitting.fit_file("observations.parquet", "duration_s")
best = ranked[0]
sandbox.execute_query(best.query_class, best.parameters)
```

If the KS p-value rejects every parametric family, skip them and sample the
data itself:

```python
sandbox.execute_query("empirical", {"quantiles": ecdf})
```

Reads CSV, Parquet and NDJSON; scans lazily so wide files cost only the column
you name.

---

## 10. Choosing sample size

Don't guess — state the precision you need:

```python
result = sandbox.run_until_precision(
    "gaussian", {"mean": 20.0, "std_dev": 3.0},
    target_standard_error=1e-3,
    batch_size=250_000,
)
# n=9,000,000, achieved 1.000e-03, 0.80s
```

The floor is σ/√n, so the cost of precision is quadratic:

| Target SE | n needed (σ=3) | Time |
|---|---|---|
| 1e-2 | 90,000 | 0.02 s |
| 1e-3 | 9,000,000 | 0.80 s |
| 2e-4 | 225,000,000 | 23.6 s |
| 1e-5 | 9×10¹⁰ | impractical |

If the target can't be met inside `max_samples`, the engine says so in
`caveats` rather than presenting a stopped-early run as a success.

**Rough guidance:** 10⁵ for exploration, 10⁶–10⁷ for reported figures, and
importance sampling rather than brute force for anything rarer than ~10⁻⁶.
And remember — past a certain point, more samples buy nothing that matters
(Section 3).

---

## 11. Variance reduction

Free precision when the estimator is monotone in U:

```python
result = sandbox.execute_query("lognormal", {"mean": 0.0, "std_dev": 0.6},
                               antithetic=True)
```

Measured 11.6× variance reduction on a lognormal functional at equal cost.
Works best on smooth monotone transforms; near-useless on symmetric targets
where pairs cancel to a constant.

`uss.variance` also exposes `control_variate` for when you have a correlated
quantity with a known analytic mean.

> Standard errors from these estimators are computed across *pairs*, not pooled
> draws. Never recompute an antithetic SE with `stats.sem` — it treats
> negatively-correlated partners as independent and overstates error ~3.4×.

---

## 12. Behavioral modelling

For human-choice queries, build the probability properly rather than scaling it:

```python
from uss import behavioral

# Part I logit form: P = 1/(1 + exp(-(b0 + sum bk Xk)))
p = behavioral.logistic_link(intercept=-8.0, coefficients=[0.22], covariates=[33.7])

# Or a multiplicative effect on the ODDS scale — never leaves (0,1):
p = behavioral.odds_ratio_shift(0.28, odds_ratio=1.15)   # 0.309
```

Do **not** multiply a probability and clip. At p=0.9, multipliers of 1.2 and
5.0 both give exactly 1.0 — a 4× difference in effect collapses to certainty.
The engine rejects `bias_multiplier` for this reason.

Also available: `apply_loss_aversion` (λ=2.25), `prospect_value`,
`social_proof_cascade`, `hyperbolic_discount`, `multinomial_logit`.

---

## 13. Extending

Any non-decreasing `F⁻¹(U)` works and inherits the whole machinery:

```python
import numpy as np
from uss import register, QueryClass

def weibull(u, shape=1.5, scale=1.0, **_):
    return scale * (-np.log1p(-u)) ** (1.0 / shape)

register(QueryClass("weibull", weibull, "continuous", "Failure times"))

sandbox.execute_query("weibull", {"shape": 1.5, "scale": 2.0})
```

`kind` must be one of `proportion`, `count`, or `continuous` — it selects the
interval construction (Wilson, Garwood, Student-t respectively).

To work inside a `ScenarioGraph`, accept vector parameters: use `np.asarray`
and validate elementwise rather than calling `float()`.

---

## 14. Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Quoting the MC interval as your answer | Overstates certainty by 100×+ | Use `execute_with_priors`, quote `total_interval` |
| Raising `sample_size` to "improve accuracy" | Buys nothing once MC error < parameter sd | Check the caveat; go get data instead |
| `p * multiplier` for a covariate effect | Clips to 1.0, discards effect size | `odds_ratio_shift` or `logistic_link` |
| Normal interval on a rare-event rate | Invalid at low counts | Engine auto-selects Garwood; keep `kind="count"` |
| Pooling antithetic halves into `stats.sem` | Overstates error 3.4× | Use the returned `standard_error` |
| Plain MC for a 10⁻⁹ event | Reports exactly 0 | `rare_event_probability` |
| Small `inner_sample_size` with rare events | Interval reports a discretisation grid | Raise it until ~30+ events per replication |
| Ignoring `caveats` | Miss every warning above | Read them |

---

## 15. Command line

```bash
uss gui                                            # local web interface
uss classes                                        # list query classes
uss domains                                        # Part IV confidence ceilings
uss query gaussian -p mean=20 -p std_dev=3 -n 1000000
uss query poisson -p lambda=0.000005 -n 10000000
uss query lognormal -p mean=0 -p std_dev=0.6 --antithetic
uss query bernoulli -p probability=0.3 --domain macro_behavioral --json
uss fit data.parquet column_name
```

---

## 16. Knowing what the engine can't do

`uss domains` prints the blueprint's own honest ceiling per domain:

| Domain | Achievable | Limited by |
|---|---|---|
| Closed physical systems | 98–99.9% | floating-point, initial conditions |
| Aggregated macro behaviour | 90–95% | exogenous shift events |
| Complex non-linear networks | 50–80% | chaotic divergence |
| Quantum anomalies / spontaneous creation | ~0% | no empirical priors exist |

Pass `domain="macro_behavioral"` to attach the relevant ceiling to a result.

That last row is the honest one. For a query like "rate of a balloon
materialising in a box", there is no prior to condition on — the engine returns
the λ you supplied, with an interval describing only its own arithmetic. The
machinery is real; the input isn't. No amount of sampling fixes that, and the
engine is built to say so rather than let a tight interval imply otherwise.
