# Use Cases: the blueprint's two worked examples

Both queries from the original design blueprint, run end to end.
They go through the identical pipeline and land in completely different places —
which is the most useful thing this pair can teach you.

```bash
.venv\Scripts\python examples\case1_tanktops.py
.venv\Scripts\python examples\case2_balloon.py
```

| | Case 1 — tank tops | Case 2 — balloon |
|---|---|---|
| Evidence available | yes, countable | none, and none obtainable |
| Output shape | point estimate + interval | upper bound + floor of zero |
| Answer | 40.2%, 95% CI 32.8–48.6% | `P < 4.3e-06`, `P > 0` |
| More data helps? | yes, directly | not collectable |
| More simulation helps? | no | no |
| Blueprint Part IV ceiling | 90–95% | approaches 0% |

---

# Case 1 — What % of a population wears a tank top on a hot day?

**Answerable.** You can go outside and count.

## Step 1 — Collect counts, not opinions

Two afternoons at two temperatures. The script's numbers are placeholders —
replace them with yours:

```python
COOL_TEMP_C, COOL_YES, COOL_TOTAL = 18.0,  9, 140
HOT_TEMP_C,  HOT_YES,  HOT_TOTAL  = 34.0, 61, 152
```

Two points is the minimum that pins a temperature response. More temperatures
is better; more people per temperature is better still.

## Step 2 — The direct answer

You surveyed on a hot day, so the hot-day count answers the question directly:

```
surveyed 61 of 152 people at 34 C
estimate        : 40.2%
95% interval    : 32.8% to 48.6%
simulation-only : 40.2% to 40.3%   <- do NOT quote this
```

**Quote the first interval.** The second is 0.1 percentage points wide and would
be exactly that narrow if you had surveyed three people — it measures the
arithmetic, not the world. This is the single most important habit with this
engine.

## Step 3 — Any temperature, via the logistic link

Two survey points determine `logit(p) = β₀ + β₁·T` exactly, and both endpoints
carry uncertainty, so β₀ and β₁ inherit it:

```
slope per degree C : +0.1399  (95% +0.0967 to +0.1864)
odds multiply by   : 1.150x per degree

  temp   estimate       95% interval
  15 C      4.8%     2.0% to  8.9%
  20 C      9.0%     4.8% to 13.9%
  25 C     16.2%    11.1% to 21.6%
  30 C     27.9%    22.2% to 34.0%
  34 C     40.3%    32.9% to 47.8%
  38 C     54.0%    43.3% to 64.1%
```

Intervals widen away from 18 °C and 34 °C — the two temperatures you actually
measured. The 38 °C row is extrapolation: a modelling assumption, not data.

## Step 4 — Turn a fraction into a headcount

A `ScenarioGraph` couples day-to-day temperature variation into the outcome:

```python
temp = graph.add("temperature", "gaussian", mean=34.0, std_dev=2.5)
prob = graph.derive("p", lambda T: expit(b0 + b1 * T), temp)
graph.add("wears", "bernoulli", probability=prob)
```

```
crowd of 50,000 on a 34 C day
expected wearing : 20,216 people (40.4%)
temp/outfit corr : +0.164

sampling + weather only : 20,140 to 20,292 people
including survey error  : 16,376 to 24,295 people
```

The narrow range is a trap for the same reason as Step 2. **Quote 16,376 to
24,295.**

## Step 5 — Which survey to expand

```
hot_day_rate    100.0%  ########################################
cool_day_rate     0.0%
```

The cool-day survey contributes nothing to a prediction *at* 34 °C — correct,
since the hot survey sits exactly there. Expand the cool-day count only if you
care about the curve at intermediate temperatures.

## Step 6 — What the answer is not

It is the rate among people you could have sampled **the way you sampled**:
same streets, hours, city, season. It is not a national figure and not next
summer's. Part IV caps aggregated macro behaviour at 90–95%, limited by
"exogenous shift events" — fashion is exactly that.

The engine bounds sampling error. It cannot tell you whether you sampled the
right people.

---

# Case 2 — What are the chances of a balloon appearing out of thin air?

**Not answerable as a probability.** Worth running anyway, because it shows
exactly where the wall is.

## Step 1 — The trap

```
assumed lambda 1e-03  ->  estimate 1.022e-03
assumed lambda 1e-06  ->  estimate 1.400e-06
assumed lambda 1e-09  ->  estimate 0.000e+00
```

Each input came back out. No observation of the world entered the calculation,
so the interval describes the random number generator. Case 1 escaped this
because 61-of-152 was a real count. **Here this route is circular — don't use
it.**

## Step 2 — What you *can* compute

You cannot say how often balloons spontaneously appear. You can say: *if it
happened more often than X, we would have seen one.* Zero observations is
enough evidence for that.

```
observation coverage                                 95% upper bound
one room, one year                                   1.90e-09 /m3/s
one room, a human lifetime                           2.37e-11 /m3/s
a city's indoor volume, a century                    1.90e-18 /m3/s
Earth's lower atmosphere, all recorded history       4.75e-30 /m3/s
```

These coverage figures are **assumptions about observation, not measurements** —
own them when you quote the result.

The lower bound is exactly zero in every row and always will be. No amount of
not-seeing-something rules out a true rate of zero. The answer is an interval
`[0, X]`, never a point.

## Step 3 — The engine's exact interval at zero events

```
exposure      : 1.262e+11 m3-seconds
garwood 95%   : [0.000e+00, 2.922e-11] per m3 per second
rule of three : 2.373e-11  (one-sided, for comparison)
```

The two differ only by convention — Garwood splits 5% across both tails, the
rule of three puts it all in the upper one. This is why `summarize(kind="count")`
selects Garwood automatically: the normal approximation is invalid at zero.

## Step 4 — What one sighting would do

```
  events      central rate                  95% interval
       0    not identified                 [0, 2.92e-11]
       1         1.188e-11        [8.55e-13, 3.70e-11]
       5         4.357e-11        [1.51e-11, 8.68e-11]
      50         4.001e-10        [2.97e-10, 5.18e-10]
```

The machinery is not useless here — it is idle, waiting for data. One credible
instrumented observation turns an upper bound into an estimate with both ends
finite.

## Step 5 — If you insist on a probability

Watching a 50 m³ room for one hour, using the lifetime-of-non-observation bound
as a ceiling:

```
P(at least one balloon)  <  4.27e-06
P(at least one balloon)  >  0          (exactly, and irreducibly)
```

That is the complete honest answer. Any single number quoted between those is a
choice you made, not a result the data produced. If a downstream calculation
needs a point value, **say that you assumed it.**

---

# What the pair demonstrates

Both ran the same pipeline: `U ~ Uniform(0,1) → F⁻¹(U) → statistics → bounded
report`. The engine did not need to know that one question was about clothing
and the other about spontaneous matter creation. That is the "non-case-limited"
design working.

What differed was entirely upstream:

- **Tank tops** — a real count entered. Output is quotable and narrows with more
  surveying.
- **Balloons** — only an assumption entered. Output is a bound. More simulation
  changes nothing; more surveying is not possible.

Blueprint Part IV predicted this split before either was run, putting
spontaneous creation in the "approaches 0%" row limited by "lack of empirical
priors". That row is the most valuable line in the spec — it is the engine
telling you in advance which of your questions it can and cannot answer.

Read the `caveats` on every result. They are where the engine says which
situation you are in.

## Adapting these

| Your question resembles | Start from | Key move |
|---|---|---|
| a rate you can survey | `case1_tanktops.py` | count something, use `update_bernoulli` |
| a rate driven by a measurable factor | `case1_tanktops.py` §3 | logistic link + `ScenarioGraph` |
| a rare event that *has* occurred | `case2_balloon.py` §4 | `update_poisson(events, exposure)` |
| a rare event never observed | `case2_balloon.py` §2 | report a bound, not a probability |

See [GUIDE.md](GUIDE.md) for the full API, and `uss gui` for the same workflows
without writing code.
