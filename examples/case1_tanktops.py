"""CASE 1 -- What fraction of a population wears a tank top on a hot day?

A question with obtainable data. Everything below is driven by counts you could
actually collect by standing on a street corner for two afternoons.

    .venv\\Scripts\\python examples\\case1_tanktops.py

The numbers marked  # <<< SURVEY  are illustrative stand-ins. Replace them with
your own counts -- nothing here is a published statistic.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit

from uss import (
    ScenarioGraph,
    UniversalStochasticSandbox,
    parameter_sensitivity,
    update_bernoulli,
)

rng = np.random.default_rng(2026)


def rule(t: str) -> None:
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


# ---------------------------------------------------------------------------
# The evidence. Two afternoons of counting, at two different temperatures.
# ---------------------------------------------------------------------------
COOL_TEMP_C, COOL_YES, COOL_TOTAL = 18.0, 9, 140      # <<< SURVEY
HOT_TEMP_C, HOT_YES, HOT_TOTAL = 34.0, 61, 152        # <<< SURVEY
TARGET_TEMP_C = 34.0                                   # <<< the day you care about

p_cool = update_bernoulli(COOL_YES, COOL_TOTAL)
p_hot = update_bernoulli(HOT_YES, HOT_TOTAL)


rule("1. THE DIRECT ANSWER -- you surveyed on a hot day, so just use it")

sandbox = UniversalStochasticSandbox(sample_size=2_000_000, seed=42)
direct = sandbox.execute_with_priors(
    "bernoulli",
    {"probability": p_hot},
    n_parameter_draws=800,
    inner_sample_size=200_000,
    domain="macro_behavioral",
)
lo, hi = direct.report.total_interval
print(f"  surveyed {HOT_YES} of {HOT_TOTAL} people at {HOT_TEMP_C:.0f} C")
print(f"  estimate           : {direct.report.point_estimate:.1%}")
print(f"  95% interval       : {lo:.1%} to {hi:.1%}")
print(f"  simulation-only    : {direct.report.monte_carlo_interval[0]:.1%} to "
      f"{direct.report.monte_carlo_interval[1]:.1%}  <- do NOT quote this")
print("\n  Quote the first interval. The second one only says the arithmetic")
print("  converged; it would be just as narrow if you had surveyed 3 people.")


rule("2. ANY TEMPERATURE -- interpolate with the logistic link")

# Two survey points pin a logistic curve exactly:
#   logit(p) = b0 + b1 * T
# Both endpoints are uncertain, so b0 and b1 inherit that uncertainty.
def logistic_params(pc: float, ph: float) -> tuple[float, float]:
    b1 = (logit(ph) - logit(pc)) / (HOT_TEMP_C - COOL_TEMP_C)
    b0 = logit(pc) - b1 * COOL_TEMP_C
    return b0, b1


draws_cool = p_cool.sample(4000, rng)
draws_hot = p_hot.sample(4000, rng)
b0s, b1s = np.vectorize(logistic_params)(draws_cool, draws_hot)

print(f"  slope per degree C : {b1s.mean():+.4f} "
      f"(95% {np.quantile(b1s, .025):+.4f} to {np.quantile(b1s, .975):+.4f})")
print(f"  odds multiply by   : {np.exp(b1s.mean()):.3f}x per degree\n")

print(f"  {'temp':>6}  {'estimate':>9}  {'95% interval':>22}")
for t in [15, 20, 25, 30, 34, 38]:
    curve = expit(b0s + b1s * t)
    print(f"  {t:>4} C  {curve.mean():>8.1%}  "
          f"{np.quantile(curve, .025):>9.1%} to {np.quantile(curve, .975):<9.1%}")
print("\n  Intervals widen away from the two temperatures you actually")
print("  measured. Extrapolating past them is a modelling assumption, not data.")


rule("3. HOW MANY PEOPLE -- turn a fraction into a crowd count")

CROWD = 50_000  # <<< EDIT

graph = ScenarioGraph(seed=7)
temp = graph.add("temperature", "gaussian", mean=TARGET_TEMP_C, std_dev=2.5)
prob = graph.derive("p", lambda T: expit(b0s.mean() + b1s.mean() * T), temp)
graph.add("wears", "bernoulli", probability=prob)

res = graph.run(400_000)
frac = res["wears"].mean()
report = res.report("wears")
print(f"  crowd of {CROWD:,} on a {TARGET_TEMP_C:.0f} C day (day-to-day temp varies +/- 2.5 C)")
print(f"  expected wearing   : {frac * CROWD:,.0f} people ({frac:.1%})")
print(f"  temp/outfit corr   : {res.correlation('temperature', 'wears'):+.3f}")
print("\n  Note this interval covers sampling and weather only:")
print(f"    {report.monte_carlo_interval[0] * CROWD:,.0f} to "
      f"{report.monte_carlo_interval[1] * CROWD:,.0f} people")
print("  Fold in survey uncertainty from step 1 and the honest range is:")
print(f"    {lo * CROWD:,.0f} to {hi * CROWD:,.0f} people")


rule("4. WHICH SURVEY SHOULD YOU EXPAND?")


def predict_at_target(params: dict[str, float]) -> float:
    b0, b1 = logistic_params(params["cool_day_rate"], params["hot_day_rate"])
    return float(expit(b0 + b1 * TARGET_TEMP_C))


sens = parameter_sensitivity(
    predict_at_target,
    {"cool_day_rate": p_cool, "hot_day_rate": p_hot},
    4000,
    rng,
)
print(f"  {'survey':<18}{'share of spread':>18}")
for name, _first, total in sens.ranked():
    bar = "#" * int(total * 40)
    print(f"  {name:<18}{total:>17.1%}  {bar}")
for w in sens.warnings:
    print(f"  ! {w}")
top = sens.ranked()[0]
print(f"\n  -> more counting on {top[0].replace('_', ' ')}s narrows the answer most.")


rule("5. WHAT THIS ANSWER IS AND IS NOT")

print("""  IS : the tank-top rate among people you could have sampled the same
       way you sampled -- same streets, same hours, same city, same season.
  NOT: a national figure, and not next summer's figure. Blueprint Part IV
       caps aggregated macro behaviour at 90-95% confidence, limited by
       'exogenous shift events'. Fashion is exactly such a shift.

  Widen your sampling before widening your claim. The engine bounds
  sampling error; it cannot bound whether you sampled the right people.""")

for c in direct.report.caveats:
    print(f"\n  ! {c}")
