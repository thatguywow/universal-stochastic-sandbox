"""YOUR SCENARIO -- edit this file and re-run it.

    .venv/Scripts/python my_question.py

The worked example below asks a question with a real decision attached:

    "We stock 18 litres of oat milk for the morning rush. What's the chance
     we run out -- and is that risk worth restocking over?"

Every block marked  # <<< EDIT  is something you swap for your own question.
Work top to bottom; each step explains what decision you're making and why.
"""

from __future__ import annotations

import numpy as np

from uss import (
    combine_uncertainty,
    parameter_sensitivity,
    propagate,
    summarize,
    update_bernoulli,
    update_gaussian_mean,
)

rng = np.random.default_rng(42)


# =============================================================================
# STEP 1 -- State the question, and be explicit about what you actually know.
# =============================================================================
# The hard part of modelling is not the maths, it's being honest here. For each
# input, write down where the number came from. If the answer is "I guessed",
# say so -- STEP 2 will carry that guess through as real uncertainty instead of
# letting it masquerade as fact.

RUSH_HOURS = 3.0            # <<< EDIT  length of the window you care about
ML_PER_ORDER = 150.0        # <<< EDIT  consumption per event
STOCK_ML = 18000.0          # <<< EDIT  the threshold that defines "bad"

# What we know about arrival rate: 14 mornings of till counts.
OBSERVED_ARRIVALS_PER_HOUR = np.array(  # <<< EDIT  your observations
    [118, 131, 105, 127, 142, 109, 122, 136, 115, 128, 133, 111, 124, 139],
    dtype=float,
)

# What we know about oat preference: we asked 40 customers, 11 wanted oat.
OAT_SURVEY_YES = 11         # <<< EDIT
OAT_SURVEY_TOTAL = 40       # <<< EDIT


# =============================================================================
# STEP 2 -- Turn each "what we know" into a posterior.
# =============================================================================
# A posterior says "here is the range of values consistent with my evidence",
# not "here is the number". Fourteen mornings pins the arrival rate reasonably
# well; forty survey responses pins the oat fraction only loosely. That
# difference is the whole point -- it will show up in the final answer.

posteriors = {
    "arrival_rate": update_gaussian_mean(OBSERVED_ARRIVALS_PER_HOUR),
    "oat_fraction": update_bernoulli(OAT_SURVEY_YES, OAT_SURVEY_TOTAL),
}

print("STEP 2 -- what we know, as ranges rather than points")
for name, post in posteriors.items():
    lo, hi = post.interval(0.95)
    print(f"  {name:<16}central {post.mean:>8.4f}   95% range [{lo:.4f}, {hi:.4f}]")
print()


# =============================================================================
# STEP 3 -- Write the simulation for ONE fixed set of parameters.
# =============================================================================
# This function must take a dict of parameters and return a single number: the
# quantity you care about. Do not put uncertainty about the parameters in here
# -- that is STEP 4's job. In here, the parameters are known and you are only
# simulating the randomness of the world.

INNER_DRAWS = 20_000  # realisations per parameter set; see the sizing note below


def simulate(params: dict[str, float]) -> float:
    """P(we run out of oat milk), for one specific arrival rate and oat share.

    Note the fixed seed. Drawing from a shared generator would make this
    function return slightly different answers for identical inputs, and STEP 5
    (Sobol sensitivity) is only defined for a deterministic function -- the
    run-to-run jitter would be attributed to the inputs. Seeding here makes the
    simulation reproducible given its parameters, which is what that step needs.
    """
    lam_total = params["arrival_rate"] * RUSH_HOURS
    p_oat = params["oat_fraction"]

    # Poisson thinning: if arrivals are Poisson(L) and each independently wants
    # oat with probability p, then oat orders are exactly Poisson(L * p).
    # That identity saves us simulating every customer individually.
    local = np.random.default_rng(20260805)
    oat_orders = local.poisson(lam_total * p_oat, size=INNER_DRAWS)

    volume_needed = oat_orders * ML_PER_ORDER
    return float(np.mean(volume_needed > STOCK_ML))  # <<< EDIT  your output metric


# Sanity check first: run it once at the central estimate. Always do this --
# if this number is absurd, the model is wrong and no amount of uncertainty
# machinery will save it.
central = {name: post.mean for name, post in posteriors.items()}
print("STEP 3 -- sanity check at the central estimate")
print(f"  expected oat orders : {central['arrival_rate'] * RUSH_HOURS * central['oat_fraction']:.1f}")
print(f"  expected volume     : {central['arrival_rate'] * RUSH_HOURS * central['oat_fraction'] * ML_PER_ORDER:.0f} ml"
      f"   (stock is {STOCK_ML:.0f} ml)")
print(f"  P(stockout)         : {simulate(central):.4f}")
print()


# =============================================================================
# STEP 4 -- Propagate the uncertainty.
# =============================================================================
# Now run the simulation once per plausible parameter set. The spread of the
# results is the part of your uncertainty that does NOT go away by simulating
# harder -- it only goes away by collecting more data.

PARAM_DRAWS = 400  # <<< EDIT  more = smoother interval, linear cost

draws = propagate(simulate, posteriors, PARAM_DRAWS, rng)

report = summarize(draws, kind="continuous", confidence_level=0.95)
report = combine_uncertainty(report, draws)

lo, hi = report.total_interval
print("STEP 4 -- the answer, with honest bounds")
print(f"  P(stockout), central : {report.point_estimate:.4f}")
print(f"  95% interval         : [{lo:.4f}, {hi:.4f}]")
print(f"  simulation error     : {report.monte_carlo_error:.5f}   (shrinks with INNER_DRAWS)")
print(f"  parameter spread     : {report.parameter_uncertainty:.5f}   (shrinks only with more data)")
print()
print("  Read it like this: the central number is your best guess, and the")
print("  interval is the range you should actually plan against.")
print()


# =============================================================================
# STEP 5 -- Ask what would most improve the answer.
# =============================================================================
# If the interval above is too wide to act on, this tells you which input to go
# measure. Spending a morning counting customers vs. surveying oat preference
# are different costs with very different payoffs -- this says which one wins.

sens = parameter_sensitivity(simulate, posteriors, 600, rng)
print("STEP 5 -- what to measure next")
print(f"  {'input':<18}{'first-order':>13}{'total':>9}")
for name, first, total in sens.ranked():
    print(f"  {name:<18}{first:>13.4f}{total:>9.4f}")
top = sens.ranked()[0]
print(f"\n  -> {top[0]} accounts for {top[2]:.0%} of the spread in the answer.")
print("     Collect more of that before collecting anything else.")
for warning in sens.warnings:  # never drop these: they invalidate the ranking
    print(f"\n  ! {warning}")
print()


# =============================================================================
# STEP 6 -- Make the decision.
# =============================================================================
RISK_TOLERANCE = 0.10  # <<< EDIT  the threshold you'd actually act on

print("STEP 6 -- decision")
if hi < RISK_TOLERANCE:
    verdict = f"SAFE - even the pessimistic end ({hi:.1%}) is under your {RISK_TOLERANCE:.0%} bar."
elif lo > RISK_TOLERANCE:
    verdict = f"ACT - even the optimistic end ({lo:.1%}) exceeds your {RISK_TOLERANCE:.0%} bar."
else:
    verdict = (
        f"UNRESOLVED - the interval [{lo:.1%}, {hi:.1%}] straddles your "
        f"{RISK_TOLERANCE:.0%} bar.\n           More simulation will not settle this. "
        f"Go measure {top[0]}."
    )
print(f"  {verdict}")


# =============================================================================
# SIZING NOTE
# =============================================================================
# INNER_DRAWS controls simulation precision; PARAM_DRAWS controls how smoothly
# the parameter uncertainty is mapped. If "parameter spread" is much larger
# than "simulation error" (the usual case), raising INNER_DRAWS is wasted
# effort -- the answer is limited by your data, not your compute.
#
# Watch for a caveat about the interval being "quantised by the inner
# simulation grid": that means INNER_DRAWS is too small to resolve a rare
# outcome, and the interval is reporting rounding rather than uncertainty.
for caveat in report.caveats:
    print(f"\n  ! {caveat}")
