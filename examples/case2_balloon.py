"""CASE 2 -- What are the chances of a balloon appearing out of thin air?

The blueprint's second worked example, and the opposite situation to Case 1.
There is no survey to run and no observation to condition on, so the honest
output is not a probability. It is a bound.

    .venv\\Scripts\\python examples\\case2_balloon.py

Blueprint Part IV already says this: spontaneous-creation queries sit in the
'approaches 0%' row, limited by 'Heisenberg uncertainty & lack of empirical
priors'. This script shows what that means operationally -- what the engine
still computes, and what it refuses to invent.
"""

from __future__ import annotations

import numpy as np

from uss import UniversalStochasticSandbox, garwood_interval, update_poisson

SECONDS_PER_YEAR = 365.25 * 24 * 3600


def rule(t: str) -> None:
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


rule("1. THE TRAP -- hand the engine a rate and it hands the rate back")

sandbox = UniversalStochasticSandbox(sample_size=5_000_000, seed=42)
for assumed in [1e-3, 1e-6, 1e-9]:
    r = sandbox.execute_query("poisson", {"lambda": assumed})
    print(f"  assumed lambda {assumed:.0e}  ->  estimate {r.report.point_estimate:.3e}")

print("""
  The engine reproduced each input to the precision requested. That is the
  arithmetic working correctly, and it is worth nothing: no observation of
  the world entered the calculation. Whatever number you put in is the number
  you get out, wrapped in a confidence interval that describes only the
  random number generator.

  Case 1 avoided this because 61-of-152 was a real count. Here there is no
  count, so this route is circular. Do not use it.""")


rule("2. WHAT CAN LEGITIMATELY BE COMPUTED -- a bound from non-observation")

print("""  You cannot say how often balloons spontaneously appear. You CAN say:
  'if it happened more often than X, we would have seen one by now.'
  That is a real statistical statement, and zero observations is enough
  evidence to make it.

  Pick how much watched space-time you are willing to claim. These are
  assumptions about observation coverage, not measurements -- own them.
""")

SCENARIOS = [
    ("one room, one year", 50.0, 1.0),
    ("one room, a human lifetime", 50.0, 80.0),
    ("a city's indoor volume, a century", 5.0e8, 100.0),
    ("Earth's lower atmosphere, all recorded history", 4.0e18, 5000.0),
]

print(f"  {'observation coverage':<48}{'95% upper bound':>20}")
for label, volume_m3, years in SCENARIOS:
    exposure = volume_m3 * years * SECONDS_PER_YEAR  # m^3 * seconds
    # Rule of three: with zero events, the one-sided 95% bound is -ln(0.05)/E.
    one_sided = -np.log(0.05) / exposure
    print(f"  {label:<48}{one_sided:>13.2e} /m3/s")

print("""
  Read the last row as: 'fewer than about that many per cubic metre per
  second, with 95% confidence.' The lower bound is exactly zero in every
  row and always will be -- no amount of not-seeing-something can rule out
  that the true rate is zero. The answer is an interval [0, X], never a point.""")


rule("3. THE ENGINE'S OWN EXACT INTERVAL AT ZERO EVENTS")

exposure = 50.0 * 80.0 * SECONDS_PER_YEAR  # one room, a human lifetime
lo, hi = garwood_interval(0, exposure, 0.95)
print(f"  exposure           : {exposure:.3e} m3-seconds")
print(f"  garwood 95%        : [{lo:.3e}, {hi:.3e}] per m3 per second")
print(f"  rule of three      : {-np.log(0.05) / exposure:.3e}  (one-sided, for comparison)")
print("""
  The two differ only by convention -- Garwood splits 5% across both tails,
  the rule of three puts it all in the upper one. Both give a lower bound of
  exactly zero, which is the honest part.""")


rule("4. WHAT ONE SIGHTING WOULD DO")

print("""  The machinery is not useless here -- it is idle, waiting for data.
  If a credible, instrumented observation ever occurred, the posterior
  updates immediately:
""")
print(f"  {'events':>8}{'central rate':>18}{'95% interval':>30}")
for events in [0, 1, 5, 50]:
    if events == 0:
        _, hi0 = garwood_interval(0, exposure, 0.95)
        print(f"  {events:>8}{'not identified':>18}{f'[0, {hi0:.2e}]':>30}")
        continue
    post = update_poisson(events, exposure)
    plo, phi = post.interval(0.95)
    print(f"  {events:>8}{post.mean:>18.3e}{f'[{plo:.2e}, {phi:.2e}]':>30}")

print("""
  One sighting turns an upper bound into an estimate with both ends finite.
  That is the transition from 'we cannot say' to 'we can say, roughly'.""")


rule("5. IF YOU INSIST ON A PROBABILITY -- read this first")

room_volume, watch_hours = 50.0, 1.0
watch_exposure = room_volume * watch_hours * 3600
bound = -np.log(0.05) / (50.0 * 80.0 * SECONDS_PER_YEAR)
p_upper = 1.0 - np.exp(-bound * watch_exposure)

print(f"  Watching a {room_volume:.0f} m3 room for {watch_hours:.0f} hour, using the")
print("  lifetime-of-non-observation bound as a ceiling on the rate:")
print(f"\n    P(at least one balloon)  <  {p_upper:.2e}")
print("    P(at least one balloon)  >  0    (exactly, and irreducibly)")
print("""
  That is the complete honest answer: a ceiling and a floor of zero. Any
  single number quoted between them is a choice you made, not a result the
  data produced. If you need a point estimate for a downstream calculation,
  state that you assumed it.""")


rule("6. CASE 1 vs CASE 2 -- why one worked and one did not")

print("""  Both ran through the same pipeline. The difference is upstream.

    tank tops : a real count (61 of 152) entered the calculation.
                Output: 40.2%, 95% interval 32.8% to 48.6%. Quotable.
                More surveying narrows it.

    balloons  : nothing entered but an assumption. Output: an upper bound
                and a floor of zero. More simulation changes nothing;
                more surveying is not possible.

  The engine treats both identically -- that is the 'non-case-limited'
  design working as intended. Whether the answer means anything is decided
  by the evidence you fed it, and the caveats are where it tells you which
  situation you are in.""")
