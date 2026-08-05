"""Blueprint Part V's two test executions, run through the built engine.

Shows the same two queries the spec names -- the tank-top behavioural query and
the balloon-materialisation spatial query -- and then shows what changes once
parameter uncertainty is admitted.

    python examples/blueprint_demo.py
"""

from __future__ import annotations

import numpy as np

from uss import UniversalStochasticSandbox, behavioral, des, update_bernoulli, update_poisson


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    sandbox = UniversalStochasticSandbox(sample_size=10_000_000, seed=42)

    # ---------------------------------------------------------------------
    rule("1. Macro human behaviour -- tank tops under a heatwave")

    # Part V wrote this as probability=0.28, bias_multiplier=1.15.
    # The multiplier form clips at 1.0 and discards covariate signal, so the
    # heatwave effect is expressed on the odds scale instead.
    p_base = 0.28
    p_heatwave = behavioral.odds_ratio_shift(p_base, odds_ratio=1.15)
    print(f"baseline p            : {p_base}")
    print(f"p after 1.15x odds    : {p_heatwave:.6f}")
    print(f"(the old p*1.15 path  : {p_base * 1.15:.6f})\n")

    result = sandbox.execute_query(
        "bernoulli", {"probability": p_heatwave}, domain="macro_behavioral"
    )
    print(result.summary())

    # ---------------------------------------------------------------------
    rule("2. Rare spatial event -- balloon materialisation, lambda = 5e-6")

    balloon = sandbox.execute_query("poisson", {"lambda": 0.000005})
    print(balloon.summary())

    # ---------------------------------------------------------------------
    rule("3. The same balloon query as a jump-sampled event stream")

    # Rather than drawing 10^7 mostly-zero counts, jump straight between events.
    stream = des.jump_sample(rate=5e-6, horizon=10_000_000.0, rng=np.random.default_rng(7))
    print(f"horizon               : {stream.horizon:,.0f} time units")
    print(f"events realised       : {stream.count}")
    print(f"empirical rate        : {stream.empirical_rate:.3e}  (true 5.000e-06)")
    print(f"first 5 timestamps    : {np.round(stream.timestamps[:5], 1)}")

    # ---------------------------------------------------------------------
    rule("4. What the confidence interval actually means")

    print(
        "Query 1 reports a very tight interval, but it is measuring the RNG:\n"
        "  the engine was handed p and recovered p.\n"
        "Suppose that p came from a survey of only 10 people, 3 of whom wore\n"
        "tank tops. The honest interval is far wider:\n"
    )

    honest = sandbox.execute_with_priors(
        "bernoulli",
        {"probability": update_bernoulli(successes=3, trials=10)},
        n_parameter_draws=512,
        inner_sample_size=50_000,
        domain="macro_behavioral",
    )
    print(honest.summary())

    mc_lo, mc_hi = honest.report.monte_carlo_interval
    tot_lo, tot_hi = honest.report.total_interval
    print(
        f"\n  MC interval width   : {mc_hi - mc_lo:.6f}\n"
        f"  TOTAL width         : {tot_hi - tot_lo:.6f}"
        f"   ({(tot_hi - tot_lo) / (mc_hi - mc_lo):.0f}x wider)"
    )
    print("  Raising sample_size shrinks the first and leaves the second alone.")

    # ---------------------------------------------------------------------
    rule("5. Rare-event probability beyond the reach of plain Monte Carlo")

    from scipy import stats

    est = sandbox.rare_event_probability(
        indicator=lambda x: (x > 7.0).astype(np.float64),
        log_target_pdf=lambda x: -0.5 * x**2,
        log_proposal_pdf=lambda x: -0.5 * (x - 7.0) ** 2,
        proposal_sampler=lambda n, rng: rng.standard_normal(n) + 7.0,
        sample_size=1_000_000,
    )
    truth = float(stats.norm.sf(7.0))
    print(f"P(Z > 7) truth        : {truth:.6e}")
    print(f"importance sampling   : {est.value:.6e} +/- {1.96 * est.standard_error:.2e}")
    print(f"relative error        : {abs(est.value - truth) / truth:.3%}")
    print(f"effective sample size : {est.effective_sample_size:,.0f}")
    print(
        f"plain MC at n=1e6     : expected hits = {truth * 1e6:.2e}"
        "  (i.e. it would report exactly 0)"
    )

    # ---------------------------------------------------------------------
    rule("6. Poisson rate under uncertainty about lambda itself")

    observed = sandbox.execute_with_priors(
        "poisson",
        {"lam": update_poisson(event_count=50, exposure=10_000_000.0)},
        n_parameter_draws=256,
        inner_sample_size=100_000,
        domain="quantum_anomaly",
    )
    print(observed.summary())


if __name__ == "__main__":
    main()
