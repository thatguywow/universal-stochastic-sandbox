"""The four additions beyond the blueprint, on one worked problem.

Question: what fraction of a city wears tank tops tomorrow, and how much do we
actually know about that number?

    python examples/elevated_demo.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from uss import (
    ScenarioGraph,
    UniversalStochasticSandbox,
    calibration,
    parameter_sensitivity,
    update_bernoulli,
    update_gaussian_mean,
    wilson_interval,
)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    # ---------------------------------------------------------------------
    rule("1. COMPOSITION -- the question is three stages deep, not one")

    g = ScenarioGraph(seed=42)
    temperature = g.add("temperature", "gaussian", mean=31.0, std_dev=4.0)
    humidity = g.add("humidity", "gaussian", mean=55.0, std_dev=12.0)
    comfort = g.derive(
        "discomfort_index",
        lambda t, h: t + 0.05 * h,
        temperature,
        humidity,
    )
    p_tanktop = g.derive("p_tanktop", lambda d: expit(-8.0 + 0.22 * d), comfort)
    g.add("wears_tanktop", "bernoulli", probability=p_tanktop)

    print(g.to_mermaid())
    print()
    result = g.run(500_000)
    print(result.summary())
    print(
        f"\n  corr(temperature, wears_tanktop) = "
        f"{result.correlation('temperature', 'wears_tanktop'):+.4f}"
    )
    print(
        "  The outcome inherits the driver's spread. Estimating each stage\n"
        "  separately and multiplying point estimates would discard exactly this."
    )

    # ---------------------------------------------------------------------
    rule("2. SENSITIVITY -- which assumption should I go measure?")

    # Three uncertain inputs, each known to a different degree.
    posteriors = {
        "baseline_rate": update_bernoulli(28, 100),          # 100-person survey
        "temp_mean": update_gaussian_mean(
            np.random.default_rng(1).normal(31.0, 4.0, 30)   # 30 days of records
        ),
        "slope": update_gaussian_mean(
            np.random.default_rng(2).normal(0.22, 0.09, 12)  # 12 weak studies
        ),
    }

    def predict(p: dict[str, float]) -> float:
        base = np.log(p["baseline_rate"] / (1 - p["baseline_rate"]))
        return float(expit(base + p["slope"] * (p["temp_mean"] - 31.0)))

    sens = parameter_sensitivity(
        predict, posteriors, 20_000, np.random.default_rng(3)
    )
    print(sens.summary())
    print(
        "\n  Actionable: effort spent narrowing the top input reduces output\n"
        "  variance the most. Effort on the bottom one is close to wasted."
    )

    # ---------------------------------------------------------------------
    rule("3. CALIBRATION -- do the stated intervals actually hold?")

    def trial(level: float, rng: np.random.Generator):
        """One replication: survey 400 people, report the engine's interval."""
        true_p = 0.31
        observed = int(rng.binomial(400, true_p))
        return wilson_interval(observed, 400, level), true_p

    cov = calibration.coverage_curve(trial, 800, np.random.default_rng(4))
    print(cov.summary())

    print("\n  Now the same check on a deliberately overconfident procedure:")

    def bad_trial(level: float, rng: np.random.Generator):
        true_p = 0.31
        observed = int(rng.binomial(400, true_p))
        p_hat = observed / 400
        half = 1.96 * np.sqrt(p_hat * (1 - p_hat) / 400) / 3.0  # 3x too narrow
        return (p_hat - half, p_hat + half), true_p

    bad = calibration.coverage_curve(bad_trial, 400, np.random.default_rng(5))
    print(f"  verdict: {bad.verdict}   mean error: {bad.calibration_error:.3f}")
    print("  The harness catches it rather than reporting a confident wrong answer.")

    # ---------------------------------------------------------------------
    rule("4. ADAPTIVE PRECISION -- stop when precise enough, not at a guess")

    sandbox = UniversalStochasticSandbox(seed=42)
    for target in (1e-2, 1e-3, 2e-4):
        r = sandbox.run_until_precision(
            "gaussian",
            {"mean": 20.0, "std_dev": 3.0},
            target_standard_error=target,
            batch_size=250_000,
            max_samples=300_000_000,
        )
        met = "ok" if r.report.monte_carlo_error <= target else "HIT CAP"
        print(
            f"  target SE {target:<8.0e} -> n={r.sample_size:>12,}  "
            f"achieved {r.report.monte_carlo_error:.3e}  "
            f"({r.elapsed_seconds:5.2f}s)  [{met}]"
        )

    # A target that cannot be met inside the cap must say so, not quietly stop.
    strict = sandbox.run_until_precision(
        "gaussian",
        {"mean": 20.0, "std_dev": 3.0},
        target_standard_error=1e-5,
        batch_size=1_000_000,
        max_samples=5_000_000,
    )
    print(f"\n  target SE 1e-05  -> n={strict.sample_size:,} then stopped:")
    for caveat in strict.report.caveats:
        if "above the target" in caveat:
            print(f"    ! {caveat}")
    print(
        "\n  sigma/sqrt(n) sets the floor: at sigma=3, an SE of 1e-5 needs\n"
        "  n = 9e10 draws. The engine reports the shortfall instead of\n"
        "  presenting a stopped-early run as if it met the target."
    )

    # ---------------------------------------------------------------------
    rule("PUTTING IT TOGETHER")

    naive = result.report("wears_tanktop")
    print(
        f"Point estimate               : {naive.point_estimate:.4f}\n"
        f"Simulation-precision interval: "
        f"[{naive.monte_carlo_interval[0]:.4f}, {naive.monte_carlo_interval[1]:.4f}]"
    )
    draws = np.array([predict({
        "baseline_rate": posteriors["baseline_rate"].sample(1, np.random.default_rng(i))[0],
        "temp_mean": posteriors["temp_mean"].sample(1, np.random.default_rng(i + 1000))[0],
        "slope": posteriors["slope"].sample(1, np.random.default_rng(i + 2000))[0],
    }) for i in range(2000)])
    print(
        f"Honest interval given priors : "
        f"[{np.quantile(draws, 0.025):.4f}, {np.quantile(draws, 0.975):.4f}]"
    )
    print(
        f"\nThe second is {(np.quantile(draws, 0.975) - np.quantile(draws, 0.025)) / (naive.monte_carlo_interval[1] - naive.monte_carlo_interval[0]):.0f}x wider "
        "and is the one to quote.\nSensitivity says which input to measure to narrow it."
    )


if __name__ == "__main__":
    main()
