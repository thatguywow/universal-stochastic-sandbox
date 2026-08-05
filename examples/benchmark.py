"""Throughput benchmark against the blueprint's 10^7 events/second claim.

    python examples/benchmark.py
"""

from __future__ import annotations

import time

import numpy as np
from scipy import stats

from uss import UniversalStochasticSandbox, distributions

N = 10_000_000


def timed(label: str, fn, *args, **kwargs) -> float:
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    rate = N / elapsed
    print(f"{label:<44}{elapsed:>8.3f}s{rate:>16,.0f} draws/s")
    return elapsed


def main() -> None:
    print(f"n = {N:,} draws per query\n")
    print(f"{'operation':<44}{'time':>9}{'throughput':>17}")
    print("-" * 70)

    rng = np.random.default_rng(42)
    u = rng.random(N)

    timed("uniform core generation", rng.random, N)
    timed("bernoulli  (inverse CDF)", distributions.get("bernoulli").sample, u, probability=0.31)
    timed("gaussian   (inverse CDF)", distributions.get("gaussian").sample, u, mean=0.0, std_dev=1.0)
    timed("gumbel     (inverse CDF)", distributions.get("extreme_value").sample, u, loc=0.0, scale=1.0)
    timed("exponential(inverse CDF)", distributions.get("exponential").sample, u, rate=1.0)

    print("-" * 70)
    print("Poisson, lambda = 5e-6:")
    t_fast = timed("  searchsorted CDF table (this engine)", distributions.get("poisson").sample, u, lam=5e-6)
    t_slow = timed("  scipy.stats.poisson.ppf (blueprint)", stats.poisson.ppf, u, mu=5e-6)
    print(f"{'  speedup':<44}{t_slow / t_fast:>8.1f}x")

    print("-" * 70)
    print("Poisson, lambda = 25:")
    t_fast = timed("  searchsorted CDF table (this engine)", distributions.get("poisson").sample, u, lam=25.0)
    t_slow = timed("  scipy.stats.poisson.ppf (blueprint)", stats.poisson.ppf, u, mu=25.0)
    print(f"{'  speedup':<44}{t_slow / t_fast:>8.1f}x")

    print("-" * 70)
    sandbox = UniversalStochasticSandbox(sample_size=N, seed=42)
    for qc, params in [
        ("bernoulli", {"probability": 0.31}),
        ("poisson", {"lambda": 5e-6}),
        ("gaussian", {"mean": 20.0, "std_dev": 3.0}),
        ("extreme_value", {"loc": 0.0, "scale": 1.0}),
    ]:
        r = sandbox.execute_query(qc, params)
        print(
            f"full pipeline: {qc:<29}{r.elapsed_seconds:>8.3f}s"
            f"{N / r.elapsed_seconds:>16,.0f} draws/s"
        )

    print("-" * 70)
    print(f"blueprint target: {10**7:,} events/second")


if __name__ == "__main__":
    main()
