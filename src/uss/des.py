"""Discrete Event Simulation jump sampling (blueprint Part II.2).

The clock never ticks.  Inter-arrival times are drawn analytically from the
exponential distribution,

    dt = -ln(U) / lambda

so the simulation steps directly from one event to the next and spends no
compute on empty intervals.  For a rate of 5e-6 per unit time this is the
difference between 200,000 idle iterations per event and one array operation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventStream:
    """Event timestamps produced by a jump-sampling run."""

    timestamps: np.ndarray
    horizon: float
    rate: float | Callable[[np.ndarray], np.ndarray]

    @property
    def count(self) -> int:
        return int(self.timestamps.size)

    @property
    def empirical_rate(self) -> float:
        return self.count / self.horizon if self.horizon > 0 else 0.0


def jump_sample(
    rate: float,
    horizon: float,
    rng: np.random.Generator,
    *,
    block_size: int = 4096,
) -> EventStream:
    """Homogeneous Poisson process over [0, horizon) by exponential jumps.

    Draws inter-arrival times in blocks and accumulates until the horizon is
    crossed, so the cost scales with the number of *events*, not the length of
    the interval.
    """
    if rate < 0:
        raise ValueError(f"rate must be non-negative, got {rate}")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if rate == 0:
        return EventStream(np.empty(0, dtype=np.float64), horizon, rate)

    chunks: list[np.ndarray] = []
    elapsed = 0.0
    # Expected event count sets a sensible first block; at least `block_size`.
    n = max(block_size, int(rate * horizon * 1.1) + 16)

    while True:
        u = rng.random(n)
        gaps = -np.log1p(-u) / rate
        times = elapsed + np.cumsum(gaps)
        if times.size and times[-1] >= horizon:
            keep = times[times < horizon]
            chunks.append(keep)
            break
        chunks.append(times)
        elapsed = float(times[-1]) if times.size else elapsed
        n = block_size

    stamps = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
    return EventStream(stamps, horizon, rate)


def thinned_sample(
    rate_fn: Callable[[np.ndarray], np.ndarray],
    rate_max: float,
    horizon: float,
    rng: np.random.Generator,
) -> EventStream:
    """Non-homogeneous Poisson process via Lewis-Shedler thinning.

    Generates a homogeneous stream at the dominating rate `rate_max`, then keeps
    each candidate with probability lambda(t) / rate_max.  This covers queries
    where intensity varies over time -- diurnal foot traffic, seasonal demand,
    a heatwave building through an afternoon.
    """
    if rate_max <= 0:
        raise ValueError(f"rate_max must be positive, got {rate_max}")

    candidates = jump_sample(rate_max, horizon, rng).timestamps
    if candidates.size == 0:
        return EventStream(candidates, horizon, rate_fn)

    intensities = np.asarray(rate_fn(candidates), dtype=np.float64)
    if np.any(intensities < 0):
        raise ValueError("rate_fn returned a negative intensity")
    if np.any(intensities > rate_max * (1 + 1e-9)):
        raise ValueError(
            "rate_fn exceeded rate_max; the dominating rate must bound lambda(t)"
        )

    keep = rng.random(candidates.size) < (intensities / rate_max)
    return EventStream(candidates[keep], horizon, rate_fn)


def next_event_times(
    rate: float, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Vectorised inter-arrival gaps: the raw dt = -ln(U)/lambda transform."""
    if rate <= 0:
        raise ValueError(f"rate must be positive, got {rate}")
    return -np.log1p(-rng.random(size)) / rate


def time_to_first_event(
    rate: float, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Waiting time until the first occurrence, drawn `size` times.

    Answers the 'how long until X happens' form of a rare-event query directly,
    without simulating the intervening emptiness.
    """
    return next_event_times(rate, size, rng)
