"""Behavioral economics layer (blueprint Part III.1).

These operators turn contextual covariates into the effective probability that
the Bernoulli sampler consumes.  The blueprint's Part I specifies a logistic
link,

    P_i = 1 / (1 + exp(-(beta_0 + sum_k beta_k X_k)))

which is the form implemented here.  Part V's script instead used
`clip(p_base * bias_multiplier, 0, 1)`; that shortcut is not equivalent and is
not offered, because multiplying a probability leaves the [0, 1] range for any
p_base > 1/multiplier and then clips -- collapsing every strong-signal case to
certainty regardless of covariate magnitude.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Kahneman-Tversky loss aversion coefficient.
LOSS_AVERSION_LAMBDA = 2.25


def logistic_link(
    intercept: float,
    coefficients: Sequence[float] | np.ndarray,
    covariates: Sequence[float] | np.ndarray,
) -> float:
    """P = 1 / (1 + exp(-(beta_0 + sum beta_k X_k))).

    `covariates` may be a 1-D vector (one scenario) or 2-D (n_scenarios,
    n_features), in which case a vector of probabilities is returned.
    """
    beta = np.atleast_1d(np.asarray(coefficients, dtype=np.float64))
    x = np.asarray(covariates, dtype=np.float64)

    if x.ndim == 1:
        if x.shape != beta.shape:
            raise ValueError(
                f"covariates {x.shape} incompatible with coefficients {beta.shape}"
            )
        linear = float(intercept) + float(np.dot(beta, x))
    elif x.ndim == 2:
        if x.shape[1] != beta.size:
            raise ValueError(
                f"covariates has {x.shape[1]} features, coefficients has {beta.size}"
            )
        linear = float(intercept) + x @ beta
    else:
        raise ValueError("covariates must be 1-D or 2-D")

    # expit is overflow-safe at large |linear|, unlike a bare 1/(1+exp(-z)).
    from scipy.special import expit

    return expit(linear)


def odds_ratio_shift(base_probability: float, odds_ratio: float) -> float:
    """Apply a multiplicative effect on the *odds* scale, not the probability.

    This is the correct way to express "a heatwave makes this 1.15x more
    likely": the result stays inside (0, 1) for every positive odds ratio, so
    no clipping is ever required.
    """
    p = float(base_probability)
    if not 0.0 < p < 1.0:
        raise ValueError(f"base_probability must lie strictly in (0, 1), got {p}")
    if odds_ratio <= 0:
        raise ValueError(f"odds_ratio must be positive, got {odds_ratio}")
    odds = p / (1.0 - p) * float(odds_ratio)
    return odds / (1.0 + odds)


def apply_loss_aversion(
    outcomes: np.ndarray, lambda_loss: float = LOSS_AVERSION_LAMBDA
) -> np.ndarray:
    """Scale negative outcomes by lambda_loss, leaving gains untouched."""
    if lambda_loss < 0:
        raise ValueError(f"lambda_loss must be non-negative, got {lambda_loss}")
    arr = np.asarray(outcomes, dtype=np.float64)
    return np.where(arr < 0, arr * float(lambda_loss), arr)


def prospect_value(
    outcomes: np.ndarray,
    alpha: float = 0.88,
    beta: float = 0.88,
    lambda_loss: float = LOSS_AVERSION_LAMBDA,
) -> np.ndarray:
    """Kahneman-Tversky value function with diminishing sensitivity.

    v(x) = x^alpha for gains, -lambda * (-x)^beta for losses.
    """
    arr = np.asarray(outcomes, dtype=np.float64)
    gains = np.power(np.clip(arr, 0.0, None), alpha)
    losses = -float(lambda_loss) * np.power(np.clip(-arr, 0.0, None), beta)
    return np.where(arr >= 0, gains, losses)


def social_proof_cascade(
    initial_probability: np.ndarray | float,
    alpha: float,
    steps: int,
    rng: np.random.Generator,
    *,
    population: int | None = None,
) -> np.ndarray:
    """P_i(t+1) = P_i(0) + alpha * (N_active / N_total), iterated `steps` times.

    Returns the adoption fraction at each step (length `steps + 1`, including
    the initial state).  Adoption is resampled each step from the current
    probability, so the trajectory carries genuine stochastic variation rather
    than tracing a deterministic logistic curve.
    """
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    p0 = np.asarray(initial_probability, dtype=np.float64)
    if p0.ndim == 0:
        if population is None:
            raise ValueError(
                "population is required when initial_probability is a scalar"
            )
        p0 = np.full(population, float(p0))
    n_total = p0.size

    active = rng.random(n_total) < p0
    trajectory = np.empty(steps + 1, dtype=np.float64)
    trajectory[0] = active.mean()

    for t in range(steps):
        adoption_rate = active.mean()
        p_t = np.clip(p0 + float(alpha) * adoption_rate, 0.0, 1.0)
        active = rng.random(n_total) < p_t
        trajectory[t + 1] = active.mean()

    return trajectory


def hyperbolic_discount(
    value: np.ndarray | float, delay: np.ndarray | float, k: float = 1.0
) -> np.ndarray:
    """V = A / (1 + k * t) -- present comfort over future utility."""
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    t = np.asarray(delay, dtype=np.float64)
    if np.any(t < 0):
        raise ValueError("delay must be non-negative")
    return np.asarray(value, dtype=np.float64) / (1.0 + float(k) * t)


def multinomial_logit(
    utilities: np.ndarray,
) -> np.ndarray:
    """Softmax choice probabilities across mutually exclusive alternatives.

    Backs the "Aggregated Macro Behavior" row of the Part IV confidence table,
    where the governing equation is a multinomial logit.
    """
    u = np.asarray(utilities, dtype=np.float64)
    shifted = u - np.max(u, axis=-1, keepdims=True)
    exp_u = np.exp(shifted)
    return exp_u / np.sum(exp_u, axis=-1, keepdims=True)
