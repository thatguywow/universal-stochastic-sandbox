"""The checks a reviewing statistician runs first.

Each test here corresponds to a defect that survived the earlier audits because
the code produced plausible numbers: an invalid p-value, an index with no error
bar, a non-converged chain, and a degenerate importance-sampling estimate.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from uss import fitting, metropolis_hastings, sobol_indices, variance
from uss.inference import effective_sample_size, gelman_rubin


# --------------------------------------------------------------------------
# Goodness-of-fit p-values must be calibrated under the null
# --------------------------------------------------------------------------
def test_naive_ks_pvalue_is_uncalibrated_and_says_so() -> None:
    """Documents the defect: fitted parameters invalidate the KS p-value."""
    rng = np.random.default_rng(0)
    pvalues = []
    for _ in range(120):
        x = rng.normal(10.0, 2.0, 150)
        pvalues.append(fitting.fit_family(x, "norm").ks_pvalue)
    # Under a valid test this would be ~0.5; the naive form runs far higher.
    assert np.mean(pvalues) > 0.7
    assert not fitting.fit_family(rng.normal(size=100), "norm").pvalue_is_valid


def test_bootstrap_ks_pvalue_is_calibrated_under_the_null() -> None:
    """The recalibrated p-value must be roughly Uniform(0,1) for a true model."""
    rng = np.random.default_rng(1)
    pvalues = [
        fitting.fit_family(
            rng.normal(10.0, 2.0, 120), "norm", n_bootstrap=99, rng=rng
        ).ks_pvalue
        for _ in range(120)
    ]
    arr = np.array(pvalues)
    assert 0.35 < arr.mean() < 0.65               # uniform mean is 0.5
    assert 0.0 <= np.mean(arr < 0.05) < 0.15      # nominal 5% rejection
    assert stats.kstest(arr, "uniform").pvalue > 0.01


def test_bootstrap_ks_still_rejects_a_genuinely_wrong_family() -> None:
    """Calibration must not come at the cost of power."""
    rng = np.random.default_rng(2)
    heavy_tailed = rng.standard_t(2.0, 400)
    res = fitting.fit_family(heavy_tailed, "norm", n_bootstrap=99, rng=rng)
    assert res.pvalue_is_valid
    assert res.ks_pvalue < 0.05


def test_best_fit_validates_only_the_winner() -> None:
    rng = np.random.default_rng(3)
    data = rng.lognormal(1.0, 0.5, 2_000)
    ranked = fitting.best_fit(data, families=["lognorm", "norm"], n_bootstrap=49, rng=rng)
    assert ranked[0].family == "lognorm"
    assert ranked[0].pvalue_is_valid and ranked[0].n_bootstrap > 0
    assert not ranked[1].pvalue_is_valid  # runner-up left uncalibrated, by design


def test_fit_result_marks_uncalibrated_pvalues_in_its_repr() -> None:
    res = fitting.fit_family(np.random.default_rng(4).normal(size=200), "norm")
    assert "(uncalibrated)" in str(res)


# --------------------------------------------------------------------------
# Sobol indices need error bars
# --------------------------------------------------------------------------
def test_sobol_reports_bootstrap_intervals() -> None:
    res = sobol_indices(
        lambda u: 3.0 * u[:, 0] + u[:, 1], 2, 4_000, np.random.default_rng(5)
    )
    assert res.total_effect_interval is not None
    assert res.first_order_interval is not None
    for i in range(2):
        lo, hi = res.total_effect_interval[i]
        assert 0.0 <= lo <= res.total_effect[i] <= hi <= 1.0


def test_sobol_intervals_cover_the_analytic_truth() -> None:
    """Coverage, measured over seeds -- the only honest test of an interval.

    Additive model y = 3*u0 + u1 has S1 = 9/10 and 1/10 exactly. Asserting on a
    single seed would be a coin flip near the boundary; what matters is that the
    intervals contain the truth about 95% of the time. Measured 92-94%, the mild
    under-coverage normal for a percentile bootstrap, with bias below 0.005.
    """
    truth = (0.9, 0.1)
    model = lambda u: 3.0 * u[:, 0] + u[:, 1]
    trials = 60
    hits = [0, 0]
    bias = [[], []]

    for seed in range(trials):
        res = sobol_indices(
            model, 2, 4_000, np.random.default_rng(1000 + seed), n_bootstrap=200
        )
        for i in range(2):
            lo, hi = res.first_order_interval[i]
            hits[i] += lo <= truth[i] <= hi
            bias[i].append(res.first_order[i] - truth[i])

    for i in range(2):
        assert hits[i] / trials > 0.85, f"index {i} coverage {hits[i] / trials:.0%}"
        assert abs(np.mean(bias[i])) < 0.02, f"index {i} biased by {np.mean(bias[i]):+.4f}"


def test_sobol_intervals_narrow_as_n_base_grows() -> None:
    def width(n):
        r = sobol_indices(
            lambda u: 3.0 * u[:, 0] + u[:, 1], 2, n, np.random.default_rng(7)
        )
        return r.total_effect_interval[0][1] - r.total_effect_interval[0][0]

    assert width(20_000) < width(1_000) / 2


def test_sobol_flags_an_unresolved_ranking() -> None:
    """Two near-equal inputs at small n must not be presented as ranked."""
    res = sobol_indices(
        lambda u: u[:, 0] + u[:, 1], 2, 300, np.random.default_rng(8)
    )
    assert not res.separates_top_two()
    assert "not resolved" in res.summary()


def test_sobol_confirms_a_resolved_ranking() -> None:
    res = sobol_indices(
        lambda u: 8.0 * u[:, 0] + u[:, 1], 2, 20_000, np.random.default_rng(9)
    )
    assert res.separates_top_two()
    assert "resolved at the stated confidence" in res.summary()


def test_sobol_bootstrap_can_be_disabled() -> None:
    res = sobol_indices(
        lambda u: u[:, 0], 1, 2_000, np.random.default_rng(10), n_bootstrap=0
    )
    assert res.total_effect_interval is None
    assert not res.separates_top_two()


# --------------------------------------------------------------------------
# MCMC convergence
# --------------------------------------------------------------------------
def _log_post(x: float) -> float:
    return -0.5 * ((x - 2.0) / 0.5) ** 2


def test_well_tuned_chain_passes_diagnostics() -> None:
    post = metropolis_hastings(
        _log_post, 0.0, 8_000, np.random.default_rng(11), proposal_scale=1.0
    )
    assert post.params["r_hat"] < 1.01
    assert post.params["ess"] > 200
    assert post.warnings == ()
    draws = np.asarray(post.params["draws"])
    assert draws.mean() == pytest.approx(2.0, abs=0.05)
    assert draws.std() == pytest.approx(0.5, rel=0.1)


def test_badly_tuned_chain_is_caught() -> None:
    """proposal_scale=0.005 gives a mean of ~0.19 against a truth of 2.0."""
    post = metropolis_hastings(
        _log_post, 0.0, 4_000, np.random.default_rng(12), proposal_scale=0.005
    )
    assert post.warnings, "a non-converged chain must not report clean"
    joined = " ".join(post.warnings)
    assert "R-hat" in joined or "effective sample size" in joined


def test_high_acceptance_rate_is_flagged() -> None:
    post = metropolis_hastings(
        _log_post, 2.0, 2_000, np.random.default_rng(13), proposal_scale=0.001
    )
    assert any("too small" in w or "R-hat" in w for w in post.warnings)


def test_effective_sample_size_detects_autocorrelation() -> None:
    rng = np.random.default_rng(14)
    independent = rng.normal(size=4_000)
    # AR(1) with strong persistence carries far less information.
    correlated = np.empty(4_000)
    correlated[0] = rng.normal()
    for i in range(1, 4_000):
        correlated[i] = 0.98 * correlated[i - 1] + rng.normal() * 0.2

    assert effective_sample_size(independent) > 2_000
    assert effective_sample_size(correlated) < 400


def test_effective_sample_size_edge_cases() -> None:
    assert effective_sample_size(np.array([1.0, 2.0])) == 2.0
    assert effective_sample_size(np.full(100, 3.0)) == 1.0


def test_gelman_rubin_detects_disagreeing_chains() -> None:
    rng = np.random.default_rng(15)
    agreeing = rng.normal(0.0, 1.0, (4, 2_000))
    disagreeing = agreeing.copy()
    disagreeing[0] += 8.0  # one chain stuck somewhere else entirely

    assert gelman_rubin(agreeing) < 1.01
    assert gelman_rubin(disagreeing) > 1.2


def test_gelman_rubin_needs_multiple_chains() -> None:
    assert np.isnan(gelman_rubin(np.random.default_rng(16).normal(size=(1, 500))))


def test_mcmc_validates_arguments() -> None:
    rng = np.random.default_rng(17)
    with pytest.raises(ValueError, match="n_chains must be at least 1"):
        metropolis_hastings(_log_post, 0.0, 100, rng, n_chains=0)
    with pytest.raises(ValueError, match="thin must be at least 1"):
        metropolis_hastings(_log_post, 0.0, 100, rng, thin=0)


# --------------------------------------------------------------------------
# Importance sampling degeneracy
# --------------------------------------------------------------------------
def test_degenerate_proposal_is_flagged() -> None:
    """A proposal that misses the target gives 0 with no hint anything is wrong."""
    est = variance.importance_sample(
        indicator=lambda x: (x > 6.0).astype(np.float64),
        log_target_pdf=lambda x: -0.5 * x**2,
        log_proposal_pdf=lambda x: -0.5 * (x - 1.0) ** 2,
        proposal_draws=np.random.default_rng(18).standard_normal(100_000) + 1.0,
    )
    assert est.warnings
    assert "not be trusted" in " ".join(est.warnings) or "0 by default" in " ".join(est.warnings)


def test_good_proposal_produces_no_warning() -> None:
    est = variance.importance_sample(
        indicator=lambda x: (x > 6.0).astype(np.float64),
        log_target_pdf=lambda x: -0.5 * x**2,
        log_proposal_pdf=lambda x: -0.5 * (x - 6.0) ** 2,
        proposal_draws=np.random.default_rng(19).standard_normal(200_000) + 6.0,
    )
    assert est.warnings == ()
    assert est.efficiency > 0.05
    assert est.value == pytest.approx(float(stats.norm.sf(6.0)), rel=0.05)


def test_efficiency_is_reported() -> None:
    est = variance.plain(np.random.default_rng(20).normal(size=1000))
    assert est.efficiency == pytest.approx(1.0)
