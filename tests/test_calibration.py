"""Calibration diagnostics must flag miscalibration, not just pass everything."""

from __future__ import annotations

import numpy as np
import pytest

from uss import calibration, wilson_interval


def test_correct_procedure_is_reported_calibrated() -> None:
    """A properly-constructed normal interval should track the diagonal."""

    def trial(level: float, rng: np.random.Generator):
        from scipy import stats

        truth = 5.0
        n = 200
        sample = rng.normal(truth, 2.0, n)
        z = stats.t.ppf((1 + level) / 2, df=n - 1)
        half = z * sample.std(ddof=1) / np.sqrt(n)
        return (sample.mean() - half, sample.mean() + half), truth

    res = calibration.coverage_curve(trial, 600, np.random.default_rng(41))
    assert res.verdict == "calibrated"
    assert res.calibration_error < 0.03


def test_overconfident_procedure_is_caught() -> None:
    """Intervals deliberately shrunk by 3x must be flagged, not passed."""

    def trial(level: float, rng: np.random.Generator):
        from scipy import stats

        truth = 5.0
        n = 200
        sample = rng.normal(truth, 2.0, n)
        z = stats.t.ppf((1 + level) / 2, df=n - 1)
        half = z * sample.std(ddof=1) / np.sqrt(n) / 3.0  # too narrow
        return (sample.mean() - half, sample.mean() + half), truth

    res = calibration.coverage_curve(trial, 400, np.random.default_rng(42))
    assert res.verdict == "overconfident"
    assert res.calibration_error > 0.1


def test_conservative_procedure_is_caught() -> None:
    def trial(level: float, rng: np.random.Generator):
        truth = 5.0
        sample = rng.normal(truth, 2.0, 200)
        half = 5.0  # absurdly wide
        return (sample.mean() - half, sample.mean() + half), truth

    res = calibration.coverage_curve(trial, 200, np.random.default_rng(43))
    assert res.verdict == "conservative"


def test_wilson_interval_coverage_via_the_harness() -> None:
    """Exercise the engine's own interval through the calibration path."""

    def trial(level: float, rng: np.random.Generator):
        p, n = 0.3, 400
        k = int(rng.binomial(n, p))
        return wilson_interval(k, n, level), p

    res = calibration.coverage_curve(
        trial, 500, np.random.default_rng(44), levels=(0.8, 0.95)
    )
    assert res.empirical_coverage[0] == pytest.approx(0.8, abs=0.05)
    assert res.empirical_coverage[1] == pytest.approx(0.95, abs=0.035)


def test_pit_uniform_for_a_correct_predictive_distribution() -> None:
    rng = np.random.default_rng(45)
    n_obs, n_draws = 2000, 500
    truth = rng.normal(0.0, 1.0, n_obs)
    preds = rng.normal(0.0, 1.0, (n_obs, n_draws))

    res = calibration.pit_report(preds, truth)
    assert res.verdict == "calibrated"
    assert res.ks_pvalue > 0.05


def test_pit_detects_under_dispersed_forecasts() -> None:
    """Forecast sd too small => observations land in the tails => U-shaped PIT."""
    rng = np.random.default_rng(46)
    n_obs, n_draws = 2000, 500
    truth = rng.normal(0.0, 1.0, n_obs)
    preds = rng.normal(0.0, 0.3, (n_obs, n_draws))  # far too confident

    res = calibration.pit_report(preds, truth)
    assert res.ks_pvalue < 0.01
    assert "under-dispersed" in res.verdict


def test_pit_detects_over_dispersed_forecasts() -> None:
    rng = np.random.default_rng(47)
    n_obs, n_draws = 2000, 500
    truth = rng.normal(0.0, 1.0, n_obs)
    preds = rng.normal(0.0, 4.0, (n_obs, n_draws))  # far too vague

    res = calibration.pit_report(preds, truth)
    assert res.ks_pvalue < 0.01
    assert "over-dispersed" in res.verdict


def test_pit_detects_bias() -> None:
    rng = np.random.default_rng(48)
    truth = rng.normal(2.0, 1.0, 2000)
    preds = rng.normal(0.0, 1.0, (2000, 400))  # centred wrong
    res = calibration.pit_report(preds, truth)
    assert res.ks_pvalue < 0.01


def test_pit_shape_validation() -> None:
    with pytest.raises(ValueError, match="predictive rows"):
        calibration.pit_values(np.zeros((5, 10)), np.zeros(6))
    with pytest.raises(ValueError, match="must be 2-D"):
        calibration.pit_values(np.zeros(10), np.zeros(10))


def test_pit_histogram_bins_sum_to_n() -> None:
    rng = np.random.default_rng(49)
    res = calibration.pit_report(rng.normal(size=(500, 200)), rng.normal(size=500))
    counts, edges = res.histogram(bins=10)
    assert counts.sum() == 500
    assert edges.size == 11
    assert "PIT over" in res.summary()


def test_interval_score_prefers_narrow_intervals_that_still_cover() -> None:
    truth = np.zeros(1000)
    tight = calibration.interval_score(np.full(1000, -1.0), np.full(1000, 1.0), truth)
    wide = calibration.interval_score(np.full(1000, -5.0), np.full(1000, 5.0), truth)
    assert tight < wide


def test_interval_score_penalises_misses() -> None:
    truth = np.zeros(1000)
    covering = calibration.interval_score(np.full(1000, -1.0), np.full(1000, 1.0), truth)
    missing = calibration.interval_score(np.full(1000, 2.0), np.full(1000, 3.0), truth)
    assert missing > covering * 10


def test_interval_score_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="below lower bound"):
        calibration.interval_score(np.ones(3), np.zeros(3), np.zeros(3))


def test_sharpness_is_mean_width() -> None:
    assert calibration.sharpness(np.zeros(10), np.full(10, 3.0)) == pytest.approx(3.0)


def test_coverage_rejects_invalid_levels() -> None:
    with pytest.raises(ValueError, match="strictly in"):
        calibration.coverage_curve(
            lambda lvl, rng: ((0.0, 1.0), 0.5), 10, np.random.default_rng(50), levels=(1.5,)
        )
