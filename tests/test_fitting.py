"""Empirical fitting from real files."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from uss import fitting


@pytest.fixture
def normal_csv(tmp_path):
    data = np.random.default_rng(61).normal(20.0, 3.0, 20_000)
    path = tmp_path / "temps.csv"
    pl.DataFrame({"temperature_c": data, "station": ["a"] * data.size}).write_csv(path)
    return path


@pytest.fixture
def lognormal_parquet(tmp_path):
    data = np.random.default_rng(62).lognormal(1.0, 0.5, 20_000)
    path = tmp_path / "durations.parquet"
    pl.DataFrame({"duration_s": data}).write_parquet(path)
    return path


def test_load_column_from_csv(normal_csv) -> None:
    values = fitting.load_column(normal_csv, "temperature_c")
    assert values.size == 20_000
    assert values.mean() == pytest.approx(20.0, abs=0.1)


def test_load_column_from_parquet(lognormal_parquet) -> None:
    values = fitting.load_column(lognormal_parquet, "duration_s")
    assert values.size == 20_000
    assert np.all(values > 0)


def test_load_column_rejects_missing_column(normal_csv) -> None:
    with pytest.raises(ValueError, match="not in"):
        fitting.load_column(normal_csv, "humidity")


def test_load_column_rejects_unsupported_suffix(tmp_path) -> None:
    path = tmp_path / "data.xlsx"
    path.write_text("nope")
    with pytest.raises(ValueError, match="unsupported file type"):
        fitting.load_column(path, "col")


def test_load_column_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        fitting.load_column("no_such_file.csv", "col")


def test_load_column_drops_nulls_and_non_finite(tmp_path) -> None:
    path = tmp_path / "gappy.csv"
    pl.DataFrame({"x": [1.0, None, 3.0, float("inf"), 5.0]}).write_csv(path)
    values = fitting.load_column(path, "x")
    assert values.size == 3
    assert np.all(np.isfinite(values))


def test_best_fit_identifies_the_generating_family(normal_csv) -> None:
    data = fitting.load_column(normal_csv, "temperature_c")
    ranked = fitting.best_fit(data, families=["norm", "expon", "gumbel_r"])
    assert ranked[0].family == "norm"
    assert ranked[0].ks_pvalue > 0.01


def test_best_fit_recovers_parameters(normal_csv) -> None:
    data = fitting.load_column(normal_csv, "temperature_c")
    best = fitting.best_fit(data, families=["norm"])[0]
    assert best.parameters["mean"] == pytest.approx(20.0, abs=0.1)
    assert best.parameters["std_dev"] == pytest.approx(3.0, abs=0.1)
    assert best.query_class == "gaussian"


def test_best_fit_on_lognormal_data(lognormal_parquet) -> None:
    data = fitting.load_column(lognormal_parquet, "duration_s")
    ranked = fitting.best_fit(data, families=["lognorm", "norm", "expon"])
    assert ranked[0].family == "lognorm"
    assert ranked[0].parameters["mean"] == pytest.approx(1.0, abs=0.1)


def test_best_fit_skips_families_that_fail(normal_csv) -> None:
    """A family that cannot fit must not abort the whole comparison."""
    data = fitting.load_column(normal_csv, "temperature_c")
    ranked = fitting.best_fit(data)  # includes families invalid for negative data
    assert len(ranked) >= 1
    assert all(np.isfinite(r.aic) for r in ranked)


def test_empirical_quantiles_round_trip_through_sampler() -> None:
    source = np.random.default_rng(63).gamma(2.0, 3.0, 100_000)
    q = fitting.empirical_quantiles(source)

    from uss import distributions

    u = np.random.default_rng(64).random(200_000)
    resampled = distributions.get("empirical").sample(u, quantiles=q)
    assert resampled.mean() == pytest.approx(source.mean(), rel=0.02)
    assert np.quantile(resampled, 0.9) == pytest.approx(np.quantile(source, 0.9), rel=0.03)


def test_empirical_quantiles_requires_data() -> None:
    with pytest.raises(ValueError, match="at least 2 observations"):
        fitting.empirical_quantiles(np.array([1.0]))


def test_fit_file_returns_ranking_and_ecdf(normal_csv) -> None:
    ranked, ecdf = fitting.fit_file(normal_csv, "temperature_c", families=["norm", "expon"])
    assert ranked[0].family == "norm"
    assert ecdf.size == 4096
    assert np.all(np.diff(ecdf) >= 0)
