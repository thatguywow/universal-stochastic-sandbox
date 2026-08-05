"""Regressions for the second bug sweep (graph determinism, binning, reporting)."""

from __future__ import annotations

import numpy as np
import pytest

from uss import ScenarioGraph, UniversalStochasticSandbox, webui


# --------------------------------------------------------------------------
# ScenarioGraph.run must be idempotent
# --------------------------------------------------------------------------
def test_repeated_run_on_the_same_graph_is_identical() -> None:
    """A stored SeedSequence advanced its spawn counter, so run() drifted."""
    g = ScenarioGraph(seed=42)
    g.add("a", "gaussian", mean=0.0, std_dev=1.0)
    g.add("b", "bernoulli", probability=0.3)

    first, second, third = g.run(5_000), g.run(5_000), g.run(5_000)
    assert np.array_equal(first["a"], second["a"])
    assert np.array_equal(second["a"], third["a"])
    assert np.array_equal(first["b"], second["b"])


def test_replicate_gives_independent_draws() -> None:
    g = ScenarioGraph(seed=42)
    g.add("a", "gaussian", mean=0.0, std_dev=1.0)

    base = g.run(20_000)["a"]
    rep1 = g.run(20_000, replicate=1)["a"]
    rep2 = g.run(20_000, replicate=2)["a"]

    assert not np.array_equal(base, rep1)
    assert not np.array_equal(rep1, rep2)
    # Independent, but the same distribution.
    for arr in (base, rep1, rep2):
        assert arr.mean() == pytest.approx(0.0, abs=0.03)
    assert abs(np.corrcoef(base, rep1)[0, 1]) < 0.05


def test_replicate_is_itself_reproducible() -> None:
    def build():
        g = ScenarioGraph(seed=7)
        g.add("a", "gaussian", mean=0.0)
        return g

    assert np.array_equal(build().run(3_000, replicate=4)["a"],
                          build().run(3_000, replicate=4)["a"])


def test_negative_replicate_rejected() -> None:
    g = ScenarioGraph(seed=1)
    g.add("a", "gaussian", mean=0.0)
    with pytest.raises(ValueError, match="replicate must be non-negative"):
        g.run(100, replicate=-1)


def test_seedless_graph_still_runs_and_replicates() -> None:
    g = ScenarioGraph(seed=None)
    g.add("a", "gaussian", mean=0.0)
    assert g.run(1_000)["a"].size == 1_000
    assert not np.array_equal(g.run(1_000)["a"], g.run(1_000, replicate=1)["a"])


def test_run_stays_deterministic_after_appending_a_node() -> None:
    g = ScenarioGraph(seed=42)
    g.add("a", "gaussian", mean=0.0)
    before = g.run(4_000)["a"]
    g.add("b", "gaussian", mean=5.0)
    after = g.run(4_000)["a"]
    assert np.array_equal(before, after)


# --------------------------------------------------------------------------
# Histogram binning
# --------------------------------------------------------------------------
def test_continuous_data_with_integral_max_is_not_binned_as_integers() -> None:
    """A uniform on [0,10] whose max lands on 10.0 was drawn as 11 integer bars."""
    arr = np.concatenate([np.random.default_rng(0).random(999) * 9.0, [10.0]])
    out = webui._histogram(arr, "continuous")
    assert out["type"] == "histogram"
    assert len(out["counts"]) == webui.MAX_HIST_BINS
    assert sum(out["counts"]) == arr.size


def test_genuinely_integral_data_keeps_integer_bins() -> None:
    arr = np.random.default_rng(1).poisson(4.0, 5_000).astype(np.float64)
    out = webui._histogram(arr, "continuous")
    assert len(out["counts"]) == int(arr.max() - arr.min()) + 1
    assert sum(out["counts"]) == arr.size


def test_count_kind_always_uses_integer_bins() -> None:
    arr = np.random.default_rng(2).poisson(3.0, 2_000)
    out = webui._histogram(arr, "count")
    assert out["width"] == pytest.approx(1.0)


def test_degenerate_and_two_point_data() -> None:
    flat = webui._histogram(np.full(400, 3.0), "continuous")
    assert sum(flat["counts"]) == 400
    two = webui._histogram(np.array([1.0] * 50 + [2.0] * 50), "continuous")
    assert sum(two["counts"]) == 100


def test_proportion_kind_is_categorical() -> None:
    arr = (np.random.default_rng(3).random(1_000) < 0.4).astype(np.int8)
    out = webui._histogram(arr, "proportion")
    assert out["type"] == "categorical"
    assert sum(out["counts"]) == 1_000


# --------------------------------------------------------------------------
# Antithetic sample-size reporting
# --------------------------------------------------------------------------
def test_odd_sample_size_reports_what_actually_ran() -> None:
    """n=1001 evaluated 1000 draws but reported 1001."""
    r = UniversalStochasticSandbox(sample_size=1001, seed=1).execute_query(
        "gaussian", {"mean": 0.0}, antithetic=True
    )
    assert r.sample_size == 1000


def test_even_sample_size_unchanged() -> None:
    r = UniversalStochasticSandbox(sample_size=1000, seed=1).execute_query(
        "gaussian", {"mean": 0.0}, antithetic=True
    )
    assert r.sample_size == 1000


def test_antithetic_rejects_sample_size_that_floors_below_a_pair() -> None:
    sb = UniversalStochasticSandbox(sample_size=3, seed=1)
    # sample_size=1 floors to 0 draws -- no pair exists, so refuse rather than
    # return an estimate built from nothing.
    with pytest.raises(ValueError, match="at least 2"):
        sb.execute_query("gaussian", {"mean": 0.0}, antithetic=True, sample_size=1)
    # n=3 floors to a single valid pair and must still work.
    assert sb.execute_query("gaussian", {"mean": 0.0}, antithetic=True).sample_size == 2
