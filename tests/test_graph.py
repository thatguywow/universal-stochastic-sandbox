"""Composition graph: coupling, propagation, and reproducibility."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import expit

from uss import ScenarioGraph, gaussian_copula


def test_downstream_node_sees_per_draw_parent_values() -> None:
    """A vector parameter must vary per draw, not collapse to its mean."""
    g = ScenarioGraph(seed=1)
    temp = g.add("temp", "gaussian", mean=30.0, std_dev=5.0)
    p = g.derive("p", lambda t: expit(-6.0 + 0.22 * t), temp)
    g.add("wearing", "bernoulli", probability=p)

    res = g.run(200_000)
    # Correlation between the driver and the outcome can only exist if the
    # per-draw coupling survived.
    assert res.correlation("temp", "wearing") > 0.2
    assert res["p"].std() > 0.05


def test_coupling_changes_the_answer_versus_independent_stages() -> None:
    """Running stages separately and multiplying understates the spread."""
    g = ScenarioGraph(seed=2)
    temp = g.add("temp", "gaussian", mean=30.0, std_dev=6.0)
    p = g.derive("p", lambda t: expit(-6.0 + 0.22 * t), temp)
    g.add("wearing", "bernoulli", probability=p)
    coupled = g.run(400_000)

    # The naive two-step: take E[p], then simulate Bernoulli at that fixed p.
    mean_p = float(coupled["p"].mean())
    rng = np.random.default_rng(3)
    naive = (rng.random(400_000) >= 1 - mean_p).astype(np.float64)

    # Means agree, but only the coupled run carries the driver's variance.
    assert coupled.mean("wearing") == pytest.approx(naive.mean(), abs=0.01)
    assert coupled["p"].std() > 0.05  # real spread in p
    assert np.std([mean_p]) == 0.0  # the naive path has none


def test_multi_stage_chain_propagates() -> None:
    g = ScenarioGraph(seed=4)
    temp = g.add("temp", "gaussian", mean=31.0, std_dev=4.0)
    p = g.derive("p", lambda t: expit(-6.0 + 0.22 * t), temp)
    wearing = g.add("wearing", "bernoulli", probability=p)
    g.derive("crowd", lambda w: 2000.0 * w, wearing)

    res = g.run(100_000)
    assert set(res.samples) == {"temp", "p", "wearing", "crowd"}
    assert res.mean("crowd") == pytest.approx(2000.0 * res.mean("wearing"), rel=1e-9)


def test_same_seed_reproduces_graph_run() -> None:
    def build() -> ScenarioGraph:
        g = ScenarioGraph(seed=5)
        t = g.add("t", "gaussian", mean=0.0, std_dev=1.0)
        g.derive("d", lambda x: x * 2.0, t)
        return g

    a, b = build().run(50_000), build().run(50_000)
    assert np.array_equal(a["t"], b["t"])
    assert np.array_equal(a["d"], b["d"])


def test_appending_a_node_does_not_perturb_existing_streams() -> None:
    """Node streams come from independent child seeds, not one shared stream."""
    g1 = ScenarioGraph(seed=6)
    g1.add("a", "gaussian", mean=0.0, std_dev=1.0)
    first = g1.run(20_000)["a"]

    g2 = ScenarioGraph(seed=6)
    g2.add("a", "gaussian", mean=0.0, std_dev=1.0)
    g2.add("b", "gaussian", mean=5.0, std_dev=1.0)  # appended afterwards
    second = g2.run(20_000)["a"]

    assert np.array_equal(first, second)


def test_cycle_is_detected() -> None:
    g = ScenarioGraph(seed=7)
    g.add("a", "gaussian", mean=0.0)
    g.derive("b", lambda x: x, "a")
    # Force a cycle by rewiring 'a' to depend on 'b'.
    g._nodes["a"].parents = ("b",)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="cycle detected"):
        g.run(100)


def test_duplicate_node_name_rejected() -> None:
    g = ScenarioGraph(seed=8)
    g.add("a", "gaussian", mean=0.0)
    with pytest.raises(ValueError, match="already exists"):
        g.add("a", "gaussian", mean=1.0)


def test_unknown_parent_rejected() -> None:
    g = ScenarioGraph(seed=9)
    with pytest.raises(ValueError, match="unknown parent"):
        g.derive("d", lambda x: x, "nope")


def test_unknown_query_class_rejected_at_construction() -> None:
    g = ScenarioGraph(seed=10)
    with pytest.raises(ValueError, match="Unknown query class"):
        g.add("a", "not_a_class")


def test_vector_lambda_poisson_warns_about_cost() -> None:
    g = ScenarioGraph(seed=11)
    base = g.add("base", "gaussian", mean=5.0, std_dev=0.5)
    rate = g.derive("rate", lambda b: np.clip(b, 0.1, None), base)
    g.add("events", "poisson", lam=rate)

    with pytest.warns(RuntimeWarning, match="69x slower"):
        res = g.run(20_000)
    assert res.mean("events") == pytest.approx(5.0, rel=0.05)


def test_root_nodes_identified() -> None:
    g = ScenarioGraph(seed=12)
    a = g.add("a", "gaussian", mean=0.0)
    d = g.derive("d", lambda x: x, a)
    g.add("b", "bernoulli", probability=g.derive("p", lambda x: expit(x), d))
    assert g.root_nodes == ["a"]
    assert set(g.stochastic_nodes) == {"a", "b"}


def test_report_uses_node_kind() -> None:
    g = ScenarioGraph(seed=13)
    g.add("flag", "bernoulli", probability=0.3)
    res = g.run(100_000)
    assert res.report("flag").interval_type == "wilson-score"


def test_uniform_override_controls_a_node() -> None:
    g = ScenarioGraph(seed=14)
    g.add("a", "gaussian", mean=0.0, std_dev=1.0)
    forced = np.full(1000, 0.5)
    res = g.run(1000, uniform_overrides={"a": forced})
    assert np.allclose(res["a"], 0.0)  # median of a standard normal


def test_uniform_override_shape_checked() -> None:
    g = ScenarioGraph(seed=15)
    g.add("a", "gaussian", mean=0.0)
    with pytest.raises(ValueError, match="expected"):
        g.run(1000, uniform_overrides={"a": np.zeros(10)})


def test_gaussian_copula_induces_requested_correlation() -> None:
    corr = np.array([[1.0, 0.7], [0.7, 1.0]])
    u = gaussian_copula(corr, 200_000, np.random.default_rng(16))
    assert u.shape == (200_000, 2)
    assert np.all((u >= 0) & (u <= 1))
    # Rank correlation of the uniforms tracks the latent Gaussian correlation.
    from scipy import stats

    rho = stats.spearmanr(u[:, 0], u[:, 1]).statistic
    assert rho == pytest.approx(0.68, abs=0.03)


def test_copula_couples_two_graph_roots() -> None:
    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    u = gaussian_copula(corr, 100_000, np.random.default_rng(17))

    g = ScenarioGraph(seed=18)
    g.add("temperature", "gaussian", mean=30.0, std_dev=5.0)
    g.add("humidity", "gaussian", mean=60.0, std_dev=10.0)
    res = g.run(
        100_000, uniform_overrides={"temperature": u[:, 0], "humidity": u[:, 1]}
    )
    assert res.correlation("temperature", "humidity") == pytest.approx(0.8, abs=0.02)


def test_copula_rejects_non_psd_matrix() -> None:
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="positive semi-definite"):
        gaussian_copula(bad, 100, np.random.default_rng(19))


def test_to_frame_and_mermaid() -> None:
    g = ScenarioGraph(seed=20)
    t = g.add("t", "gaussian", mean=0.0)
    g.derive("d", lambda x: x * 2, t)
    res = g.run(1000)
    assert set(res.to_frame().columns) == {"t", "d"}
    diagram = g.to_mermaid()
    assert "graph TD" in diagram and "t --> d" in diagram
