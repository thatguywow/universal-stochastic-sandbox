"""Web interface: endpoints exercised against a live server on a real socket."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from uss import webui


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), webui._Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=60) as r:
        return json.loads(r.read())


def post(base: str, path: str, payload: dict):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def post_expect_error(base: str, path: str, payload: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_index_page_served(server) -> None:
    with urllib.request.urlopen(server + "/", timeout=30) as r:
        body = r.read().decode()
    assert r.status == 200
    assert "Universal Stochastic Sandbox" in body
    assert "/api/query" in body


def test_classes_endpoint_describes_every_registered_class(server) -> None:
    data = get(server, "/api/classes")
    names = {c["name"] for c in data["classes"]}
    assert {"bernoulli", "poisson", "gaussian", "extreme_value"} <= names
    gaussian = next(c for c in data["classes"] if c["name"] == "gaussian")
    assert {p["name"] for p in gaussian["params"]} == {"mean", "std_dev"}
    assert gaussian["kind"] == "continuous"
    assert set(data["domains"]) == {
        "closed_physical", "macro_behavioral", "complex_network", "quantum_anomaly",
    }


def test_query_endpoint_returns_a_correct_estimate(server) -> None:
    r = post(server, "/api/query", {
        "query_class": "gaussian",
        "parameters": {"mean": 20.0, "std_dev": 3.0},
        "sample_size": 200_000,
    })
    assert r["point_estimate"] == pytest.approx(20.0, abs=0.05)
    assert r["interval_type"] == "student-t"
    lo, hi = r["monte_carlo_interval"]
    assert lo < 20.0 < hi
    assert r["chart"]["type"] == "histogram"
    assert sum(r["chart"]["counts"]) > 0
    assert any("does not bound the accuracy" in c for c in r["caveats"])


def test_query_bernoulli_returns_categorical_chart(server) -> None:
    r = post(server, "/api/query", {
        "query_class": "bernoulli",
        "parameters": {"probability": 0.3},
        "sample_size": 100_000,
    })
    assert r["chart"]["type"] == "categorical"
    assert r["chart"]["labels"] == ["no", "yes"]
    total = sum(r["chart"]["counts"])
    assert r["chart"]["counts"][1] / total == pytest.approx(0.3, abs=0.01)
    assert r["interval_type"] == "wilson-score"


def test_query_poisson_uses_exact_interval(server) -> None:
    r = post(server, "/api/query", {
        "query_class": "poisson",
        "parameters": {"lam": 0.000005},
        "sample_size": 500_000,
    })
    assert r["interval_type"] == "garwood-exact"
    assert r["monte_carlo_interval"][0] >= 0.0


def test_antithetic_chart_shows_the_marginal_not_pair_means(server) -> None:
    """Regression: the plot must not inherit the pair-mean tail compression."""
    r = post(server, "/api/query", {
        "query_class": "gaussian",
        "parameters": {"mean": 0.0, "std_dev": 1.0},
        "sample_size": 200_000,
        "antithetic": True,
    })
    assert r["quantiles"]["p99"] == pytest.approx(2.326, abs=0.08)
    assert r["max_observed"] > 3.0


def test_domain_ceiling_accepted(server) -> None:
    r = post(server, "/api/query", {
        "query_class": "bernoulli",
        "parameters": {"probability": 0.3},
        "sample_size": 50_000,
        "domain": "macro_behavioral",
    })
    assert r["point_estimate"] == pytest.approx(0.3, abs=0.01)


def test_priors_endpoint_widens_the_interval(server) -> None:
    r = post(server, "/api/priors", {
        "query_class": "bernoulli",
        "posteriors": {"probability": {"kind": "proportion", "successes": 3, "trials": 10}},
        "inner_sample_size": 20_000,
        "parameter_draws": 200,
    })
    mlo, mhi = r["monte_carlo_interval"]
    tlo, thi = r["total_interval"]
    assert (thi - tlo) > (mhi - mlo) * 10
    assert r["parameter_uncertainty"] > r["monte_carlo_error"]
    assert r["posteriors"]["probability"]["n_observations"] == 10


def test_priors_accepts_measurement_evidence(server) -> None:
    r = post(server, "/api/priors", {
        "query_class": "gaussian",
        "posteriors": {"mean": {"kind": "measurements",
                                "observations": [31, 28, 34, 30, 33, 29, 32]}},
        "fixed": {"std_dev": 4.0},
        "inner_sample_size": 20_000,
        "parameter_draws": 150,
    })
    assert r["point_estimate"] == pytest.approx(31.0, abs=1.5)
    assert r["total_interval"][0] < r["point_estimate"] < r["total_interval"][1]


def test_priors_accepts_rate_evidence(server) -> None:
    r = post(server, "/api/priors", {
        "query_class": "poisson",
        "posteriors": {"lam": {"kind": "rate", "events": 50, "exposure": 10.0}},
        "inner_sample_size": 50_000,
        "parameter_draws": 120,
    })
    assert r["point_estimate"] == pytest.approx(5.0, rel=0.25)


def test_zero_event_rate_is_flagged_not_presented_as_an_estimate(server) -> None:
    """Balloon case: no observations means a bound, not a probability.

    The interface keys its 'quote this one' badge off these two signals, so if
    they stop being emitted the UI silently starts recommending a meaningless
    [0, 0] interval.
    """
    r = post(server, "/api/priors", {
        "query_class": "poisson",
        "posteriors": {"lam": {"kind": "rate", "events": 0, "exposure": 1.262e11}},
        "inner_sample_size": 100_000,
        "parameter_draws": 200,
    })
    assert r["posteriors"]["lam"]["n_observations"] == 0
    assert any("quantised by the inner" in c for c in r["caveats"])
    assert r["total_interval"] == [0.0, 0.0]
    # The usable answer is the posterior's upper end, which stays finite.
    lo, hi = r["posteriors"]["lam"]["interval"]
    assert lo > 0.0
    assert 1e-12 < hi < 1e-10


def test_surveyed_proportion_is_not_flagged_as_unresolvable(server) -> None:
    """Tank-top case: real counts must keep the quotable interval."""
    r = post(server, "/api/priors", {
        "query_class": "bernoulli",
        "posteriors": {"probability": {"kind": "proportion", "successes": 61, "trials": 152}},
        "inner_sample_size": 100_000,
        "parameter_draws": 400,
    })
    lo, hi = r["total_interval"]
    assert hi - lo > 0.05
    assert not any("quantised by the inner" in c for c in r["caveats"])
    assert r["posteriors"]["probability"]["n_observations"] == 152
    assert r["point_estimate"] == pytest.approx(0.40, abs=0.02)


def test_sensitivity_endpoint_ranks_inputs(server) -> None:
    r = post(server, "/api/sensitivity", {
        "query_class": "gaussian",
        "posteriors": {
            "mean": {"kind": "measurements", "observations": [10, 30, 20, 40, 15]},
            "std_dev": {"kind": "measurements", "observations": [5.0, 5.1, 4.9, 5.05]},
        },
        "n_base": 400,
    })
    assert len(r["rows"]) == 2
    assert r["rows"][0]["name"] == "mean"  # location dominates the mean estimate
    assert r["rows"][0]["total_effect"] > 0.5


def test_sensitivity_model_is_deterministic_so_no_warning(server) -> None:
    """The endpoint fixes the uniform stream; Sobol requires that."""
    r = post(server, "/api/sensitivity", {
        "query_class": "gaussian",
        "posteriors": {
            "mean": {"kind": "measurements", "observations": [10, 30, 20, 40, 15]},
            "std_dev": {"kind": "measurements", "observations": [5.0, 5.1, 4.9, 5.05]},
        },
        "n_base": 400,
    })
    assert not any("not deterministic" in w for w in r["warnings"])


def test_errors_are_reported_not_swallowed(server) -> None:
    status, body = post_expect_error(server, "/api/query", {
        "query_class": "gaussian", "parameters": {"std_dev": -1.0}, "sample_size": 1000,
    })
    assert status == 400
    assert "std_dev must be positive" in body["error"]


def test_unknown_class_is_a_client_error(server) -> None:
    status, body = post_expect_error(
        server, "/api/query", {"query_class": "nope", "sample_size": 1000}
    )
    assert status == 400
    assert "Unknown query class" in body["error"]


def test_sensitivity_requires_two_inputs(server) -> None:
    status, body = post_expect_error(server, "/api/sensitivity", {
        "query_class": "bernoulli",
        "posteriors": {"probability": {"kind": "proportion", "successes": 3, "trials": 10}},
    })
    assert status == 400
    assert "at least two" in body["error"]


def test_priors_requires_evidence(server) -> None:
    status, body = post_expect_error(
        server, "/api/priors", {"query_class": "bernoulli", "posteriors": {}}
    )
    assert status == 400
    assert "at least one" in body["error"]


def test_unknown_route_404s(server) -> None:
    status, body = post_expect_error(server, "/api/nope", {})
    assert status == 404
    assert "no such endpoint" in body["error"]


def test_unknown_route_with_a_body_does_not_abort_the_connection(server) -> None:
    """Responding before draining the request body aborts the socket on Windows."""
    payload = {"padding": "x" * 200_000}
    status, body = post_expect_error(server, "/api/nope", payload)
    assert status == 404
    assert "no such endpoint" in body["error"]
    # The server must still be healthy for the next request.
    assert get(server, "/api/classes")["classes"]


def test_sample_size_is_capped(server) -> None:
    r = post(server, "/api/query", {
        "query_class": "bernoulli", "parameters": {"probability": 0.5},
        "sample_size": 10**12,
    })
    assert r["sample_size"] == webui.MAX_SAMPLES


def test_histogram_helper_handles_degenerate_input() -> None:
    out = webui._histogram(np.full(100, 3.0), "continuous")
    assert sum(out["counts"]) == 100
