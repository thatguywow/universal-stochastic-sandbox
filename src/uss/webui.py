"""Local web interface for the sandbox.

Runs on Python's stdlib http.server -- no web framework, no extra dependencies,
nothing leaves the machine. Start it with `uss gui`.

The interface exists because the numbers this engine produces are easy to
misread. A terminal dump prints the simulation-precision interval and the
parameter-uncertainty interval as two similar-looking lines; on screen they can
be drawn at different weights, so the one that actually bounds a real-world
claim is the one that stands out.
"""

from __future__ import annotations

import json
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from . import distributions
from .core import UniversalStochasticSandbox
from .estimators import CONFIDENCE_DOMAINS
from .inference import Posterior, update_bernoulli, update_gaussian_mean, update_poisson
from .sensitivity import parameter_sensitivity

_STATIC = Path(__file__).parent / "static"

# Every request runs unbounded numerical work, so each knob a caller can turn
# needs a ceiling. Without caps on the replication counts, `parameter_draws`
# multiplied by `inner_sample_size` reaches trillions of samples from a single
# request and the process hangs or is killed by the OOM reaper.
MAX_SAMPLES = 20_000_000
MAX_PARAMETER_DRAWS = 5_000
MAX_SOBOL_BASE = 20_000
MAX_HIST_BINS = 60


def _histogram(samples: np.ndarray, kind: str) -> dict[str, Any]:
    """Bin samples for plotting, using integer bins for discrete quantities."""
    arr = samples.astype(np.float64, copy=False)
    lo, hi = float(arr.min()), float(arr.max())

    if kind == "proportion":
        n_one = int(np.count_nonzero(arr))
        return {
            "type": "categorical",
            "labels": ["no", "yes"],
            "counts": [int(arr.size - n_one), n_one],
        }

    # Integer bins only when the data really is integral. Testing just the
    # maximum misfires: a continuous uniform on [0, 10] whose largest draw
    # happens to land on 10.0 was being drawn as 11 integer bars, hiding the
    # actual shape of the distribution.
    integral = kind == "count" or (
        arr.size > 0 and bool(np.all(arr == np.round(arr)))
    )
    if integral and (hi - lo) < MAX_HIST_BINS:
        top = int(min(hi, lo + MAX_HIST_BINS))
        edges = np.arange(int(lo), top + 2) - 0.5
    elif hi > lo:
        edges = np.linspace(lo, hi, MAX_HIST_BINS + 1)
    else:
        edges = np.array([lo - 0.5, lo + 0.5])

    counts, edges = np.histogram(arr, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2.0
    return {
        "type": "histogram",
        "centres": [float(c) for c in centres],
        "counts": [int(c) for c in counts],
        "width": float(edges[1] - edges[0]),
    }


def _build_posterior(spec: dict[str, Any]) -> Posterior:
    kind = spec.get("kind")
    if kind == "proportion":
        return update_bernoulli(int(spec["successes"]), int(spec["trials"]))
    if kind == "rate":
        return update_poisson(int(spec["events"]), float(spec["exposure"]))
    if kind == "measurements":
        values = np.asarray(spec["observations"], dtype=np.float64)
        if values.size < 2:
            raise ValueError("need at least 2 observations to estimate a mean")
        return update_gaussian_mean(values)
    raise ValueError(f"unknown evidence type: {kind!r}")


class _Handler(BaseHTTPRequestHandler):
    server_version = "uss"

    def log_message(self, *args: Any) -> None:  # keep the console quiet
        pass

    # -- plumbing ---------------------------------------------------------
    def _send(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            html = (_STATIC / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return
        if self.path == "/api/classes":
            self._send(
                {
                    "classes": [
                        {
                            "name": name,
                            "kind": (qc := distributions.get(name)).kind,
                            "describe": qc.describe,
                            "params": [p.as_dict() for p in qc.params],
                        }
                        for name in distributions.available()
                    ],
                    "domains": {
                        key: {
                            "label": meta["label"],
                            "band": list(meta["achievable_confidence"]),
                            "bottleneck": meta["bottleneck"],
                        }
                        for key, meta in CONFIDENCE_DOMAINS.items()
                    },
                }
            )
            return
        self._send({"error": "not found"}, 404)

    def do_POST(self) -> None:
        try:
            # Always drain the request body, including on the 404 path.
            # Responding without consuming it leaves unread bytes in the socket
            # and the client sees the connection abort instead of the status.
            body = self._read_json()
            handler = {
                "/api/query": self._api_query,
                "/api/priors": self._api_priors,
                "/api/sensitivity": self._api_sensitivity,
                "/api/reference": self._api_reference,
            }.get(self.path)
            if handler is None:
                self._send({"error": f"no such endpoint: {self.path}"}, 404)
                return
            self._send(handler(body))
        except ValueError as exc:
            # Expected: bad input. The message is the whole diagnosis.
            self._send({"error": str(exc)}, 400)
        except Exception as exc:
            # Unexpected: a real defect. HTTP logging is suppressed to keep the
            # console readable, so without this the traceback goes nowhere and a
            # bug in the interface leaves no diagnostic trace at all.
            traceback.print_exc()
            self._send({"error": f"{type(exc).__name__}: {exc}"}, 500)

    # -- endpoints --------------------------------------------------------
    def _api_query(self, body: dict[str, Any]) -> dict[str, Any]:
        name = body.get("query_class", "gaussian")
        qc = distributions.get(name)
        n = min(int(body.get("sample_size", 500_000)), MAX_SAMPLES)
        params = {k: float(v) for k, v in (body.get("parameters") or {}).items()}

        sandbox = UniversalStochasticSandbox(sample_size=n, seed=body.get("seed", 42))
        result = sandbox.execute_query(
            name,
            params,
            confidence_level=float(body.get("confidence_level", 0.95)),
            antithetic=bool(body.get("antithetic", False)),
            domain=body.get("domain") or None,
        )

        # Re-draw for the plot so the histogram shows the marginal distribution
        # rather than antithetic pair means.
        plot_n = min(n, 200_000)
        plot_samples = qc.sample(
            np.random.default_rng(body.get("seed", 42)).random(plot_n), **params
        )

        report = result.report
        return {
            "point_estimate": report.point_estimate,
            "variance": report.variance,
            "monte_carlo_error": report.monte_carlo_error,
            "monte_carlo_interval": list(report.monte_carlo_interval),
            "interval_type": report.interval_type,
            "confidence_level": report.confidence_level,
            "sample_size": result.sample_size,
            "elapsed_seconds": result.elapsed_seconds,
            "quantiles": result.quantiles,
            "min_observed": result.min_observed,
            "max_observed": result.max_observed,
            "caveats": report.caveats,
            "chart": _histogram(plot_samples, qc.kind),
            "kind": qc.kind,
        }

    def _api_priors(self, body: dict[str, Any]) -> dict[str, Any]:
        name = body.get("query_class", "bernoulli")
        n = min(int(body.get("sample_size", 200_000)), MAX_SAMPLES)
        inner = min(int(body.get("inner_sample_size", 50_000)), MAX_SAMPLES)
        draws = min(int(body.get("parameter_draws", 400)), MAX_PARAMETER_DRAWS)
        confidence = float(body.get("confidence_level", 0.95))

        posteriors = {
            key: _build_posterior(spec)
            for key, spec in (body.get("posteriors") or {}).items()
        }
        if not posteriors:
            raise ValueError("supply at least one piece of evidence")

        sandbox = UniversalStochasticSandbox(sample_size=n, seed=body.get("seed", 42))
        result = sandbox.execute_with_priors(
            name,
            posteriors,
            fixed={k: float(v) for k, v in (body.get("fixed") or {}).items()},
            n_parameter_draws=draws,
            inner_sample_size=inner,
            confidence_level=confidence,
            domain=body.get("domain") or None,
        )
        report = result.report

        rng = np.random.default_rng(7)
        posterior_previews = {
            key: {
                "mean": post.mean,
                "interval": list(post.interval(confidence)),
                "n_observations": post.n_observations,
                "samples": [float(v) for v in post.sample(4000, rng)],
            }
            for key, post in posteriors.items()
        }

        return {
            "point_estimate": report.point_estimate,
            "monte_carlo_error": report.monte_carlo_error,
            "monte_carlo_interval": list(report.monte_carlo_interval),
            "parameter_uncertainty": report.parameter_uncertainty,
            "total_interval": list(report.total_interval) if report.total_interval else None,
            "confidence_level": report.confidence_level,
            "caveats": report.caveats,
            "posteriors": posterior_previews,
            "elapsed_seconds": result.elapsed_seconds,
        }

    def _api_reference(self, _body: dict[str, Any]) -> dict[str, Any]:
        """Live reference data for the Info tab.

        Sample-size and interval figures are computed on request rather than
        hard-coded into the page, so the guidance shown can never drift away
        from what the estimators actually do.
        """
        from .planning import proportion_tradeoff, samples_for_proportion

        return {
            "classes": [
                {
                    "name": name,
                    "kind": (qc := distributions.get(name)).kind,
                    "describe": qc.describe,
                    "params": [p.name for p in qc.params],
                }
                for name in distributions.available()
            ],
            "domains": {
                key: {
                    "label": meta["label"],
                    "band": list(meta["achievable_confidence"]),
                    "governing": meta["governing"],
                    "bottleneck": meta["bottleneck"],
                }
                for key, meta in CONFIDENCE_DOMAINS.items()
            },
            "tradeoff": [
                {"n": n, "half_width": w}
                for n, w in proportion_tradeoff(assumed_rate=0.3)
            ],
            "targets": [
                {
                    "half_width": t,
                    "n": samples_for_proportion(t, assumed_rate=0.3).n_required,
                }
                for t in (0.10, 0.05, 0.02)
            ],
        }

    def _api_sensitivity(self, body: dict[str, Any]) -> dict[str, Any]:
        name = body.get("query_class", "bernoulli")
        qc = distributions.get(name)
        posteriors = {
            key: _build_posterior(spec)
            for key, spec in (body.get("posteriors") or {}).items()
        }
        if len(posteriors) < 2:
            raise ValueError("sensitivity needs at least two uncertain inputs")

        fixed = {k: float(v) for k, v in (body.get("fixed") or {}).items()}
        inner = min(int(body.get("inner_sample_size", 20_000)), 200_000)
        base = min(int(body.get("n_base", 2_000)), MAX_SOBOL_BASE)
        confidence = float(body.get("confidence_level", 0.95))

        # Deterministic given the parameters: the uniform stream is fixed, so
        # repeated evaluations at identical inputs agree. Sobol requires that.
        fixed_u = np.random.default_rng(99).random(inner)

        def simulate(params: dict[str, float]) -> float:
            merged = {**fixed, **params}
            return float(qc.sample(fixed_u, **merged).astype(np.float64).mean())

        res = parameter_sensitivity(
            simulate,
            posteriors,
            base,
            np.random.default_rng(11),
            fixed=fixed,
            confidence_level=confidence,
        )

        def interval_for(name: str) -> list[float] | None:
            if res.total_effect_interval is None:
                return None
            return [float(v) for v in res.total_effect_interval[res.names.index(name)]]

        return {
            "rows": [
                {
                    "name": n_,
                    "first_order": f,
                    "total_effect": t,
                    "total_interval": interval_for(n_),
                }
                for n_, f, t in res.ranked()
            ],
            "output_variance": res.output_variance,
            "interaction_strength": res.interaction_strength,
            "unexplained_variance": res.unexplained_variance,
            "resolved": res.separates_top_two(),
            "warnings": list(res.warnings),
            "n_model_evaluations": res.n_model_evaluations,
        }


_LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Start the local interface and block until interrupted.

    Binds to loopback by default. The server has no authentication and runs
    unbounded numerical work on request, so exposing it beyond the local
    machine hands anyone who can reach the port a way to consume all available
    CPU and memory. Binding elsewhere is allowed but announced loudly.
    """
    if not (_STATIC / "index.html").exists():
        raise FileNotFoundError(f"interface assets missing: {_STATIC / 'index.html'}")

    try:
        httpd = ThreadingHTTPServer((host, port), _Handler)
    except OSError as exc:
        raise OSError(
            f"cannot bind {host}:{port} -- {exc}. Another instance may already "
            f"be running; try a different --port."
        ) from exc

    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{display_host}:{port}/"

    if host not in _LOOPBACK:
        print("!" * 70)
        print(f"WARNING: binding to {host}, which is reachable from other machines.")
        print("This interface has no authentication and will run arbitrarily large")
        print("simulations on request. Do not expose it on an untrusted network.")
        print("!" * 70)

    print(f"Universal Stochastic Sandbox -- interface running at {url}")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
