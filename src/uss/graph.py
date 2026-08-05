"""Scenario composition: wire distributions together into one coupled model.

A single `execute_query` call answers a one-stage question.  Real questions are
usually several stages deep:

    temperature ~ Gaussian(31, 4)
    p_tanktop   = logit(-6.0 + 0.22 * temperature)     <- depends on the draw
    wearing     ~ Bernoulli(p_tanktop)                  <- parameter is a vector
    crowd       ~ Poisson(2000 * wearing_fraction)

Each node here draws its own independent uniform vector and every downstream
node sees the *per-draw* value of its parents, so uncertainty and correlation
propagate through the whole graph in one vectorized pass.  Estimating the
stages separately and multiplying the point estimates throws that away and
understates the spread of the answer.

Node streams are derived from independent child seeds, so appending a node
never perturbs the draws of existing ones -- which is what makes graph runs
reproducible and makes the sensitivity analysis in `uss.sensitivity` valid.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import distributions, estimators
from .estimators import UncertaintyReport


class Node:
    """Handle referring to one variable in a graph."""

    __slots__ = ("_graph", "name")

    def __init__(self, name: str, graph: ScenarioGraph) -> None:
        self.name = name
        self._graph = graph

    def __repr__(self) -> str:
        return f"Node({self.name!r})"


@dataclass
class _Stochastic:
    name: str
    query_class: str
    parameters: dict[str, Any]
    parents: tuple[str, ...]
    seed_index: int


@dataclass
class _Derived:
    name: str
    fn: Callable[..., np.ndarray]
    parents: tuple[str, ...]


@dataclass
class GraphResult:
    """Samples for every node of one graph run."""

    samples: dict[str, np.ndarray]
    n: int
    kinds: dict[str, str] = field(default_factory=dict)

    def __getitem__(self, name: str) -> np.ndarray:
        return self.samples[name]

    def mean(self, name: str) -> float:
        return float(np.mean(self.samples[name]))

    def report(
        self, name: str, *, confidence_level: float = 0.95, domain: str | None = None
    ) -> UncertaintyReport:
        """Full uncertainty report for one node, using the right interval type."""
        return estimators.summarize(
            self.samples[name],
            kind=self.kinds.get(name, "continuous"),
            confidence_level=confidence_level,
            domain=domain,
        )

    def correlation(self, a: str, b: str) -> float:
        """Correlation induced between two nodes by their shared ancestry."""
        x = self.samples[a].astype(np.float64, copy=False)
        y = self.samples[b].astype(np.float64, copy=False)
        return float(np.corrcoef(x, y)[0, 1])

    def to_frame(self):
        """Materialise the run as a polars DataFrame."""
        import polars as pl

        return pl.DataFrame(
            {k: v.astype(np.float64, copy=False) for k, v in self.samples.items()}
        )

    def summary(self) -> str:
        lines = [f"graph run  n={self.n:,}"]
        for name, values in self.samples.items():
            arr = values.astype(np.float64, copy=False)
            lines.append(
                f"  {name:<22}mean={arr.mean():>14.6g}  sd={arr.std():>12.6g}"
                f"  [{np.quantile(arr, 0.025):>11.6g}, {np.quantile(arr, 0.975):>11.6g}]"
            )
        return "\n".join(lines)


class ScenarioGraph:
    """A directed acyclic graph of stochastic and derived variables."""

    def __init__(self, seed: int | None = 42) -> None:
        self.seed = seed
        self._nodes: dict[str, _Stochastic | _Derived] = {}
        self._order: list[str] = []
        self._n_stochastic = 0

    # -- construction ------------------------------------------------------
    def add(self, name: str, query_class: str, **parameters: Any) -> Node:
        """Add a stochastic node. Parameter values may be scalars or Nodes."""
        self._check_name(name)
        distributions.get(query_class)  # fail fast on unknown class

        parents = tuple(
            v.name for v in parameters.values() if isinstance(v, Node)
        )
        self._nodes[name] = _Stochastic(
            name=name,
            query_class=query_class,
            parameters=dict(parameters),
            parents=parents,
            seed_index=self._n_stochastic,
        )
        self._n_stochastic += 1
        self._order.append(name)
        return Node(name, self)

    def derive(
        self, name: str, fn: Callable[..., np.ndarray], *parents: Node | str
    ) -> Node:
        """Add a deterministic node computed from its parents' draws."""
        self._check_name(name)
        parent_names = tuple(p.name if isinstance(p, Node) else p for p in parents)
        for p in parent_names:
            if p not in self._nodes:
                raise ValueError(f"unknown parent node {p!r}")
        self._nodes[name] = _Derived(name=name, fn=fn, parents=parent_names)
        self._order.append(name)
        return Node(name, self)

    def _check_name(self, name: str) -> None:
        if name in self._nodes:
            raise ValueError(f"node {name!r} already exists")
        if not name:
            raise ValueError("node name must be non-empty")

    # -- execution ---------------------------------------------------------
    def run(
        self,
        n: int,
        *,
        uniform_overrides: dict[str, np.ndarray] | None = None,
        replicate: int = 0,
    ) -> GraphResult:
        """Draw `n` coupled realisations of every node.

        Deterministic: the same graph, `n` and `replicate` always produce the
        same draws, however many times `run` is called.  An earlier revision
        spawned children from a stored SeedSequence, whose internal counter
        advances on every call -- so a second `run()` on the same object
        silently returned different numbers, making config comparisons and
        graph-level sensitivity analysis invalid.

        Pass `replicate=1, 2, ...` for independent repeat runs of the same
        graph, which is the explicit way to ask for fresh randomness.

        `uniform_overrides` replaces a node's uniform stream with a supplied
        vector; `uss.sensitivity` uses this to hold factors fixed across runs.
        """
        if n < 2:
            raise ValueError(f"n must be at least 2, got {n}")
        if replicate < 0:
            raise ValueError(f"replicate must be non-negative, got {replicate}")
        order = self._topological_order()
        overrides = uniform_overrides or {}
        # Rebuilt per call, so spawning never mutates state that outlives it.
        entropy = (self.seed, replicate) if self.seed is not None else (replicate,)
        child_seeds = np.random.SeedSequence(entropy).spawn(
            max(self._n_stochastic, 1)
        )

        values: dict[str, np.ndarray] = {}
        kinds: dict[str, str] = {}

        for name in order:
            node = self._nodes[name]

            if isinstance(node, _Derived):
                args = [values[p] for p in node.parents]
                out = np.asarray(node.fn(*args))
                if out.shape != (n,):
                    out = np.broadcast_to(out, (n,)).copy()
                values[name] = out
                kinds[name] = "continuous"
                continue

            qc = distributions.get(node.query_class)
            resolved = {
                k: (values[v.name] if isinstance(v, Node) else v)
                for k, v in node.parameters.items()
            }

            if name in overrides:
                u = np.asarray(overrides[name], dtype=np.float64)
                if u.shape != (n,):
                    raise ValueError(
                        f"override for {name!r} has shape {u.shape}, expected ({n},)"
                    )
            else:
                u = np.random.default_rng(child_seeds[node.seed_index]).random(n)

            values[name] = self._sample(qc, u, resolved, name)
            kinds[name] = qc.kind

        return GraphResult(samples=values, n=n, kinds=kinds)

    @staticmethod
    def _sample(
        qc: distributions.QueryClass,
        u: np.ndarray,
        params: dict[str, Any],
        name: str,
    ) -> np.ndarray:
        has_vector = any(isinstance(v, np.ndarray) for v in params.values())
        if has_vector and qc.name == "poisson":
            # The searchsorted CDF table is built per scalar lambda, so a
            # per-draw lambda has to fall back to scipy's elementwise ppf --
            # exact, but ~69x slower. Warn rather than silently stalling.
            import warnings

            warnings.warn(
                f"node {name!r} uses a per-draw lambda; Poisson falls back to "
                "scipy's elementwise ppf (~69x slower than the scalar path). "
                "Consider discretising lambda or using a Gaussian approximation "
                "when lambda is large.",
                RuntimeWarning,
                stacklevel=3,
            )
            from scipy import stats

            return stats.poisson.ppf(u, mu=params["lam"]).astype(np.int64)
        return qc.sample(u, **params)

    def _topological_order(self) -> list[str]:
        visited: dict[str, int] = {}
        order: list[str] = []

        def visit(name: str, stack: tuple[str, ...]) -> None:
            state = visited.get(name, 0)
            if state == 1:
                cycle = " -> ".join((*stack, name))
                raise ValueError(f"cycle detected in scenario graph: {cycle}")
            if state == 2:
                return
            visited[name] = 1
            for parent in self._nodes[name].parents:
                visit(parent, (*stack, name))
            visited[name] = 2
            order.append(name)

        for name in self._order:
            visit(name, ())
        return order

    # -- introspection -----------------------------------------------------
    @property
    def stochastic_nodes(self) -> list[str]:
        return [n for n, v in self._nodes.items() if isinstance(v, _Stochastic)]

    @property
    def root_nodes(self) -> list[str]:
        """Stochastic nodes with no parents -- the graph's independent inputs."""
        return [
            n
            for n, v in self._nodes.items()
            if isinstance(v, _Stochastic) and not v.parents
        ]

    def to_mermaid(self) -> str:
        """Render the graph as a mermaid diagram."""
        lines = ["graph TD"]
        for name, node in self._nodes.items():
            if isinstance(node, _Stochastic):
                lines.append(f'    {name}["{name}<br/>~ {node.query_class}"]')
            else:
                lines.append(f'    {name}("{name}")')
        for name, node in self._nodes.items():
            for parent in node.parents:
                lines.append(f"    {parent} --> {name}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ScenarioGraph(nodes={len(self._nodes)}, "
            f"stochastic={self._n_stochastic}, seed={self.seed})"
        )


def gaussian_copula(
    correlation: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Correlated uniform vectors for coupling otherwise-independent roots.

    Returns an (n, k) array of U(0,1) columns with the requested rank
    correlation structure.  Feed the columns in as `uniform_overrides` when
    inputs should move together -- temperature and humidity, say, rather than
    being sampled independently.
    """
    from scipy import stats

    corr = np.asarray(correlation, dtype=np.float64)
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError("correlation must be a square matrix")
    if not np.allclose(corr, corr.T):
        raise ValueError("correlation matrix must be symmetric")

    eigenvalues = np.linalg.eigvalsh(corr)
    if np.min(eigenvalues) < -1e-8:
        raise ValueError("correlation matrix is not positive semi-definite")

    chol = np.linalg.cholesky(corr + np.eye(corr.shape[0]) * 1e-12)
    normals = rng.standard_normal((n, corr.shape[0]))
    return stats.norm.cdf(normals @ chol.T)
