"""Command-line interface: `uss query`, `uss fit`, `uss domains`."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import distributions
from .core import UniversalStochasticSandbox
from .estimators import CONFIDENCE_DOMAINS


def _parse_params(pairs: list[str]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"parameter must be key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        try:
            params[key.strip()] = float(raw)
        except ValueError:
            params[key.strip()] = raw.strip()
    return params


def _cmd_query(args: argparse.Namespace) -> int:
    sandbox = UniversalStochasticSandbox(sample_size=args.n, seed=args.seed)
    result = sandbox.execute_query(
        args.query_class,
        _parse_params(args.param),
        confidence_level=args.confidence,
        antithetic=args.antithetic,
        domain=args.domain,
    )
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, default=str))
    else:
        print(result.summary())
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    from .fitting import best_fit, load_column

    data = load_column(args.path, args.column)
    results = best_fit(data)
    print(f"{len(data):,} observations from {args.path}::{args.column}\n")
    print(f"{'family':<14}{'AIC':>14}{'KS stat':>12}{'KS p':>12}  parameters")
    for r in results:
        params = ", ".join(f"{k}={v:.6g}" for k, v in r.parameters.items())
        print(
            f"{r.family:<14}{r.aic:>14.2f}{r.ks_statistic:>12.5f}"
            f"{r.ks_pvalue:>12.4g}  {params or '-'}"
        )
    best = results[0]
    print(f"\nbest by AIC: {best.family} -> query_class={best.query_class!r}")
    if best.ks_pvalue < 0.05:
        print(
            "  ! KS test rejects this family at the 5% level; prefer the "
            "'empirical' query class for this data"
        )
    return 0


def _cmd_gui(args: argparse.Namespace) -> int:
    from .webui import serve

    serve(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def _cmd_domains(_: argparse.Namespace) -> int:
    print(f"{'domain':<20}{'confidence':<16}{'bottleneck'}")
    for key, meta in CONFIDENCE_DOMAINS.items():
        lo, hi = meta["achievable_confidence"]
        band = "approaches 0%" if hi == 0 else f"{lo:.0%} - {hi:.1%}"
        print(f"{key:<20}{band:<16}{meta['bottleneck']}")
    return 0


def _cmd_classes(_: argparse.Namespace) -> int:
    for name in distributions.available():
        qc = distributions.get(name)
        print(f"{name:<16}{qc.kind:<14}{qc.describe}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="uss", description="Universal Stochastic Sandbox Engine"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    q = sub.add_parser("query", help="run a simulation query")
    q.add_argument("query_class", choices=distributions.available())
    q.add_argument("-p", "--param", action="append", default=[], metavar="KEY=VALUE")
    q.add_argument("-n", type=int, default=1_000_000, help="sample size")
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--confidence", type=float, default=0.95)
    q.add_argument("--antithetic", action="store_true")
    q.add_argument("--domain", choices=sorted(CONFIDENCE_DOMAINS), default=None)
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=_cmd_query)

    f = sub.add_parser("fit", help="fit distributions to a data column")
    f.add_argument("path")
    f.add_argument("column")
    f.set_defaults(func=_cmd_fit)

    g = sub.add_parser("gui", help="open the local web interface")
    g.add_argument("--port", type=int, default=8765)
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--no-browser", action="store_true")
    g.set_defaults(func=_cmd_gui)

    d = sub.add_parser("domains", help="show Part IV confidence ceilings")
    d.set_defaults(func=_cmd_domains)

    c = sub.add_parser("classes", help="list registered query classes")
    c.set_defaults(func=_cmd_classes)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
