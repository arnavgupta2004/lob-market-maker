"""Experiment command-line entry point.

    python -m experiments.run --list
    python -m experiments.run as_assumptions --seed 42
    python -m experiments.run inventory --seed 42 --workers 4
    python -m experiments.run inventory --seed 42 --quick     # smoke-test sized

Each experiment writes to ``results/<name>/``: machine-readable results (CSV/JSON), figures, and
``provenance.json`` (seed, parameters, git commit, environment, result hashes).
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

from experiments.common import RunContext

# name -> module exposing DESCRIPTION and run(ctx)
REGISTRY = {
    "as_assumptions": "experiments.baseline.as_assumptions",
    "inventory": "experiments.inventory.gamma_sweep",
    "imbalance": "experiments.imbalance.obi_predictiveness",
    "queue": "experiments.queue.fill_probability",
    "adverse_selection": "experiments.adverse_selection.post_fill",
    "price_impact": "experiments.price_impact.impact_curve",
    "adaptive_demo": "experiments.ablations.adaptive_demo",
    "benchmark_engine": "benchmarks.engine_benchmark",
    "benchmark_complexity": "benchmarks.complexity",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m experiments.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help="experiment name (see --list)")
    ap.add_argument("--seed", type=int, default=42, help="base seed (sessions use seed*100000 + i)")
    ap.add_argument("--sessions", type=int, default=None, help="independent sessions/seeds per arm")
    ap.add_argument("--workers", type=int, default=1, help="worker processes")
    ap.add_argument("--quick", action="store_true", help="tiny run (smoke test)")
    ap.add_argument("--out", default="results", help="output root directory")
    ap.add_argument("--list", action="store_true", help="list experiments")
    args = ap.parse_args(argv)

    if args.list or not args.name:
        for k, mod in REGISTRY.items():
            print(f"{k:18s} {importlib.import_module(mod).DESCRIPTION}")
        return 0
    if args.name not in REGISTRY:
        print(f"unknown experiment {args.name!r}; try --list", file=sys.stderr)
        return 2
    out_dir = Path(args.out) / args.name / ("quick" if args.quick else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(args.name, args.seed, args.sessions, args.workers, args.quick, out_dir)
    t0 = time.time()
    importlib.import_module(REGISTRY[args.name]).run(ctx)
    print(f"[{args.name}] done in {time.time() - t0:.1f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
