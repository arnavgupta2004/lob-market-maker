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
    "stylized_facts": "experiments.validation.stylized_facts",
    "resilience": "experiments.validation.resilience",
    "replay_roundtrip": "experiments.validation.replay_roundtrip",
    "adaptive_demo": "experiments.ablations.adaptive_demo",
    "real_data": "experiments.real_data.real_data",
    "real_mm": "experiments.real_data.real_mm",
    "latency": "experiments.latency.latency_sweep",
    "ablation": "experiments.ablations.ladder",
    "sweep_gamma_size": "experiments.parameter_sweeps.sweeps:sweep_gamma_size",
    "sweep_gamma_latency": "experiments.parameter_sweeps.sweeps:sweep_gamma_latency",
    "sweep_inventory": "experiments.parameter_sweeps.sweeps:sweep_inventory",
    "sweep_market": "experiments.parameter_sweeps.sweeps:sweep_market",
    "sweep_adaptive": "experiments.parameter_sweeps.sweeps:sweep_adaptive",
    "sweep_spread": "experiments.parameter_sweeps.sweeps:sweep_spread",
    "benchmark_engine": "benchmarks.engine_benchmark",
    "benchmark_complexity": "benchmarks.complexity",
}


def _resolve(name: str):
    """Registry value ``"pkg.mod"`` (module with ``run(ctx)``) or ``"pkg.mod:func"`` (function ``func(ctx)``)."""
    target = REGISTRY[name]
    mod, _, attr = target.partition(":")
    m = importlib.import_module(mod)
    return getattr(m, attr) if attr else m.run


def _describe(name: str) -> str:
    mod = importlib.import_module(REGISTRY[name].partition(":")[0])
    d = getattr(mod, "DESCRIPTIONS", None)
    return d[name] if d and name in d else getattr(mod, "DESCRIPTION", "")


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
        for k in REGISTRY:
            print(f"{k:18s} {_describe(k)}")
        return 0
    if args.name not in REGISTRY:
        print(f"unknown experiment {args.name!r}; try --list", file=sys.stderr)
        return 2
    out_dir = Path(args.out) / args.name / ("quick" if args.quick else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(args.name, args.seed, args.sessions, args.workers, args.quick, out_dir)
    t0 = time.time()
    _resolve(args.name)(ctx)
    print(f"[{args.name}] done in {time.time() - t0:.1f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
