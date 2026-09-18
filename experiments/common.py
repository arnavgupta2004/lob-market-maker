"""Experiment plumbing: provenance, output files, common CLI context."""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]


@dataclass
class RunContext:
    """What every experiment receives from the CLI."""

    name: str
    seed: int  # base seed; sessions use seed_base + i
    sessions: Optional[int]  # None => experiment default
    workers: int
    quick: bool  # tiny run for smoke tests / CI
    out_dir: Path

    def n(self, default: int, quick: int = 3) -> int:
        """Number of sessions: CLI override > quick mode > experiment default."""
        if self.sessions is not None:
            return self.sessions
        return quick if self.quick else default

    def seeds(self, default: int, quick: int = 3) -> list[int]:
        return [self.seed * 100_000 + i for i in range(self.n(default, quick))]


def git_info() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()
    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--", "."))
    except Exception:  # pragma: no cover
        return {"commit": None, "dirty": None}
    return {"commit": commit or None, "dirty": dirty}


def _jsonable(o: Any):
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _jsonable(dataclasses.asdict(o))
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return o


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=2, sort_keys=True, allow_nan=True) + "\n")


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows([{k: _jsonable(v) for k, v in r.items()} for r in rows])


def write_provenance(ctx: RunContext, params: dict, dataset: str = "synthetic (no external data)",
                     files: Optional[list[str]] = None, elapsed_s: Optional[float] = None) -> None:
    """Record everything needed to reproduce and audit a result (spec section 12)."""
    import scipy
    out_files = {}
    for f in files or []:
        p = ctx.out_dir / f
        if p.exists():
            out_files[f] = hashlib.sha256(p.read_bytes()).hexdigest()
    save_json(ctx.out_dir / "provenance.json", {
        "experiment": ctx.name,
        "command": "python -m experiments.run " + " ".join(sys.argv[1:]),
        "seed": ctx.seed,
        "session_seeds": {"base": ctx.seed * 100_000, "scheme": "seed = base + session_index"},
        "quick_mode": ctx.quick,
        "dataset": dataset,
        "parameters": params,
        "git": git_info(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform()},
        "result_file_sha256": out_files,
        "elapsed_s": elapsed_s,
        "written_at_unix": int(time.time()),
    })


def pmap(fn, args: list, workers: int) -> list:
    """Order-preserving map over ``args``, in worker processes when ``workers > 1``. ``fn`` must be a
    picklable top-level function."""
    if workers > 1 and len(args) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(fn, args))
    return [fn(a) for a in args]


def save_parquet(path: Path, rows: list[dict]) -> None:
    """Rows -> Parquet (columns keep their numeric types)."""
    import pandas as pd
    pd.DataFrame(rows).to_parquet(path, index=False)
