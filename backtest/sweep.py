"""Reproducible parameter sweeps.

A sweep is a base :class:`~backtest.runner.Scenario` plus a list of :class:`Axis` objects; the Cartesian product of the axes'
values defines the cells. Every cell is evaluated on the *same seeds* (common random numbers), so differences between cells are
paired. Axes are ordinary functions ``(scenario, value) -> scenario``; :func:`path_axis` builds one that replaces a (nested)
dataclass field by dotted path (``"mm.gamma"``, ``"mm.latency"``, ``"sim.fundamental.sigma"``, ``"noise.limit_rate"``), so any
strategy, simulator or order-flow parameter can be swept without new code. Each cell yields a record with the axis values, the
per-metric mean / median / std / bootstrap CI over sessions, and the number of sessions; per-session rows are kept too. Everything is
deterministic given ``(scenario, axes, seeds)`` and serialisable (CSV / JSON / Parquet via ``experiments.common``).
"""
from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np

from backtest.runner import Scenario, run_many
from backtest.statistics import summarize
from simulator.latency.models import LatencyConfig


def replace_path(obj: Any, path: str, value: Any) -> Any:
    """Return a copy of the (frozen) dataclass ``obj`` with the field at dotted ``path`` replaced by ``value``."""
    head, _, rest = path.partition(".")
    if not dataclasses.is_dataclass(obj) or head not in {f.name for f in dataclasses.fields(obj)}:
        raise AttributeError(f"{type(obj).__name__} has no field {head!r} (path {path!r})")
    if not rest:
        return dataclasses.replace(obj, **{head: value})
    child = getattr(obj, head)
    if child is None:
        raise AttributeError(f"cannot descend into None at {head!r} (path {path!r})")
    return dataclasses.replace(obj, **{head: replace_path(child, rest, value)})


@dataclass(frozen=True)
class Axis:
    name: str
    values: tuple
    apply: Callable[[Scenario, Any], Scenario]
    label: Callable[[Any], Any] = lambda v: v  # what is written to the results for a value


def path_axis(path: str, values: Sequence, name: Optional[str] = None) -> Axis:
    return Axis(name or path, tuple(values), lambda sc, v: replace_path(sc, path, v))


def latency_axis(values_ms: Sequence[float], name: str = "latency_ms") -> Axis:
    """One-way latency (ms) on both legs of the market maker."""
    return Axis(name, tuple(values_ms), lambda sc, v: replace_path(sc, "mm.latency", LatencyConfig.symmetric(v)))


def cells(axes: Sequence[Axis]):
    for combo in itertools.product(*[a.values for a in axes]):
        yield combo


def run_sweep(base: Scenario, axes: Sequence[Axis], seeds: Sequence[int], metrics: Sequence[str], workers: int = 1,
              progress: Optional[Callable[[dict], None]] = None) -> tuple[list[dict], list[dict]]:
    """Evaluate every cell. Returns ``(cell_records, session_rows)``."""
    records, sessions = [], []
    for combo in cells(axes):
        sc = base
        for ax, v in zip(axes, combo):
            sc = ax.apply(sc, v)
        rows = run_many(sc, list(seeds), workers=workers)
        rec: dict = {ax.name: ax.label(v) for ax, v in zip(axes, combo)}
        rec["n"] = len(rows)
        for m in metrics:
            vals = [r.get(m, np.nan) for r in rows]
            s = summarize(vals, n_boot=1000)
            rec.update({f"{m}_mean": s["mean"], f"{m}_median": s["median"], f"{m}_std": s["std"], f"{m}_ci_lo": s["ci_lo"], f"{m}_ci_hi": s["ci_hi"]})
        records.append(rec)
        for r in rows:
            sessions.append({**{ax.name: ax.label(v) for ax, v in zip(axes, combo)}, **r})
        if progress:
            progress(rec)
    return records, sessions


def pivot(records: Sequence[dict], x: str, y: str, metric: str, stat: str = "mean") -> tuple[list, list, np.ndarray]:
    """Grid ``Z[j, i]`` of ``metric_stat`` over axis values ``xs[i]``, ``ys[j]`` (other axes must be constant in ``records``)."""
    xs = sorted({r[x] for r in records})
    ys = sorted({r[y] for r in records})
    Z = np.full((len(ys), len(xs)), np.nan)
    for r in records:
        Z[ys.index(r[y]), xs.index(r[x])] = r[f"{metric}_{stat}"]
    return xs, ys, Z


def plot_heatmaps(records: Sequence[dict], x: str, y: str, metrics: Sequence[tuple[str, str]], path, title: str = "",
                  log_x: bool = False, log_y: bool = False, ncols: int = 3) -> None:
    """One heatmap per ``(metric, label)`` with the cell means annotated. Saves a PNG at ``path``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(metrics)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.3 * nrows), squeeze=False)
    for ax, (m, lab) in zip(axes.flat, metrics):
        xs, ys, Z = pivot(records, x, y, m)
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in xs], rotation=30)
        ax.set_yticks(range(len(ys))); ax.set_yticklabels([f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in ys])
        ax.set(xlabel=x, ylabel=y, title=lab)
        fig.colorbar(im, ax=ax, fraction=0.046)
        thr = np.nanmean(Z)
        for j in range(Z.shape[0]):
            for i in range(Z.shape[1]):
                if np.isfinite(Z[j, i]):
                    ax.text(i, j, f"{Z[j, i]:.2g}", ha="center", va="center", fontsize=7, color="w" if Z[j, i] < thr else "k")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
