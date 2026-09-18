"""Helpers shared by the microstructure research modules (all read exchange ground truth)."""
from __future__ import annotations

import numpy as np

from simulator.simulator import NS, SimResult


def mid_at(res: SimResult, t_ns: np.ndarray) -> np.ndarray:
    """True mid (ticks) in force at each time. The mid is a step function; while a side is empty
    it keeps its last valid value, and before the first observation it equals the first."""
    t = np.asarray(res.mid_history.t)
    m = np.asarray(res.mid_history.mid)
    idx = np.clip(np.searchsorted(t, np.asarray(t_ns), side="right") - 1, 0, len(t) - 1)
    return m[idx]


def ffill(x: np.ndarray) -> np.ndarray:
    """Forward-fill nans (leading nans take the first valid value)."""
    x = np.array(x, dtype=float)
    ok = ~np.isnan(x)
    if not ok.any():
        return x
    idx = np.where(ok, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    out[: np.argmax(ok)] = out[np.argmax(ok)]
    return out


def obi_from_samples(samples: dict[str, np.ndarray], levels: int = 1) -> np.ndarray:
    """Order-book imbalance ``(Vb - Va) / (Vb + Va)`` from recorded samples (nan if both empty).
    ``levels=1`` uses the touch, ``levels=5`` the recorded 5-level depth."""
    vb = samples["bid_qty1"] if levels == 1 else samples["bid_depth5"]
    va = samples["ask_qty1"] if levels == 1 else samples["ask_depth5"]
    tot = vb + va
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, (vb - va) / tot, np.nan)


def obi_at(res: SimResult, t_ns: np.ndarray, levels: int = 1) -> np.ndarray:
    """OBI at the most recent sample at or before each time (sample grid resolution)."""
    s = res.samples
    obi = obi_from_samples(s, levels)
    idx = np.clip(np.searchsorted(s["t_ns"], np.asarray(t_ns), side="right") - 1, 0, len(obi) - 1)
    return obi[idx]
