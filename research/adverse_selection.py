"""Adverse selection of passive fills, measured directly (no VPIN / toxicity proxies).

For a passive fill at time ``t``, price ``p`` (ticks) and pre-trade mid ``m^-`` (the mid just before the
aggressive order that caused it), with ``s = +1`` if the maker *bought* and ``-1`` if it sold, per lot::

    effective half-spread     E    =  s (m^- - p)                what the maker earned versus the mid
    signed post-fill move     M_h  =  s (m_{t+h} - m^-)          < 0 means the price moved against the maker
    adverse-selection cost    A_h  = -M_h
    realised half-spread      R_h  =  E + M_h  =  s (m_{t+h} - p)    the markout

so ``R_h = E - A_h``: spread capture net of what is lost to subsequent price movement. Horizons follow the
spec (10 ms ... 1 s). All figures are in ticks and quantity-weighted unless stated.

Null control: ``random_time_markouts`` computes the same statistic at random times with random side.
Because the mid is (close to) a martingale over these horizons its mean must be ~0; any non-zero mean of
``M_h`` at *fills* is therefore attributable to the fill (selection), not to drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from research.common import mid_at, obi_at
from simulator.simulator import NS, SimResult

HORIZONS_NS: dict[str, int] = {"10ms": 10_000_000, "50ms": 50_000_000, "100ms": 100_000_000,
                               "500ms": 500_000_000, "1s": 1_000_000_000}


@dataclass
class PassiveFills:
    """Passive (maker-side) execution legs from the exchange tape."""

    t: np.ndarray  # ns
    price: np.ndarray  # ticks
    side: np.ndarray  # +1 maker bought, -1 maker sold
    qty: np.ndarray
    mid_before: np.ndarray
    taker_owner: np.ndarray
    maker_owner: np.ndarray
    maker_id: np.ndarray

    def __len__(self) -> int:
        return len(self.t)

    def select(self, mask: np.ndarray) -> "PassiveFills":
        return PassiveFills(*[a[mask] for a in (self.t, self.price, self.side, self.qty, self.mid_before,
                                                 self.taker_owner, self.maker_owner, self.maker_id)])


def passive_fills(res: SimResult, maker_owner: Optional[int] = None) -> PassiveFills:
    """Maker legs (optionally only for one owner). Fills with a one-sided pre-trade book are dropped."""
    tp = res.trades
    m = ~np.isnan(tp["mid_before"])
    if maker_owner is not None:
        m &= tp["maker_owner"] == maker_owner
    return PassiveFills(tp["t_ns"][m], tp["price"][m].astype(float), -tp["aggressor"][m], tp["qty"][m].astype(float),
                        tp["mid_before"][m], tp["taker_owner"][m], tp["maker_owner"][m], tp["maker_id"][m])


def markouts(res: SimResult, f: PassiveFills, horizons: dict[str, int] = HORIZONS_NS) -> dict[str, np.ndarray]:
    """Per-fill arrays: ``E`` and, per horizon label, ``M_<h>`` and ``R_<h>`` (ticks per lot).
    Fills whose look-ahead window extends beyond the simulation end are excluded (NaN)."""
    end = res.config.horizon_s * NS
    out = {"E": f.side * (f.mid_before - f.price)}
    for lab, h in horizons.items():
        m_h = mid_at(res, f.t + h)
        M = f.side * (m_h - f.mid_before)
        M = np.where(f.t + h <= end, M, np.nan)
        out[f"M_{lab}"] = M
        out[f"R_{lab}"] = out["E"] + M
    return out


def wmean(x: np.ndarray, w: np.ndarray) -> float:
    ok = ~np.isnan(x)
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok])) if ok.any() and np.sum(w[ok]) > 0 else float("nan")


def summarize_fills(res: SimResult, f: PassiveFills, horizons: dict[str, int] = HORIZONS_NS) -> dict[str, float]:
    """Quantity-weighted means of E, M_h, R_h plus the fill count/volume (one session)."""
    mo = markouts(res, f, horizons)
    out = {"n_fills": float(len(f)), "volume": float(f.qty.sum())}
    for k, v in mo.items():
        out[k] = wmean(v, f.qty)
    return out


def random_time_markouts(res: SimResult, n: int, rng: np.random.Generator,
                         horizons: dict[str, int] = HORIZONS_NS, start_s: float = 5.0) -> dict[str, float]:
    """Null control: mean of ``s (m_{t+h} - m_t)`` at random times with random side ``s``."""
    end = res.config.horizon_s * NS
    t = rng.integers(int(start_s * NS), int(end - max(horizons.values())), n)
    s = rng.choice([-1, 1], n)
    m0 = mid_at(res, t)
    return {f"M_{lab}": float(np.mean(s * (mid_at(res, t + h) - m0))) for lab, h in horizons.items()}


def by_group(res: SimResult, f: PassiveFills, groups: dict[str, np.ndarray], horizon: str = "100ms") -> dict[str, dict]:
    """Quantity-weighted (E, M, R) at one horizon within each boolean mask in ``groups``."""
    mo = markouts(res, f, {horizon: HORIZONS_NS[horizon]})
    out = {}
    for name, mask in groups.items():
        w = f.qty[mask]
        out[name] = {"n": int(mask.sum()), "volume": float(w.sum()), "E": wmean(mo["E"][mask], w),
                     "M": wmean(mo[f"M_{horizon}"][mask], w), "R": wmean(mo[f"R_{horizon}"][mask], w)}
    return out


def signed_obi_at_fills(res: SimResult, f: PassiveFills, levels: int = 1) -> np.ndarray:
    """``s * OBI`` just before each fill (sample-grid resolution): positive = the book leans toward
    the maker's side (more resting volume on the side it just traded on)."""
    return f.side * obi_at(res, f.t - 1, levels)
