"""Empirical tests of the Avellaneda-Stoikov assumptions inside the simulator.

A2 (fill intensity ``lambda(delta) = A exp(-k delta)``): a passive *probe* posts 1-lot
post-only orders at randomised distances from the mid, leaves each for ``dwell_s`` seconds, and
records exposure time and whether it filled. Because each probe is right-censored (cancelled
after the dwell), the intensity in a distance bin is estimated by the censored-exponential MLE
``lambda_hat = fills / total_exposure_time`` (standard error ``lambda_hat / sqrt(fills)``),
not by a raw fill fraction. ``ln lambda_hat`` is then regressed on ``delta`` (weighted by fills)
to estimate ``(A, k)`` and R^2. The probe also reports the hazard in the first vs second half of
the dwell: a constant-intensity Poisson fill process predicts equal hazards; queue effects make
them differ.

A1 (Brownian mid): variance ratios ``VR(q) = Var(m_{t+q} - m_t) / (q Var(m_{t+1} - m_t))``
(equal to 1 for a random walk; < 1 indicates mean reversion such as bid-ask bounce), the
volatility "signature" ``sigma_hat(dt) = sqrt(Var(dm) / dt)`` across sampling intervals
(constant for Brownian motion) and the excess kurtosis / zero-return share of increments.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from engine.commands import Cancel, Command, NewLimit
from engine.common import EventType as ET, Side
from simulator.order_flow.arrivals import NS
from simulator.participant import Participant


@dataclass
class ProbeRecord:
    side: int
    delta: float  # ticks from the mid at posting
    spread: float
    exposure_s: float = 0.0
    filled: bool = False
    fill_after_s: float = float("nan")


class ProbeQuoter(Participant):
    """Measurement participant: posts 1-lot passive probes at randomised distances."""

    feed = "own"

    def __init__(self, deltas: Sequence[float] = (0.5, 1, 1.5, 2, 3, 4, 5, 6, 8), rate: float = 20.0,
                 dwell_s: float = 0.5, start_s: float = 5.0, name: str = "probe"):
        self.name = name
        self.deltas = list(deltas)
        self.rate = rate
        self.dwell_ns = int(dwell_s * NS)
        self.dwell_s = dwell_s
        self.start_ns = int(start_s * NS)
        self.records: list[ProbeRecord] = []
        self._by_id: dict[int, tuple[int, int]] = {}  # order_id -> (record index, post time)
        self._timers: list[tuple[int, int]] = []
        self._next = 0

    def on_start(self, sim) -> None:
        self.rng = sim.rng_for(self.name)
        self._next = self.start_ns + int(self.rng.exponential(1 / self.rate) * NS)

    def next_time(self) -> Optional[int]:
        return min(self._next, self._timers[0][0]) if self._timers else self._next

    def on_wakeup(self, sim, now: int) -> list[Command]:
        cmds: list[Command] = []
        while self._timers and self._timers[0][0] <= now:
            _, oid = heapq.heappop(self._timers)
            idx, t0 = self._by_id[oid]
            rec = self.records[idx]
            if not rec.filled:
                rec.exposure_s = (now - t0) / NS
                if oid in sim.book:
                    cmds.append(Cancel(oid))
        if self._next <= now:
            self._next = now + max(1, int(self.rng.exponential(1 / self.rate) * NS))
            cmds += self._post(sim, now)
        return cmds

    def _post(self, sim, now: int) -> list[Command]:
        b, a = sim.book.best_bid(), sim.book.best_ask()
        if b is None or a is None:
            return []
        mid = (a + b) / 2
        delta = float(self.rng.choice(self.deltas))
        side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
        price = math.floor(mid - delta) if side is Side.BUY else math.ceil(mid + delta)
        if price < 1 or (side is Side.BUY and price >= a) or (side is Side.SELL and price <= b):
            return []
        oid = sim.new_id()
        self._by_id[oid] = (len(self.records), now)
        self.records.append(ProbeRecord(int(side), abs(mid - price), a - b))
        heapq.heappush(self._timers, (now + self.dwell_ns, oid))
        return [NewLimit(oid, side, price, 1, owner=self.owner_id, post_only=True)]

    def on_events(self, sim, now: int, events) -> list[Command]:
        for ev in events:
            if ev.type is ET.TRADE and ev.maker_owner == self.owner_id and ev.maker_id in self._by_id:
                idx, t0 = self._by_id[ev.maker_id]
                rec = self.records[idx]
                if not rec.filled:
                    rec.filled = True
                    rec.fill_after_s = (ev.ts - t0) / NS
                    rec.exposure_s = rec.fill_after_s
        return []


# --------------------------------------------------------------------------- estimators
STAT_FIELDS = ("n", "fills", "exposure", "f1", "e1", "f2", "e2")


def session_bin_stats(records: Sequence[ProbeRecord], dwell_s: float, bin_width: float = 0.5) -> dict[float, np.ndarray]:
    """Per-distance-bin sufficient statistics ``[n, fills, exposure, f1, e1, f2, e2]`` for one session
    (``f1/e1``: fills/exposure in the first half of the dwell, ``f2/e2``: second half). Sufficient
    statistics add across sessions, which makes session-level bootstrapping cheap."""
    half = dwell_s / 2
    bins: dict[float, np.ndarray] = {}
    for r in records:
        if r.exposure_s <= 0:
            continue
        key = round(round(r.delta / bin_width) * bin_width, 6)
        b = bins.setdefault(key, np.zeros(len(STAT_FIELDS)))
        b[0] += 1
        b[1] += r.filled
        b[2] += r.exposure_s
        b[4] += min(r.exposure_s, half)
        b[6] += max(0.0, r.exposure_s - half)
        if r.filled:
            b[3 if r.fill_after_s < half else 5] += 1
    return bins


def rows_from_stats(stats: dict[float, np.ndarray]) -> list[dict]:
    """Censored-exponential MLE ``fills / exposure`` per bin (+ first/second-half hazards)."""
    rows = []
    for d in sorted(stats):
        n, fills, exp, f1, e1, f2, e2 = stats[d]
        lam = fills / exp if exp > 0 else float("nan")
        rows.append(dict(
            delta=d, n_probes=int(n), fills=int(fills), exposure_s=float(exp), lam=float(lam),
            lam_se=float(lam / math.sqrt(fills)) if fills else float("nan"),
            fill_frac=float(fills / n),
            lam_first_half=float(f1 / e1) if e1 else float("nan"),
            lam_second_half=float(f2 / e2) if e2 else float("nan")))
    return rows


def add_stats(a: dict[float, np.ndarray], b: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    out = {k: v.copy() for k, v in a.items()}
    for k, v in b.items():
        out[k] = out[k] + v if k in out else v.copy()
    return out


def intensity_by_distance(records: Sequence[ProbeRecord], dwell_s: float, bin_width: float = 0.5) -> list[dict]:
    """Censored-exponential MLE of fill intensity per distance bin, plus first/second-half hazards."""
    return rows_from_stats(session_bin_stats(records, dwell_s, bin_width))


def fit_exponential_intensity(rows: Sequence[dict], min_fills: int = 10) -> dict:
    """Weighted least squares of ``ln lambda`` on ``delta``: returns ``A, k, r2`` and the fit's
    standard errors. Weights are the fill counts (Poisson: ``Var(ln lam) ~ 1/fills``)."""
    use = [r for r in rows if r["fills"] >= min_fills and r["lam"] > 0]
    if len(use) < 3:
        return dict(A=float("nan"), k=float("nan"), r2=float("nan"), k_se=float("nan"), n_bins=len(use))
    x = np.array([r["delta"] for r in use])
    y = np.log([r["lam"] for r in use])
    w = np.array([r["fills"] for r in use], dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    cov = np.linalg.inv(X.T @ W @ X)
    beta = cov @ X.T @ W @ y
    resid = y - X @ beta
    ss_res = float(np.sum(w * resid ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    dof = max(1, len(use) - 2)
    sigma2 = ss_res / dof  # residual scale relative to Poisson weights (1 => exactly Poisson-consistent)
    return dict(A=float(math.exp(beta[0])), k=float(-beta[1]), r2=1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
                k_se=float(math.sqrt(cov[1, 1] * max(sigma2, 1.0))), chi2_per_dof=sigma2, n_bins=len(use))


def mid_diagnostics(mid: np.ndarray, dt_s: float, lags: Sequence[int] = (1, 2, 5, 10, 50)) -> dict:
    """Random-walk diagnostics of an evenly sampled mid series (ticks)."""
    m = np.asarray(mid, dtype=float)
    m = m[~np.isnan(m)]
    d1 = np.diff(m)
    v1 = d1.var()
    out: dict = {"n": int(len(d1)), "zero_return_share": float(np.mean(d1 == 0)),
                 "excess_kurtosis": float(((d1 - d1.mean()) ** 4).mean() / v1 ** 2 - 3) if v1 > 0 else float("nan")}
    for q in lags:
        if len(m) > 4 * q:
            dq = m[q:] - m[:-q]
            if q > 1:  # VR(1) == 1 by construction
                out[f"vr_{q}"] = float(dq.var() / (q * v1)) if v1 > 0 else float("nan")
            out[f"sigma_{q * dt_s:g}s"] = float(math.sqrt(dq.var() / (q * dt_s)))
    return out
