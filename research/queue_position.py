"""Queue position and fill probability: ``P(fill within dt | Q_t, market state)``.

``Q_t`` is the quantity resting *ahead* of our order in its price level's FIFO at time ``t`` (exact, read from
the true book). A :class:`QueueProbe` posts 1-lot post-only orders at the touch (``k = 0``) and ``k`` ticks behind
the same-side best price, keeps each for ``dwell_s`` seconds, and at every checkpoint (every ``checkpoint_s``)
records ``Q_t`` and the market state:

* spread (ticks), same-side depth (5 levels), trailing 1 s |mid change| (realised-volatility proxy) and its signed
  version ``side * (m_t - m_{t-1s})`` (positive: the mid has moved *away* from a resting bid / toward a resting ask
  side... i.e. against our order for a buy, see below), trailing 1 s traded volume (order-flow intensity), signed
  imbalance ``side * OBI`` at the touch, and the **current** distance ``d_touch`` from our price to the same-side
  best price (0 = at the touch). ``k`` is the depth at posting; ``d_touch`` is what matters at time ``t``, since
  the market moves after posting.

This yields a discrete-time hazard dataset with outcome ``y_{i,k}(dt) = 1[fill in (t_k, t_k + dt]]``.
**Censoring:** a row is used only if its whole window lies inside the dwell (``age + dt <= dwell``), for filled
and unfilled orders alike. Dropping only the unfilled rows that were cut short would bias fill probabilities
upward; dropping all incompletely observed rows does not, because the cut-off is fixed by the order's age, not by
its outcome.

Estimators: an empirical table by ``Q`` bin (additive sufficient statistics => cheap session bootstrap) and a
logistic regression on standardised features (coefficients are log-odds per +1 SD). Rows from one order and one
session are strongly dependent, so intervals come from a bootstrap over *sessions*.
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from engine.commands import Cancel, Command, NewLimit
from engine.common import EventType as ET, Side
from simulator.order_flow.arrivals import NS
from simulator.participant import Participant

FEATURES = ("log1p_Q", "d_touch", "spread", "log1p_depth5", "abs_ret1s", "signed_ret1s", "log1p_flow1s", "signed_obi")
Q_EDGES = (0, 1, 3, 6, 11, 21, 41, 10**9)  # bins: 0 | 1-2 | 3-5 | 6-10 | 11-20 | 21-40 | 41+


@dataclass
class _Probe:
    oid: int
    side: int
    k: int
    t0: int
    price: int
    q0: int
    fill_ns: float = float("nan")


class QueueProbe(Participant):
    """Measurement participant; see module docstring. Zero latency, sees the true book."""

    feed = "all"

    def __init__(self, rate: float = 3.0, dwell_s: float = 3.0, checkpoint_s: float = 0.25,
                 depths: Sequence[int] = (0, 1, 2), start_s: float = 5.0, name: str = "qprobe"):
        self.name = name
        self.rate, self.dwell_ns, self.dwell_s = rate, int(dwell_s * NS), dwell_s
        self.check_ns = int(checkpoint_s * NS)
        self.depths = list(depths)
        self.start_ns = int(start_s * NS)
        self.probes: list[_Probe] = []
        self.rows: list[tuple] = []  # see ``dataset`` for the column order
        self._live: dict[int, int] = {}  # order_id -> probe index
        self._timers: list[tuple[int, int]] = []
        self._trades: deque[tuple[int, int]] = deque()
        self._next_post = 0
        self._next_check = 0

    def on_start(self, sim) -> None:
        self.rng = sim.rng_for(self.name)
        self._next_post = self.start_ns + int(self.rng.exponential(1 / self.rate) * NS)
        self._next_check = self.start_ns + self.check_ns

    def next_time(self) -> Optional[int]:
        t = min(self._next_post, self._next_check)
        return min(t, self._timers[0][0]) if self._timers else t

    def on_events(self, sim, now: int, events) -> list[Command]:
        for ev in events:
            if ev.type is ET.TRADE:
                self._trades.append((ev.ts, ev.qty))
                idx = self._live.pop(ev.maker_id, None) if ev.maker_owner == self.owner_id else None
                if idx is not None:
                    self.probes[idx].fill_ns = ev.ts
        return []

    def on_wakeup(self, sim, now: int) -> list[Command]:
        cmds: list[Command] = []
        if self._next_check <= now:
            self._checkpoint(sim, now)
            self._next_check += self.check_ns
        while self._timers and self._timers[0][0] <= now:
            _, oid = heapq.heappop(self._timers)
            if self._live.pop(oid, None) is not None:
                cmds.append(Cancel(oid))
        if self._next_post <= now:
            self._next_post = now + max(1, int(self.rng.exponential(1 / self.rate) * NS))
            cmds += self._post(sim, now)
        return cmds

    def _post(self, sim, now: int) -> list[Command]:
        side = Side.BUY if self.rng.random() < 0.5 else Side.SELL
        k = int(self.rng.choice(self.depths))
        best = sim.book.best_bid() if side is Side.BUY else sim.book.best_ask()
        if best is None:
            return []
        price = best - k if side is Side.BUY else best + k
        opp = sim.book.best_ask() if side is Side.BUY else sim.book.best_bid()
        if price < 1 or (opp is not None and (price >= opp if side is Side.BUY else price <= opp)):
            return []
        q0 = sim.book.level_qty(side, price)
        oid = sim.new_id()
        self.probes.append(_Probe(oid, int(side), k, now, price, q0))
        self._live[oid] = len(self.probes) - 1
        heapq.heappush(self._timers, (now + self.dwell_ns, oid))
        return [NewLimit(oid, side, price, 1, owner=self.owner_id, post_only=True)]

    def _checkpoint(self, sim, now: int) -> None:
        if not self._live:
            return
        while self._trades and self._trades[0][0] < now - NS:
            self._trades.popleft()
        flow = sum(q for _, q in self._trades)
        hist = sim.mid_history
        m_now, m_lag = sim.reference_mid(), hist.at(now - NS)
        vol = abs(m_now - m_lag) if m_lag is not None else 0.0
        spread = sim.book.spread()
        obi = sim.book.imbalance(1)
        for oid, idx in self._live.items():
            p = self.probes[idx]
            pos = sim.book.queue_position(oid)
            if pos is None:
                continue
            side = Side(p.side)
            depth5 = sim.book.volume(side, 5)
            best = sim.book.best_bid() if p.side == 1 else sim.book.best_ask()
            d_touch = (best - p.price) if p.side == 1 else (p.price - best)
            signed_ret = p.side * (m_now - m_lag) if m_lag is not None else 0.0
            self.rows.append((idx, (now - p.t0) / NS, pos[0], pos[1], p.k, p.side, spread if spread is not None else np.nan,
                              depth5, vol, flow, p.side * obi if obi is not None else 0.0, d_touch, signed_ret))

    # ------------------------------------------------------------------- outputs
    def dataset(self) -> dict[str, np.ndarray]:
        r = np.array(self.rows, dtype=float).reshape(-1, 13)
        names = ("probe", "age_s", "Q", "n_ahead", "k", "side", "spread", "depth5", "vol1s", "flow1s", "signed_obi",
                 "d_touch", "signed_ret1s")
        d = {n: r[:, i] for i, n in enumerate(names)}
        fill = np.array([p.fill_ns for p in self.probes], dtype=float)
        t0 = np.array([p.t0 for p in self.probes], dtype=float)
        d["t_check_ns"] = t0[d["probe"].astype(int)] + d["age_s"] * NS if len(r) else np.array([])
        d["fill_ns"] = fill[d["probe"].astype(int)] if len(r) else np.array([])
        return d

    def probe_table(self) -> dict[str, np.ndarray]:
        return {"oid": np.array([p.oid for p in self.probes]), "q0": np.array([p.q0 for p in self.probes]),
                "k": np.array([p.k for p in self.probes]), "side": np.array([p.side for p in self.probes]),
                "filled": np.array([not math.isnan(p.fill_ns) for p in self.probes])}


# ---------------------------------------------------------------------------------- analysis
def outcomes(d: dict[str, np.ndarray], delta_s: float, dwell_s: float) -> tuple[np.ndarray, np.ndarray]:
    """``(y, valid)``: fill within ``delta_s`` after each checkpoint, and whether that window was fully observed."""
    valid = d["age_s"] + delta_s <= dwell_s + 1e-9
    after = (d["fill_ns"] - d["t_check_ns"]) / NS
    y = (~np.isnan(after)) & (after > -1e-9) & (after <= delta_s)
    return y.astype(float), valid


def feature_matrix(d: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([np.log1p(d["Q"]), d["d_touch"], np.nan_to_num(d["spread"], nan=1.0), np.log1p(d["depth5"]),
                            d["vol1s"], d["signed_ret1s"], np.log1p(d["flow1s"]), d["signed_obi"]])


def q_bin_stats(Q: np.ndarray, y: np.ndarray, edges: Sequence[int] = Q_EDGES) -> dict[int, np.ndarray]:
    """Per-Q-bin ``[n, fills]`` (additive across sessions)."""
    b = np.digitize(Q, edges) - 1
    return {int(i): np.array([np.sum(b == i), y[b == i].sum()], dtype=float) for i in np.unique(b)}


def add_stats(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    out = {k: v.copy() for k, v in a.items()}
    for k, v in b.items():
        out[k] = out[k] + v if k in out else v.copy()
    return out


def q_bin_label(i: int, edges: Sequence[int] = Q_EDGES) -> str:
    lo, hi = edges[i], edges[i + 1] - 1
    return "0" if hi == 0 else (f"{lo}+" if hi > 10**8 else f"{lo}-{hi}")


def fit_logit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 30) -> dict:
    """L2-regularised logistic regression by Newton's method on standardised features.

    Returns ``{"coef": per-SD log-odds, "intercept", "mean", "std"}`` (features listed in ``FEATURES``)."""
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    beta = np.zeros(Z.shape[1])
    pen = l2 * np.eye(Z.shape[1]); pen[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ beta, -30, 30)))
        g = Z.T @ (p - y) + pen @ beta
        H = (Z * (p * (1 - p))[:, None]).T @ Z + pen + 1e-9 * np.eye(len(beta))
        step = np.linalg.solve(H, g)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return {"intercept": float(beta[0]), "coef": beta[1:].copy(), "mean": mu, "std": sd}


def predict_logit(model: dict, X: np.ndarray) -> np.ndarray:
    z = model["intercept"] + ((X - model["mean"]) / model["std"]) @ model["coef"]
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------------------- fill-probability model
@dataclass(frozen=True)
class FillModel:
    """Parametric fill probability ``P(fill within horizon_s | Q, d) = sigmoid(a + b_q ln(1+Q) + b_d d)``.

    ``Q`` = quantity ahead in the queue (lots), ``d`` = distance in ticks from the same-side touch (0 = at the
    touch). Fitted from :class:`QueueProbe` data by :func:`fit_fill_model` (Experiment B found these two variables
    dominate fill probability). Used by queue-aware strategies to value queue priority.
    """

    intercept: float
    b_q: float
    b_d: float
    horizon_s: float = 1.0

    def p_fill(self, q: float, d: float) -> float:
        z = self.intercept + self.b_q * math.log1p(max(q, 0.0)) + self.b_d * max(d, 0.0)
        return 1.0 / (1.0 + math.exp(-max(min(z, 30.0), -30.0)))


def fit_fill_model(datasets: Sequence[dict[str, np.ndarray]], delta_s: float = 1.0, dwell_s: float = 3.0) -> FillModel:
    """Fit a :class:`FillModel` on probe datasets (only fully observed windows are used, see module docstring)."""
    Xs, ys = [], []
    for d in datasets:
        y, valid = outcomes(d, delta_s, dwell_s)
        Xs.append(np.column_stack([np.log1p(d["Q"]), d["d_touch"]])[valid])
        ys.append(y[valid])
    X, y = np.vstack(Xs), np.concatenate(ys)
    m = fit_logit(X, y)
    b = m["coef"] / m["std"]  # back to raw feature units
    return FillModel(float(m["intercept"] - np.sum(m["coef"] * m["mean"] / m["std"])), float(b[0]), float(b[1]), delta_s)
