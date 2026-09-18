"""Adaptive microstructure-aware market maker.

Design rule: **every adjustment is an explicit, named function of one measurable state variable, and every
component has its own switch**, so the ablation study can add them one at a time and measure each one's
incremental contribution (nothing here is "tuned until it works"; defaults come from theory or from the measurement
experiments, as noted). All quantities are in ticks; ``I`` is signed inventory in lots.

Quote centre (reservation price) and half-spreads::

    r        = m  -  lambda_I * I            (inventory_skew)   m - lambda I, the spec's example
                  +  beta_obi * OBI_t         (obi)              OBI over the top ``obi_levels`` levels
                  +  beta_ret * (m_t - m_{t-h})                  (returns)   optional, off in the main ladder
    h_bid    = h0 + c_A * A_bid + c_L * sigma * sqrt(L)          h_ask analogous with A_ask
    bid      = floor(r - h_bid),   ask = ceil(r + h_ask)          (rounded outward, ask >= bid + 1)

* ``h0 = 1/k``: the risk-neutral Avellaneda-Stoikov half-spread (gamma -> 0), ``k`` the *measured* fill-intensity decay.
* ``A_side`` (adverse_selection): EWMA of the realised adverse cost ``-s (m_{t+h} - m^-)`` of the maker's own fills
  on that side (``m^-`` = replica mid immediately before the fill; horizon ``adverse_horizon_s``), clipped to
  [0, ``adverse_cap``]. ``c_A = 1`` is the break-even pass-through: the half-spread covers the expected loss to
  informed flow (Glosten-Milgrom logic); Experiment C measures this loss directly.
* ``L`` (latency): the measured delay between sending an order and seeing it acknowledged on the feed (order +
  feed latency). ``sigma sqrt(L)`` is one standard deviation of the price move the maker cannot see; ``c_L = 1``.
* ``beta_obi`` (obi): estimated by regressing forward mid change on OBI at the typical quote lifetime (Experiment A).
* size (inventory_size): ``q_bid = Q (1 - kappa I / I_max)``, ``q_ask = Q (1 + kappa I / I_max)`` (clipped at 0): size
  falls linearly as the position approaches the limit ``I_max`` (the soft inventory limit) on the side that would extend it.
* queue (queue): keep a resting quote at its current price when its *expected value* is at least that of moving to
  the desired price and joining the back of that queue: ``EV(p) = P_fill(Q_p, d_p) * (s (r - p) - A_side)``, with
  ``P_fill`` the fill model fitted from probe data (Experiment B). No hand-set stickiness threshold.

With every flag off the strategy quotes symmetrically at ``mid +- 1/k`` - the ablation baseline.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from engine.common import Side
from research.queue_position import FillModel
from strategies.base import NS, MMConfig, MarketMaker, Quote


class AdverseSelectionTracker:
    """Measures the maker's own realised adverse-selection cost per side.

    ``on_fill`` registers a fill with the mid just before it; once ``horizon_s`` has elapsed, ``evaluate`` measures the
    signed move ``M = s (m_now - m^-)`` (``s = +1`` if the maker bought) and updates a per-side, quantity-weighted EWMA of
    the cost ``-M``. The estimate also decays toward 0 with half-life ``halflife_s`` so stale information fades.
    """

    def __init__(self, horizon_s: float = 0.5, alpha: float = 0.1, halflife_s: float = 30.0, cap: float = 5.0):
        self.horizon_ns = int(horizon_s * NS)
        self.alpha, self.halflife_s, self.cap = alpha, halflife_s, cap
        self._pending: deque[tuple[int, int, int, float]] = deque()  # (t_fill_ns, side, qty, pre_mid)
        self._est = {1: 0.0, -1: 0.0}  # +1 = maker bought (bid fills), -1 = maker sold (ask fills)
        self._last_t: Optional[int] = None
        self.n_measured = 0

    def on_fill(self, t_ns: int, side: int, qty: int, pre_mid: Optional[float]) -> None:
        if pre_mid is not None:
            self._pending.append((t_ns, side, qty, pre_mid))

    def evaluate(self, now: int, mid: Optional[float]) -> None:
        if self._last_t is not None and self.halflife_s > 0:
            f = 0.5 ** ((now - self._last_t) / NS / self.halflife_s)
            self._est = {k: v * f for k, v in self._est.items()}
        self._last_t = now
        if mid is None:
            return
        while self._pending and now - self._pending[0][0] >= self.horizon_ns:
            _, side, qty, pre = self._pending.popleft()
            cost = -side * (mid - pre)
            a = 1.0 - (1.0 - self.alpha) ** qty  # a qty-lot fill counts as qty unit fills
            self._est[side] += a * (cost - self._est[side])
            self.n_measured += 1

    def estimate(self, side: Side) -> float:
        """Adverse cost estimate (ticks, >= 0) for fills on ``side`` (BUY = the maker's bid)."""
        return min(max(self._est[int(side)], 0.0), self.cap)


@dataclass(frozen=True)
class AdaptiveConfig(MMConfig):
    k: float = 0.5  # measured fill-intensity decay (1/tick); base half-spread h0 = 1/k
    base_half_spread: Optional[float] = None  # overrides 1/k if set
    # ---- component switches (the ablation ladder) ----
    use_inventory_skew: bool = True
    use_inventory_size: bool = True
    use_obi: bool = True
    use_adverse: bool = True
    use_queue: bool = True
    use_latency: bool = True
    use_returns: bool = False  # extra signal, not part of the main ladder
    # ---- coefficients ----
    lambda_inv: float = 0.1  # ticks per lot (AS skew gamma sigma^2 tau at gamma = .01, sigma = 2, tau = 2.5 s)
    kappa_size: float = 1.0
    beta_obi: float = 0.3  # ticks per unit OBI; ESTIMATE per environment (see experiments.ablations.calibration)
    obi_levels: int = 1
    beta_ret: float = 0.0
    ret_horizon_s: float = 1.0
    c_adverse: float = 1.0
    adverse_horizon_s: float = 0.5
    adverse_alpha: float = 0.1
    adverse_halflife_s: float = 30.0
    adverse_cap: float = 5.0
    c_latency: float = 1.0
    fill_model: Optional[FillModel] = None  # required for use_queue
    queue_eps: float = 0.0  # move only if EV improves by more than this many ticks (0 = strictly better)
    modify_to_shrink: bool = True  # size changes must not destroy queue priority

    def build(self, name: str = "mm") -> "AdaptiveMM":
        return AdaptiveMM(self, name)


class AdaptiveMM(MarketMaker):
    cfg: AdaptiveConfig

    def __init__(self, cfg: AdaptiveConfig, name: str = "mm"):
        if cfg.use_queue and cfg.fill_model is None:
            raise ValueError("use_queue requires a fill_model (fit it with research.queue_position.fit_fill_model)")
        super().__init__(cfg, name)
        self.tracker = AdverseSelectionTracker(cfg.adverse_horizon_s, cfg.adverse_alpha, cfg.adverse_halflife_s, cfg.adverse_cap)
        self._mid_hist: deque[tuple[int, float]] = deque()
        self._pre_mid_of_fill: Optional[float] = None
        self._center: Optional[float] = None
        # diagnostics: mean absolute contribution of each component (ticks), for the write-up
        self.diag_sum = {"inv": 0.0, "obi": 0.0, "ret": 0.0, "adverse": 0.0, "latency": 0.0, "h0": 0.0}
        self.diag_n = 0
        self.n_kept_by_queue = 0

    # -------------------------------------------------------------- measurement hooks
    def _pre_apply(self, ev) -> None:
        # replica mid *before* this event: for the first trade of a command this is the pre-trade mid m^-
        if ev.maker_owner == self.owner_id and ev.type.value == "TRADE":
            self.tracker.on_fill(ev.ts, -int(ev.side), ev.qty, self.replica.mid_price())

    def on_wakeup(self, sim, now: int):
        m = self.replica.mid_price()
        self.tracker.evaluate(now, m)
        if m is not None:
            self._mid_hist.append((now, m))
            horizon = int(max(self.cfg.ret_horizon_s, 1.0) * 2 * NS)
            while self._mid_hist and self._mid_hist[0][0] < now - horizon:
                self._mid_hist.popleft()
        return super().on_wakeup(sim, now)

    def _past_mid(self, t: int) -> Optional[float]:
        for ts, m in reversed(self._mid_hist):
            if ts <= t:
                return m
        return self._mid_hist[0][1] if self._mid_hist else None

    # ------------------------------------------------------------------- quoting
    def h0(self) -> float:
        c = self.cfg
        return c.base_half_spread if c.base_half_spread is not None else 1.0 / c.k

    def components(self, now: int, mid: float) -> dict:
        """The individual terms of the quote model at this instant (ticks) - also used for diagnostics/tests."""
        c = self.cfg
        inv = self.inventory
        out = {"h0": self.h0(), "inv": 0.0, "obi": 0.0, "ret": 0.0, "lat": 0.0, "adv_bid": 0.0, "adv_ask": 0.0}
        if c.use_inventory_skew:
            out["inv"] = -c.lambda_inv * inv
        if c.use_obi:
            obi = self.replica.imbalance(c.obi_levels)
            out["obi"] = c.beta_obi * (obi if obi is not None else 0.0)
        if c.use_returns:
            past = self._past_mid(now - int(c.ret_horizon_s * NS))
            out["ret"] = c.beta_ret * (mid - past) if past is not None else 0.0
        if c.use_latency and self.ack_latency_s is not None:
            out["lat"] = c.c_latency * self.sigma * math.sqrt(self.ack_latency_s)
        if c.use_adverse:
            out["adv_bid"] = c.c_adverse * self.tracker.estimate(Side.BUY)
            out["adv_ask"] = c.c_adverse * self.tracker.estimate(Side.SELL)
        return out

    def desired_quotes(self, now: int, mid: float):
        c = self.cfg
        k = self.components(now, mid)
        r = mid + k["inv"] + k["obi"] + k["ret"]
        self._center = r
        hb = k["h0"] + k["adv_bid"] + k["lat"]
        ha = k["h0"] + k["adv_ask"] + k["lat"]
        bid, ask = math.floor(r - hb), math.ceil(r + ha)
        if ask <= bid:
            ask = bid + 1
        qb = qa = float(c.quote_size)
        if c.use_inventory_size:
            room = float(c.inventory_limit)
            qb = c.quote_size * (1.0 - c.kappa_size * self.inventory / room)
            qa = c.quote_size * (1.0 + c.kappa_size * self.inventory / room)
        self.diag_n += 1
        d = self.diag_sum
        d["inv"] += abs(k["inv"]); d["obi"] += abs(k["obi"]); d["ret"] += abs(k["ret"]); d["h0"] += k["h0"]
        d["adverse"] += 0.5 * (k["adv_bid"] + k["adv_ask"]); d["latency"] += k["lat"]
        return (bid, max(0, int(round(qb)))), (ask, max(0, int(round(qa))))

    # --------------------------------------------------------------------- queue
    def _touch_distance_and_queue(self, side: Side, price: int, own_id: Optional[int]) -> tuple[float, float]:
        """``(d_touch, Q)`` for a quote at ``price``: an existing order (``own_id``) reads its true queue position from
        the replica; a hypothetical new order joins the back of the level (Q = level quantity)."""
        best = self.replica.best_bid() if side is Side.BUY else self.replica.best_ask()
        if best is None:
            return 0.0, 0.0
        d = (best - price) if side is Side.BUY else (price - best)
        if d < 0:  # would improve the touch: alone at a new best level
            return 0.0, 0.0
        if own_id is not None:
            pos = self.replica.queue_position(own_id)
            return float(d), float(pos[0]) if pos is not None else float(self.replica.level_qty(side, price))
        return float(d), float(self.replica.level_qty(side, price))

    def keep_quote(self, side: Side, cur, want: Quote) -> bool:
        c = self.cfg
        if not c.use_queue or self._center is None:
            return False
        if self.replica.queue_position(cur.order_id) is None:  # not yet visible: no information, follow the target
            return False
        fm = c.fill_model
        s = int(side)
        adv = c.c_adverse * self.tracker.estimate(side) if c.use_adverse else 0.0
        dc, qc = self._touch_distance_and_queue(side, cur.price, cur.order_id)
        dn, qn = self._touch_distance_and_queue(side, want[0], None)
        ev_cur = fm.p_fill(qc, dc) * (s * (self._center - cur.price) - adv)
        ev_new = fm.p_fill(qn, dn) * (s * (self._center - want[0]) - adv)
        keep = ev_cur >= ev_new - c.queue_eps
        self.n_kept_by_queue += int(keep)
        return keep

    def mean_components(self) -> dict:
        n = max(self.diag_n, 1)
        return {f"mean_abs_{k}": v / n for k, v in self.diag_sum.items()}
