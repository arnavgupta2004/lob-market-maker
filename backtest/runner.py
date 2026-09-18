"""Run one simulated session (scenario x seed) and compute its metrics.

A :class:`Scenario` is a frozen, picklable description (configs only, no live objects), so
sessions can be farmed out to worker processes and recorded verbatim in experiment provenance.
"""
from __future__ import annotations

import dataclasses
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional, Sequence

from backtest.costs import FeeModel
from backtest.metrics import session_metrics
from simulator.market_state.fundamental import FundamentalConfig
from simulator.order_flow.informed import InformedTrader, InformedTraderConfig
from simulator.order_flow.noise import NoiseTrader, NoiseTraderConfig
from simulator.participant import Participant
from simulator.simulator import SimConfig, SimResult, Simulator
from strategies.base import MMConfig, MarketMaker


# A session whose mid ever strays this far (ticks) from its start is flagged as diverged. Typical
# healthy sessions move a few tens of ticks; 500 ticks = 5% of the initial price at tick 0.01.
DIVERGENCE_TICKS = 500


@dataclass(frozen=True)
class Scenario:
    """Everything needed to (re)produce a session except the seed."""

    name: str
    sim: SimConfig = SimConfig()
    noise: Optional[NoiseTraderConfig] = NoiseTraderConfig()
    informed: Optional[InformedTraderConfig] = None
    mm: Optional[MMConfig] = None  # must expose .build() -> MarketMaker

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class Session:
    result: SimResult
    metrics: dict
    mm: Optional[MarketMaker]


def build_participants(sc: Scenario) -> tuple[list[Participant], Optional[MarketMaker]]:
    parts: list[Participant] = []
    if sc.noise is not None:
        parts.append(NoiseTrader(sc.noise))
    if sc.informed is not None:
        parts.append(InformedTrader(sc.informed))
    mm = sc.mm.build() if sc.mm is not None else None  # type: ignore[attr-defined]
    if mm is not None:
        parts.append(mm)
    return parts, mm


def run_session(sc: Scenario, seed: int, keep_result: bool = True) -> Session:
    """Simulate ``sc`` with ``seed`` and compute the market maker's metrics (if any)."""
    sim_cfg = dataclasses.replace(sc.sim, seed=seed)
    if sc.informed is not None and sim_cfg.fundamental is None:
        sim_cfg = dataclasses.replace(sim_cfg, fundamental=FundamentalConfig())
    parts, mm = build_participants(sc)
    res = Simulator(sim_cfg, parts).run()
    metrics: dict = {"seed": seed}
    if mm is not None:
        metrics.update(session_metrics(res, mm.owner_id, mm.cfg.fees, start_s=mm.cfg.start_s))
        metrics.update(killed=float(mm.killed), kill_time_s=(mm.kill_time or 0) / 1e9 if mm.killed else float("nan"),
                       sigma_hat_final=mm.sigma, n_sigma_clamped=float(mm.n_sigma_clamped),
                       n_offset_clamped=float(mm.n_offset_clamped))
    s, tr = res.steady()
    metrics["market_trades"] = float(len(tr["price"]))
    mid = s["mid"][~_isnan(s["mid"])]
    dev = float(_absmax(mid - sim_cfg.mid0_ticks)) if len(mid) else float("nan")
    metrics["mid_max_dev_ticks"] = dev
    metrics["diverged"] = float(dev > DIVERGENCE_TICKS)
    metrics["market_spread_mean"] = float(_nanmean(s["spread"]))
    return Session(res if keep_result else None, metrics, mm)  # type: ignore[arg-type]


def _nanmean(x):
    import numpy as np
    return np.nanmean(x) if len(x) else float("nan")


def _isnan(x):
    import numpy as np
    return np.isnan(x)


def _absmax(x):
    import numpy as np
    return np.abs(x).max()


def _worker(args):
    sc, seed = args
    try:
        return run_session(sc, seed, keep_result=False).metrics
    except OverflowError:  # numerical blow-up of an unguarded strategy: record, do not hide
        return {"seed": seed, "diverged": 1.0, "crashed": 1.0}


def run_many(sc: Scenario, seeds: Sequence[int], workers: int = 1) -> list[dict]:
    """Metrics for ``sc`` over ``seeds`` (optionally in parallel; result order = seed order)."""
    if workers <= 1 or len(seeds) < 2:
        return [_worker((sc, s)) for s in seeds]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_worker, [(sc, s) for s in seeds]))
