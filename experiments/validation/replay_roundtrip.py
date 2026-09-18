"""Historical-replay round trip: simulate -> publish as an L2 + trade feed -> replay -> compare.

This is the evidence that the replay machinery (``simulator.order_flow.historical``) is faithful *before* any real dataset is
used: if a feed with known ground truth is not reconstructed exactly, results on real data could not be trusted either.
For each seed it checks (a) the replayed touch equals the feed's touch after every level update, (b) zero trade-price
mismatches and zero crossing adds, (c) identical final (price, quantity) per level, (d) identical trade tape (time, price,
aggressor; quantity aggregated per fill group), and reports the replay speed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from experiments.common import RunContext, pmap, save_csv, write_provenance
from simulator.market_state.fundamental import FundamentalConfig
from simulator.order_flow.historical import HistoricalReplay
from simulator.order_flow.informed import InformedTrader
from simulator.order_flow.market_data import events_to_market_data, validate_market_data
from simulator.order_flow.arrivals import HawkesArrival
from simulator.order_flow.metaorder import MetaOrderTrader
from simulator.order_flow.noise import NoiseTrader
from simulator.simulator import SimConfig, Simulator

DESCRIPTION = "Round trip: simulated exchange -> L2+trades feed -> replay; fidelity of the reconstruction"


@dataclass(frozen=True)
class Job:
    seed: int
    horizon_s: float


def _tape(res) -> dict:
    t, agg = res.trades, {}
    for ts, px, q, s in zip(t["t_ns"], t["price"], t["qty"], t["aggressor"]):
        k = (int(ts), int(px), int(s))
        agg[k] = agg.get(k, 0) + int(q)
    return agg


def _levels(b):
    bids, asks = b.depth()
    return [(p, q) for p, q, _ in bids], [(p, q) for p, q, _ in asks]


def session(job: Job) -> dict:
    cfg = SimConfig(seed=job.seed, horizon_s=job.horizon_s, fundamental=FundamentalConfig(sigma=0.03), sample_interval_s=0.1)
    parts = [NoiseTrader(arrivals=HawkesArrival.with_mean_rate(65.0, 0.6, 8.0)), InformedTrader(), MetaOrderTrader()]
    src = Simulator(cfg, parts).run()
    md = events_to_market_data(src.events)
    rep = validate_market_data(md)
    rp = HistoricalReplay(md, track_fidelity=True)
    t0 = time.perf_counter()
    res = Simulator(SimConfig(seed=0, horizon_s=job.horizon_s, seed_levels=0, warmup_s=0.0, sample_interval_s=0.1), [rp]).run()
    dt = time.perf_counter() - t0
    rp.finalize(res.book)
    a, b = src.samples, res.samples
    same_samples = all(np.array_equal(a[k][1:], b[k][1:], equal_nan=True) for k in ("best_bid", "best_ask", "bid_depth5", "ask_depth5", "bid_qty1", "ask_qty1"))
    return {"seed": job.seed, "feed_events": rep["n"], "feed_trades": rep["n_trades"], "feed_level_updates": rep["n_level_updates"],
            "touch_fidelity": rp.fidelity, "trade_price_mismatches": rp.rec.n_trade_price_mismatch, "crossing_adds": rp.rec.n_crossing_adds,
            "final_levels_equal": float(_levels(src.book) == _levels(res.book)), "tape_equal": float(_tape(src) == _tape(res)),
            "sampled_book_equal": float(same_samples), "replay_events_per_s": rep["n"] / dt}


def run(ctx: RunContext) -> None:
    t0 = time.time()
    seeds = ctx.seeds(default=8, quick=2)
    horizon = 30.0 if ctx.quick else 180.0
    rows = pmap(session, [Job(s, horizon) for s in seeds], ctx.workers)
    save_csv(ctx.out_dir / "roundtrip.csv", rows)
    for k in ("touch_fidelity", "trade_price_mismatches", "crossing_adds", "final_levels_equal", "tape_equal", "sampled_book_equal"):
        print(f"{k:24s} min={min(r[k] for r in rows):.4f}  max={max(r[k] for r in rows):.4f}")
    print(f"feed events per session: {np.mean([r['feed_events'] for r in rows]):.0f}   replay speed: {np.mean([r['replay_events_per_s'] for r in rows]):,.0f} feed events/s")
    write_provenance(ctx, {"sessions": len(seeds), "horizon_s": horizon}, dataset="synthetic feed exported from the simulator",
                     files=["roundtrip.csv"], elapsed_s=time.time() - t0)
