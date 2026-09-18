"""Shared helpers for experiments on the recorded real feed."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from simulator.order_flow.historical import HistoricalReplay
from simulator.order_flow.market_data import MD, read_market_data
from simulator.participant import Participant
from simulator.simulator import NS, SimConfig, SimResult, Simulator

DEFAULT_PATH = Path("data/processed/kraken_BTCUSD_A.parquet")


def load_dataset(path: Path = DEFAULT_PATH) -> tuple[np.ndarray, dict]:
    if not Path(path).exists():
        raise SystemExit(f"{path} not found: record it with data/collect_kraken.py and run data/prepare_kraken.py first")
    arr, meta = read_market_data(path)
    return arr, meta


def dataset_id(meta: dict) -> str:
    return (f"{meta['source']}; {meta['symbol']}; raw {meta['raw_file']} sha256={meta['raw_sha256'][:16]}...; "
            f"{meta['quality']['span_s']:.0f} s; tick {meta['tick_size']} lot {meta['lot_size']}")


def unit_value(meta: dict) -> float:
    """Currency per (tick x lot): the engine's P&L unit for this dataset (USD here)."""
    return meta["tick_size"] * meta["lot_size"] * 1.0  # USD per tick per lot, given price in USD/BTC and lot in BTC


def state_at(arr: np.ndarray, t_ns: int) -> tuple[dict, dict, int]:
    """Book (bids, asks) implied by the feed just before ``t_ns`` and the index of the first event at or after it."""
    i = int(np.searchsorted(arr[:, 0], t_ns, side="left"))
    bids: dict[int, int] = {}
    asks: dict[int, int] = {}
    for _, kind, side, price, qty in arr[:i]:
        if kind == MD.RESET:
            bids.clear(); asks.clear()
        elif kind == MD.LEVEL:
            book = bids if side == 1 else asks
            if qty > 0:
                book[int(price)] = int(qty)
            else:
                book.pop(int(price), None)
    return bids, asks, i


def window(arr: np.ndarray, t0_ns: int, t1_ns: int) -> np.ndarray:
    """Feed events for ``[t0, t1)`` preceded by the book state at ``t0`` (as LEVEL events stamped ``t0``), so a replay can start mid-stream."""
    bids, asks, i = state_at(arr, t0_ns)
    j = int(np.searchsorted(arr[:, 0], t1_ns, side="left"))
    init = [(t0_ns, int(MD.LEVEL), 1, p, q) for p, q in bids.items()] + [(t0_ns, int(MD.LEVEL), -1, p, q) for p, q in asks.items()]
    return np.vstack([np.array(init, dtype=np.int64).reshape(-1, 5), arr[i:j]])


def replay(arr: np.ndarray, meta: dict, participants: Optional[list[Participant]] = None, sample_s: float = 0.1, track: bool = False,
           warmup_s: float = 0.0, seed: int = 0, record_events: bool = False) -> tuple[SimResult, HistoricalReplay]:
    """Replay ``arr`` (starting at its first timestamp = sim time 0) alongside optional participants."""
    rp = HistoricalReplay(arr, track_fidelity=track)
    span_ns = int(arr[-1, 0] - arr[0, 0])
    cfg = SimConfig(seed=seed, horizon_s=span_ns / NS + 0.001, tick_size=unit_value(meta), seed_levels=0, warmup_s=warmup_s, sample_interval_s=sample_s,
                    record_events=record_events, record_commands=False)
    res = Simulator(cfg, [rp] + list(participants or [])).run()
    rp.finalize(res.book)
    return res, rp


def blocks(span_ns: int, block_s: float, skip_s: float = 0.0) -> list[tuple[int, int]]:
    """Consecutive block boundaries ``[t0, t1)`` in sim time, skipping the first ``skip_s`` seconds (warm-up)."""
    step = int(block_s * NS)
    return [(t, t + step) for t in range(int(skip_s * NS), span_ns - step + 1, step)]


def block_ci(values, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and bootstrap CI over blocks (blocks are treated as exchangeable; adjacent blocks are in truth correlated)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))
