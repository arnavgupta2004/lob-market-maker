"""Loaders that turn public market-data files into the normalised array of ``simulator.order_flow.market_data``.

**Status: the column layouts below follow the vendors' public documentation as I understand it and are exercised only by
hand-written fixtures (``tests/unit/test_loaders.py``). They have NOT been run against real files. On first use with real
data, check the layout with ``describe_file`` and adjust the ``Layout`` (column names / timestamp unit / side vocabulary) -
every assumption is a field, not code.**

Ordering convention when a trade print and level updates share a timestamp: the TRADE comes first (a print precedes the level
update it causes); the sort is stable so intra-message order is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from simulator.order_flow.market_data import MD, to_lots, to_ticks


@dataclass(frozen=True)
class Layout:
    """Column names and vocabularies of one file type."""

    ts: str
    price: str
    qty: str
    side: str
    ts_unit_ns: int = 1_000  # nanoseconds per timestamp unit (1_000 = microseconds, 1_000_000 = milliseconds)
    is_snapshot: Optional[str] = None  # L2 files: boolean column marking snapshot rows
    bid_words: tuple[str, ...] = ("bid", "buy")  # L2: values of ``side`` meaning the bid book
    buy_words: tuple[str, ...] = ("buy", "true")  # trades: values of ``side`` meaning buyer-initiated (aggressor buys)
    has_header: bool = True
    columns: Optional[tuple[str, ...]] = None  # names to assign when the file has no header


# Layout presets (unverified against real files - see module docstring).
TARDIS_L2 = Layout(ts="timestamp", price="price", qty="amount", side="side", ts_unit_ns=1_000, is_snapshot="is_snapshot",
                   bid_words=("bid",))
TARDIS_TRADES = Layout(ts="timestamp", price="price", qty="amount", side="side", ts_unit_ns=1_000, buy_words=("buy",))
# Exchange aggregate trades: the flag says whether the BUYER was the maker, i.e. is_buyer_maker=True => aggressor SOLD.
BINANCE_AGGTRADES = Layout(ts="transact_time", price="price", qty="quantity", side="is_buyer_maker", ts_unit_ns=1_000_000,
                           buy_words=("false",), has_header=False,
                           columns=("agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time",
                                    "is_buyer_maker", "is_best_match"))


def describe_file(path: str | Path, n: int = 3) -> str:
    """First lines of a file - to eyeball a layout before trusting a loader."""
    with open(path, "r", encoding="utf-8") as fh:
        return "".join(fh.readline() for _ in range(n))


def _read(path: str | Path, lay: Layout) -> pd.DataFrame:
    if lay.has_header:
        return pd.read_csv(path)
    return pd.read_csv(path, header=None, names=list(lay.columns or ()))


def load_l2_updates(path: str | Path, lay: Layout, tick_size: float, lot_size: float) -> np.ndarray:
    """L2 incremental file -> LEVEL events (+ a RESET before each snapshot block)."""
    df = _read(path, lay)
    ts = df[lay.ts].to_numpy(dtype=np.int64) * lay.ts_unit_ns
    side = np.where(df[lay.side].astype(str).str.lower().isin(lay.bid_words), 1, -1).astype(np.int64)
    price = to_ticks(df[lay.price].to_numpy(), tick_size)
    qty = to_lots(df[lay.qty].to_numpy(), lot_size)
    kind = np.full(len(df), int(MD.LEVEL), dtype=np.int64)
    out = np.column_stack([ts, kind, side, price, qty])
    if lay.is_snapshot is not None:
        snap = df[lay.is_snapshot].astype(str).str.lower().isin(("true", "1")).to_numpy()
        starts = np.flatnonzero(snap & ~np.r_[False, snap[:-1]])
        resets = np.column_stack([ts[starts], np.full(len(starts), int(MD.RESET)), np.zeros(len(starts), np.int64),
                                  np.zeros(len(starts), np.int64), np.zeros(len(starts), np.int64)])
        out = _insert_before(out, starts, resets)
    return out


def load_trades(path: str | Path, lay: Layout, tick_size: float, lot_size: float) -> np.ndarray:
    """Trade file -> TRADE events (aggressor side from the file's convention)."""
    df = _read(path, lay)
    ts = df[lay.ts].to_numpy(dtype=np.int64) * lay.ts_unit_ns
    is_buy = df[lay.side].astype(str).str.lower().isin(lay.buy_words).to_numpy()
    side = np.where(is_buy, 1, -1).astype(np.int64)
    return np.column_stack([ts, np.full(len(df), int(MD.TRADE), dtype=np.int64), side,
                            to_ticks(df[lay.price].to_numpy(), tick_size), np.maximum(to_lots(df[lay.qty].to_numpy(), lot_size), 1)])


def _insert_before(arr: np.ndarray, idx: np.ndarray, rows: np.ndarray) -> np.ndarray:
    if len(idx) == 0:
        return arr
    return np.insert(arr, idx, rows, axis=0)


def merge_streams(*streams: np.ndarray) -> np.ndarray:
    """Merge normalised streams by time. Stable; at equal timestamps TRADE < RESET < LEVEL is *not* imposed globally - the
    input order is kept within a stream and TRADEs are placed before LEVEL/RESET rows of other streams at the same time."""
    arr = np.vstack([s for s in streams if len(s)])
    priority = np.where(arr[:, 1] == int(MD.TRADE), 0, np.where(arr[:, 1] == int(MD.RESET), 1, 2))
    order = np.lexsort((np.arange(len(arr)), priority, arr[:, 0]))
    return arr[order]


def drop_crossed_and_dedupe(arr: np.ndarray) -> np.ndarray:
    """Remove exact duplicate consecutive LEVEL updates (same side/price/qty) - they are no-ops for the reconciler."""
    keep = np.ones(len(arr), dtype=bool)
    last: dict[tuple[int, int], int] = {}
    for i, (t, k, s, p, q) in enumerate(arr):
        if k == int(MD.LEVEL):
            if last.get((int(s), int(p))) == int(q):
                keep[i] = False
            last[(int(s), int(p))] = int(q)
        elif k == int(MD.RESET):
            last.clear()
    return arr[keep]
