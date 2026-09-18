"""Normalised market-data events: the common format between historical feeds and the simulator.

Public exchange data usually comes as (a) an L2 feed - *absolute* quantity per price level, updated incrementally - and
(b) a trade tape. It rarely has order ids (no L3). Both are normalised to integer ticks / lots / nanoseconds and stored as
an ``(n, 5)`` int64 array with columns ``ts_ns, kind, side, price, qty``:

============ ======================================================================================================
``TRADE``    an execution: ``side`` = aggressor (+1 buyer-initiated, consumes asks), ``price``, ``qty``
``LEVEL``    absolute quantity at a price level after an update: ``side`` +1 bid / -1 ask, ``qty`` = new total (0 = gone)
``RESET``    the feed restarted (e.g. a snapshot follows): every level is void
============ ======================================================================================================

Conversions: price ticks = round(price / tick_size); quantity lots = round(qty / lot_size) (a non-zero quantity that rounds
to zero is kept as 1 lot so a level never silently vanishes). Files are Parquet with the metadata (tick size, lot size,
source, symbol, license note) stored alongside - a dataset without its provenance is not usable for an experiment.
"""
from __future__ import annotations

import json
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from engine.common import EventType, Side
from engine.events import Event

COLUMNS = ("ts_ns", "kind", "side", "price", "qty")


class MD(IntEnum):
    TRADE = 0
    LEVEL = 1
    RESET = 2


def to_ticks(price: np.ndarray, tick_size: float) -> np.ndarray:
    return np.rint(np.asarray(price, dtype=float) / tick_size).astype(np.int64)


def to_lots(qty: np.ndarray, lot_size: float) -> np.ndarray:
    q = np.asarray(qty, dtype=float)
    lots = np.rint(q / lot_size).astype(np.int64)
    return np.where((q > 0) & (lots == 0), 1, lots)


def write_market_data(path: str | Path, arr: np.ndarray, meta: Optional[dict] = None) -> None:
    """Write a normalised event array as Parquet with ``meta`` (JSON) in the file metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({c: arr[:, i] for i, c in enumerate(COLUMNS)})
    table = table.replace_schema_metadata({b"lob_meta": json.dumps(meta or {}, sort_keys=True).encode()})
    pq.write_table(table, path)


def read_market_data(path: str | Path) -> tuple[np.ndarray, dict]:
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    arr = np.column_stack([t.column(c).to_numpy() for c in COLUMNS]).astype(np.int64)
    meta = json.loads((t.schema.metadata or {}).get(b"lob_meta", b"{}"))
    return arr, meta


def validate_market_data(arr: np.ndarray) -> dict:
    """Sanity checks on a normalised stream; returns a report (raises ``ValueError`` on structural errors)."""
    if arr.ndim != 2 or arr.shape[1] != len(COLUMNS):
        raise ValueError("market data must be an (n, 5) array")
    ts = arr[:, 0]
    if len(ts) and np.any(np.diff(ts) < 0):
        raise ValueError("timestamps must be non-decreasing")
    kind, side, price, qty = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    if not np.isin(kind, [int(k) for k in MD]).all():
        raise ValueError("unknown event kind")
    lv, tr = kind == MD.LEVEL, kind == MD.TRADE
    if not np.isin(side[lv | tr], [-1, 1]).all():
        raise ValueError("side must be +-1 for TRADE/LEVEL")
    if (price[lv | tr] < 1).any() or (qty[lv] < 0).any() or (qty[tr] < 1).any():
        raise ValueError("non-positive price, negative level quantity or non-positive trade quantity")
    return {"n": int(len(arr)), "n_trades": int(tr.sum()), "n_level_updates": int(lv.sum()), "n_resets": int((kind == MD.RESET).sum()),
            "span_s": float((ts[-1] - ts[0]) / 1e9) if len(ts) else 0.0}


def events_to_market_data(events: Iterable[Event]) -> np.ndarray:
    """Export a LOB event stream as the L2 + trade feed an exchange would publish (used for round-trip testing).

    Trades are printed before the level update they cause (as real feeds do); order identity is discarded.
    """
    levels: dict[tuple[int, int], int] = {}
    rows: list[tuple[int, int, int, int, int]] = []

    def bump(ts: int, side: int, price: int, delta: int) -> None:
        q = levels.get((side, price), 0) + delta
        assert q >= 0, "negative level quantity in source events"
        levels[(side, price)] = q
        rows.append((ts, int(MD.LEVEL), side, price, q))

    for e in events:
        t = e.type
        if t is EventType.ADD:
            bump(e.ts, int(e.side), e.price, e.qty)
        elif t is EventType.TRADE:
            rows.append((e.ts, int(MD.TRADE), int(e.side), e.price, e.qty))
            bump(e.ts, -int(e.side), e.price, -e.qty)
        elif t is EventType.CANCEL:
            bump(e.ts, int(e.side), e.price, -e.qty)
        elif t is EventType.MODIFY:
            if e.keeps_priority:
                bump(e.ts, int(e.side), e.price, e.new_qty - e.qty)
            else:
                bump(e.ts, int(e.side), e.price, -e.qty)
                bump(e.ts, int(e.side), e.new_price, e.new_qty)
    return np.array(rows, dtype=np.int64).reshape(-1, len(COLUMNS))
