"""Loaders that turn public market-data files into the normalised array of ``simulator.order_flow.market_data``.

**Status.** The Kraken v2 JSONL loader (``load_kraken_jsonl``, bottom of this file) has been run on a real one-hour recording and is verified against the exchange's own book
checksum. The vendor-CSV layouts below (Tardis-style L2/trades, exchange aggregate trades) follow the vendors' public documentation as I understand it and are exercised only by
hand-written fixtures (``tests/unit/test_loaders.py``); they have NOT been run on real files. On first use with real data, check the layout with ``describe_file`` and adjust the
``Layout`` (column names / timestamp unit / side vocabulary) - every assumption is a field, not code.

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


# ------------------------------------------------------------------------------------------------------------ Kraken v2 (raw JSONL)
import json as _json
import zlib as _zlib
from datetime import datetime, timezone


def _iso_ns(ts: str) -> int:
    """RFC 3339 timestamp with up to 9 fractional digits -> integer ns since the epoch."""
    ts = ts.rstrip("Z")
    main, _, frac = ts.partition(".")
    base = int(datetime.strptime(main, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
    return base * 1_000_000_000 + int((frac + "000000000")[:9])


def kraken_checksum(bids: dict[float, float], asks: dict[float, float], price_decimals: int, qty_decimals: int) -> int:
    """Kraken's book checksum: CRC32 of the top 10 asks (ascending) then top 10 bids (descending), each level rendered as
    price and quantity with the decimal point and leading zeros removed (per the public v2 documentation)."""
    def fmt(x: float, d: int) -> str:
        return f"{x:.{d}f}".replace(".", "").lstrip("0") or "0"
    a = sorted(asks)[:10]
    b = sorted(bids, reverse=True)[:10]
    s = "".join(fmt(p, price_decimals) + fmt(asks[p], qty_decimals) for p in a) + "".join(fmt(p, price_decimals) + fmt(bids[p], qty_decimals) for p in b)
    return _zlib.crc32(s.encode()) & 0xFFFFFFFF


def load_kraken_jsonl(path: str | Path, tick_size: float, lot_size: float, price_decimals: int = 1, qty_decimals: int = 8,
                      depth: Optional[int] = None, verify_checksum: bool = True) -> tuple[np.ndarray, dict]:
    """Raw Kraken WebSocket v2 recording (``data/collect_kraken.py``) -> normalised events + a data-quality report.

    * book ``snapshot`` -> ``RESET`` + one ``LEVEL`` per level; ``update`` -> ``LEVEL`` events (absolute quantities, 0 removes);
    * ``trade`` -> ``TRADE`` (Kraken's ``side`` is the aggressor); a collector reconnect -> ``RESET``;
    * timestamps: the exchange timestamp of each message (microsecond resolution as published); a snapshot has none, so the local
      receive time is used (and forced non-decreasing);
    * ``depth``: if set, the reconstructed book is truncated to the best ``depth`` levels per side after each update (the exchange
      maintains a depth-limited book); levels pushed out are emitted as removals so the replay book stays consistent;
    * if ``verify_checksum``: after each update the CRC32 of the reconstructed top-10 is compared with the exchange's checksum -
      the report gives the match rate, i.e. an independent test that the reconstruction equals the exchange's book.
    """
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    rows: list[tuple[int, int, int, int, int]] = []
    rep = {"messages": 0, "bad_lines": 0, "snapshots": 0, "updates": 0, "trades": 0, "reconnects": 0, "checksum_checked": 0, "checksum_ok": 0,
           "max_levels_bid": 0, "max_levels_ask": 0, "truncated_levels_emitted": 0, "ts_inversions": 0, "first_ts_ns": None, "last_ts_ns": None}
    last_ts = 0
    RES, LVL, TRD = int(MD.RESET), int(MD.LEVEL), int(MD.TRADE)

    def ticks(p: float) -> int:
        return int(round(p / tick_size))

    def lots(q: float) -> int:
        n = int(round(q / lot_size))
        return 1 if (q > 0 and n == 0) else n

    def emit_level(ts: int, side: int, p: float, q: float) -> None:
        rows.append((ts, LVL, side, ticks(p), lots(q) if q > 0 else 0))

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = _json.loads(line)
            except ValueError:
                rep["bad_lines"] += 1  # e.g. the last line of a file still being written
                continue
            msg, recv = rec["msg"], rec["recv_ns"]
            ch, typ = msg.get("channel"), msg.get("type")
            if ch == "_collector":
                rep["reconnects"] += 1
                bids.clear(); asks.clear()
                rows.append((max(last_ts, recv), RES, 0, 0, 0))
                continue
            if ch not in ("book", "trade") or typ not in ("snapshot", "update"):
                continue
            rep["messages"] += 1
            for d in msg.get("data", []):
                ts = _iso_ns(d["timestamp"]) if d.get("timestamp") else recv
                if ts < last_ts:
                    rep["ts_inversions"] += 1
                    ts = last_ts  # forced non-decreasing; inversions are counted in the report
                last_ts = ts
                if rep["first_ts_ns"] is None:
                    rep["first_ts_ns"] = ts
                rep["last_ts_ns"] = ts
                if ch == "trade":
                    rep["trades"] += 1
                    rows.append((ts, TRD, 1 if d["side"] == "buy" else -1, ticks(d["price"]), max(lots(d["qty"]), 1)))
                    continue
                if typ == "snapshot":
                    rep["snapshots"] += 1
                    bids.clear(); asks.clear()
                    rows.append((ts, RES, 0, 0, 0))
                else:
                    rep["updates"] += 1
                for side, book, key in ((1, bids, "bids"), (-1, asks, "asks")):
                    for lv in d.get(key, []):
                        p, q = lv["price"], lv["qty"]
                        if q > 0:
                            book[p] = q
                        else:
                            book.pop(p, None)
                        emit_level(ts, side, p, q)
                if depth is not None:
                    for side, book, worst in ((1, bids, lambda b: sorted(b)[:-depth]), (-1, asks, lambda a: sorted(a, reverse=True)[:-depth])):
                        if len(book) > depth:
                            for p in worst(book):
                                book.pop(p)
                                emit_level(ts, side, p, 0.0)
                                rep["truncated_levels_emitted"] += 1
                rep["max_levels_bid"] = max(rep["max_levels_bid"], len(bids))
                rep["max_levels_ask"] = max(rep["max_levels_ask"], len(asks))
                if verify_checksum and d.get("checksum") is not None and bids and asks:
                    rep["checksum_checked"] += 1
                    rep["checksum_ok"] += int(kraken_checksum(bids, asks, price_decimals, qty_decimals) == d["checksum"])
    arr = np.array(rows, dtype=np.int64).reshape(-1, 5)
    if len(arr):  # at equal time: trades first (a print precedes the level update it causes); everything else keeps STREAM order
        prio = np.where(arr[:, 1] == TRD, 0, 1)  # (a snapshot's RESET precedes its levels in the stream; a reconnect's RESET follows the updates it voids)
        arr = arr[np.lexsort((np.arange(len(arr)), prio, arr[:, 0]))]
    rep["checksum_match_rate"] = rep["checksum_ok"] / rep["checksum_checked"] if rep["checksum_checked"] else float("nan")
    rep["span_s"] = (rep["last_ts_ns"] - rep["first_ts_ns"]) / 1e9 if rep["first_ts_ns"] else 0.0
    return arr, rep
