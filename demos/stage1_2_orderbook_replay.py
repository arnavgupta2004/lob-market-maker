"""Stage 1-2 demonstration: FIFO matching, partial fills, cancels, modifies, and exact replay.

    python demos/stage1_2_orderbook_replay.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from engine.common import Side
from engine.events import read_jsonl, stream_hash, write_jsonl
from engine.python.order_book import OrderBook
from engine.replay import replay_events
from engine.workload import WorkloadParams, generate_workload

B, S = Side.BUY, Side.SELL


def show(book: OrderBook, title: str) -> None:
    bids, asks = book.depth()
    print(f"\n--- {title}")
    for p, q, n in reversed(asks):
        print(f"    ASK {p:>5}  qty {q:>3}  ({n} orders)")
    print(f"    spread={book.spread()} mid={book.mid_price()}")
    for p, q, n in bids:
        print(f"    BID {p:>5}  qty {q:>3}  ({n} orders)")


def scenario() -> OrderBook:
    b = OrderBook(record_events=True)
    for oid, px, q in [(1, 101, 5), (2, 101, 3), (3, 102, 4)]:
        b.submit_limit(oid, S, px, q, owner=oid)
    b.submit_limit(4, B, 99, 6)
    b.submit_limit(5, B, 100, 2)
    show(b, "initial book (orders 1,2 queue FIFO at 101)")

    ev = b.submit_market(10, B, 6)
    print("\nmarket BUY 6 ->", [(e.maker_id, e.qty, e.maker_remaining) for e in ev if e.type.value == "TRADE"])
    print("  (order 1 fully filled first, then 1 lot of order 2: partial fill, keeps its place)")
    show(b, "after market order")

    b.modify(2, 101, 1)
    print("\nmodify order 2: 2 -> 1 lots  (size decrease keeps priority)")
    b.submit_limit(6, S, 101, 4)
    print("new SELL 4@101 queues behind order 2:", b.queue_position(6))
    b.modify(6, 101, 6)
    print("modify order 6 size up 4 -> 6 (loses priority):", b.queue_position(6))
    b.cancel(3)
    show(b, "after cancel of order 3 (level 102 disappears)")

    ev = b.submit_limit(11, B, 105, 20)
    print("\naggressive BUY 20@105 ->", [(e.type.value, e.price, e.qty) for e in ev])
    show(b, "after aggressive limit (remainder rests; book never crossed)")
    b.validate()
    return b


def main() -> None:
    book = scenario()

    print("\n=== Stage 2: replay ===")
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "events.jsonl"
        write_jsonl(path, book.events)
        rebuilt = replay_events(read_jsonl(path))
        print(f"{len(book.events)} events written to JSONL, replayed from disk")
        print("replayed state == live state:", rebuilt.state() == book.state())

    params = WorkloadParams(n_commands=20_000, p_invalid=0.02, p_snapshot=0.001)
    live = OrderBook(record_events=True)
    live.process_all(generate_workload(42, params))
    again = OrderBook(record_events=True)
    again.process_all(generate_workload(42, params))
    print(f"\nrandom workload (seed 42): {len(live.events)} events, resting orders={len(live)}")
    print("sha256(events) run 1:", stream_hash(live.events)[:24], "...")
    print("sha256(events) run 2:", stream_hash(again.events)[:24], "...  identical:",
          stream_hash(live.events) == stream_hash(again.events))
    print("event-only replay reproduces book:", replay_events(live.events).state() == live.state())


if __name__ == "__main__":
    main()
