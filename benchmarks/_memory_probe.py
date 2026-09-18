"""Child process for memory measurement: build a book from a command file and report the RSS growth.

    python -m benchmarks._memory_probe {python|cpp} commands.npy
Prints one JSON line. RSS is read with ``ps`` (current, not peak) before and after building the book so the
command list itself (created before the first reading) is excluded.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np


def rss_kb() -> int:
    return int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout.strip())


def main() -> None:
    impl, path = sys.argv[1], sys.argv[2]
    arr = np.load(path)
    if impl == "python":
        from engine.commands import Cancel, Modify, NewLimit, NewMarket, SnapshotRequest
        from engine.common import Side
        from engine.python.order_book import OrderBook
        cmds = []
        for op, oid, side, price, qty, ts, owner, flags in arr.tolist():
            if op == 0:
                cmds.append(NewLimit(oid, Side(side), price, qty, ts, owner, bool(flags & 1), bool(flags & 2)))
            elif op == 1:
                cmds.append(NewMarket(oid, Side(side), qty, ts, owner))
            elif op == 2:
                cmds.append(Cancel(oid, ts))
            elif op == 3:
                cmds.append(Modify(oid, price, qty, ts))
            else:
                cmds.append(SnapshotRequest(ts))
        book = OrderBook(record_events=False)
        base = rss_kb()
        for c in cmds:
            book.process(c)
    else:
        from engine.cpp_engine import CppOrderBook
        book = CppOrderBook()
        base = rss_kb()
        book.run_batch(arr)
    after = rss_kb()
    n = len(book)
    print(json.dumps({"impl": impl, "resting_orders": n, "commands": int(len(arr)), "rss_delta_kb": after - base,
                      "bytes_per_resting_order": (after - base) * 1024 / n if n else None}))


if __name__ == "__main__":
    main()
