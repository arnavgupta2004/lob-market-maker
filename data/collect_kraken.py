"""Record Kraken's public WebSocket v2 book + trade channels to raw JSON lines (no authentication, public data only).

    python data/collect_kraken.py --symbol BTC/USD --depth 100 --seconds 1800 --out data/raw/kraken_BTCUSD.jsonl

Each output line is ``{"recv_ns": <local receive time>, "msg": <exact message>}`` so the raw stream is preserved verbatim; nothing
is filtered or normalised here (that is ``simulator.order_flow.loaders.load_kraken_jsonl``). On a dropped connection it reconnects and
re-subscribes (the new book snapshot becomes a RESET downstream). A ``<out>.meta.json`` records the source, the subscription, the
start/end times, message counts, reconnects and the instrument's tick/lot precision (from the public AssetPairs REST endpoint).

Terms: this reads Kraken's public market-data streams for personal research. Kraken's terms of use were not independently verified
by the author; raw data is git-ignored and must not be redistributed.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import websockets

WS_URL = "wss://ws.kraken.com/v2"


def asset_pair_info(symbol: str) -> dict:
    """Tick/lot precision of the instrument from the public REST endpoint (v2 symbol 'BTC/USD' -> REST pair 'XBTUSD')."""
    pair = symbol.replace("/", "").replace("BTC", "XBT")
    with urllib.request.urlopen(f"https://api.kraken.com/0/public/AssetPairs?pair={pair}", timeout=15) as r:
        res = json.load(r)["result"]
    (k, v), = res.items()
    return {"rest_pair": k, "pair_decimals": v["pair_decimals"], "lot_decimals": v["lot_decimals"], "tick_size": v.get("tick_size"),
            "ordermin": v.get("ordermin"), "costmin": v.get("costmin")}


async def record(symbol: str, depth: int, seconds: float, out: Path) -> dict:
    counts: dict[str, int] = {}
    reconnects, t_start = 0, time.time_ns()
    deadline = time.time() + seconds
    with open(out, "w", encoding="utf-8") as fh:
        while time.time() < deadline:
            try:
                async with websockets.connect(WS_URL, max_size=2 ** 24, ping_interval=20) as ws:
                    for p in ({"channel": "book", "symbol": [symbol], "depth": depth}, {"channel": "trade", "symbol": [symbol], "snapshot": False}):
                        await ws.send(json.dumps({"method": "subscribe", "params": p}))
                    while time.time() < deadline:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        msg = json.loads(raw)
                        key = f"{msg.get('channel') or msg.get('method')}/{msg.get('type') or ''}"
                        counts[key] = counts.get(key, 0) + 1
                        if msg.get("channel") == "heartbeat":
                            continue
                        fh.write(json.dumps({"recv_ns": time.time_ns(), "msg": msg}, separators=(",", ":")) + "\n")
            except (websockets.ConnectionClosed, asyncio.TimeoutError, OSError) as e:
                reconnects += 1
                fh.write(json.dumps({"recv_ns": time.time_ns(), "msg": {"channel": "_collector", "type": "reconnect", "reason": repr(e)}}) + "\n")
                await asyncio.sleep(2)
    return {"t_start_ns": t_start, "t_end_ns": time.time_ns(), "message_counts": counts, "reconnects": reconnects}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USD")
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=1800)
    ap.add_argument("--out", default="data/raw/kraken_BTCUSD.jsonl")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    info = asset_pair_info(a.symbol)
    stats = asyncio.run(record(a.symbol, a.depth, a.seconds, out))
    meta = {"source": "Kraken public WebSocket API v2", "url": WS_URL, "symbol": a.symbol, "channels": ["book", "trade"], "book_depth": a.depth,
            "instrument": info, "collector": "data/collect_kraken.py", "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "bytes": out.stat().st_size, "license_note": "public market data streamed for personal research; terms not independently verified; do not redistribute",
            **stats}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(json.dumps({k: meta[k] for k in ("bytes", "message_counts", "reconnects")}, indent=2))


if __name__ == "__main__":
    main()
