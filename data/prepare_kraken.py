"""Raw Kraken recording (``data/collect_kraken.py``) -> normalised market data (Parquet) + a data-quality report.

    python data/prepare_kraken.py --raw data/raw/kraken_BTCUSD_A.jsonl --out data/processed/kraken_BTCUSD_A.parquet

Unit conversion: tick 0.1 USD, lot 1e-8 BTC (Kraken quantities have 8 decimals, so conversion is exact; a coarser lot left 1-2 lot residues at swept levels and mis-priced 1.4-2.3% of replayed trades). The reconstructed book is truncated to the subscription depth (100) exactly as the
exchange does, and its top-10 CRC32 is checked against the exchange's checksum on every update: the match rate is stored in the
metadata as an independent integrity test of the reconstruction. Output is git-ignored (redistribution terms unverified).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from simulator.order_flow.loaders import load_kraken_jsonl
from simulator.order_flow.market_data import validate_market_data, write_market_data

TICK, LOT = 0.1, 1e-8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/kraken_BTCUSD_A.jsonl")
    ap.add_argument("--out", default="data/processed/kraken_BTCUSD_A.parquet")
    ap.add_argument("--depth", type=int, default=100)
    a = ap.parse_args()
    raw = Path(a.raw)
    arr, rep = load_kraken_jsonl(raw, TICK, LOT, price_decimals=1, qty_decimals=8, depth=a.depth)
    val = validate_market_data(arr)
    meta_path = Path(str(raw) + ".meta.json")
    src = json.loads(meta_path.read_text()) if meta_path.exists() else {"note": "raw metadata missing (recording still in progress?)"}
    meta = {"source": "Kraken public WebSocket API v2 (book depth %d + trade)" % a.depth, "symbol": "BTC/USD", "tick_size": TICK, "lot_size": LOT,
            "timestamp_resolution": "microseconds (exchange timestamps as published; snapshot uses local receive time)",
            "raw_file": raw.name, "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(), "raw_bytes": raw.stat().st_size,
            "collector_meta": src, "quality": rep, "validation": val, "depth": a.depth,
            "preprocessing": "book truncated to the best %d levels per side after every update (mirrors the exchange window); trades before level "
                             "updates at equal timestamps; forced non-decreasing timestamps (%d inversions); tiny quantities kept as >= 1 lot" % (a.depth, rep["ts_inversions"]),
            "license_note": "public market data streamed for personal research; terms not independently verified; not redistributed"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_market_data(a.out, arr, meta)
    print(json.dumps({"events": len(arr), **{k: rep[k] for k in ("span_s", "trades", "updates", "checksum_match_rate", "ts_inversions", "reconnects")}}, indent=2))


if __name__ == "__main__":
    main()
