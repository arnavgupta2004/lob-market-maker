# Data

**No external dataset is bundled or has been downloaded.** Stage 8 built and validated the replay machinery on a *synthetic*
feed with known ground truth (`experiments.validation.replay_roundtrip`); a real dataset requires an explicit decision on source,
symbol, date range and licence, and a download that is approved beforehand.

## Normalised format (`simulator/order_flow/market_data.py`)

Parquet with columns `ts_ns, kind, side, price, qty` (int64: nanoseconds, kind, +-1, integer ticks, integer lots) and file metadata
(`lob_meta` JSON: tick size, lot size, source, symbol, licence note). `kind`: `TRADE` (side = aggressor), `LEVEL` (absolute quantity at a
price after an update; 0 removes it), `RESET` (feed restarted). `validate_market_data` rejects unsorted timestamps, bad sides, non-positive
prices/trade sizes and negative level quantities.

## What a real dataset must provide, and how it is used

| need | why |
|---|---|
| L2 incremental updates (absolute quantity per level) **and** a trade tape with aggressor side, with exchange timestamps | replayed through `HistoricalReplay` so the same market-maker code runs on synthetic and historical flow |
| timestamp resolution (record it in the metadata) | sets the resolution of every latency/fill statement; 1 ms data cannot support 1 ms-latency conclusions |
| snapshot rows / feed restarts | become `RESET` events |
| tick size and lot size | prices to ticks, quantities to lots |

Documented for every dataset actually used (to be filled in when one is chosen): source and URL, licence/terms and why they permit this use,
symbol, date range, timestamp resolution and clock source, field list, gaps/outages, preprocessing applied (deduplication, unit conversion),
and the SHA-256 of the raw files. `experiments.common.write_provenance(dataset=...)` records the dataset string in every result.

## Assumptions the replay makes (see `simulator/order_flow/historical.py`)

* L2 carries no order identity: quantity increases become one new order at the back of the queue; decreases that are not trades are removed
  from the back of the replay's own orders at that price. Real cancels can be anywhere in a queue, so a strategy's queue position relative to
  historical orders is approximate.
* Trades are replayed as market orders; the level update that follows is reconciled against the engine's actual state, so a trade's depletion
  is never double counted.
* A strategy's orders displace historical liquidity; the replay is not counterfactual-exact once the strategy trades.

## Loaders

`simulator/order_flow/loaders.py` has config-driven CSV loaders (`Layout`) with presets for a vendor L2-incremental + trades layout and an
exchange aggregate-trades layout. **The presets follow public documentation as I understand it and are tested only against hand-written
fixtures - they have not been run on real files.** Use `describe_file` to check a real file's layout first and adjust the `Layout`.
