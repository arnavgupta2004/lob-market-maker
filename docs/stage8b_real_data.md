# Stage 8b - real order flow: Kraken BTC/USD, one recorded hour

```bash
python data/collect_kraken.py --symbol BTC/USD --depth 100 --seconds 3600 --out data/raw/kraken_BTCUSD_A.jsonl   # record (network, public data)
python data/prepare_kraken.py                                                                                    # normalise + integrity report
python -m experiments.run real_data --seed 42        # replay + stylized facts + OBI + impact + adverse selection, vs the simulator
python -m experiments.run real_mm   --seed 42        # the market makers on replayed real flow
```

## 1. Dataset (documented as the specification requires)

| item | value |
|---|---|
| source | Kraken public WebSocket API v2 (`wss://ws.kraken.com/v2`), channels `book` (depth 100) and `trade`; no authentication; recorded by `data/collect_kraken.py` |
| instrument / window | BTC/USD; feed timestamps 2026-09-18 22:42:24 - 23:42:34 UTC (3,597 s of continuous data, 0 reconnects) |
| raw file | `kraken_BTCUSD_A.jsonl`, 61,154,809 bytes, SHA-256 `0c2227334d650118...` (full hash in `data/raw/*.meta.json` and in every result's provenance) |
| contents | 262,670 book updates + 1 snapshot, 2,141 trade messages (4,044 individual fills), 3,592 heartbeats |
| timestamp resolution | microseconds, exchange timestamps as published (the snapshot has none: local receive time is used); 208 of 459,568 events (0.045%) arrived with a timestamp earlier than their predecessor and were clamped forward |
| fields used | book: price, absolute quantity per level (0 = removed), per-update CRC32 checksum; trades: side (= aggressor), price, quantity, timestamp |
| instrument precision | tick 0.1 USD, quantity 8 decimals (from the public AssetPairs endpoint); engine units: tick 0.1 USD, **lot 1e-8 BTC** |
| licence | public market data streamed for personal research. **Kraken's terms were not independently verified.** The raw and processed data are git-ignored and not redistributed; only aggregate statistics derived from it are committed |
| preprocessing | reconstruct the book from snapshot + updates; **truncate to the best 100 levels per side after every update (mirrors the exchange window)**; trades before level updates at equal timestamps, otherwise stream order; timestamps forced non-decreasing; a collector reconnect (none occurred) would become a `RESET` |

**Integrity check - the reconstructed book equals the exchange's book.** Kraken publishes a CRC32 of its top 10 levels on every update. My reconstruction matches it on **262,671 / 262,671 updates (100.000%)**.
This mattered: without mirroring the depth-100 truncation the match rate was only 78.9% (a desync window once the price moved far enough that levels the exchange had silently dropped were still in my book); depth 50 and 25 gave 90.5% and 66.2%. So the checksum both
validated the pipeline and caught a real reconstruction error in it.

**Replay fidelity.** Replayed through `HistoricalReplay`, the engine's touch equals the feed's touch after **99.80%** of level updates; **0 of 4,044** replayed trades executed at a price different from the print; 0 crossing insertions. Two lessons recorded: (a) a lot of 1e-4 BTC left 1-2 lot residues at
swept levels and mis-priced 1.4-2.3% of trades - Kraken quantities have 8 decimals, so lot = 1e-8 makes the conversion exact and removed all mismatches; (b) a trade-price check that ignores the trades' own depletion of levels wrongly flags every multi-level sweep, so fidelity is measured on the *engine's* state.

## 2. The market is very different from the simulator's

| | simulator (`noise` .. `hawkes+meta`) | real hour |
|---|---|---|
| mid volatility | ~2 ticks/sqrt(s) | **22.4** (1 s), 26.3 (10 s), 31.4 (60 s) ticks/sqrt(s) |
| spread (ticks) | mean 1.5-2.05, median 1-1.7 | mean 1.14, median 1 (a *large-tick*, tick = 0.12 bp market) |
| trade rate | ~15-25 aggressive orders/s | 1.12 fills/s (1,988 parent orders after merging same-instant sweep prints) |
| 1 s returns exactly zero | 12-24% | **85%** |
| trend / mean reversion (variance ratio) | VR(5 s) 0.72-0.77 (mean reverting) | **VR(10 s) 1.38, VR(60 s) 1.98** (trending; one hour can do this by chance) |

Consequently every tick-denominated strategy parameter had to be rescaled for the real-flow backtest (section 4).

## 3. Stylized facts, order-book imbalance, impact, adverse selection - real vs simulator

Uncertainty: 11 consecutive 5-minute blocks of ONE recording, block bootstrap (blocks are correlated, so intervals are optimistic; only 11 blocks make them wide). `results/real_data/`.

| metric | real [95% block CI] | simulator variants |
|---|---|---|
| order-sign ACF, lag 1 (parent orders) | **0.144 [0.081, 0.198]** | 0.00 / 0.04 / 0.05 / 0.05 |
| sign-ACF power-law exponent | 0.33 [0.21, 0.45] | -0.01 / 0.32 / 0.39 / 0.41 |
| Fano factor of aggressive orders, 1 s / 10 s | 2.27 [1.67, 2.93] / 4.46 [1.9, 8.9] | 0.97-2.02 / 1.0-4.1 |
| inter-arrival CV | 1.20 [1.13, 1.29] | 0.99-1.08 |
| ACF of abs 1 s return, lag 1 s / lag 10 s | 0.087 [0.039, 0.139] / **0.020 [-0.010, 0.049]** | 0.12-0.14 / -0.02..0.00 |
| return ACF at 0.1 s, lag 1 | **+0.115 [+0.053, +0.181]** | -0.05 .. -0.06 |
| excess kurtosis of 10 s returns (pooled) | 6.2 [2.0, 14.4] | (not comparable: different horizons) |
| Hill tail index of 10 s returns, top 5% / 10% | 3.8 [2.3, 9.1] / 3.5 [2.6, 5.3] | (n/a) |
| Hill index of parent-order sizes (top 5%) | **1.37 [0.93, 2.03]** | 3.3 - 3.95 |

* **Where the simulator agrees.** Order signs are persistent with a long-memory-like exponent ~0.3-0.4, aggressive orders arrive in clusters (Fano > 1), and the simulator's Hawkes/metaorder variants land in the right region - all on the *same estimators*, so this is a like-for-like comparison. Heavy tails of the 10 s and 60 s returns (Hill ~2.7-3.8) are consistent with the widely reported ~3, but with an hour of data the CIs are far too wide (2.3 - 9.1) to test that.
* **Where it does not.** Real high-frequency returns are positively autocorrelated (trending) where the simulator's bounce; real volatility is ~10x higher in ticks; real order sizes are far heavier-tailed (alpha ~1.4); the real hour trends (VR > 1) where the simulator mean-reverts.
* **A finding that did not survive more data.** On the first 34 minutes the abs-return ACF at a 10 s lag was +0.041 [+0.014, +0.072] (persistent volatility clustering); on the full hour it is +0.020 [-0.010, +0.049]. So this hour does **not** establish persistent volatility clustering, and the simulator's lack of it is *not* confirmed as a realism gap by these data (Stage 8 called it "the clearest realism gap": that was a statement about the literature, not something this hour shows).

**Order-book imbalance** (`OBI`, touch, 0.1-10 s): forward correlation 0.127 [0.116, 0.139] at 0.1 s, 0.210 [0.173, 0.245] at 1 s, **0.261 [0.181, 0.336] at 5 s**, 0.250 [0.173, 0.315] at 10 s - it *increases* with horizon, where the simulator's decays (0.21 -> 0.09 noise, 0.18 -> 0.05 informed); the backward correlation is positive (+0.08 .. +0.20), unlike the simulator's negative touch value, so reverse causality / persistence is a live explanation here. In ticks the slope is 7.0 ticks per unit OBI at 1 s
(simulator 0.6), because real volatility is ~10x higher. **Economic size:** for |OBI| > 0.8 the mean forward move is **+8.7 ticks [+6.5, +11.0] at 1 s** and +21.3 [+12.8, +29.9] at 5 s (0.107 and 0.262 bp), against a half-spread of ~0.007 bp - so the signal is large *relative to the spread* (unlike the simulator, where it was below the half-spread) -
but a 1 bp fee is 81 ticks, so it is small relative to retail-level fees. This is an in-sample, mid-to-mid statement: it ignores latency, queue position and the impact of trading on it.

**Price impact** of a parent order (same-instant same-side prints merged), exponent alpha of `I ~ Q^alpha`: immediate 0.05 [-0.04, 0.12] (R^2 0.06), at 1 s **0.13 [0.02, 0.26]** (R^2 0.63), at 5 s **0.18 [0.01, 0.29]** (R^2 0.72); restricted to sizes >= 1e-3 BTC: 0.05 / 0.19 / 0.37. Impact is **concave over most of the range and only weakly size-dependent there**: roughly flat at ~20-30 ticks (1-5 s) from 3e-4 to 0.1 BTC (bins of 85-400 orders), then **rising for the few largest orders** (0.3-1 BTC: 38 / 84 ticks at 1 s / 5 s, n = 160; >= 1 BTC: 105 / 190 ticks, n = 18), which is why the fitted exponent is small but not zero. The simulator's impact is linear (alpha ~ 1). The commonly cited square-root law (0.5) is not supported by the all-sizes fits (upper CI 0.26-0.29), but the size range and ~2,000 orders are limited and I merge prints by timestamp, which may split or join true parent orders.

**Adverse selection of historical maker fills** (signed post-fill mid move `M_h`, ticks; 1 tick = 0.1 USD; per fill): `E` (effective half-spread) mean +3.5 / **median +0.5**; `M` at 100 ms **-17.4 [-21.9, -13.4]** (median -3.8), 500 ms -25.2 [-35.4, -17.3] (median -5.8), 1 s -27.1 [-37.9, -18.7], 5 s -37.2 [-58.1, -21.3], 10 s -45.4 [-66.5, -26.2] (median -29.9);
the random-time **null control is ~0** (|null| <= 0.31 at every horizon). Fills are followed by strong adverse moves, several times the spread captured, growing with the horizon (mean far below median: a few large sweeps dominate). The simulator's informed environment had adverse cost ~1.6 ticks per fill at ~1.7 ticks of half-spread; here the ratio is much worse.

## 4. Market makers on real flow (`real_mm`)

Calibrated on the **first 15 minutes** (probe participants inside the replay, the same procedure as in the simulator), evaluated on **8 disjoint 5-minute windows** after it (strategy starts 30 s into each): `k` = 0.0304/tick (R^2 0.89; fill intensity decays over ~33 ticks, vs 0.5-1.0 in the simulator), `beta_obi` = 1.86 ticks/OBI at 0.25 s, fill-model coefficients `(a, b_lnQ, b_d) = (-2.31, -0.031, -0.323)`.
Parameters rescaled by design (not tuned): quote 0.001 BTC, soft inventory limit 0.01 BTC (kill 0.02), volatility bounds (1, 200), quote-offset cap 200 ticks, sigma0 20, `lambda_I` 0.5 tick per quote-size at the limit; AS `gamma` chosen for the same skew per quote. Arms: AS, symmetric quotes at mid +- 1/k, + inventory, full; 0 and 10 ms one-way latency.

| 0 ms | gross P&L (bps of traded notional) [block CI] | net, maker 1 bp / taker 2.5 bp | fills / window | mean abs inventory (BTC) | E (ticks) | M_500ms (ticks) |
|---|---|---|---|---|---|---|
| AS (scaled) | **-1.25 [-1.86, -0.60]** | -2.25 | 39 | 0.0046 | +16.4 | -47.2 |
| symmetric | -1.55 [-2.26, -0.77] | -2.55 | 37 | 0.0049 | +16.2 | -47.9 |
| + inventory | -1.17 [-1.64, -0.71] | -2.17 | 43 | 0.0029 | +16.4 | -48.1 |
| full | -1.25 [-1.86, -0.69] | -2.25 | 24 | 0.0018 | +17.5 | -45.3 |

* **Every arm loses money on this real hour, gross and net of fees**; the losses are statistically distinguishable from zero. The mechanism is visible in the columns: the half-spread captured (+16 ticks) is about a third of the adverse move after the fill (-47 ticks at 500 ms). This is the same *pattern* as the simulator's informed environments, at a much larger scale.
* **Inventory control works on real flow as it did in the simulator**: mean |inventory| -0.0020 BTC [-0.0028, -0.0013] (about -40%) versus the symmetric quoter, at 0 ms.
* **The full adaptive maker "loses less" only because it trades less.** Its USD P&L is higher than +inventory's (+0.19 USD [+0.09, +0.30] at 0 ms), but it makes 19 fewer fills [-29, -9]; per unit of traded notional the loss is the same (-1.25 vs -1.17 bp). No evidence of better per-trade economics.
* **10 ms latency** changes little in P&L (gross -1.2 to -1.5 bp in every arm) but the effective half-spread falls (16.4 -> 12.3 ticks) and the post-fill move worsens (-47 -> -53 ticks), the direction the latency experiment predicts.
* **What is *not* shown.** These arms are the simulator-era designs with rescaled units, not strategies optimised for this market; the measured OBI signal is large relative to the spread and was used only through the modest `beta_obi` shift, so the result says nothing about what a better-designed strategy could earn.

## 5. Limitations (specific to the real data)

* **One hour, one instrument, one day-part**; a trending hour (VR > 1). Nothing generalises beyond it; a second hour or a different regime could change the sign of several comparisons.
* **Block bootstrap with 8-11 blocks** - wide and optimistic (adjacent blocks are correlated); ~190-340 fills per arm in total; USD amounts are tiny (a quote is 0.001 BTC, ~ $80; ~$2,000 traded per window), so read results in bps.
* **Replay is not counterfactual-exact.** The strategy's orders displace historical liquidity; historical participants do not react to it; queue position relative to historical orders is the back-of-queue approximation of an L2 feed (no order ids); the strategy sees the feed with no exchange-side effects (no rate limits, no fees in the quoting logic).
* **Parent orders are inferred** (same-timestamp same-side prints merged); Kraken prints one trade per fill.
* Licence/terms of the source were not verified; data is not redistributed.
