# Stage 8 - historical-data replay machinery and simulator validation

**Scope and status.** No real dataset has been approved or downloaded, so this stage delivers (1) the replay interface and its
proof of fidelity on a synthetic feed with known ground truth, (2) validated stylized-fact estimators and two richer flow models,
and (3) the simulator's own stylized facts and book-resilience response. **The comparison of simulated versus *empirical* properties
is not done** - it needs the dataset (see `data/README.md`). Nothing below is a calibration to a real asset.

```bash
python -m experiments.run replay_roundtrip --seed 42 --workers 8
python -m experiments.run stylized_facts   --seed 42 --workers 8
python -m experiments.run resilience       --seed 42 --workers 8
```

## 1. Replay through the common interface

`HistoricalReplay` is a `Participant` like any synthetic flow, so a market maker runs unchanged on either
(`simulator/order_flow/historical.py`). A normalised feed (`market_data.py`: `TRADE` / `LEVEL` (absolute level quantity) / `RESET`) is turned
into order-level flow by `L2Reconciler`: quantity up = a new order at the back, quantity down = removal from the back of the replay's own
orders (cancel or in-place shrink), trade = market order, reset = cancel all. One feed event is processed per wake-up so each
reconciliation sees the engine after the previous event. `owner_orders_at()` was added to both engines (differentially tested) for this.

**Round-trip evidence** (`replay_roundtrip`, 8 sessions x 180 s, ~27,750 feed events each, flow = Hawkes noise + informed + metaorder):
a simulated exchange is exported as an L2 + trade feed and replayed into an empty engine. Result: after **every** level update the replayed
touch equals the feed's touch (fidelity 1.000, min = max over sessions); 0 trade-price mismatches; 0 crossing adds; identical final
(price, quantity) per level; identical trade tape (time, price, aggressor; quantity aggregated per fill group); identical sampled top-of-book and
5-level depth; ~84,000 feed events/s. The number of *orders* per level differs by construction (L2 discards order granularity), and this is
asserted, not hidden.

A bug I hit on the way: fidelity first read 93% because I compared *after trade prints*, when the feed's own book is transiently inconsistent
(the trade has depleted a level the feed has not yet updated). The correct comparison is after `LEVEL`/`RESET` events.

**Limits of what this proves.** It proves the reconstruction is exact *when the feed is a consistent L2 stream from an exchange with these
matching rules*. Real feeds have gaps, out-of-order messages, and cancels that are not at the back of the queue; those are the open risks.

## 2. Estimators and flow models (all tested against known ground truth)

`research/stylized_facts.py`: moments, Hill tail index, ACF, power-law ACF exponent, Fano factor, inter-arrival CV, spread stats. Tests use
Student-t / Pareto (Hill), AR(1) and sign chains (ACF), Laplace / uniform (kurtosis), Poisson vs cluster processes (Fano), GARCH (vol clustering).
Two findings from testing: Student-t(5) gives Hill 3.6 / 4.0 / 4.5 / 4.7 at the top 5% / 2% / 0.5% / 0.1% (true 5) - the Hill index is biased low
and depends on k, so it is always reported at two k; and the sample kurtosis of t(5) has infinite variance and cannot be tested against 6.

`HawkesArrival` (exact Ogata thinning; tests: long-run rate, clustering, Fano ~ theory range) and `MetaOrderTrader` (order splitting; test: sign-ACF
exponent recovered within 0.2 of the Lillo-Mike-Farmer prediction alpha - 1 for a Pareto(alpha = 1.5) metaorder-size tail).

## 3. Experiment G - the simulator's stylized facts (`stylized_facts`, 16 sessions x 600 s per variant, session-bootstrap CIs)

Four variants adding one mechanism at a time. Qualitative checks are *sign/shape* checks against widely reported facts, not a quantitative match.

| check (mean; PASS = CI on the right side) | noise | informed | hawkes | hawkes+meta |
|---|---|---|---|---|
| excess kurtosis of 1 s returns > 0 | PASS +10.6 | PASS +5.8 | PASS +6.2 | PASS +5.1 |
| ACF(abs return, lag 1 s) > 0 | PASS 0.13 | PASS 0.12 | PASS 0.12 | PASS 0.14 |
| ACF(abs return, lag 10 s) > 0 (persistence) | **fail** -0.02 | fail 0.00 | fail 0.00 | fail -0.01 |
| ACF(order sign, lag 1) > 0 | fail 0.00 | PASS 0.04 | PASS 0.05 | PASS 0.05 |
| sign-ACF exponent in (0.1, 1) | fail -0.01 | PASS 0.32 | PASS 0.39 | PASS 0.41 |
| Fano factor of aggressive orders (1 s) > 1.1 | fail 0.97 | fail 1.02 | PASS 1.68 | PASS 2.02 |
| spread varies (99th pct > median) | PASS | PASS | PASS | PASS |

Each mechanism does what it was added for: informed flow creates sign persistence (exponent 0.32 [0.28, 0.36]); Hawkes arrivals create arrival
clustering (Fano 1.02 -> 1.68 at 1 s, 2.2 at 10 s; inter-arrival CV 1.00 -> 1.07); metaorders add a little more (exponent 0.41 [0.36, 0.45]).

**Honest reading of the numbers behind the table.**
* *Kurtosis PASS is not evidence of heavy tails.* The noise-only value is the highest (+10.6) and 24% of its 1 s returns (60% at 0.1 s) are exactly zero:
  tick discreteness alone inflates it. The Hill index at 1 s is 3.0-3.4 at the top 5% but 3.7-4.6 at the top 1%, i.e. not a clean power law; I do not claim
  the "inverse cubic law".
* *Volatility clustering is short-lived only.* ACF(|r|) is +0.13 at 1 s lag but ~0 by 10 s in every variant: no persistent clustering. The fundamental has
  constant volatility and no mechanism couples volatility to activity. **This is the clearest realism gap and is not fixed.**
* *Order-sign persistence is weak and its exponent is not well determined.* Lag-1 sign ACF is 0.04-0.05 (empirical values for liquid assets are
  reported to be considerably larger); the power-law fit has R^2 of only 0.25-0.34; 0.41 is below the alpha - 1 = 0.5 theory value, plausibly because
  metaorder children are only ~10% of aggressive orders (0.3 parents/s x ~9 children vs ~25 other market orders/s) - a hypothesis, not tested. The
  informed variant already has exponent 0.32 *without* metaorders (a persistent value-driven trader), so exponent alone does not identify order splitting.
* Return autocorrelation at 0.1 s is -0.05 to -0.06 (bid-ask bounce / mean reversion, as in Stage 4) and ~0 at 1 s.
* Spread: median 1 tick, mean 1.5 (noise) to 2.0 (metaorder), 99th percentile 5-8 ticks. No empirical comparison is available.
* No parameter was tuned to move any of these numbers; the Hawkes (branching 0.6, beta 8/s) and metaorder (Pareto 1.5, 0.3/s) settings were fixed before the run.

## 4. Book resilience (`resilience`, 24 sessions x 19 shocks per size, paired against a no-shock control with identical seeds)

One aggressive market order of 20 or 60 lots every 30 s; effect = shock minus control, averaged per session, 95% session-bootstrap CIs.

| effect | noise, 60 lots | informed, 60 lots |
|---|---|---|
| mid impact right after / at 1 s / at 10 s (ticks) | +1.62 / +1.77 [1.55, 2.01] / **+1.74 [1.52, 1.96]** | +1.63 / +1.30 [1.09, 1.54] / **+0.09 [-0.36, +0.42]** |
| mid half-life | none: **permanent** | 2.5 s (20 lots: 0.8 s, CI wide) |
| spread excess | +1.02 initially, back to ~0 within ~0.1-0.15 s | +0.75, ~0.16 s |
| depth deficit (5 levels, side hit) | +0.45 initially, half-life ~0.7 s, 0 by 10 s | +0.51, ~0.4 s |

Larger shocks displace the book more (mid 0.53 -> 1.62, spread 0.32 -> 1.02, depth 0.15 -> 0.45 for 20 -> 60 lots in the noise market): H1 supported.
Spread and depth recover (H2 supported); the spread half-life is at the resolution floor (0.05 s sampling, 3-point smoothing). The mid **does not** recover
in the noise-only market (H3 supported): noise traders re-quote around the moved mid, so there is no restoring force - the same result as Stage 5's
permanent price impact. With informed flow the mid mean-reverts toward the fundamental, so recovery there is *not liquidity replenishment* but the fundamental
value reasserting itself, which is a different mechanism from the resilience seen in real markets. Depth being fully replenished while the price stays displaced is
an unrealistic decoupling that a real-data comparison would test.

## Verification

Tests added: `test_replay.py` (10: reconciler semantics, format validation, Parquet round trip with provenance, exact round trip, strategy sharing the book),
`test_loaders.py` (4, fixtures only), `test_stylized_facts.py` (9, ground-truth processes), `test_flow_models.py` (7), `test_resilience.py` (4), plus
`owner_orders_at` in both engines' differential tests and quick-mode CLI runs of all new experiments.
