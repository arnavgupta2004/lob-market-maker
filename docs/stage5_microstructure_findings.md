# Stage 5 - microstructure measurements (Experiments A, B, C and price impact)

Reproduce (deterministic; outputs and `provenance.json` in `results/<name>/`):

```bash
python -m experiments.run imbalance         --seed 42 --workers 8   # Experiment A
python -m experiments.run queue             --seed 42 --workers 8   # Experiment B
python -m experiments.run adverse_selection --seed 42 --workers 8   # Experiment C
python -m experiments.run price_impact      --seed 42 --workers 8   # impact exponent (feeds Experiment G)
```

Everything is measured from exchange ground truth (tape, true mid history, recorded book), in *simulated* markets only:
"noise" = Poisson noise flow; "informed" = noise + an informed fundamental-value trader (10 wake-ups/s, Brownian
fundamental, sigma = 0.03 price units/sqrt(s), tick = 0.01). 24 independent sessions per environment. Confidence intervals
bootstrap **sessions** (observations inside a session overlap and are not independent). Nothing here is calibrated to real
data; conclusions describe this simulator and are hypotheses for the real-data comparison in Stage 8.

---------------------------------------------------------------------------------------------------

## Experiment A - does order-book imbalance predict short-horizon price moves?

`OBI = (Vb - Va)/(Vb + Va)`, 10 ms sampling, horizons 10 ms ... 5 s, shuffle null, backward-looking control.

| noise, L1 (touch) | 10 ms | 100 ms | 500 ms | 1 s | 5 s |
|---|---|---|---|---|---|
| correlation with forward mid change | 0.09 | 0.209 [0.202, 0.215] | 0.211 [0.200, 0.222] | 0.171 | 0.088 |
| OLS slope (ticks per unit OBI) | 0.04 | 0.26 | 0.54 | 0.61 | 0.68 |
| directional accuracy - 1/2 (moves only) | 0.144 | 0.151 | 0.123 | 0.089 | 0.038 |
| backward correlation (OBI vs *past* move) | -0.075 | -0.164 | -0.148 | -0.114 | -0.053 |
| shuffled-OBI null | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Informed environment: same shape, lower correlation (0.182 at 100 ms) but larger slope (0.435 ticks/OBI at 100 ms).
Conditioning on spread = 1 tick raises the correlation to 0.27 (noise) and 0.25 (informed) at 100 ms.
Deep imbalance (5 levels) is weaker at short horizons (0.10 at 100 ms) but peaks later (0.14-0.15 at 0.5-1 s).

**Verdicts.** H1 (positive prediction at short horizons): supported. H2 (decays with horizon): supported - the
correlation peaks at 100-500 ms and falls to ~0.09 (noise) / 0.05 (informed) at 5 s.
**H3 not supported as stated**: I expected imbalance to be partly *caused* by recent price moves (positive backward
correlation). For the touch it is *negative* (a fresh level after a price move is thin on the side the price moved
toward), so persistence of past moves does not explain the forward result. For the 5-level measure the backward
correlation is positive (+0.04 to +0.32), so that variant is more exposed to reverse causality.
H4 (usable for skewing, not for crossing): supported - the most extreme bin (OBI > 0.8) predicts only +0.26 ticks in
100 ms and +0.58 ticks in 1 s (noise; informed +0.43 / +0.70) against a half-spread of ~0.75 (noise) / ~1.1 (informed) ticks.

**Interpretation and caveat.** The noise flow contains *no* information by construction, yet OBI predicts moves. The
predictability is mechanical: a thin touch queue is likely to be depleted by the next market order, so the mid ticks
in that direction. Whether real books show the same effect, at what size, is a Stage 8 question.

---------------------------------------------------------------------------------------------------

## Experiment B - queue position, fill probability and execution quality

1-lot passive probes at the touch and 1-2 ticks behind; exact `Q_t` (quantity ahead) read from the book every 250 ms;
outcome = fill within dt in {0.1, 0.5, 1, 2} s; only fully observed windows are used (no censoring bias).

**A pitfall worth recording.** Pooled over all orders, fill probability is *not* monotone in Q and a Q-only model has held-out
AUC 0.50 (noise) / 0.53 (informed). This is a confound, not a market fact: orders sitting further from the touch have small
Q but rarely fill. My first analysis stratified by the depth at *posting* and still showed non-monotone curves, because
the market moves after posting. Stratifying by the **current** distance from the touch `d_touch` gives the expected result:

| P(fill within 1 s) | Q = 0 | 1-2 | 3-5 | 6-10 | 11-20 | 21-40 | 41+ |
|---|---|---|---|---|---|---|---|
| noise, at the touch | 0.94 | 0.93 | 0.91 | 0.89 | 0.84 | 0.72 | 0.52 |
| noise, 1 tick behind | 0.55 | 0.55 | 0.51 | 0.49 | 0.43 | 0.36 | 0.26 |
| informed, at the touch | 0.93 | 0.92 | 0.90 | 0.86 | 0.81 | 0.71 | 0.60 |

Held-out AUC (train on even sessions, test on odd): noise Q only 0.50 -> +d_touch 0.81 -> full model 0.83; informed
0.53 -> 0.78 -> 0.79. Logistic coefficients (log-odds per +1 SD, 95% session-bootstrap CI), noise: `log1p Q` -0.23
[-0.26, -0.21], `d_touch` -2.14, spread -0.29, same-side depth -0.35, signed imbalance (`side*OBI`) -0.33 (more volume on our
own side = more competition), |1 s return| +0.10; flow intensity is not distinguishable from 0 in the noise environment
(+0.04 [-0.01, +0.10]) - expected, since noise arrivals are Poisson with a constant rate.

**Verdicts.** H1 (fill probability falls with Q): supported *conditional on distance to the touch*, with a qualification. At the
touch and one tick behind it declines steadily (dt = 0.5 s and 1 s were stratified); at >= 2 ticks behind the dependence on Q is weak
(roughly flat up to ~20 lots, then lower) because reaching the order at all is the binding constraint. The pooled curve is a confounded
artefact. H2 (market state adds power beyond Q): supported - spread, depth, signed imbalance and the returns have non-zero
coefficients, but distance to the touch dominates.
**H3 rejected**: I predicted fills from deep in the queue would be *more* adverse. Within each posting depth the opposite
holds: signed 500 ms post-fill move at the touch (noise) is -0.50 [-0.59, -0.42] ticks for q0 = 1-5, -0.39 for 6-15 and
-0.32 [-0.37, -0.28] for 16+ (informed: -0.82 -> -0.67 -> -0.65); realised half-spread rises accordingly (0.39 -> 0.47 -> 0.57
noise). A mechanism (e.g. front-of-queue orders are the ones filled by the first, most impactful sweeps) is a hypothesis
I have not tested. Practical reading: deeper queue = fewer fills but better per-fill quality - a trade-off, not a free lunch.

**Limitations.** 1-lot probes; probes perturb the book slightly (a few lots in flight at any time); queue position is exact
(no uncertainty about it, unlike real venues); fills within a session are dependent, hence session-level intervals.

---------------------------------------------------------------------------------------------------

## Experiment C - do market-maker fills predict subsequent price movement?

Avellaneda-Stoikov maker (gamma = 0.01, `k` calibrated per environment on disjoint seeds), 180 s sessions. Per fill:
effective half-spread `E = s(m^- - p)`, signed post-fill move `M_h = s(m_{t+h} - m^-)`, realised half-spread `R_h = E + M_h`
(ticks per lot, quantity-weighted; no VPIN - the measure is the post-fill price move itself).

| | E | M 10 ms | M 100 ms | M 500 ms | M 1 s | R at 500 ms |
|---|---|---|---|---|---|---|
| noise, MM fills | 1.28 | -0.71 | -0.77 | -0.84 [-0.91, -0.79] | -0.86 | 0.44 |
| informed, MM fills | 1.73 | -1.35 | -1.49 | -1.63 [-1.73, -1.54] | -1.64 | 0.10 |
| null: random time, random side | - | 0.000 | 0.000 | -0.001 | -0.004 | - |

By group (M at 500 ms, ticks): informed env - fills against **informed takers -2.96 [-3.14, -2.77]** vs against noise takers
-0.96 [-1.04, -0.89]; fills that **extend** inventory -2.21 vs **reduce** it -1.37 (noise: -1.45 vs -0.54); fills when the
book leans against the maker (`side*OBI <= -0.5`) -2.00 vs -1.03 when it leans with it (noise: -1.08 vs -0.56).

**Verdicts.** H0 (null control ~ 0): supported (|mean| <= 0.004 ticks at every horizon), so the mid has no drift and the
non-zero fill-conditioned values are attributable to the fills. H1 (fills are adverse): supported, CIs far from 0.
H2 (concentrated in informed takers): supported (3x). H3 (inventory-extending fills worse): supported. H4 (book leaning
against the maker is worse): supported, and consistent with Experiment A's OBI predictability.

**Timing finding.** About 80-85% of the total adverse move is already present 10 ms after the fill (noise: -0.71 of -0.86;
informed: -1.35 of -1.64). Most of the cost is the *mechanical impact of the aggressing order itself*, which no post-fill
reaction can avoid; the part a faster maker could avoid is being hit *before* the price moves - a latency question for
Experiment D. In the informed environment the spread captured is almost fully consumed: realised half-spread 0.10 ticks.

---------------------------------------------------------------------------------------------------

## Price impact `I(Q) ~ c Q^alpha` (input to Experiment G)

Aggressive orders (all trades of one taker order); `alpha` by weighted log-log fit of per-bin mean impact, CI by session bootstrap.

| impact horizon | noise env, all | informed env, noise takers | informed env, informed takers |
|---|---|---|---|
| immediate (book walk) | 1.03 [1.01, 1.04] | 0.97 [0.95, 0.98] | 0.97 [0.95, 0.98] |
| 0.1 s | 1.01 [0.97, 1.05] | 0.95 [0.92, 0.98] | 0.47 [0.44, 0.51] |
| 1 s | 0.99 [0.92, 1.03] | 0.84 [0.67, 0.96] | 0.10 [0.08, 0.11] |
| 5 s | 1.02 [0.86, 1.14] | 0.51 (CI 0.20-1.31, R^2 0.61) | 0.01 (R^2 0.17: no size dependence) |

Impact is **linear in size** (alpha ~ 1; ~0.02 ticks per lot) in this simulator and, in the noise market, **permanent**: mean
impact of a ~25-lot order is 0.66 ticks immediately and 0.70 ticks after 5 s - no resilience, because noise traders re-quote
around the moved mid. Informed orders carry a size-independent information component (~2 ticks at 1 s, ~4.2 at 5 s), so their
fitted alpha collapses toward 0. The slippage exponent (~0.17-0.20) is not meaningful: slippage is dominated by the constant
half-spread, so a pure power law is the wrong functional form for it.

**Verdicts.** H1 (impact increases with Q at every horizon): supported for noise takers and for immediate impact;
**rejected for informed orders at 5 s** (no size dependence). H2 (alpha != 1): rejected in the noise environment (alpha ~ 1 at
every horizon), supported for lagged impact of informed flow. H3 (informed impact larger at long horizons): supported
(~4.2 ticks vs ~0.1-0.3 for noise orders at similar size at 5 s), while immediate impact is identical - as expected.

**Realism gap (for Stage 8).** Empirical markets typically show *concave* impact (exponent well below 1) that partly
decays. Both properties are absent from the noise-only simulator, so its impact and resilience must not be presented as
realistic until compared with data or improved (e.g. order-flow with size-dependent depth, resilient liquidity).

---------------------------------------------------------------------------------------------------

## What this means for the adaptive strategy (Stage 7)

Measured, usable state variables: touch OBI (positive predictor of the next 0.1-1 s move, worth <= ~0.6 ticks - a quote-skew
signal, not a crossing signal); distance to the touch and Q (fill probability); side*OBI and inventory-extending state at
the time of a fill (adverse-selection markers). What the measurements do **not** show: that any of these is *profitable* after
costs - that is the ablation study's job.
