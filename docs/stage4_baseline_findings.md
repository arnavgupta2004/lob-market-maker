# Stage 4 - Avellaneda-Stoikov baseline: assumption tests and inventory experiment

Reproduce (deterministic; outputs in `results/as_assumptions/` and `results/inventory/`, each with a
`provenance.json` recording seed, parameters, git commit and result hashes):

```bash
python -m experiments.run as_assumptions --seed 42 --workers 8
python -m experiments.run inventory      --seed 42 --workers 8
```

Environment for both: synthetic Poisson noise flow (defaults in `NoiseTraderConfig`) either alone
("noise") or with an informed fundamental-value trader at 10 wake-ups/s on a Brownian fundamental
with sigma = 0.03 price units/sqrt(s), tick = 0.01 ("informed"). No external data is used.
The 24 evaluation sessions (as_assumptions, 300 s) / 30 sessions (inventory, 180 s) per arm are independent
seeds; all inventory arms share the same seeds (paired comparison).

---------------------------------------------------------------------------------------------------

## Experiment 1 - `as_assumptions`: do the AS assumptions hold in this simulator?

**Hypotheses** (stated before running). H1: fill intensity decays exponentially with distance from the mid.
H2: intensity is constant over an order's life (Poisson fills). H3: the mid is a random walk with constant
volatility (variance ratios ~ 1, flat volatility signature).

**Method.** 1-lot post-only probes at randomised distances, left for 0.5 s then cancelled. Intensity per
distance bin by the censored-exponential MLE `fills / exposure time` (a raw fill fraction is biased by the
censoring - unit-tested). `ln(lambda)` regressed on distance with Poisson weights; CIs by bootstrapping whole sessions.

**Results** (95% CI in brackets)

| | noise | informed |
|---|---|---|
| k (1/tick) | 0.965 [0.939, 0.991] | 0.572 [0.560, 0.587] |
| A (1/s) | 3.11 | 4.04 |
| R^2 of ln-lambda fit | 0.925 | 0.927 |
| chi^2/dof of that fit (1 = Poisson-consistent) | 144 | 201 |
| VR(0.5 s) / VR(1 s) / VR(5 s) | 0.855 / 0.808 / 0.773 | 0.831 / 0.773 / 0.724 |
| volatility (ticks/sqrt(s)) at 0.1 s -> 5 s sampling | 1.88 -> 1.65 | 3.59 -> 3.05 |

**Interpretation.**
* H1 - *approximately, not exactly.* The curve is close to log-linear (R^2 ~ 0.93) but the deviations are
  far larger than sampling error (chi^2/dof >> 1); noise-flow intensities at large distances fall below the
  fitted line, and a half-tick saw-tooth appears from mid parity. Using a single `k` is a coarse approximation.
* `k` is **environment-dependent** (0.97 vs 0.57): informed flow makes deep quotes fill ~1.7x more readily per
  tick of distance. `k` must therefore be estimated for the regime; it is not a universal constant.
* H2 - *environment-dependent.* Ratio of second-half to first-half hazard (pooled per bin range, `intensity.csv`):
  noise, delta <= 3 ticks: median 1.03 (range 0.86-1.42, ~20k fills) - consistent with constant hazard near the touch, but
  1.42 for delta > 3 (only ~1.1k fills, noisy); informed: 0.74 for delta <= 3 (range 0.56-0.91, ~37k fills) - hazard
  falls ~25% over the dwell, i.e. rejected. A plausible reading (untested here): under informed flow orders that are
  going to be picked off are picked off early, so survivors face a lower hazard.
* H3 - rejected: VR < 1 at all horizons (mean reversion, e.g. bid-ask bounce and liquidity replenishment) and
  volatility estimates depend on the sampling interval (~12% lower at 5 s than at 0.1 s in the noise environment).

**Limitations.** The probe is a 1-lot passive order, so this measures fill *intensity per order*, including queue
effects, rather than the market-order arrival intensity at a price the AS model assumes. Half-tick binning mixes
mid-parity effects into the distance dependence. Probes perturb the book slightly (about 10 lots resting). Only two
environments and one parameter set of the noise model; nothing here is calibrated to real data yet (Stage 8).

---------------------------------------------------------------------------------------------------

## Experiment 2 - `inventory` (Experiment E): how does risk aversion gamma change behaviour and outcomes?

**Hypotheses** (before running). H1: larger gamma reduces inventory dispersion. H2: larger gamma reduces
session-to-session P&L dispersion. H3: the mean-P&L cost of that risk reduction is small relative to its
benefit. H4: skew helps more under informed flow (adverse selection) than under noise flow.

**Design.** `k` is estimated per environment with the probe calibration on seeds *disjoint* from the evaluation
seeds (0.939 noise, 0.565 informed). Arms: gamma in {0, .005, .01, .02, .05, .1, .2} (1/tick), AS horizon
T = 5 s (`restart` mode), EWMA volatility (half-life 10 s), quote size 5, soft limit 50, kill at 100. gamma = 0 is
the inventory-blind symmetric quoter (control). Paired bootstrap CIs on differences vs the control.
Safety guards (sigma bounds [0.05, 10], quote offset cap 20 ticks) apply identically to all arms - see Failure case.

**Results** (means over 30 sessions; full table incl. CIs in `results/inventory/summary.csv`)

| env | gamma | net P&L | P&L std | per-second Sharpe | max drawdown | mean abs inventory | P&L: edge / inventory |
|---|---|---|---|---|---|---|---|
| noise | 0 | +7.14 | 6.78 | 0.117 | 6.11 | 22.96 | +15.3 / -8.1 |
| noise | 0.005 | +5.81 | 2.52 | 0.263 | 1.62 | 4.88 | +16.3 / -10.5 |
| noise | 0.01 | +4.93 | 2.51 | 0.268 | 1.50 | 3.38 | +16.0 / -11.1 |
| noise | 0.05 | -0.28 | 2.40 | 0.036 | 2.15 | 1.46 | +14.1 / -14.4 |
| noise | 0.2 | -2.25 | 2.20 | -0.088 | 2.75 | 0.70 | +9.8 / -12.1 |
| informed | 0 | +10.55 | 10.74 | 0.078 | 12.76 | 24.75 | +39.5 / -29.0 |
| informed | 0.005 | +6.12 | 5.10 | 0.156 | 3.49 | 3.68 | +40.5 / -34.3 |
| informed | 0.02 | -7.38 | 5.77 | -0.125 | 9.27 | 1.57 | +36.6 / -43.9 |
| informed | 0.2 | -17.61 | 8.61 | -0.189 | 18.61 | 0.59 | +31.2 / -48.8 |

Paired differences vs gamma = 0 (95% CI): mean |inventory| falls by 18.1 [15.8, 20.2] lots (noise, gamma = .005) and 21.1
[19.7, 22.4] (informed) - every arm's CI excludes 0 by a wide margin. Net P&L: noise gamma = .005: -1.33 [-3.62, +0.98]
(not significant); gamma = .02: -4.05 [-6.47, -1.60]; gamma = .2: -9.39 [-12.08, -6.66]. Informed gamma = .005: -4.43
[-8.22, -0.53]; gamma = .02: -17.9 [-22.0, -14.1].

**Verdict on hypotheses.**
* **H1 supported, strongly.** Inventory std falls from 18 to 6 lots (noise) and 25 to 5 (informed) already at gamma = .005.
* **H2 partly supported.** Most of the P&L-dispersion reduction happens at the smallest gamma (6.8 -> 2.5; 10.7 -> 5.1); further
  increases in gamma do not reduce it, and dispersion *rises* again at large gamma under informed flow.
* **H3 supported only for small gamma.** For gamma <= 0.01 the mean-P&L difference is small/insignificant in the noise
  environment while Sharpe more than doubles and drawdown falls ~4x. At larger gamma the cost dominates: noise P&L is
  indistinguishable from 0 at gamma = 0.05 (-0.28, CI [-1.16, +0.53]) and negative from gamma = 0.1; informed P&L is
  negative from gamma = 0.02 (CI excludes 0). Heavy skewing costs more than it protects in this simulator.
* **H4 not supported.** The P&L cost of skew is *larger* under informed flow. Caveat: the same gamma produces a ~3.6x larger
  reservation shift there (`gamma * sigma^2 * tau`; sigma-hat is ~1.9x larger), so this compares different effective skews.

**Mechanism observed (not yet proven).** Spread capture ("edge") is roughly flat in gamma, while inventory P&L becomes *more*
negative as gamma grows even though average inventory shrinks 30x. A larger skew means quoting more aggressively to unwind, and
each fill is followed by a mid move against the maker (price impact of the order flow and, in the informed env, of information).
Whether the loss is realised by unwinding into adverse moves is a hypothesis for Experiment C (post-fill returns conditioned on
inventory state), not a conclusion of this experiment.

**Failure case (kept, not hidden).** With the volatility bounds and quote-offset cap switched off, the baseline diverges: the
EWMA volatility is driven by mid moves the maker's own quotes cause, the `gamma * sigma^2 * tau` skew pushes the bid through the
mid, and noise traders (who quote around the mid) follow it. Sessions in which the mid ever moved > 500 ticks:

| env | gamma | diverged sessions (95% CI) |
|---|---|---|
| noise | 0.05 | 0% [0, 0] |
| noise | 0.2 | 20% [7, 33] |
| informed | 0.05 | 70% [53, 87] |
| informed | 0.2 | 97% [90, 100] |

The inventory kill-switch never triggers in these runs because inventory stays near zero while the price runs away. With the guards on,
no arm diverged. The guards are engineering safeguards, not part of the AS theory, and they *do* bind at large gamma
(informed gamma = .2: ~1,400 quote-offset clamp events per session against ~2,500 quotes posted), so results for gamma >= 0.05
are guard-affected; at gamma <= 0.02 the guards are almost inactive (<= 45 clamp events per session, mostly 0).

**Limitations.**
* The control (gamma = 0) is bounded by the soft inventory limit (max |q| hits ~50), so its risk is truncated, not unlimited.
* One parameter set of the noise model, a single volatility half-life, a single AS horizon; no latency yet (0 ms) and no fees.
* A reflexive, single-venue market: the mid is set by the same participants being studied, with no external anchor.
* "Sharpe" is per-second, in-session; annualised values (in the CSV) are extrapolations and should not be quoted.
* Sessions are 180 s; 30 sessions per arm gives CIs that are still wide for P&L differences of order 1-2 currency units.
