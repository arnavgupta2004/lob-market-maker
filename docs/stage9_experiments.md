# Stage 9 - latency, ablation, parameter sweeps and statistical analysis

All results are from the simulator (no real data), deterministic given the seed (`--seed 42`), with `results/<name>/provenance.json` recording seed,
parameters, git commit and result hashes. Every comparison between arms is *paired* (same seeds in every arm), intervals are 95% bootstrap CIs
over sessions, and P&L is in currency units per 180 s session (tick = 0.01, quote size 5 lots). Parameters the strategies estimate (`k`, `beta_obi`,
the fill model) are measured per environment on seeds disjoint from evaluation and then fixed; nothing was tuned on these results.

```bash
python -m experiments.run latency  --seed 42 --workers 8          # Experiment D        (24 sessions x 3 strategies x 6 latencies x 2 envs)
python -m experiments.run ablation --seed 42 --workers 8          # Experiment F        (40 sessions x 12 arms x 3 envs)
python -m experiments.run sweep_gamma_size | sweep_gamma_latency | sweep_inventory | sweep_market | sweep_adaptive | sweep_spread   # response surfaces (12 sessions per cell)
```

Environments: `noise` (Poisson noise flow), `informed` (+ an informed fundamental-value trader), `informed_jump` (informed flow + news: fundamental
jumps of ~5 ticks std every ~5 s on top of diffusion; a stress environment whose parameters were not tuned to any result).

## 1. Experiment D - latency

**Hypotheses (stated before running).** H1 P&L falls with latency, faster where prices jump. H2 adverse selection worsens. H3 fills fall. H4 the adaptive
maker's latency term (widen by `sigma sqrt(measured latency)`) limits the loss. One-way latency 0-50 ms on both legs of the maker only.

| net P&L, paired change vs 0 ms (seed 42) | informed 25 ms | informed 50 ms | informed_jump 25 ms | informed_jump 50 ms |
|---|---|---|---|---|
| adaptive full | -1.21 [-2.86, +0.40] | **-2.49 [-4.17, -0.92]** | **-2.60 [-4.77, -0.58]** | **-5.01 [-6.91, -3.22]** |
| adaptive without latency term | **-3.26 [-5.93, -0.72]** | **-5.35 [-7.35, -3.47]** | **-6.15 [-7.85, -4.43]** | **-7.26 [-9.78, -4.81]** |
| AS baseline (gamma = .01) | **+3.77 [+1.39, +6.36]** | **+3.40 [+1.13, +5.94]** | **+8.87 [+5.74, +12.05]** | **+6.85 [+3.68, +10.09]** |

**Replication.** The experiment was re-run with a second base seed (`--seed 7`, `results/latency_seed7`) and the two runs are pooled (48 paired sessions) below. Findings are stated at the level that survives replication.

* **H1 - adaptive maker: supported.** P&L falls with latency: pooled 50 ms vs 0 ms **-1.96 [-3.17, -0.69]** (informed) and **-4.62 [-6.00, -3.35]** (informed_jump), i.e. larger with news jumps. (Seed 42 alone: -2.49 and -5.01; at seed 7 the
  diffusion-only effect was not individually significant, -1.42 [-3.26, +0.70].) **AS baseline: rejected** - its P&L *rises* with latency in both seeds (pooled **+3.44 [+1.85, +5.08]**, **+6.49 [+4.22, +8.85]**); see the mechanism below.
* **H2 - adverse selection worsens: not established.** Adaptive adverse cost +0.14 [+0.01, +0.28] / +0.16 [+0.01, +0.30] at seed 42, but pooled only +0.08 [-0.01, +0.17] / +0.10 [0.00, +0.20]. AS's adverse cost *falls* with latency (1.64 -> 1.18 at seed 42).
* **H3 - fills fall: supported and replicated.** Adaptive fills at 50 ms vs 0: pooled **-60 [-68, -51]** (informed), **-58 [-66, -50]** (jump); fill rate per quote 0.273 -> 0.143 and quote survival >= 250 ms 0.365 -> 0.233 (seed 42, informed).
* **H4 - latency term helps: supported only at 50 ms, and smaller than the seed-42 numbers suggested.** Pooled benefit **+1.93 [+0.67, +3.11]** (informed) and **+2.68 [+0.83, +4.55]** (jump) at 50 ms; borderline at 25 ms (+1.41 [+0.02, +2.82], +1.50 [-0.01, +2.94]); not significant at 10 ms
  (+1.13, +0.82). Seed 42 alone had shown +3.55 [+1.96, +5.12] at 25 ms with jumps; that did not replicate (seed 7: -0.56 [-2.88, +1.54]) - a reminder that single-seed significance is fragile.

**Why AS improves with latency - a mechanism, tested.** At 0 -> 50 ms AS's inventory P&L improves (-37.2 -> -27.6), its adverse cost falls and its mean |inventory| rises
(2.4 -> 4.2): latency *damps an over-reactive inventory-skew loop* (Stage 4: heavy skew realises losses by chasing adverse moves). The gamma x latency sweep tests this: the paired
effect of 25 ms is -1.98 [-7.51, +4.16] at gamma = 0 (and **-7.21 [-12.57, -1.46]** at 50 ms) - latency hurts when there is no skew to damp - and grows steadily with gamma
(+1.32 at .005, +6.93 [+2.42, +12.96] at .01, +11.31 [+7.01, +15.54] at .1). So AS's "benefit from latency" is an interaction with its skew aggressiveness, **not evidence that being slow
is good**; a fair latency study needs each latency evaluated at a well-chosen gamma, which is what the sweep provides. *Caveat:* diffusion alone moves the fundamental ~0.7 ticks in 50 ms,
so weak latency sensitivity without jumps is expected; real latency arbitrage involves faster information than this model contains.

## 2. Experiment F and the ablation study - which signals add value after costs?

Cumulative ladder L0 (symmetric quotes) -> L1 +inventory -> L2 +OBI -> L3 +adverse selection -> L4 +queue -> L5 +latency (= full), plus leave-one-out (LOO) and AS as reference;
10 ms one-way latency. Fee regimes applied post hoc from exact traded notional (strategies are not fee-aware, so fees are linear): A = 0; **B = maker 0.5 bp / taker 2 bp (primary)**; C = 1 / 4 bp.
The 15 ladder comparisons on primary net P&L are Holm-corrected; everything else is secondary and uncorrected.

**Incremental net P&L of each ladder step (regime B), paired [95% CI], Holm p:**

| step | noise | informed | informed_jump |
|---|---|---|---|
| L1 +inventory | **-3.09 [-4.85, -1.27]** p=.006 | +1.98 [-1.44, +5.74] | **+7.36 [+3.44, +11.65]** p=.004 |
| L2 +OBI | **+0.82 [+0.50, +1.15]** p=.004 | **+1.33 [+0.51, +2.21]** p=.033 | +1.04 [-0.09, +2.18] |
| L3 +adverse | +0.88 [+0.23, +1.48] p=.105 | +1.47 [+0.10, +2.76] p=.259 | +1.24 [-0.41, +2.91] |
| L4 +queue | **+1.56 [+1.01, +2.14]** p=.004 | +1.84 [+0.24, +3.44] p=.160 | +2.44 [+0.49, +4.34] p=.126 |
| L5 +latency | +0.18 [-0.17, +0.55] | +1.18 [-0.73, +3.01] | +1.59 [-0.29, +3.37] |

**Leave-one-out (value of the component = full minus full-without-it, regime B):** inventory -2.88 [-4.86, -0.93] (noise), +1.68 [-3.42, +6.49] (informed), **+6.04 [+1.78, +10.37]** (jump);
OBI +0.20 / +0.38 / +0.85, none significant; adverse +0.01 / -0.18 / +1.40, none significant; queue **+1.33 [+0.99, +1.66]** (noise), +0.67 [-0.87, +2.19], +1.82 [-0.06, +3.60]; latency +0.18 / +1.18 / +1.59, none significant at 10 ms.

**What the ablation supports.**
* **Inventory control is a risk tool, not a profit source.** In every environment it cuts mean |inventory| by ~20 lots and raises the per-second Sharpe by +0.13 to +0.16 (LOO; +0.19 to +0.26 in the
  ladder). Its P&L effect is environment-dependent and only partly robust: under news jumps it *earns* (+7.4 ladder / +6.0 LOO at seed 42; +8.5 / +5.1 at seed 7, significant both times); under noise flow it *looked* costly
  at seed 42 (-3.09 [-4.85, -1.27]; LOO -2.88) but **did not replicate** at seed 7 (-1.31, not significant; LOO +0.81), so no claim is made about its sign there.
* **The queue rule is the most consistently positive component** (significant under noise in both the ladder and LOO; positive point estimates elsewhere with wider CIs).
* **Adverse-selection widening trades fills for quality and worsens risk-adjusted results.** It removes ~180-240 fills per session (ladder) and 104-129 (LOO), raises mean |inventory| by 1.4-3.3 lots
  and lowers Sharpe by 0.14-0.19 (ladder) with no reliably positive P&L gain. This is the component behind the Stage 7 observation that the full strategy had fewer fills and a lower Sharpe than +inventory alone.
* **OBI is a small effect**: a modest significant ladder step under noise and informed flow (+0.8, +1.3), but no significant LOO value anywhere. Consistent with Experiment A (predicted moves << half-spread).
* **Costs dominate the level of results.** Full strategy net P&L for regimes A / B / C: noise 5.30 / 2.13 / -1.04, informed 10.27 / 5.53 / 0.78, jump 9.78 / 5.11 / 0.43; AS: noise 5.06 / -0.99 / -7.03,
  informed 4.41 / -6.12 / -16.65, jump 2.13 / -8.58 / -19.30. In regime B the full maker beats AS by +3.12 [2.59, 3.63] (noise), +11.65 [9.60, 13.71] (informed), +13.69 [11.56, 15.78] (jump) - but its
  Sharpe is not clearly higher (-0.04, +0.04, +0.09) and its inventory is larger (+1.2 to +4.2 lots). The gap is largely *fewer, wider fills* (270 vs 600 fills/session; effective half-spread 3.1-3.3 vs 1.8 ticks) that pay less
  in fees, not proof that microstructure information beats AS at equal risk. It also means the comparison partly reflects spread width, not only signal content.

**Replication with a different base seed (`results/ablation_seed7`, same design, seed 7).** All **15/15** primary ladder-increment signs agree with seed 42. Robust: adverse-selection widening lowers Sharpe (-0.18, -0.18, -0.13 vs -0.19,
-0.16, -0.14), raises inventory (+2.1 to +3.2 lots) and cuts fills; inventory control raises Sharpe by ~0.2-0.26 and cuts inventory ~20 lots; the queue rule is positive in all three environments (significant under noise and jumps; +1.70 n.s. under informed
flow); OBI and latency-term increments are small and positive. Not robust: *statistical significance* of the smaller increments (e.g. informed L2/L3/L4 significant at seed 42, not at seed 7), and the sign of the LOO effect of inventory control under noise flow.
Effect sizes are stable; p-values of effects near +1 currency unit are not. Treat a finding as established only if it appears in both seeds.

**Limits of this ablation.** Ladder increments depend on the order in which components are added (the LOO arms complement, not replace, this); two latencies/one fee-quote-size setting; 40 sessions
per arm resolve differences of ~1 currency unit; only the primary family is multiplicity-corrected; the environments are synthetic and the informed trader is a single fixed design.

## 3. Parameter sweeps (response surfaces; 12 sessions per cell - cell means have a standard error of ~1.2 currency units, so differences below ~3 are not resolved)

* **gamma x quote size (AS, informed).** Quote size is harmless at gamma = 0 (P&L 3.1 -> 10.8 from 1 to 20 lots, Sharpe ~0.1) but ruinous with skew: at gamma = .02, 2.0 (1 lot) -> -29.3 (20 lots);
  at 1 lot, Sharpe peaks at an intermediate gamma (0.31 at gamma = .01 at seed 42; 0.33 at gamma = .02 at seed 7 - the location is not stable). Larger orders amplify chasing losses; adverse cost rises with gamma (1.24 -> 2.26 ticks at 1 lot).
* **half-spread x quote size (symmetric maker, informed; `sweep_spread`; single seed).** The classical spread trade-off: with 5-lot quotes, net P&L is -19.8 at a 0.5-tick half-spread, -5.2 at 1, +1.9 at 1.5, +8.1 at 2 and then plateaus at
  +11 to +12 for 3-6 ticks; fills fall from ~1,400 to ~65 per session and the adverse cost per fill rises (0.53 -> 2.28 ticks) as the quote moves out. The measured risk-neutral half-spread 1/k = 1.77 ticks sits at the start of the plateau.
  P&L scales roughly with size at wide spreads and losses scale with size at narrow spreads. The plateau (no interior optimum in mean P&L) is why Sharpe, not mean P&L, is the more discriminating metric here (Sharpe rises with spread at 1 lot: -0.06 -> 0.16).
* **gamma x latency (AS, jumps).** See section 1: surface P&L at gamma = .05 rises from -11.9 (0 ms) to +2.1 (50 ms); at gamma = 0 it falls from +3.4 to -3.8.
* **inventory limit x gamma (AS, informed).** A *hard limit with symmetric quoting* is an effective risk control: at gamma = 0, limit 5 gives Sharpe 0.35 and max drawdown 0.94 vs limit 100: Sharpe 0.10, drawdown 13.4,
  at lower mean P&L (6.8 vs 11.7). For gamma >= .02 the P&L is negative at every limit (the limit does not bind above ~20 lots because the skew keeps inventory small).
* **market conditions (volatility x order-flow intensity), AS vs adaptive.** AS is fragile at low order-flow intensity (0.5x): P&L -12.6 (sigma .01) to **-74.8** (sigma .08); at 1x it ranges +5.2 to -13.4; at 2x +3.5 to +2.2. The adaptive
  maker ranges +17.1 to -9.2 (best at low intensity / low volatility). *Confound:* informed flow is fixed at 10 wake-ups/s, so lower noise intensity raises the informed share of aggressive flow (a more toxic mix); and `k` was measured
  at default conditions and held fixed, so this also measures sensitivity to its misspecification.
  *Replicated at seed 7 (`results/sweep_market_seed7`):* AS at 0.5x intensity ranges -18.4 (sigma .01) to -82.0 (sigma .08) (seed 42: -12.6 to -74.8); adaptive +18.4 to +1.8 (seed 42: +17.1 to -9.2). The qualitative surface (AS fragile at low intensity; adaptive best at low
  intensity and low volatility) replicates; individual cells differ by up to ~10 currency units.
* **adaptive coefficient sensitivity (informed, 10 ms).** P&L varies only within +5.6 .. +9.4 across `lambda_inv` x `c_adverse`; larger `lambda_inv` monotonically lowers |inventory| (8.8 -> 2.8 at `c_adverse` = 0) and raises Sharpe (0.21 -> 0.32);
  larger `c_adverse` lowers Sharpe (0.29 -> 0.17 at `lambda_inv` = .2) and raises |inventory| - consistent with the ablation. The `beta_obi` multiplier x `c_latency` grid shows no pattern distinguishable from noise (P&L +5.4 .. +9.6, session std ~5, 12 sessions per cell).

**Replication status.** Every experiment in this stage was re-run with a second base seed (`results/*_seed7`): the ablation (15/15 primary signs agree; section 2), the latency experiment (pooled; section 1), and all six response surfaces.
The qualitative patterns replicate: gamma x latency (gamma = .05: -11.9 -> +2.1 from 0 to 50 ms at seed 42, -13.8 -> +1.7 at seed 7; at gamma = 0: +3.4 -> -3.8 vs +3.3 -> -3.6); the hard inventory limit (limit 5, gamma = 0: Sharpe 0.35 / 0.38, max drawdown 0.9 / 0.9;
limit 100: 0.10 / 0.14 and 13.4 / 12.1); quote size ruinous with skew (gamma = .02, 1 -> 20 lots: +2.0 -> -29.3 vs +2.2 -> -19.4); adaptive-coefficient grids (P&L 5.6-9.4 vs 4.8-9.7; Sharpe falls with `c_adverse`: 0.29 -> 0.14 vs 0.35 -> 0.18; |inventory| falls with `lambda_inv`:
8.8 -> 2.8 vs 9.5 -> 2.8). What does *not* replicate is fine structure: the location of the Sharpe peak over gamma at 1 lot (gamma = .01 at seed 42; gamma = .02-.05 at seed 7), and single-cell values (differences up to ~10 currency units in the market sweep).

## 4. Statistical methodology (as implemented)

Sessions are the unit of inference (samples inside a session overlap). `backtest/statistics.py`: percentile bootstrap CIs, **paired** difference CIs (common random numbers remove between-seed variance; unit-tested that pairing shrinks the CI by >10x on shared-variance data),
a bootstrap two-sided p-value, Holm-Bonferroni across a declared family, and the paired effect size d_z. Reported for each metric: mean, median, standard deviation, CI. Not done: a formal power analysis, or multiplicity correction beyond the
primary ablation family; the many secondary comparisons in the CSVs are uncorrected and should be read as exploratory.

## Limitations

Synthetic markets only (real data comparison is Stage 8's open item); one order-flow parameter set per environment; the informed trader and news process are fixed designs; latency applies to the maker only; the adaptive maker's `lambda_inv` and `kappa` are design
choices; results at 12-40 sessions per arm cannot resolve small effects.
