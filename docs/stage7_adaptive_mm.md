# Stage 7 - adaptive microstructure-aware market maker

Reproduce (deterministic; `results/adaptive_demo/` has `provenance.json`):

```bash
python -m experiments.run adaptive_demo --seed 42 --workers 8
```

## Design: explicit, switchable components

`strategies/adaptive_mm.py`. All quantities in ticks, `I` = signed inventory (lots). Full derivations are in the module docstring.

```
r      = m  - lambda_I * I                (inventory_skew)   the spec's  r = m - lambda I
            + beta_obi * OBI              (obi)
            + beta_ret * (m_t - m_{t-h})  (returns; optional, not in the main ladder)
h_side = h0 + c_A * A_side + c_L * sigma * sqrt(L)     (adverse_selection, latency)
bid = floor(r - h_bid),  ask = ceil(r + h_ask)          size_side = Q * (1 -/+ kappa I / I_max)   (inventory_size)
queue: keep a resting quote iff EV(current) >= EV(desired), EV(p) = P_fill(Q_p, d_p) * (s (r - p) - A_side)
```

With every flag off it quotes symmetrically at `mid +- 1/k` - the ablation baseline. Each of the six ladder components
(inventory skew, inventory size, OBI, adverse selection, queue, latency) plus the optional returns signal has its own switch, so the
Stage 9 ablation can add them one at a time.

| quantity | value / source | kind |
|---|---|---|
| `h0 = 1/k` | risk-neutral AS half-spread (gamma -> 0); `k` = 0.939 (noise), 0.565 (informed) | **measured** (probe fit, disjoint seeds) |
| `beta_obi` | 0.43 (noise), 0.63 (informed) ticks per unit OBI: OLS slope of the forward mid change at 0.25 s (~ mean quote lifetime) | **measured** (Experiment A code, disjoint seeds) |
| fill model | `P(fill<=1s) = sigmoid(1.41 - 0.22 ln(1+Q) - 1.07 d)` noise; `(1.31, -0.17, -0.57)` informed | **measured** (probe data, Experiment B) |
| `A_side` | EWMA of the maker's own realised adverse cost at 0.5 s (alpha 0.1, half-life 30 s, cap 5) | measured online |
| `L` | measured order-send -> own-ADD-seen delay (one-way order + feed latency) | measured online |
| `c_A = 1`, `c_L = 1` | break-even pass-through of expected adverse loss; one sigma of the unobserved price move | **theory-motivated, not fitted** |
| `lambda_I = 0.1` ticks/lot | the AS skew `gamma sigma^2 tau` at gamma = .01, sigma = 2, tau = 2.5 s | **chosen** (see limitations) |
| `kappa = 1` | size falls to 0 at the soft inventory limit | chosen |

Supporting infrastructure (also usable by other strategies): `MarketMaker` gained a `keep_quote` hook, a `_pre_apply` hook (replica state
before each event, used to capture the pre-trade mid), measured `ack_latency_s`, and optional in-place size shrinking
(`modify_to_shrink`, a size decrease keeps queue priority; on for the adaptive maker, **off for the AS baseline**, whose Stage 4 results
are bit-for-bit unchanged - checked on a committed session). `research.queue_position.FillModel/fit_fill_model` turn probe data into
the fill model. `experiments.ablations.calibration` estimates the three measured inputs on seeds disjoint from evaluation.

## Verification

* 26 unit tests (`tests/unit/test_adaptive_mm.py`): each component's sign and magnitude in isolation against hand-computed values; the
  all-off baseline is exactly `mid +- 1/k`; the tracker's EWMA / quantity weighting / cap / decay / sign convention for both sides;
  pre-trade-mid capture; the queue rule with hand-computed expected values (front-of-queue kept, hopeless position moved, unknown order and
  disabled flag default to moving); fill-model fit recovers known coefficients; in-place shrink vs cancel+repost; ack latency measured exactly
  (5 ms order + 10 ms feed = 15 ms); every ladder configuration runs, is deterministic and ends with a valid book.
* Mutation check: 9 deliberate bugs (inventory/OBI sign flips, adverse cost sign, widening the wrong side, `sigma*L` instead of
  `sigma*sqrt(L)`, inverted queue rule, size scaling on the wrong side, post- instead of pre-trade mid, tracker ignoring its horizon) - all
  9 caught, each in ~0.1 s.

## Preliminary comparison (NOT the ablation; that is Stage 9)

24 paired sessions x 180 s per arm and environment, one-way latency 0 and 5 ms. Paired bootstrap CIs; the adaptive parameters above were
fixed before the run and the run was executed once (no iteration on results).

| noise, 0 ms | net P&L | P&L std | Sharpe/s | mean abs inventory | fills |
|---|---|---|---|---|---|
| AS baseline (gamma = .01) | +4.73 [3.67, 5.69] | 2.60 | 0.262 | 3.3 | 353 |
| adaptive, all off | +7.00 [4.17, 9.74] | 7.11 | 0.120 | 22.8 | 288 |
| adaptive + inventory | +4.75 [4.04, 5.43] | 1.78 | 0.284 | 2.9 | 367 |
| adaptive full | +5.91 [5.10, 6.73] | 2.14 | 0.269 | 4.9 | 195 |

| informed, 0 ms | net P&L | P&L std | Sharpe/s | mean abs inventory | fills |
|---|---|---|---|---|---|
| AS baseline (gamma = .01) | +1.32 [-0.68, 3.12] | 5.02 | 0.062 | 2.4 | 631 |
| adaptive, all off | +11.42 [7.49, 15.21] | 9.98 | 0.083 | 24.3 | 558 |
| adaptive + inventory | +10.56 [9.59, 11.54] | 2.49 | 0.299 | 3.5 | 598 |
| adaptive full | +11.73 [10.13, 13.26] | 4.07 | 0.199 | 7.2 | 317 |

Paired differences (95% CI), informed / 0 ms: adaptive+inventory minus AS **+9.24 [+7.17, +11.41]** P&L, Sharpe +0.24;
full minus +inventory: P&L +1.17 [-0.33, +2.69] (not significant) but Sharpe **-0.10 [-0.14, -0.06]**, mean |inventory| **+3.7 lots**, fills
**-280**. Noise / 0 ms: full minus +inventory: P&L **+1.16 [+0.61, +1.74]**, Sharpe -0.02 [-0.06, +0.03], |inventory| +2.0, fills -172.
The 5 ms results are similar to 0 ms (no visible latency effect for these arms at this scale; the latency experiment is Stage 9).

**Component diagnostics** (adaptive full, mean absolute contribution per requote): adverse-selection widening 0.66 (noise) / 1.09 (informed)
ticks per side; OBI shift 0.19 / 0.29 ticks; inventory shift 0.49 / 0.72 ticks; latency term 0.18 / 0.34 ticks at 10 ms measured ack latency;
the queue rule kept ~1,000-1,600 resting quotes per session.

### What can and cannot be concluded

1. **Inventory adjustment is a pure risk control here.** Adaptive+inventory vs adaptive-off: mean |inventory| falls by ~20 lots, Sharpe
   rises by 0.16-0.25 (CIs exclude 0) while the P&L difference is not significant (e.g. noise 0 ms: -2.25 [-5.04, +0.67]).
2. **The large P&L gap between adaptive+inventory and the AS baseline under informed flow (+9.2) must NOT be read as "the adaptive
   framework is better".** The two arms differ in skew magnitude: AS skews by `gamma sigma-hat^2 tau`, which is ~3.6x larger under informed
   flow (sigma-hat is larger), while the adaptive rule skews by a fixed 0.1 ticks/lot. Stage 4 showed heavy skewing costs P&L in exactly this
   environment. So this comparison is confounded by effective skew; the like-for-like comparison is *within* the adaptive framework (Stage 9).
3. **The microstructure components trade fills for quality.** Adding OBI, adverse-selection, queue and latency components roughly halves the
   fill count and raises mean P&L slightly (significantly in noise), but *increases* inventory and lowers the Sharpe ratio (significantly under
   informed flow). Which component causes this is not identified by this run. Untested candidates: the queue rule keeping quotes at prices
   the inventory skew would move, one-sided adverse widening reducing the fills that would rebalance inventory. Stage 9 is designed to
   attribute it; I am not asserting a mechanism.
4. **No claim of superiority is made.** A higher P&L with a lower Sharpe and larger inventory is not "better"; the deciding analysis is the
   ablation with parameter sweeps and multiple environments.

## Limitations

* `lambda_I` and `kappa` are design choices (documented derivation, not estimated) and `lambda_I` is the parameter that most affects these
  results; its sensitivity is a Stage 9 sweep.
* `c_A = c_L = 1` are theory-motivated but their validity in this simulator is untested; the fill model was fit for a 1 s horizon on
  1-lot probes and the queue rule inherits its errors.
* Two synthetic environments, one noise-model parameter set, 180 s sessions, 24 seeds. All effects are simulator effects until compared with real data (Stage 8).
