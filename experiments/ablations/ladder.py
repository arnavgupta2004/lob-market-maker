"""Experiments F (adaptive strategy) and ablation study: which microstructure signals add measurable value after costs?

The adaptive maker is built up one component at a time (cumulative ladder) and, separately, one component is removed from
the full strategy (leave-one-out). All arms share the same seeds (common random numbers), so every contribution is a *paired* difference.

    L0  symmetric quoting at mid +- 1/k                    (baseline: all components off)
    L1  + inventory (skew and size)
    L2  + order-book imbalance
    L3  + adverse-selection widening (own measured post-fill cost)
    L4  + queue information (expected-value keep/move rule)
    L5  + latency model  (= full)
    LOO full minus {inventory, OBI, adverse, queue, latency}      AS  Avellaneda-Stoikov (gamma = .01) for reference

Hypotheses (stated before running; each is a claim about a *paired difference* with its CI)
    H1  Inventory control reduces inventory and P&L variance without a significant loss of mean net P&L.
    H2  Adverse-selection widening reduces the realised adverse cost per fill and improves net P&L under informed flow, not under noise.
    H3  OBI shifting adds value where OBI predicts the next move (Experiment A): it should help under both flows, weakly.
    H4  The queue rule reduces cancellations / raises quote survival and fill probability per quote.
    H5  The latency term helps only when latency is non-trivial (here 10 ms one-way).
Costs: fees are applied *post hoc* from the exact traded notional (the strategies do not quote fee-aware, so fees are linear in
notional): regime A = 0 bps; B = maker 0.5 bp / taker 2 bp; C = maker 1 bp / taker 4 bp. The PRIMARY metric is net P&L in regime B;
its 15 ladder comparisons (5 steps x 3 environments) are Holm-corrected. Every other number is secondary and uncorrected.
Environments: noise; informed; informed + jumps. One-way latency 10 ms on both legs. Parameters (k, beta_obi, fill model) are measured
per environment on disjoint seeds, then fixed; nothing is tuned on these results.
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backtest.runner import Scenario, run_many
from backtest.statistics import holm, paired_diff_ci, summarize
from experiments.ablations.calibration import calibrate_environment
from experiments.baseline.as_assumptions import ALL_ENVS as ENVS, env_fundamental
from experiments.common import RunContext, save_csv, save_json, save_parquet, write_provenance
from simulator.latency.models import LatencyConfig
from simulator.simulator import SimConfig
from strategies.adaptive_mm import AdaptiveConfig
from strategies.baseline_mm import ASConfig

DESCRIPTION = "Ablation ladder and leave-one-out for the adaptive market maker, net of costs (Experiment F)"

ENV_LIST = ("noise", "informed", "informed_jump")
LATENCY_MS = 10.0
FEES = {"A_0bps": (0.0, 0.0), "B_0.5/2bps": (0.5, 2.0), "C_1/4bps": (1.0, 4.0)}
PRIMARY_FEE = "B_0.5/2bps"
OFF = dict(use_inventory_skew=False, use_inventory_size=False, use_obi=False, use_adverse=False, use_queue=False, use_latency=False)
LADDER = [
    ("L0_symmetric", {}),
    ("L1_inventory", dict(use_inventory_skew=True, use_inventory_size=True)),
    ("L2_obi", dict(use_inventory_skew=True, use_inventory_size=True, use_obi=True)),
    ("L3_adverse", dict(use_inventory_skew=True, use_inventory_size=True, use_obi=True, use_adverse=True)),
    ("L4_queue", dict(use_inventory_skew=True, use_inventory_size=True, use_obi=True, use_adverse=True, use_queue=True)),
    ("L5_latency(full)", dict(use_inventory_skew=True, use_inventory_size=True, use_obi=True, use_adverse=True, use_queue=True, use_latency=True)),
]
FULL = LADDER[-1][1]
LOO = [("full-inventory", dict(FULL, use_inventory_skew=False, use_inventory_size=False)), ("full-obi", dict(FULL, use_obi=False)),
       ("full-adverse", dict(FULL, use_adverse=False)), ("full-queue", dict(FULL, use_queue=False)), ("full-latency", dict(FULL, use_latency=False))]
ARM_FLAGS = dict(LADDER + LOO)
ARMS = [a for a, _ in LADDER] + [a for a, _ in LOO] + ["AS_gamma0.01"]
SECONDARY = ("sharpe_1s", "inv_abs_mean", "n_fills", "adverse_cost_500ms", "eff_half_spread_ticks", "fill_rate_orders", "n_cancels", "quote_surv_250ms", "max_drawdown", "pnl_vol_1s")


def net_at(row: dict, fee: tuple[float, float]) -> float:
    """Net P&L under a fee regime from the fee-free session result (exact: fees are linear in traded notional)."""
    return row["pnl_gross"] - 1e-4 * (fee[0] * row["notional_maker"] + fee[1] * row["notional_taker"])


def arm_config(arm: str, cal: dict, lat: LatencyConfig):
    common = dict(k=cal["k"], latency=lat, quote_size=5, inventory_limit=50, kill_inventory=100)
    if arm == "AS_gamma0.01":
        return ASConfig(gamma=0.01, horizon_s=5.0, **common)
    flags = {**OFF, **ARM_FLAGS[arm]}
    return AdaptiveConfig(**common, beta_obi=cal["beta_obi"], fill_model=cal["fill_model"] if flags["use_queue"] else None, **flags)


def run(ctx: RunContext) -> None:
    t0 = time.time()
    horizon = 40.0 if ctx.quick else 180.0
    seeds = ctx.seeds(default=40, quick=3)
    calib_seeds = [ctx.seed * 100_000 + 50_000 + i for i in range(3 if ctx.quick else 12)]
    lat = LatencyConfig.symmetric(LATENCY_MS)
    cum_rows, inc_rows, loo_rows, sessions, cals = [], [], [], [], {}
    for env in ENV_LIST:
        cal = calibrate_environment(env, calib_seeds, horizon_s=40.0 if ctx.quick else 120.0, workers=ctx.workers)
        cals[env] = {k: v for k, v in cal.items() if k != "seeds"}
        per = {}
        for arm in ARMS:
            sc = Scenario(f"{env}_{arm}", SimConfig(horizon_s=horizon, fundamental=env_fundamental(env)), informed=ENVS[env]["informed"],
                          mm=arm_config(arm, cal, lat))
            per[arm] = sorted(run_many(sc, seeds, workers=ctx.workers), key=lambda r: r["seed"])
            for r in per[arm]:
                sessions.append(dict(env=env, arm=arm, **r, **{f"pnl_net_{f}": net_at(r, fee) for f, fee in FEES.items()}))

        def series(arm, metric):
            if metric.startswith("pnl_net_"):
                return np.array([net_at(r, FEES[metric[8:]]) for r in per[arm]])
            return np.array([r.get(metric, np.nan) for r in per[arm]], dtype=float)

        metrics = [f"pnl_net_{f}" for f in FEES] + list(SECONDARY)
        for arm in ARMS:
            rec = dict(env=env, arm=arm, n=len(seeds))
            for m in metrics:
                s = summarize(series(arm, m))
                rec.update({f"{m}_mean": s["mean"], f"{m}_ci_lo": s["ci_lo"], f"{m}_ci_hi": s["ci_hi"], f"{m}_std": s["std"]})
            cum_rows.append(rec)

        def contrast(a, b):
            rec = {}
            for m in metrics:
                d = paired_diff_ci(series(a, m), series(b, m))
                rec.update({f"d_{m}": d["mean_diff"], f"d_{m}_lo": d["ci_lo"], f"d_{m}_hi": d["ci_hi"], f"d_{m}_p": d["p_value"]})
            return rec
        for (prev, _), (cur, _) in zip(LADDER[:-1], LADDER[1:]):
            inc_rows.append(dict(env=env, step=cur, versus=prev, **contrast(cur, prev)))
        for name, _ in LOO:
            loo_rows.append(dict(env=env, removed=name, versus="L5_latency(full)", **contrast("L5_latency(full)", name)))
        for arm in ("L1_inventory", "L5_latency(full)"):
            inc_rows.append(dict(env=env, step=arm, versus="AS_gamma0.01", **contrast(arm, "AS_gamma0.01")))
    # ---- Holm correction over the primary family: the 15 ladder increments on net P&L (regime B)
    fam = [r for r in inc_rows if r["versus"] != "AS_gamma0.01"]
    adj = holm([r[f"d_pnl_net_{PRIMARY_FEE}_p"] for r in fam])
    for r, a in zip(fam, adj):
        r["primary_p_holm"] = a
    save_csv(ctx.out_dir / "cumulative.csv", cum_rows)
    save_csv(ctx.out_dir / "incremental.csv", inc_rows)
    save_csv(ctx.out_dir / "leave_one_out.csv", loo_rows)
    save_parquet(ctx.out_dir / "sessions.parquet", sessions)
    save_json(ctx.out_dir / "calibration.json", cals)
    _report(inc_rows, loo_rows, cum_rows)
    _plot(ctx, cum_rows, inc_rows, loo_rows)
    _plot_distributions(ctx, sessions)
    write_provenance(ctx, {"sessions": len(seeds), "horizon_s": horizon, "latency_ms_one_way": LATENCY_MS, "fees_bps(maker,taker)": FEES, "primary": PRIMARY_FEE,
                           "arms": ARMS, "ladder": LADDER, "leave_one_out": LOO, "environments": {e: ENVS[e] for e in ENV_LIST}, "calibration": cals,
                           "calibration_seeds": calib_seeds, "multiplicity": "Holm over 15 ladder comparisons on net P&L (regime B)"},
                     files=["cumulative.csv", "incremental.csv", "leave_one_out.csv", "sessions.parquet", "calibration.json", "ablation.png", "ablation_distributions.png"], elapsed_s=time.time() - t0)


def _report(inc_rows, loo_rows, cum_rows) -> None:
    print(f"\nINCREMENTAL effect of each ladder step on net P&L (regime B: maker 0.5bp / taker 2bp), paired, 95% CI; Holm-adjusted p")
    for r in inc_rows:
        if r["versus"] == "AS_gamma0.01":
            continue
        k = f"d_pnl_net_{PRIMARY_FEE}"
        print(f"  {r['env']:14s} {r['step']:17s} vs {r['versus']:15s} dP&L {r[k]:+6.2f} [{r[k + '_lo']:+.2f},{r[k + '_hi']:+.2f}]  holm p={r['primary_p_holm']:.3f}   "
              f"dSharpe {r['d_sharpe_1s']:+.3f}  d|inv| {r['d_inv_abs_mean']:+5.1f}  dfills {r['d_n_fills']:+6.0f}  dadv {r['d_adverse_cost_500ms']:+.2f}")


def _plot(ctx: RunContext, cum, inc, loo) -> None:
    fig, ax = plt.subplots(len(ENV_LIST), 4, figsize=(20, 4.2 * len(ENV_LIST)), squeeze=False)
    steps = [a for a, _ in LADDER]
    for i, env in enumerate(ENV_LIST):
        c = {r["arm"]: r for r in cum if r["env"] == env}
        x = np.arange(len(steps))
        for fee, col in zip(FEES, ("tab:green", "tab:orange", "tab:red")):
            y = np.array([c[s][f"pnl_net_{fee}_mean"] for s in steps])
            ax[i, 0].errorbar(x, y, [y - [c[s][f"pnl_net_{fee}_ci_lo"] for s in steps], [c[s][f"pnl_net_{fee}_ci_hi"] for s in steps] - y], fmt="o-", color=col, capsize=2, label=f"fees {fee}")
        ax[i, 0].axhline(c["AS_gamma0.01"]["pnl_net_A_0bps_mean"], color="gray", ls=":", label="AS (0 bps)")
        ax[i, 0].axhline(0, color="k", lw=0.4)
        ax[i, 0].set(title=f"{env}: net P&L along the ladder", xticks=x, xticklabels=[s.split("_")[0] for s in steps], ylabel="currency / session"); ax[i, 0].legend(fontsize=7)
        for m, col, ls in (("sharpe_1s", "tab:blue", "o-"), ("inv_abs_mean", "tab:purple", "s--")):
            y = np.array([c[s][f"{m}_mean"] for s in steps]); a2 = ax[i, 1] if m == "sharpe_1s" else ax[i, 1].twinx()
            a2.errorbar(x, y, [y - [c[s][f"{m}_ci_lo"] for s in steps], [c[s][f"{m}_ci_hi"] for s in steps] - y], fmt=ls, color=col, capsize=2)
            a2.set_ylabel(m, color=col)
        ax[i, 1].set(title=f"{env}: Sharpe (blue) and mean |inventory| (purple)", xticks=x, xticklabels=[s.split("_")[0] for s in steps])
        rows = [r for r in inc if r["env"] == env and r["versus"] != "AS_gamma0.01"]
        k = f"d_pnl_net_{PRIMARY_FEE}"
        y = np.array([r[k] for r in rows])
        ax[i, 2].bar(range(len(rows)), y, color="tab:green", yerr=[y - [r[k + "_lo"] for r in rows], [r[k + "_hi"] for r in rows] - y], capsize=3)
        ax[i, 2].set(title=f"{env}: INCREMENTAL net P&L of each step (95% CI)", xticks=range(len(rows)), xticklabels=[r["step"].split("_")[0] + "\n" + r["step"].split("_", 1)[1][:9] for r in rows]); ax[i, 2].axhline(0, color="k", lw=0.5)
        rows = [r for r in loo if r["env"] == env]
        y = np.array([r[k] for r in rows])
        ax[i, 3].bar(range(len(rows)), y, color="tab:red", yerr=[y - [r[k + "_lo"] for r in rows], [r[k + "_hi"] for r in rows] - y], capsize=3)
        ax[i, 3].set(title=f"{env}: value of each component = full minus (full without it)", xticks=range(len(rows)), xticklabels=[r["removed"].replace("full-", "") for r in rows]); ax[i, 3].axhline(0, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig(ctx.out_dir / "ablation.png", dpi=140); plt.close(fig)


def _plot_distributions(ctx: RunContext, sessions: list[dict]) -> None:
    """Per-session distributions (not just means): violins of net P&L (regime B) and of mean |inventory| for every arm, per environment."""
    fig, ax = plt.subplots(len(ENV_LIST), 2, figsize=(17, 4.2 * len(ENV_LIST)), squeeze=False)
    for i, env in enumerate(ENV_LIST):
        for j, (key, ttl) in enumerate(((f"pnl_net_{PRIMARY_FEE}", f"net P&L per session (regime B)"), ("inv_abs_mean", "mean |inventory| per session"))):
            data = [[r[key] for r in sessions if r["env"] == env and r["arm"] == arm and np.isfinite(r[key])] for arm in ARMS]
            parts = ax[i, j].violinplot(data, showmeans=True, showmedians=True, widths=0.8)
            for pc in parts["bodies"]:
                pc.set_alpha(0.5)
            ax[i, j].set_xticks(range(1, len(ARMS) + 1)); ax[i, j].set_xticklabels([a.replace("_", "\n") for a in ARMS], fontsize=6)
            ax[i, j].set_title(f"{env}: {ttl} (n = {len(data[0])} sessions)")
            if j == 0:
                ax[i, j].axhline(0, color="k", lw=0.5)
    fig.tight_layout(); fig.savefig(ctx.out_dir / "ablation_distributions.png", dpi=130); plt.close(fig)
