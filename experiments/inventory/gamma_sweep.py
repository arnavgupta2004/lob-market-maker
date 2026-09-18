"""Experiment E: how does inventory risk aversion change quoting behaviour and outcomes?

Hypotheses (stated before running; each is falsifiable with the reported CIs)
    H1  Larger risk aversion gamma reduces inventory dispersion (std and mean |q|).
    H2  Larger gamma reduces session-to-session P&L dispersion.
    H3  The mean-P&L cost of this risk reduction is small relative to its dispersion benefit.
        (AS optimises expected CARA utility, not mean P&L, so a mean-P&L cost is expected; how
        large it is here is the empirical question.)
    H4  The benefit of inventory skew is larger under informed flow (adverse selection) than
        under pure noise flow.
Design: two environments (noise; noise + informed). ``k`` is *estimated* per environment with
the probe calibration on seeds disjoint from the evaluation seeds. All gamma arms use the same
evaluation seeds (common random numbers) so differences are paired. ``gamma = 0`` is the
inventory-blind symmetric quoter and serves as the control.
"""
from __future__ import annotations

import dataclasses
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backtest.runner import Scenario, run_many
from backtest.statistics import bootstrap_ci, paired_diff_ci, summarize
from experiments.baseline.as_assumptions import ENVS, calibrate
from experiments.common import RunContext, save_csv, save_json, write_provenance
from simulator.market_state.fundamental import FundamentalConfig
from simulator.simulator import SimConfig
from strategies.baseline_mm import ASConfig

DESCRIPTION = "AS baseline: inventory / P&L as a function of risk aversion gamma, noise vs informed flow"

GAMMAS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
UNGUARDED_GAMMAS = (0.05, 0.2)  # failure-case arms: volatility bounds and quote-offset cap switched off
HORIZON_T = 5.0  # AS time horizon (s), 'restart' mode
METRICS = ("pnl_net", "pnl_edge", "pnl_inventory", "pnl_vol_1s", "sharpe_1s", "max_drawdown", "inv_std",
           "inv_abs_mean", "inv_max_abs", "n_fills", "fill_rate_orders", "avg_edge_ticks", "quote_to_trade",
           "killed", "diverged", "n_sigma_clamped", "n_offset_clamped")
PAIRED = ("pnl_net", "pnl_edge", "pnl_inventory", "inv_abs_mean", "inv_std", "n_fills")


def scenario(env: str, gamma: float, k: float, horizon_s: float, guarded: bool = True) -> Scenario:
    return Scenario(
        name=f"{env}_g{gamma}",
        sim=SimConfig(horizon_s=horizon_s, fundamental=FundamentalConfig(sigma=0.03)),
        informed=ENVS[env]["informed"],
        mm=ASConfig(gamma=gamma, k=k, horizon_s=HORIZON_T, quote_size=5, inventory_limit=50, kill_inventory=100,
                    **({} if guarded else dict(sigma_bounds=None, max_quote_offset_ticks=None))))


def run(ctx: RunContext) -> None:
    t0 = time.time()
    horizon = 30.0 if ctx.quick else 180.0
    seeds = ctx.seeds(default=30, quick=3)
    calib_seeds = [ctx.seed * 100_000 + 50_000 + i for i in range(3 if ctx.quick else 12)]  # disjoint from `seeds`
    gammas = GAMMAS[:3] if ctx.quick else GAMMAS

    table, sessions, ks = [], [], {}
    for env in ENVS:
        cal = calibrate(env, calib_seeds, horizon_s=40.0 if ctx.quick else 120.0, workers=ctx.workers)
        k = cal["fit"]["k"]
        ks[env] = {"k": k, "k_ci": cal["fit"]["k_ci"], "A": cal["fit"]["A"], "r2": cal["fit"]["r2"]}
        print(f"[{env}] calibrated k = {k:.3f}")
        per_gamma = {}
        for g in gammas:
            rows = run_many(scenario(env, g, k, horizon), seeds, workers=ctx.workers)
            per_gamma[g] = rows
            for r in rows:
                sessions.append(dict(env=env, gamma=g, **r))
        base = per_gamma[gammas[0]]
        for g in gammas:
            rows = per_gamma[g]
            rec: dict = dict(env=env, gamma=g, k=k, n=len(rows))
            for m in METRICS:
                v = [r[m] for r in rows]
                s = summarize(v)
                rec.update({f"{m}_mean": s["mean"], f"{m}_ci_lo": s["ci_lo"], f"{m}_ci_hi": s["ci_hi"],
                            f"{m}_std": s["std"], f"{m}_median": s["median"]})
            if g != gammas[0]:
                for m in PAIRED:
                    d = paired_diff_ci([r[m] for r in rows], [r[m] for r in base])
                    rec.update({f"d_{m}": d["mean_diff"], f"d_{m}_lo": d["ci_lo"], f"d_{m}_hi": d["ci_hi"],
                                f"d_{m}_sig": d["significant"]})
            table.append(rec)
            print(f"  gamma={g:<6} pnl={rec['pnl_net_mean']:+.3f} [{rec['pnl_net_ci_lo']:+.3f},{rec['pnl_net_ci_hi']:+.3f}] "
                  f"|q|={rec['inv_abs_mean_mean']:.2f} maxq={rec['inv_max_abs_mean']:.1f} fills={rec['n_fills_mean']:.0f} "
                  f"edge={rec['pnl_edge_mean']:+.2f} invpnl={rec['pnl_inventory_mean']:+.2f} kill={rec['killed_mean']:.2f} "
                  f"div={rec['diverged_mean']:.2f} clamp(s/o)={rec['n_sigma_clamped_mean']:.0f}/{rec['n_offset_clamped_mean']:.0f}")
    # failure-case arms (unguarded): how often does the baseline blow up?
    failures = []
    for env in ENVS:
        for g in UNGUARDED_GAMMAS if not ctx.quick else UNGUARDED_GAMMAS[:1]:
            rows = run_many(scenario(env, g, ks[env]["k"], horizon, guarded=False), seeds, workers=ctx.workers)
            div = [r.get("diverged", 0.0) for r in rows]
            crashed = [r.get("crashed", 0.0) for r in rows]
            lo, hi = bootstrap_ci(div)
            failures.append(dict(env=env, gamma=g, n=len(rows), diverged_rate=float(np.mean(div)), diverged_ci_lo=lo,
                                 diverged_ci_hi=hi, numerically_crashed=float(np.sum(crashed))))
            print(f"  UNGUARDED {env} gamma={g}: diverged {np.mean(div):.0%} of sessions [{lo:.0%},{hi:.0%}]")
            for r in rows:
                sessions.append(dict(env=env, gamma=g, guarded=False, **r))
    save_csv(ctx.out_dir / "failures_unguarded.csv", failures)
    save_csv(ctx.out_dir / "sessions.csv", sessions)
    save_csv(ctx.out_dir / "summary.csv", table)
    save_json(ctx.out_dir / "summary.json", {"calibration": ks, "table": table})
    _plot(ctx, table, gammas)
    write_provenance(ctx, {"gammas": gammas, "horizon_s": horizon, "sessions": len(seeds), "AS_horizon_T_s": HORIZON_T,
                           "example_scenario": scenario("noise", 0.05, ks["noise"]["k"], horizon),
                           "calibration": ks, "calibration_seeds": calib_seeds},
                     files=["sessions.csv", "summary.csv", "summary.json", "failures_unguarded.csv", "inventory.png"], elapsed_s=time.time() - t0)


def _plot(ctx: RunContext, table: list[dict], gammas) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    colors = {"noise": "tab:blue", "informed": "tab:red"}
    x_of = lambda g: g if g > 0 else 0.002  # place gamma = 0 on the log axis
    panels = [("pnl_net", "mean net P&L per session (currency)"), ("inv_std", "inventory std (lots)"),
              ("inv_max_abs", "max |inventory| (lots)"), ("pnl_std", "std of net P&L across sessions"),
              ("edge_vs_inv", "P&L decomposition"), ("n_fills", "fills per session")]
    for env, c in colors.items():
        rows = [r for r in table if r["env"] == env]
        xs = [x_of(r["gamma"]) for r in rows]
        for a, (m, title) in zip(ax.flat, panels):
            if m == "pnl_std":
                a.plot(xs, [r["pnl_net_std"] for r in rows], "o-", color=c, label=env)
            elif m == "edge_vs_inv":
                a.plot(xs, [r["pnl_edge_mean"] for r in rows], "o-", color=c, label=f"{env}: edge")
                a.plot(xs, [r["pnl_inventory_mean"] for r in rows], "s--", color=c, label=f"{env}: inventory")
            else:
                mean = np.array([r[f"{m}_mean"] for r in rows])
                lo = np.array([r[f"{m}_ci_lo"] for r in rows]); hi = np.array([r[f"{m}_ci_hi"] for r in rows])
                a.errorbar(xs, mean, [mean - lo, hi - mean], fmt="o-", color=c, capsize=3, label=env)
            a.set(title=title, xscale="log", xlabel="risk aversion γ (1/tick; 0 plotted at 0.002)")
    for a in ax.flat:
        a.legend(fontsize=7)
    ax[0, 0].axhline(0, color="k", lw=0.6)
    fig.tight_layout()
    fig.savefig(ctx.out_dir / "inventory.png", dpi=140)
    plt.close(fig)
