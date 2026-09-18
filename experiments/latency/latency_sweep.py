"""Experiment D: how does latency affect market-making profitability, fills, adverse selection, inventory and quote survival?

Hypotheses (stated before running)
    H1  Higher latency lowers net P&L, and the loss grows faster where the price *jumps* (news) than where it only diffuses,
        because stale quotes are picked off by fast informed flow.
    H2  Adverse selection (post-fill signed mid move) worsens with latency.
    H3  Fill count / fill rate fall with latency (later arrival = worse queue position, stale prices).
    H4  The adaptive strategy's latency component (widening by sigma * sqrt(measured latency)) reduces the P&L loss relative to
        the same strategy without it.
Design: one-way latency in {0, 1, 5, 10, 25, 50} ms applied to both legs (round trip = 2x) for the market maker only (noise and
informed flow are exogenous, latency-free). Environments: informed flow with pure diffusion, and informed flow + jump "news"
(``informed_jump``). Strategies: AS baseline (gamma = .01), the full adaptive maker, and the adaptive maker with its latency
component switched off. Same 24 seeds in every arm (paired). Parameters (k, beta_obi, fill model) are measured per environment on
disjoint seeds. Diffusion moves the fundamental only ~0.7 ticks in 50 ms, so a weak latency effect in the diffusion environment
is expected *a priori*, not a finding about the strategies.
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backtest.runner import Scenario, run_many
from backtest.statistics import paired_diff_ci, summarize
from experiments.ablations.calibration import calibrate_environment
from experiments.baseline.as_assumptions import ALL_ENVS as ENVS, env_fundamental
from experiments.common import RunContext, save_csv, save_json, save_parquet, write_provenance
from simulator.latency.models import LatencyConfig
from simulator.simulator import SimConfig
from strategies.adaptive_mm import AdaptiveConfig
from strategies.baseline_mm import ASConfig

DESCRIPTION = "Latency vs P&L, fills, adverse selection, inventory and quote survival (Experiment D)"

LATENCIES_MS = (0.0, 1.0, 5.0, 10.0, 25.0, 50.0)
ENV_LIST = ("informed", "informed_jump")
ARMS = ("as_gamma0.01", "adaptive_full", "adaptive_no_latency_term")
METRICS = ("pnl_net", "pnl_edge", "pnl_inventory", "sharpe_1s", "n_fills", "fill_rate_orders", "fill_ratio_qty", "adverse_cost_500ms",
           "postfill_10ms", "postfill_100ms", "eff_half_spread_ticks", "inv_abs_mean", "inv_max_abs", "quote_lifetime_mean_s",
           "quote_surv_250ms", "quote_to_trade", "max_drawdown", "killed", "diverged")
PAIRED = ("pnl_net", "n_fills", "fill_rate_orders", "adverse_cost_500ms", "inv_abs_mean", "quote_surv_250ms")


def arm_config(arm: str, cal: dict, lat: LatencyConfig):
    common = dict(k=cal["k"], latency=lat, quote_size=5, inventory_limit=50, kill_inventory=100)
    if arm == "as_gamma0.01":
        return ASConfig(gamma=0.01, horizon_s=5.0, **common)
    full = dict(common, beta_obi=cal["beta_obi"], fill_model=cal["fill_model"])
    if arm == "adaptive_full":
        return AdaptiveConfig(**full)
    if arm == "adaptive_no_latency_term":
        return AdaptiveConfig(**full, use_latency=False)
    raise KeyError(arm)


def run(ctx: RunContext) -> None:
    t0 = time.time()
    horizon = 40.0 if ctx.quick else 180.0
    seeds = ctx.seeds(default=24, quick=3)
    lats = (0.0, 10.0) if ctx.quick else LATENCIES_MS
    calib_seeds = [ctx.seed * 100_000 + 50_000 + i for i in range(3 if ctx.quick else 12)]
    table, sessions, cals = [], [], {}
    for env in ENV_LIST:
        cal = calibrate_environment(env, calib_seeds, horizon_s=40.0 if ctx.quick else 120.0, workers=ctx.workers)
        cals[env] = {k: v for k, v in cal.items() if k != "seeds"}
        print(f"[{env}] k={cal['k']:.3f} beta_obi={cal['beta_obi']:.3f}")
        per: dict = {}
        for arm in ARMS:
            for lat_ms in lats:
                sc = Scenario(f"{env}_{arm}_{lat_ms}", SimConfig(horizon_s=horizon, fundamental=env_fundamental(env)),
                              informed=ENVS[env]["informed"], mm=arm_config(arm, cal, LatencyConfig.symmetric(lat_ms)))
                per[(arm, lat_ms)] = run_many(sc, seeds, workers=ctx.workers)
                for r in per[(arm, lat_ms)]:
                    sessions.append(dict(env=env, arm=arm, latency_ms=lat_ms, **r))
        for arm in ARMS:
            for lat_ms in lats:
                rows = per[(arm, lat_ms)]
                rec = dict(env=env, arm=arm, latency_ms=lat_ms, n=len(rows))
                for m in METRICS:
                    s = summarize([r.get(m, np.nan) for r in rows])
                    rec.update({f"{m}_mean": s["mean"], f"{m}_ci_lo": s["ci_lo"], f"{m}_ci_hi": s["ci_hi"], f"{m}_std": s["std"]})
                if lat_ms != lats[0]:
                    for m in PAIRED:
                        d = paired_diff_ci([r.get(m, np.nan) for r in rows], [r.get(m, np.nan) for r in per[(arm, lats[0])]])
                        rec.update({f"d0_{m}": d["mean_diff"], f"d0_{m}_lo": d["ci_lo"], f"d0_{m}_hi": d["ci_hi"], f"d0_{m}_p": d["p_value"]})
                table.append(rec)
        for arm in ARMS:
            line = "  ".join(f"{l:g}ms:{next(r for r in table if r['env'] == env and r['arm'] == arm and r['latency_ms'] == l)['pnl_net_mean']:+.1f}" for l in lats)
            print(f"  {arm:24s} pnl_net  {line}")
    save_csv(ctx.out_dir / "summary.csv", table)
    save_parquet(ctx.out_dir / "sessions.parquet", sessions)
    save_json(ctx.out_dir / "calibration.json", cals)
    _plot(ctx, table, lats)
    write_provenance(ctx, {"sessions": len(seeds), "horizon_s": horizon, "latencies_ms": lats, "arms": ARMS, "environments": {e: ENVS[e] for e in ENV_LIST},
                           "calibration": cals, "calibration_seeds": calib_seeds},
                     files=["summary.csv", "sessions.parquet", "calibration.json", "latency.png"], elapsed_s=time.time() - t0)


def _plot(ctx: RunContext, table: list[dict], lats) -> None:
    metrics = [("pnl_net", "net P&L"), ("n_fills", "fills"), ("adverse_cost_500ms", "adverse-selection cost (ticks, 500 ms)"),
               ("inv_abs_mean", "mean |inventory|"), ("quote_surv_250ms", "quote survival >= 250 ms")]
    fig, ax = plt.subplots(len(ENV_LIST), len(metrics), figsize=(4.2 * len(metrics), 3.8 * len(ENV_LIST)), squeeze=False)
    colors = dict(zip(ARMS, ("tab:gray", "tab:green", "tab:olive")))
    for i, env in enumerate(ENV_LIST):
        for j, (m, ttl) in enumerate(metrics):
            for arm in ARMS:
                rr = [next(r for r in table if r["env"] == env and r["arm"] == arm and r["latency_ms"] == l) for l in lats]
                y = np.array([r[f"{m}_mean"] for r in rr])
                ax[i, j].errorbar(lats, y, [y - [r[f"{m}_ci_lo"] for r in rr], [r[f"{m}_ci_hi"] for r in rr] - y], fmt="o-", color=colors[arm], capsize=2, label=arm)
            ax[i, j].set(title=f"{env}: {ttl}", xlabel="one-way latency (ms)", xscale="symlog", xticks=lats, xticklabels=[f"{l:g}" for l in lats])
            if m == "pnl_net":
                ax[i, j].axhline(0, color="k", lw=0.5)
            if i == 0 and j == 0:
                ax[i, j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ctx.out_dir / "latency.png", dpi=140); plt.close(fig)
