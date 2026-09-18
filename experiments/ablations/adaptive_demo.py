"""Stage 7 demonstration: the adaptive market maker next to the Avellaneda-Stoikov baseline.

PRELIMINARY comparison - the controlled ablation ladder (each component added one at a time, with paired uncertainty
and latency/parameter sweeps) is the Stage 9 experiment. This run only shows that the strategy works end to end and how
four arms compare on identical seeds:

  as_gamma0.01        Avellaneda-Stoikov baseline (Stage 4), gamma = 0.01
  adaptive_off        adaptive framework with every component off: symmetric quotes at mid +- 1/k
  adaptive_inventory  + inventory skew and inventory-dependent size
  adaptive_full       + OBI, adverse-selection, queue and latency components

Parameters are *measured* per environment on disjoint seeds (``experiments.ablations.calibration``), not tuned. Two
one-way latencies (0 ms and 5 ms on both legs). Paired bootstrap CIs on differences versus the AS baseline.
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
from experiments.baseline.as_assumptions import ENVS
from experiments.common import RunContext, save_csv, save_json, write_provenance
from simulator.latency.models import LatencyConfig
from simulator.market_state.fundamental import FundamentalConfig
from simulator.simulator import SimConfig
from strategies.adaptive_mm import AdaptiveConfig
from strategies.baseline_mm import ASConfig

DESCRIPTION = "Adaptive market maker vs the AS baseline (preliminary; formal ablations are Stage 9)"

OFF = dict(use_inventory_skew=False, use_inventory_size=False, use_obi=False, use_adverse=False, use_queue=False, use_latency=False)
ARMS = ("as_gamma0.01", "adaptive_off", "adaptive_inventory", "adaptive_full")
METRICS = ("pnl_net", "pnl_edge", "pnl_inventory", "pnl_vol_1s", "sharpe_1s", "max_drawdown", "inv_abs_mean", "inv_max_abs",
           "n_fills", "fill_rate_orders", "quote_to_trade", "quote_lifetime_mean_s", "avg_edge_ticks", "diverged")
PAIRED = ("pnl_net", "pnl_edge", "pnl_inventory", "inv_abs_mean", "n_fills", "sharpe_1s")


def arm_config(arm: str, cal: dict, latency: LatencyConfig):
    common = dict(k=cal["k"], latency=latency, quote_size=5, inventory_limit=50, kill_inventory=100)
    if arm == "as_gamma0.01":
        return ASConfig(gamma=0.01, horizon_s=5.0, **common)
    ad = dict(common, beta_obi=cal["beta_obi"], fill_model=cal["fill_model"])
    if arm == "adaptive_off":
        return AdaptiveConfig(**{**ad, **OFF, "fill_model": None})
    if arm == "adaptive_inventory":
        return AdaptiveConfig(**{**ad, **OFF, "use_inventory_skew": True, "use_inventory_size": True, "fill_model": None})
    if arm == "adaptive_full":
        return AdaptiveConfig(**{**ad, **{k: True for k in OFF}})
    raise KeyError(arm)


def run(ctx: RunContext) -> None:
    t0 = time.time()
    horizon = 30.0 if ctx.quick else 180.0
    seeds = ctx.seeds(default=24, quick=3)
    calib_seeds = [ctx.seed * 100_000 + 50_000 + i for i in range(3 if ctx.quick else 12)]
    table, sessions, cals = [], [], {}
    for env in ENVS:
        cal = calibrate_environment(env, calib_seeds, horizon_s=40.0 if ctx.quick else 120.0, workers=ctx.workers)
        cals[env] = {k: v for k, v in cal.items() if k != "seeds"}
        fm = cal["fill_model"]
        print(f"[{env}] k={cal['k']:.3f}  beta_obi={cal['beta_obi']:.3f} ticks/OBI  fill model: a={fm.intercept:.2f} "
              f"b_lnQ={fm.b_q:.2f} b_d={fm.b_d:.2f}")
        for lat_ms in (0.0, 5.0) if not ctx.quick else (0.0,):
            lat = LatencyConfig.symmetric(lat_ms)
            per = {}
            for arm in ARMS:
                sc = Scenario(f"{env}_{arm}_{lat_ms}", SimConfig(horizon_s=horizon, fundamental=FundamentalConfig(sigma=0.03)),
                              informed=ENVS[env]["informed"], mm=arm_config(arm, cal, lat))
                per[arm] = run_many(sc, seeds, workers=ctx.workers)
                for r in per[arm]:
                    sessions.append(dict(env=env, latency_ms=lat_ms, arm=arm, **r))
            for arm in ARMS:
                rec = dict(env=env, latency_ms=lat_ms, arm=arm, n=len(seeds))
                for m in METRICS:
                    s = summarize([r[m] for r in per[arm]])
                    rec.update({f"{m}_mean": s["mean"], f"{m}_ci_lo": s["ci_lo"], f"{m}_ci_hi": s["ci_hi"], f"{m}_std": s["std"]})
                for ref in ("as_gamma0.01", "adaptive_off"):
                    if arm != ref:
                        for m in PAIRED:
                            d = paired_diff_ci([r[m] for r in per[arm]], [r[m] for r in per[ref]])
                            rec.update({f"d_vs_{ref}_{m}": d["mean_diff"], f"d_vs_{ref}_{m}_lo": d["ci_lo"], f"d_vs_{ref}_{m}_hi": d["ci_hi"]})
                table.append(rec)
                print(f"  {lat_ms:>3.0f}ms {arm:19s} pnl={rec['pnl_net_mean']:+7.2f} [{rec['pnl_net_ci_lo']:+.2f},{rec['pnl_net_ci_hi']:+.2f}] "
                      f"std={rec['pnl_net_std']:5.2f} sharpe={rec['sharpe_1s_mean']:+.3f} |q|={rec['inv_abs_mean_mean']:5.2f} fills={rec['n_fills_mean']:4.0f} "
                      f"edge={rec['pnl_edge_mean']:+.2f} inv={rec['pnl_inventory_mean']:+.2f}")
    save_csv(ctx.out_dir / "sessions.csv", sessions)
    save_csv(ctx.out_dir / "summary.csv", table)
    save_json(ctx.out_dir / "calibration.json", cals)
    _plot(ctx, table)
    write_provenance(ctx, {"sessions": len(seeds), "horizon_s": horizon, "arms": ARMS, "calibration": cals, "calibration_seeds": calib_seeds,
                           "example_adaptive_config": arm_config("adaptive_full", next(iter(cals.values())) | {"k": 1.0}, LatencyConfig())},
                     files=["sessions.csv", "summary.csv", "calibration.json", "adaptive_demo.png"], elapsed_s=time.time() - t0)


def _plot(ctx: RunContext, table: list[dict]) -> None:
    lats = sorted({r["latency_ms"] for r in table})
    fig, ax = plt.subplots(len(lats), 3, figsize=(15, 4.3 * len(lats)), squeeze=False)
    colors = dict(zip(ARMS, ("tab:gray", "tab:olive", "tab:orange", "tab:green")))
    for i, lat in enumerate(lats):
        for j, (m, ttl) in enumerate((("pnl_net", "net P&L per session"), ("sharpe_1s", "per-second Sharpe"), ("inv_abs_mean", "mean |inventory| (lots)"))):
            a = ax[i, j]
            for e, env in enumerate(ENVS):
                for k, arm in enumerate(ARMS):
                    r = next(x for x in table if x["env"] == env and x["latency_ms"] == lat and x["arm"] == arm)
                    y = r[f"{m}_mean"]
                    a.bar(e * 5 + k, y, 0.85, color=colors[arm], label=arm if e == 0 else None,
                          yerr=[[y - r[f"{m}_ci_lo"]], [r[f"{m}_ci_hi"] - y]] if m != "sharpe_1s" else None, capsize=3)
            a.set_xticks([1.5, 6.5]); a.set_xticklabels(list(ENVS)); a.axhline(0, color="k", lw=0.5)
            a.set_title(f"{ttl} - one-way latency {lat:g} ms")
            if i == 0 and j == 0:
                a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ctx.out_dir / "adaptive_demo.png", dpi=140); plt.close(fig)
