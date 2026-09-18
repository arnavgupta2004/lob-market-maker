"""Reproducible parameter sweeps -> response surfaces (spec section 12).

Each sweep evaluates a grid of two parameters on identical seeds (common random numbers) and writes, to ``results/<name>/``:
``cells.csv`` (per-cell mean / median / std / bootstrap CI of every metric), ``sessions.parquet`` (every session), heatmaps of P&L,
Sharpe, maximum drawdown, fill rate, adverse-selection cost and inventory, and ``provenance.json`` (seed, parameters, git commit, ...).

    sweep_gamma_size      AS: risk aversion gamma x quote size                (informed flow)
    sweep_gamma_latency   AS: gamma x one-way latency                         (informed flow + jumps)
    sweep_inventory       AS: soft inventory limit (kill at 2x) x gamma       (informed flow)
    sweep_market          market conditions: fundamental volatility x order-flow intensity, for AS and the adaptive maker
    sweep_adaptive        adaptive maker: (lambda_inv x c_adverse) and (beta_obi multiplier x c_latency)

A sweep describes *how metrics respond to parameters*; it does not select "the best cell". Parameters that the strategy estimates
(``k``, ``beta_obi``, the fill model) are measured once per environment at the default market conditions and then held fixed, so the
market sweep also shows sensitivity to their misspecification.
"""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from backtest.runner import Scenario
from backtest.sweep import Axis, latency_axis, path_axis, plot_heatmaps, replace_path, run_sweep
from experiments.ablations.calibration import calibrate_environment
from experiments.baseline.as_assumptions import ALL_ENVS as ENVS, calibrate, env_fundamental
from experiments.common import RunContext, save_csv, save_parquet, write_provenance
from simulator.latency.models import LatencyConfig
from simulator.order_flow.noise import NoiseTraderConfig
from simulator.simulator import SimConfig
from strategies.adaptive_mm import AdaptiveConfig
from strategies.baseline_mm import ASConfig

DESCRIPTIONS = {
    "sweep_gamma_size": "AS response surface: risk aversion x quote size (informed flow)",
    "sweep_gamma_latency": "AS response surface: risk aversion x one-way latency (informed flow + jumps)",
    "sweep_inventory": "AS response surface: soft inventory limit x risk aversion (informed flow)",
    "sweep_market": "Strategy response to market conditions: volatility x order-flow intensity (AS vs adaptive)",
    "sweep_adaptive": "Adaptive maker coefficient sensitivity: lambda_inv x c_adverse and beta_obi x c_latency",
    "sweep_spread": "Symmetric maker: quoted half-spread x quote size (informed flow) - the classic spread trade-off",
}
METRICS = ("pnl_net", "pnl_edge", "pnl_inventory", "sharpe_1s", "max_drawdown", "fill_rate_orders", "adverse_cost_500ms", "n_fills", "inv_abs_mean",
           "inv_max_abs", "pnl_vol_1s", "quote_lifetime_mean_s", "killed", "diverged")
HEAT = [("pnl_net", "net P&L / session"), ("sharpe_1s", "per-second Sharpe"), ("max_drawdown", "max drawdown"),
        ("fill_rate_orders", "fill rate (orders)"), ("adverse_cost_500ms", "adverse-selection cost (ticks)"), ("inv_abs_mean", "mean |inventory|")]


def _q(ctx: RunContext, values):
    return tuple(values[:2]) if ctx.quick else tuple(values)


def _finish(ctx: RunContext, prefix: str, records, sessions, x: str, y: str, params: dict, t0: float, title: str, log_x=False) -> None:
    tag = f"{prefix}_" if prefix else ""
    save_csv(ctx.out_dir / f"{tag}cells.csv", records)
    save_parquet(ctx.out_dir / f"{tag}sessions.parquet", sessions)
    plot_heatmaps(records, x, y, HEAT, ctx.out_dir / f"{tag}heatmaps.png", title=title)
    write_provenance(ctx, params, files=[f"{tag}cells.csv", f"{tag}sessions.parquet", f"{tag}heatmaps.png"], elapsed_s=time.time() - t0)


def _base(ctx: RunContext, env: str, mm, horizon: float = 120.0, noise=None) -> Scenario:
    return Scenario(f"sweep_{env}", SimConfig(horizon_s=40.0 if ctx.quick else horizon, fundamental=env_fundamental(env)),
                    noise=noise if noise is not None else NoiseTraderConfig(), informed=ENVS[env]["informed"], mm=mm)


def _k(ctx: RunContext, env: str) -> float:
    seeds = [ctx.seed * 100_000 + 50_000 + i for i in range(2 if ctx.quick else 8)]
    return calibrate(env, seeds, horizon_s=40.0 if ctx.quick else 120.0, workers=ctx.workers)["fit"]["k"]


def _run(ctx, base, axes, prefix, x, y, title, params_extra) -> None:
    t0 = time.time()
    seeds = ctx.seeds(default=12, quick=2)
    recs, sess = run_sweep(base, axes, seeds, METRICS, workers=ctx.workers)
    for r in recs[:: max(1, len(recs) // 6)]:
        print("  " + "  ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in r.items() if k in {a.name for a in axes}) +
              f"   pnl={r['pnl_net_mean']:+.2f} sharpe={r['sharpe_1s_mean']:+.2f} fill_rate={r['fill_rate_orders_mean']:.2f} adv={r['adverse_cost_500ms_mean']:.2f}")
    _finish(ctx, prefix, recs, sess, x, y, {"sessions": len(seeds), "axes": {a.name: a.values for a in axes}, "base": base, **params_extra}, t0, title)


def sweep_gamma_size(ctx: RunContext) -> None:
    k = _k(ctx, "informed")
    base = _base(ctx, "informed", ASConfig(gamma=0.01, k=k, horizon_s=5.0, inventory_limit=50, kill_inventory=100))
    axes = [path_axis("mm.gamma", _q(ctx, (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)), "gamma"), path_axis("mm.quote_size", _q(ctx, (1, 2, 5, 10, 20)), "quote_size")]
    _run(ctx, base, axes, "", "gamma", "quote_size", "AS: gamma x quote size (informed flow)", {"k": k})


def sweep_gamma_latency(ctx: RunContext) -> None:
    k = _k(ctx, "informed_jump")
    base = _base(ctx, "informed_jump", ASConfig(gamma=0.01, k=k, horizon_s=5.0, inventory_limit=50, kill_inventory=100))
    axes = [path_axis("mm.gamma", _q(ctx, (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)), "gamma"), latency_axis(_q(ctx, (0.0, 1.0, 5.0, 10.0, 25.0, 50.0)))]
    _run(ctx, base, axes, "", "gamma", "latency_ms", "AS: gamma x one-way latency (informed flow + jumps)", {"k": k})


def sweep_inventory(ctx: RunContext) -> None:
    k = _k(ctx, "informed")
    base = _base(ctx, "informed", ASConfig(gamma=0.01, k=k, horizon_s=5.0, inventory_limit=50, kill_inventory=100))
    limit_axis = Axis("inventory_limit", _q(ctx, (5, 10, 20, 50, 100)),
                      lambda sc, v: replace_path(replace_path(sc, "mm.inventory_limit", v), "mm.kill_inventory", 2 * v))
    axes = [path_axis("mm.gamma", _q(ctx, (0.0, 0.005, 0.02, 0.05)), "gamma"), limit_axis]
    _run(ctx, base, axes, "", "gamma", "inventory_limit", "AS: soft inventory limit (kill at 2x) x gamma (informed flow)", {"k": k})


def sweep_market(ctx: RunContext) -> None:
    t0 = time.time()
    env = "informed"
    seeds_c = [ctx.seed * 100_000 + 50_000 + i for i in range(2 if ctx.quick else 12)]
    cal = calibrate_environment(env, seeds_c, horizon_s=40.0 if ctx.quick else 120.0, workers=ctx.workers)
    common = dict(k=cal["k"], quote_size=5, inventory_limit=50, kill_inventory=100)
    mms = {"AS": ASConfig(gamma=0.01, horizon_s=5.0, **common),
           "adaptive": AdaptiveConfig(**common, beta_obi=cal["beta_obi"], fill_model=cal["fill_model"])}
    base = _base(ctx, env, mms["AS"])
    scale = Axis("flow_intensity_x", _q(ctx, (0.5, 1.0, 2.0)), lambda sc, v: replace_path(replace_path(sc, "noise.limit_rate", 50.0 * v), "noise.market_rate", 15.0 * v))
    sig = Axis("fundamental_sigma", _q(ctx, (0.01, 0.02, 0.03, 0.05, 0.08)), lambda sc, v: replace_path(sc, "sim.fundamental.sigma", v))
    seeds = ctx.seeds(default=12, quick=2)
    all_recs, all_sess = [], []
    for name, mm in mms.items():
        recs, sess = run_sweep(dataclasses.replace(base, mm=mm), [sig, scale], seeds, METRICS, workers=ctx.workers)
        for r in recs:
            r["strategy"] = name
        for r in sess:
            r["strategy"] = name
        all_recs += recs; all_sess += sess
        print(f"  {name}: pnl range over the grid {min(r['pnl_net_mean'] for r in recs):+.1f} .. {max(r['pnl_net_mean'] for r in recs):+.1f}")
        plot_heatmaps(recs, "fundamental_sigma", "flow_intensity_x", HEAT, ctx.out_dir / f"{name}_heatmaps.png", title=f"{name}: volatility x order-flow intensity")
    save_csv(ctx.out_dir / "cells.csv", all_recs)
    save_parquet(ctx.out_dir / "sessions.parquet", all_sess)
    write_provenance(ctx, {"sessions": len(seeds), "axes": {"fundamental_sigma": sig.values, "flow_intensity_x": scale.values}, "strategies": mms, "calibration": {k: v for k, v in cal.items() if k != "seeds"},
                           "note": "k, beta_obi, fill model measured once at default conditions and held fixed across the grid"},
                     files=["cells.csv", "sessions.parquet", "AS_heatmaps.png", "adaptive_heatmaps.png"], elapsed_s=time.time() - t0)


def sweep_adaptive(ctx: RunContext) -> None:
    t0 = time.time()
    env = "informed"
    seeds_c = [ctx.seed * 100_000 + 50_000 + i for i in range(2 if ctx.quick else 12)]
    cal = calibrate_environment(env, seeds_c, horizon_s=40.0 if ctx.quick else 120.0, workers=ctx.workers)
    lat = LatencyConfig.symmetric(10.0)
    mm = AdaptiveConfig(k=cal["k"], beta_obi=cal["beta_obi"], fill_model=cal["fill_model"], quote_size=5, inventory_limit=50, kill_inventory=100, latency=lat)
    base = _base(ctx, env, mm)
    seeds = ctx.seeds(default=12, quick=2)
    beta = cal["beta_obi"]
    grids = {
        "A": [path_axis("mm.lambda_inv", _q(ctx, (0.02, 0.05, 0.1, 0.2, 0.4)), "lambda_inv"), path_axis("mm.c_adverse", _q(ctx, (0.0, 0.5, 1.0, 1.5, 2.0)), "c_adverse")],
        "B": [Axis("beta_obi_x", _q(ctx, (0.0, 0.5, 1.0, 1.5, 2.0)), lambda sc, v: replace_path(sc, "mm.beta_obi", beta * v)), path_axis("mm.c_latency", _q(ctx, (0.0, 0.5, 1.0, 2.0)), "c_latency")],
    }
    for tag, axes in grids.items():
        recs, sess = run_sweep(base, axes, seeds, METRICS, workers=ctx.workers)
        save_csv(ctx.out_dir / f"grid{tag}_cells.csv", recs)
        save_parquet(ctx.out_dir / f"grid{tag}_sessions.parquet", sess)
        plot_heatmaps(recs, axes[0].name, axes[1].name, HEAT, ctx.out_dir / f"grid{tag}_heatmaps.png", title=f"adaptive maker: {axes[0].name} x {axes[1].name} (informed flow, 10 ms)")
        print(f"  grid {tag}: pnl range {min(r['pnl_net_mean'] for r in recs):+.1f} .. {max(r['pnl_net_mean'] for r in recs):+.1f}")
    write_provenance(ctx, {"sessions": len(seeds), "grids": {t: {a.name: a.values for a in ax} for t, ax in grids.items()}, "base": base, "calibration": {k: v for k, v in cal.items() if k != "seeds"}},
                     files=[f"grid{t}_{s}" for t in grids for s in ("cells.csv", "sessions.parquet", "heatmaps.png")], elapsed_s=time.time() - t0)


def sweep_spread(ctx: RunContext) -> None:
    """The spread parameter itself: symmetric quotes at mid +- h (adaptive framework, all components off) for a grid of half-spreads h (ticks) and quote sizes."""
    k = _k(ctx, "informed")
    off = dict(use_inventory_skew=False, use_inventory_size=False, use_obi=False, use_adverse=False, use_queue=False, use_latency=False)
    base = _base(ctx, "informed", AdaptiveConfig(k=k, quote_size=5, inventory_limit=50, kill_inventory=100, **off))
    axes = [path_axis("mm.base_half_spread", _q(ctx, (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)), "half_spread_ticks"), path_axis("mm.quote_size", _q(ctx, (1, 2, 5, 10)), "quote_size")]
    _run(ctx, base, axes, "", "half_spread_ticks", "quote_size", "Symmetric quoting: half-spread x quote size (informed flow)", {"k": k, "note": "1/k = risk-neutral AS half-spread = %.2f ticks" % (1 / k)})
