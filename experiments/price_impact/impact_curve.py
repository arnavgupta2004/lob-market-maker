"""Price impact of aggressive orders: estimate ``I(Q) ~ c Q^alpha`` (alpha is *estimated*, never assumed).

Hypotheses (stated before running)
    H1  Impact increases with order size Q at every horizon.
    H2  The exponent alpha is not 1 (impact is not linear in size); its value differs by horizon and by who trades.
    H3  Impact of informed orders exceeds that of noise orders of the same size at longer horizons (information),
        whereas immediate (book-walking) impact is mechanical and similar.
Method: aggressive orders = all trades sharing a taker order id; impact I_h = s (m_{t+h} - m^-) with h = 0 (immediately
after the order), 0.1 s, 1 s, 5 s; slippage = s (VWAP - m^-). Log-spaced size bins; weighted log-log fit of the per-bin
mean impact vs mean size; alpha CI by bootstrapping whole sessions. Bins with non-positive mean impact are excluded from
the fit (and reported). Simulated flow only: the exponent found here describes this simulator; the empirical exponent is
compared in Stage 8 on real data.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.baseline.as_assumptions import ENVS
from experiments.common import RunContext, pmap, save_csv, save_json, write_provenance
from research.price_impact import add_stats, aggressive_orders, bin_table, fit_power_law, impacts, size_bin_stats
from simulator.market_state.fundamental import FundamentalConfig
from simulator.order_flow.informed import InformedTrader
from simulator.order_flow.noise import NoiseTrader
from simulator.simulator import SimConfig, Simulator

DESCRIPTION = "Estimate the price-impact exponent alpha in I(Q) ~ Q^alpha at several horizons"

EDGES = (1, 2, 3, 4, 6, 9, 14, 21, 32, 50, 10_000)
KEYS = ("I_0", "I_0.1s", "I_1s", "I_5s", "slippage")
GROUPS = {"all": None, "noise": 1, "informed": 2}


@dataclass(frozen=True)
class Job:
    env: str
    seed: int
    horizon_s: float


def session(job: Job) -> dict:
    cfg = SimConfig(seed=job.seed, horizon_s=job.horizon_s, fundamental=FundamentalConfig(sigma=0.03), record_events=False, record_commands=False)
    parts = [NoiseTrader()] + ([InformedTrader(ENVS[job.env]["informed"])] if ENVS[job.env]["informed"] else [])
    res = Simulator(cfg, parts).run()
    out = {}
    for g, owner in GROUPS.items():
        ao = aggressive_orders(res, owner)
        ao = type(ao)(*[a[ao.t >= 5_000_000_000] for a in (ao.t, ao.side, ao.qty, ao.mid_before, ao.vwap, ao.taker_owner)])
        if len(ao.t) == 0:
            continue
        im = impacts(res, ao)
        out[g] = {k: size_bin_stats(ao.qty, im[k], EDGES) for k in KEYS}
    return out


def run(ctx: RunContext) -> None:
    t0 = time.time()
    seeds = ctx.seeds(default=24, quick=3)
    horizon = 40.0 if ctx.quick else 300.0
    rng = np.random.default_rng(ctx.seed)
    fits, tables = [], []
    for env in ENVS:
        sess = pmap(session, [Job(env, s, horizon) for s in seeds], ctx.workers)
        for g in GROUPS:
            if env == "noise" and g == "informed":
                continue
            per = [s[g] for s in sess if g in s]
            for k in KEYS:
                total: dict = {}
                for s in per:
                    total = add_stats(total, s[k])
                fit = fit_power_law(total)
                boots = []
                for _ in range(60 if ctx.quick else 300):
                    pick = rng.integers(0, len(per), len(per))
                    agg: dict = {}
                    for i in pick:
                        agg = add_stats(agg, per[i][k])
                    boots.append(fit_power_law(agg)["alpha"])
                fits.append(dict(env=env, group=g, impact=k, alpha=fit["alpha"], alpha_lo=float(np.nanquantile(boots, .025)), alpha_hi=float(np.nanquantile(boots, .975)),
                                 c=fit["c"], r2=fit["r2"], bins_used=fit["n_bins"], orders=int(sum(v[0] for v in total.values()))))
                for row in bin_table(total, EDGES):
                    tables.append(dict(env=env, group=g, impact=k, **row))
        f = {(r["group"], r["impact"]): r for r in fits if r["env"] == env}
        print(f"[{env}] alpha (95% CI) for all aggressive orders: " + "  ".join(
            f"{k}={f[('all', k)]['alpha']:.2f}[{f[('all', k)]['alpha_lo']:.2f},{f[('all', k)]['alpha_hi']:.2f}]" for k in KEYS))
        if env == "informed":
            print("   by taker (I_1s): " + "  ".join(f"{g}={f[(g, 'I_1s')]['alpha']:.2f}" for g in GROUPS))
    save_csv(ctx.out_dir / "alpha_fits.csv", fits)
    save_csv(ctx.out_dir / "impact_by_size.csv", tables)
    _plot(ctx, tables, fits)
    write_provenance(ctx, {"sessions": len(seeds), "horizon_s": horizon, "size_edges": EDGES, "impact_keys": KEYS, "envs": {k: v for k, v in ENVS.items()}},
                     files=["alpha_fits.csv", "impact_by_size.csv", "price_impact.png"], elapsed_s=time.time() - t0)


def _plot(ctx: RunContext, tables: list[dict], fits: list[dict]) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    cols = dict(zip(KEYS, ("tab:gray", "tab:green", "tab:blue", "tab:red", "tab:orange")))
    for a, (env, g) in zip(ax, (("noise", "all"), ("informed", "noise"), ("informed", "informed"))):
        for k in KEYS:
            rr = [r for r in tables if r["env"] == env and r["group"] == g and r["impact"] == k and r["n"] >= 30 and r["mean_impact"] > 0]
            if not rr:
                continue
            q = np.array([r["mean_q"] for r in rr]); y = np.array([r["mean_impact"] for r in rr]); se = np.array([r["se"] for r in rr])
            a.errorbar(q, y, 1.96 * se, fmt="o", color=cols[k], ms=4, capsize=2)
            f = next(r for r in fits if r["env"] == env and r["group"] == g and r["impact"] == k)
            xx = np.geomspace(q.min(), q.max(), 30)
            a.plot(xx, f["c"] * xx ** f["alpha"], "-", color=cols[k], lw=1, label=f"{k}: α={f['alpha']:.2f} [{f['alpha_lo']:.2f},{f['alpha_hi']:.2f}]")
        a.set(xscale="log", yscale="log", xlabel="order size Q (lots)", ylabel="mean impact (ticks)", title=f"{env} env, {g} takers"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ctx.out_dir / "price_impact.png", dpi=140); plt.close(fig)
