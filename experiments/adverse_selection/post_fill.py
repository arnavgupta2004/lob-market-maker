"""Experiment C: do market-maker fills predict subsequent price movement (adverse selection)?

Hypotheses (stated before running)
    H0  Control: at random times with random side the signed mid change ``s (m_{t+h} - m_t)`` has mean ~ 0
        (the mid has no drift over these horizons), so any non-zero mean at *fills* is caused by the fill.
    H1  Maker fills are followed by adverse mid moves: ``M_h = s (m_{t+h} - m^-) < 0`` for h in 10 ms ... 1 s.
    H2  The cost is concentrated in fills against informed takers (measured directly: the simulator knows who traded).
    H3  Fills that extend the maker's inventory are more adverse than fills that reduce it.
    H4  Fills taken when the book leans *against* the maker (thin own side) are more adverse than when it leans with it.
Decomposition per fill (ticks per lot): effective half-spread E = s(m^- - p); signed move M_h; realised half-spread
R_h = E + M_h. Reported per session (quantity-weighted), then averaged over sessions with bootstrap CIs. No VPIN or other
toxicity proxy is used; the measure is the post-fill price movement itself.
Market maker: the Avellaneda-Stoikov baseline with modest risk aversion (gamma = 0.01), ``k`` calibrated per environment on
disjoint seeds. Environments: noise flow; noise + informed flow.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backtest.runner import Scenario, run_session
from backtest.statistics import summarize
from experiments.baseline.as_assumptions import ENVS, calibrate
from experiments.common import RunContext, pmap, save_csv, save_json, write_provenance
from research.adverse_selection import (
    HORIZONS_NS, by_group, passive_fills, random_time_markouts, signed_obi_at_fills, summarize_fills,
)
from simulator.market_state.fundamental import FundamentalConfig
from simulator.simulator import NS, SimConfig
from strategies.baseline_mm import ASConfig

DESCRIPTION = "Post-fill price movement of market-maker fills, by taker type / inventory effect / imbalance (Experiment C)"

GAMMA = 0.01
GROUP_H = ("100ms", "500ms", "1s")


@dataclass(frozen=True)
class Job:
    env: str
    seed: int
    horizon_s: float
    k: float


def session(job: Job) -> dict:
    sc = Scenario(name=job.env, sim=SimConfig(horizon_s=job.horizon_s, sample_interval_s=0.01, fundamental=FundamentalConfig(sigma=0.03),
                                              record_events=False),
                  informed=ENVS[job.env]["informed"], mm=ASConfig(gamma=GAMMA, k=job.k, horizon_s=5.0))
    s = run_session(sc, job.seed)
    res, mm = s.result, s.mm
    informed_owner = 2 if ENVS[job.env]["informed"] else -1
    out: dict = {"pnl_net": s.metrics["pnl_net"]}
    mmf = passive_fills(res, mm.owner_id)
    allf = passive_fills(res)
    allf = allf.select(allf.t >= 5 * NS)
    out["mm_profile"] = summarize_fills(res, mmf)
    out["all_profile"] = summarize_fills(res, allf)
    out["null"] = random_time_markouts(res, 20_000, np.random.default_rng(job.seed))
    taker_groups = {"taker_informed": mmf.taker_owner == informed_owner, "taker_noise": mmf.taker_owner == 1}
    inv_before = np.cumsum(mmf.side * mmf.qty) - mmf.side * mmf.qty  # maker fills only (no kill-switch in these runs)
    inv_groups = {"extends_inventory": mmf.side * inv_before > 0, "reduces_inventory": mmf.side * inv_before < 0,
                  "from_flat": inv_before == 0}
    sobi = signed_obi_at_fills(res, mmf, levels=1)
    obi_groups = {"book_leans_against(s*OBI<=-.5)": sobi <= -0.5, "neutral": (sobi > -0.5) & (sobi < 0.5),
                  "book_leans_with(s*OBI>=.5)": sobi >= 0.5}
    out["groups"] = {}
    for scope, groups in (("taker", taker_groups), ("inventory", inv_groups), ("obi", obi_groups)):
        for h in GROUP_H:
            for g, v in by_group(res, mmf, groups, horizon=h).items():
                out["groups"][(scope, g, h)] = v
    return out


def agg(sess: list[dict], getter) -> dict:
    vals = [getter(s) for s in sess]
    return summarize([v for v in vals if v is not None], n_boot=3000, seed=5)


def run(ctx: RunContext) -> None:
    t0 = time.time()
    seeds = ctx.seeds(default=24, quick=3)
    horizon = 40.0 if ctx.quick else 180.0
    calib = [ctx.seed * 100_000 + 50_000 + i for i in range(3 if ctx.quick else 12)]
    rows, ks = [], {}
    for env in ENVS:
        cal = calibrate(env, calib, horizon_s=40.0 if ctx.quick else 120.0, workers=ctx.workers)
        k = cal["fit"]["k"]; ks[env] = k
        sess = pmap(session, [Job(env, s, horizon, k) for s in seeds], ctx.workers)
        print(f"[{env}] k={k:.3f}  fills/session (MM)={np.mean([s['mm_profile']['n_fills'] for s in sess]):.0f}")
        for scope in ("mm_profile", "all_profile"):
            for key in ("E", *[f"{m}_{h}" for h in HORIZONS_NS for m in ("M", "R")]):
                a = agg(sess, lambda s, scope=scope, key=key: s[scope][key])
                rows.append(dict(env=env, scope=scope, group="all", horizon=key.split("_")[-1] if "_" in key else "-", metric=key.split("_")[0],
                                 mean=a["mean"], ci_lo=a["ci_lo"], ci_hi=a["ci_hi"], sessions=a["n"], mean_fills=float(np.mean([s[scope]["n_fills"] for s in sess]))))
        for h in HORIZONS_NS:
            a = agg(sess, lambda s, h=h: s["null"][f"M_{h}"])
            rows.append(dict(env=env, scope="null_random_times", group="all", horizon=h, metric="M", mean=a["mean"], ci_lo=a["ci_lo"], ci_hi=a["ci_hi"], sessions=a["n"], mean_fills=np.nan))
        keys = sorted({k_ for s in sess for k_ in s["groups"]})
        for (scope, g, h) in keys:
            for metric in ("E", "M", "R"):
                a = agg(sess, lambda s, key=(scope, g, h), metric=metric: (s["groups"].get(key) or {}).get(metric) if (s["groups"].get(key) or {}).get("n", 0) > 0 else None)
                rows.append(dict(env=env, scope=f"mm_by_{scope}", group=g, horizon=h, metric=metric, mean=a["mean"], ci_lo=a["ci_lo"], ci_hi=a["ci_hi"], sessions=a["n"],
                                 mean_fills=float(np.mean([(s["groups"].get((scope, g, h)) or {}).get("n", 0) for s in sess]))))
        m = {r["horizon"]: r for r in rows if r["env"] == env and r["scope"] == "mm_profile" and r["metric"] == "M"}
        e = next(r for r in rows if r["env"] == env and r["scope"] == "mm_profile" and r["metric"] == "E")
        print(f"   MM fills: E={e['mean']:.2f}  " + "  ".join(f"M_{h}={m[h]['mean']:+.2f}[{m[h]['ci_lo']:+.2f},{m[h]['ci_hi']:+.2f}]" for h in HORIZONS_NS))
        nl = {r["horizon"]: r["mean"] for r in rows if r["env"] == env and r["scope"] == "null_random_times"}
        print("   null (random times): " + "  ".join(f"{h}={v:+.3f}" for h, v in nl.items()))
        for scope in ("taker", "inventory", "obi"):
            for r in rows:
                if r["env"] == env and r["scope"] == f"mm_by_{scope}" and r["horizon"] == "500ms" and r["metric"] == "M":
                    print(f"   {scope:9s} {r['group']:32s} M_500ms={r['mean']:+.2f} [{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]  fills/session={r['mean_fills']:.0f}")
    save_csv(ctx.out_dir / "post_fill_markouts.csv", rows)
    save_json(ctx.out_dir / "summary.json", {"k": ks, "gamma": GAMMA})
    _plot(ctx, rows)
    write_provenance(ctx, {"sessions": len(seeds), "horizon_s": horizon, "gamma": GAMMA, "k": ks, "horizons": HORIZONS_NS,
                           "envs": {k: v for k, v in ENVS.items()}, "calibration_seeds": calib},
                     files=["post_fill_markouts.csv", "summary.json", "adverse_selection.png"], elapsed_s=time.time() - t0)


def _plot(ctx: RunContext, rows: list[dict]) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    colors = {"noise": "tab:blue", "informed": "tab:red"}
    hs = list(HORIZONS_NS)
    xs = np.arange(len(hs))
    def eb(a, xx, rr, **kw):
        y = np.array([r["mean"] for r in rr])
        a.errorbar(xx, y, [y - [r["ci_lo"] for r in rr], [r["ci_hi"] for r in rr] - y], capsize=2, **kw)
    for env, c in colors.items():
        for j, (m, ttl) in enumerate((("M", "Signed post-fill mid move M_h (ticks; <0 = adverse)"), ("R", "Realised half-spread R_h = E + M_h"))):
            rr = [next(r for r in rows if r["env"] == env and r["scope"] == "mm_profile" and r["metric"] == m and r["horizon"] == h) for h in hs]
            eb(ax[0, j], xs, rr, fmt="o-", color=c, label=f"{env}: MM fills")
            if m == "M":
                rr = [next(r for r in rows if r["env"] == env and r["scope"] == "all_profile" and r["metric"] == "M" and r["horizon"] == h) for h in hs]
                eb(ax[0, j], xs + 0.1, rr, fmt="s--", color=c, alpha=0.6, label=f"{env}: all passive fills")
                rr = [next(r for r in rows if r["env"] == env and r["scope"] == "null_random_times" and r["horizon"] == h) for h in hs]
                eb(ax[0, j], xs - 0.1, rr, fmt="^:", color=c, alpha=0.6, label=f"{env}: null (random times)")
            ax[0, j].set(xticks=xs, xticklabels=hs, title=ttl, ylabel="ticks per lot"); ax[0, j].axhline(0, color="k", lw=0.5)
        e = next(r for r in rows if r["env"] == env and r["scope"] == "mm_profile" and r["metric"] == "E")
        ax[0, 2].bar(0 if env == "noise" else 1, e["mean"], color=c, yerr=[[e["mean"] - e["ci_lo"]], [e["ci_hi"] - e["mean"]]], capsize=4)
    ax[0, 2].set(xticks=[0, 1], xticklabels=["noise", "informed"], title="Effective half-spread E (spread captured)", ylabel="ticks")
    for a in ax[0, :2]:
        a.legend(fontsize=7)
    for i, (scope, ttl) in enumerate((("taker", "By taker type (500 ms)"), ("inventory", "By effect on inventory (500 ms)"), ("obi", "By book imbalance at fill, s*OBI (500 ms)"))):
        for env, c, off in (("noise", "tab:blue", -0.15), ("informed", "tab:red", 0.15)):
            rr = [r for r in rows if r["env"] == env and r["scope"] == f"mm_by_{scope}" and r["horizon"] == "500ms" and r["metric"] == "M" and r["sessions"] > 0]
            eb(ax[1, i], np.arange(len(rr)) + off, rr, fmt="o", color=c, label=env)
            ax[1, i].set_xticks(range(len(rr))); ax[1, i].set_xticklabels([r["group"].replace("(", "\n(") for r in rr], fontsize=7)
        ax[1, i].set(title=ttl, ylabel="M_500ms (ticks)"); ax[1, i].axhline(0, color="k", lw=0.5); ax[1, i].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ctx.out_dir / "adverse_selection.png", dpi=140); plt.close(fig)
