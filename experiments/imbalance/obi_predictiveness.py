"""Experiment A: does order-book imbalance predict short-horizon mid-price movement?

Hypotheses (stated before running)
    H1  OBI is positively related to the forward mid change at short horizons.
    H2  The relation decays with horizon (imbalance is transient information).
    H3  Part of the association is *not* prediction: imbalance is also produced by recent price moves, so the
        backward correlation (OBI_t vs m_t - m_{t-h}) is non-zero. We report it beside the forward one.
    H4  The predicted move is small relative to the half-spread, i.e. usable to *skew quotes*, not to cross the spread.
Method: fixed 10 ms sampling of the true book; per-session correlation/slope/directional accuracy at 6 horizons
(10 ms ... 5 s) for OBI over the touch (L1) and 5 levels (L5), also restricted to spread = 1 tick; shuffle null;
session-level bootstrap CIs (observations inside a session overlap, so samples are NOT the unit of inference).
Two environments: noise flow only, and noise + informed flow.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backtest.statistics import summarize
from experiments.baseline.as_assumptions import ENVS
from experiments.common import RunContext, pmap, save_csv, save_json, write_provenance
from research.imbalance import OBI_EDGES, session_stats
from simulator.market_state.fundamental import FundamentalConfig
from simulator.order_flow.noise import NoiseTrader
from simulator.order_flow.informed import InformedTrader
from simulator.simulator import SimConfig, Simulator

DESCRIPTION = "Does order-book imbalance predict short-horizon price moves? (Experiment A)"

HORIZONS = (0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
DT = 0.01
STATS = ("corr", "spearman", "slope", "dir_acc_excess", "corr_backward", "corr_shuffled", "frac_moved")
VARIANTS = {"L1": dict(levels=1, only_spread=None), "L5": dict(levels=5, only_spread=None),
            "L1_spread1": dict(levels=1, only_spread=1)}


@dataclass(frozen=True)
class Job:
    env: str
    seed: int
    horizon_s: float


def session(job: Job) -> dict:
    cfg = SimConfig(seed=job.seed, horizon_s=job.horizon_s, sample_interval_s=DT, fundamental=FundamentalConfig(sigma=0.03),
                    record_events=False, record_commands=False)
    parts = [NoiseTrader()] + ([InformedTrader(ENVS[job.env]["informed"])] if ENVS[job.env]["informed"] else [])
    res = Simulator(cfg, parts).run()
    rng = np.random.default_rng(job.seed)
    out = {v: session_stats(res.samples, DT, HORIZONS, rng=rng, **kw) for v, kw in VARIANTS.items()}
    s, _ = res.steady()
    out["mean_spread"] = float(np.nanmean(s["spread"]))
    return out


def run(ctx: RunContext) -> None:
    t0 = time.time()
    seeds = ctx.seeds(default=24, quick=3)
    horizon = 40.0 if ctx.quick else 300.0
    rows, bins = [], []
    spread = {}
    for env in ENVS:
        sess = pmap(session, [Job(env, s, horizon) for s in seeds], ctx.workers)
        spread[env] = float(np.mean([s["mean_spread"] for s in sess]))
        for v in VARIANTS:
            for h in HORIZONS:
                rec = dict(env=env, variant=v, horizon_s=h)
                for st in STATS:
                    s = summarize([x[v][h][st] for x in sess], seed=1)
                    rec.update({st: s["mean"], f"{st}_lo": s["ci_lo"], f"{st}_hi": s["ci_hi"]})
                rec["n_samples_per_session"] = float(np.mean([x[v][h]["n"] for x in sess]))
                rows.append(rec)
                if v == "L1":
                    for i in range(len(OBI_EDGES) - 1):
                        vals = [x[v][h]["bin_mean"][i] for x in sess]
                        s = summarize(vals, seed=2)
                        bins.append(dict(env=env, horizon_s=h, obi_lo=OBI_EDGES[i], obi_hi=OBI_EDGES[i + 1], mean_dm_ticks=s["mean"],
                                         ci_lo=s["ci_lo"], ci_hi=s["ci_hi"], sessions=s["n"],
                                         mean_n=float(np.mean([x[v][h]["bin_n"][i] for x in sess]))))
        r1 = next(r for r in rows if r["env"] == env and r["variant"] == "L1" and r["horizon_s"] == 0.1)
        rb = next(r for r in rows if r["env"] == env and r["variant"] == "L1" and r["horizon_s"] == 1.0)
        print(f"[{env}] L1 h=100ms: corr={r1['corr']:.3f} [{r1['corr_lo']:.3f},{r1['corr_hi']:.3f}] slope={r1['slope']:.3f} tick/OBI "
              f"back-corr={r1['corr_backward']:.3f} shuffled={r1['corr_shuffled']:.3f} | h=1s corr={rb['corr']:.3f} | mean spread {spread[env]:.2f}")
    save_csv(ctx.out_dir / "summary.csv", rows)
    save_csv(ctx.out_dir / "obi_bins.csv", bins)
    save_json(ctx.out_dir / "summary.json", {"mean_spread_ticks": spread, "horizons_s": HORIZONS})
    _plot(ctx, rows, bins)
    write_provenance(ctx, {"sessions": len(seeds), "horizon_s": horizon, "sample_dt_s": DT, "horizons_s": HORIZONS,
                           "variants": VARIANTS, "envs": {k: v for k, v in ENVS.items()}},
                     files=["summary.csv", "obi_bins.csv", "summary.json", "imbalance.png"], elapsed_s=time.time() - t0)


def _plot(ctx: RunContext, rows: list[dict], bins: list[dict]) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    colors = {"noise": "tab:blue", "informed": "tab:red"}
    for env, c in colors.items():
        def series(v, key):
            rr = [r for r in rows if r["env"] == env and r["variant"] == v]
            return ([r["horizon_s"] for r in rr], [r[key] for r in rr], [r[f"{key}_lo"] for r in rr], [r[f"{key}_hi"] for r in rr])
        h, y, lo, hi = series("L1", "corr")
        ax[0, 0].errorbar(h, y, [np.array(y) - lo, np.array(hi) - y], fmt="o-", color=c, capsize=2, label=f"{env}: forward")
        h, y, lo, hi = series("L1", "corr_backward")
        ax[0, 0].errorbar(h, y, [np.array(y) - lo, np.array(hi) - y], fmt="s--", color=c, capsize=2, label=f"{env}: backward")
        h, y, _, _ = series("L1", "corr_shuffled")
        ax[0, 0].plot(h, y, ":", color=c, label=f"{env}: shuffled null")
        for v, ls in (("L1", "o-"), ("L5", "s--"), ("L1_spread1", "^:")):
            h, y, lo, hi = series(v, "corr")
            ax[0, 1].errorbar(h, y, [np.array(y) - lo, np.array(hi) - y], fmt=ls, color=c, capsize=2, label=f"{env} {v}")
        h, y, lo, hi = series("L1", "slope")
        ax[0, 2].errorbar(h, y, [np.array(y) - lo, np.array(hi) - y], fmt="o-", color=c, capsize=2, label=env)
        h, y, lo, hi = series("L1", "dir_acc_excess")
        ax[1, 0].errorbar(h, y, [np.array(y) - lo, np.array(hi) - y], fmt="o-", color=c, capsize=2, label=env)
        for hh, ls in ((0.1, "o-"), (1.0, "s--")):
            bb = [b for b in bins if b["env"] == env and b["horizon_s"] == hh]
            mid = [0.5 * (b["obi_lo"] + b["obi_hi"]) for b in bb]
            y = np.array([b["mean_dm_ticks"] for b in bb])
            ax[1, 1 if hh == 0.1 else 2].errorbar(mid, y, [y - [b["ci_lo"] for b in bb], [b["ci_hi"] for b in bb] - y],
                                                   fmt=ls, color=c, capsize=2, label=env)
    for a in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0]):
        a.set(xscale="log", xlabel="horizon (s)")
    ax[0, 0].set(ylabel="correlation", title="L1 OBI vs mid change: forward / backward / null")
    ax[0, 1].set(ylabel="forward correlation", title="Depth (L1 vs L5) and spread-1 subset")
    ax[0, 2].set(ylabel="ticks per unit OBI", title="OLS slope of forward mid change on OBI")
    ax[1, 0].set(ylabel="P(sign match) - 1/2", title="Directional accuracy (nonzero moves)")
    ax[1, 1].set(xlabel="OBI bin centre", ylabel="mean forward mid change (ticks)", title="Conditional mean, h = 100 ms")
    ax[1, 2].set(xlabel="OBI bin centre", ylabel="mean forward mid change (ticks)", title="Conditional mean, h = 1 s")
    for a in ax.flat:
        a.axhline(0, color="k", lw=0.5); a.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(ctx.out_dir / "imbalance.png", dpi=140); plt.close(fig)
