"""Experiment B: how does queue position affect fill probability and execution quality?

Hypotheses (stated before running)
    H1  P(fill within dt) falls monotonically as the quantity ahead Q grows, for every dt.
    H2  Market state adds predictive power beyond Q: spread, same-side depth, volatility, order-flow intensity and
        imbalance have non-zero coefficients, and a model using them beats a Q-only model out of sample (AUC).
    H3  Execution quality depends on queue position: fills obtained from deep in the queue (large Q at posting) are
        more adverse (worse post-fill mid drift) than fills from near the front, because a deeper queue is only
        reached when more volume trades through.
Method: passive 1-lot probes at the touch and 1-2 ticks behind (see ``research.queue_position``); Q read exactly from
the book at 250 ms checkpoints; windows dt in {0.1, 0.5, 1, 2} s; only fully observed windows are used (no censoring
bias). Uncertainty by bootstrapping whole sessions. The logistic model is fit on even sessions and evaluated on odd ones
(out-of-sample AUC). Environments: noise flow, and noise + informed flow.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from experiments.baseline.as_assumptions import ENVS
from experiments.common import RunContext, pmap, save_csv, save_json, write_provenance
from research.adverse_selection import HORIZONS_NS, markouts, passive_fills, wmean
from research.queue_position import (
    FEATURES, Q_EDGES, QueueProbe, add_stats, feature_matrix, fit_logit, outcomes, predict_logit, q_bin_label, q_bin_stats,
)
from simulator.market_state.fundamental import FundamentalConfig
from simulator.order_flow.informed import InformedTrader
from simulator.order_flow.noise import NoiseTrader
from simulator.simulator import SimConfig, Simulator

DESCRIPTION = "Fill probability vs queue position and market state; execution quality by queue depth (Experiment B)"

DELTAS = (0.1, 0.5, 1.0, 2.0)
DWELL = 3.0
Q0_BINS = ((0, 0), (1, 5), (6, 15), (16, 10**9))
MODELS = {"Q_only": [0], "Q+d_touch": [0, 1], "Q+d_touch+spread": [0, 1, 2], "full": list(range(len(FEATURES)))}
D_BINS = (("0", lambda d: d == 0), ("1", lambda d: d == 1), ("2+", lambda d: d >= 2))


@dataclass(frozen=True)
class Job:
    env: str
    seed: int
    horizon_s: float


def session(job: Job) -> dict:
    cfg = SimConfig(seed=job.seed, horizon_s=job.horizon_s, fundamental=FundamentalConfig(sigma=0.03), record_events=False,
                    record_commands=False)
    probe = QueueProbe(dwell_s=DWELL)
    parts = [NoiseTrader()] + ([InformedTrader(ENVS[job.env]["informed"])] if ENVS[job.env]["informed"] else []) + [probe]
    res = Simulator(cfg, parts).run()
    d, tab = probe.dataset(), probe.probe_table()
    # execution quality of probe fills by initial queue depth q0
    f = passive_fills(res, probe.owner_id)
    q0 = dict(zip(tab["oid"].tolist(), tab["q0"].tolist()))
    k0 = dict(zip(tab["oid"].tolist(), tab["k"].tolist()))
    f_q0 = np.array([q0[i] for i in f.maker_id.tolist()])
    f_k = np.array([k0[i] for i in f.maker_id.tolist()])
    mo = markouts(res, f, {"500ms": HORIZONS_NS["500ms"]})
    quality = {}
    for k in (0, 1, 2):
        for lo, hi in Q0_BINS:
            m = (f_q0 >= lo) & (f_q0 <= hi) & (f_k == k)
            quality[(k, lo, hi)] = {"n": int(m.sum()), "E": wmean(mo["E"][m], f.qty[m]), "M": wmean(mo["M_500ms"][m], f.qty[m]),
                                    "R": wmean(mo["R_500ms"][m], f.qty[m])}
    return {"d": d, "tab": tab, "quality": quality, "n_probes": len(tab["oid"]), "fill_frac": float(tab["filled"].mean())}


def boot_bin_p(sess_stats: list[dict], rng: np.random.Generator, n_boot: int = 400):
    """Per-bin fill probability with session-bootstrap CI. ``sess_stats``: list of ``{bin: [n, fills]}``."""
    nb = len(Q_EDGES) - 1
    S = np.zeros((len(sess_stats), nb, 2))
    for i, st in enumerate(sess_stats):
        for b, v in st.items():
            S[i, b] = v
    tot = S.sum(0)
    p = np.where(tot[:, 0] > 0, tot[:, 1] / np.maximum(tot[:, 0], 1), np.nan)
    idx = rng.integers(0, len(S), (n_boot, len(S)))
    bs = S[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pb = bs[..., 1] / bs[..., 0]
    return p, np.nanquantile(pb, 0.025, axis=0), np.nanquantile(pb, 0.975, axis=0), tot[:, 0]


def auc(y: np.ndarray, p: np.ndarray) -> float:
    r = rankdata(p)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def run(ctx: RunContext) -> None:
    t0 = time.time()
    seeds = ctx.seeds(default=24, quick=3)
    horizon = 40.0 if ctx.quick else 300.0
    rng = np.random.default_rng(ctx.seed)
    bin_rows, logit_rows, auc_rows, quality_rows = [], [], [], []
    for env in ENVS:
        sess = pmap(session, [Job(env, s, horizon) for s in seeds], ctx.workers)
        print(f"[{env}] probes/session={np.mean([s['n_probes'] for s in sess]):.0f}  fill fraction={np.mean([s['fill_frac'] for s in sess]):.3f}")
        # ---- empirical P(fill | Q bin), by dt (all depths) and by depth-from-touch k at dt=1s
        for dt in DELTAS:
            per = []
            for s in sess:
                y, v = outcomes(s["d"], dt, DWELL)
                per.append(q_bin_stats(s["d"]["Q"][v], y[v]))
            p, lo, hi, n = boot_bin_p(per, rng)
            for b in range(len(p)):
                bin_rows.append(dict(env=env, dt_s=dt, k="all", q_bin=q_bin_label(b), n_rows=int(n[b]), p_fill=p[b], ci_lo=lo[b], ci_hi=hi[b]))
        for dlab, dfun in D_BINS:  # stratify by CURRENT distance from the touch
            for dt in (0.5, 1.0):
                per = []
                for s in sess:
                    y, v = outcomes(s["d"], dt, DWELL)
                    m = v & dfun(s["d"]["d_touch"])
                    per.append(q_bin_stats(s["d"]["Q"][m], y[m]))
                p, lo, hi, n = boot_bin_p(per, rng)
                for b in range(len(p)):
                    bin_rows.append(dict(env=env, dt_s=dt, k="d" + dlab, q_bin=q_bin_label(b), n_rows=int(n[b]), p_fill=p[b], ci_lo=lo[b], ci_hi=hi[b]))
        # ---- logistic model at dt = 1 s: per-SD coefficients (bootstrap CI over sessions), out-of-sample AUC
        Xs, ys = [], []
        for s in sess:
            y, v = outcomes(s["d"], 1.0, DWELL)
            Xs.append(feature_matrix(s["d"])[v]); ys.append(y[v])
        full = fit_logit(np.vstack(Xs), np.concatenate(ys))
        boots = []
        for _ in range(0 if ctx.quick else 40):
            pick = rng.integers(0, len(sess), len(sess))
            boots.append(fit_logit(np.vstack([Xs[i] for i in pick]), np.concatenate([ys[i] for i in pick]))["coef"])
        boots = np.array(boots) if boots else np.full((2, len(FEATURES)), np.nan)
        for j, name in enumerate(FEATURES):
            logit_rows.append(dict(env=env, feature=name, coef_per_sd=full["coef"][j], ci_lo=np.nanquantile(boots[:, j], 0.025),
                                   ci_hi=np.nanquantile(boots[:, j], 0.975), sd=full["std"][j], mean=full["mean"][j]))
        tr, te = list(range(0, len(sess), 2)), list(range(1, len(sess), 2))
        Xtr, ytr = np.vstack([Xs[i] for i in tr]), np.concatenate([ys[i] for i in tr])
        Xte, yte = np.vstack([Xs[i] for i in te]), np.concatenate([ys[i] for i in te])
        for name, cols in MODELS.items():
            m = fit_logit(Xtr[:, cols], ytr)
            auc_rows.append(dict(env=env, model=name, features=",".join(FEATURES[c] for c in cols), auc_test=auc(yte, predict_logit(m, Xte[:, cols])),
                                 n_test=len(yte), base_rate=float(yte.mean())))
        # ---- execution quality by initial queue depth
        for k in (0, 1, 2):
            for (lo_, hi_) in Q0_BINS:
                for metric in ("E", "M", "R"):
                    vals = np.array([s["quality"][(k, lo_, hi_)][metric] for s in sess], dtype=float)
                    vals = vals[~np.isnan(vals)]
                    b = np.random.default_rng(3).choice(vals, (2000, len(vals))).mean(1) if len(vals) > 1 else [np.nan, np.nan]
                    quality_rows.append(dict(env=env, k=k, q0_lo=lo_, q0_hi=hi_ if hi_ < 10**8 else "inf", metric=metric,
                                             mean=float(vals.mean()) if len(vals) else np.nan, ci_lo=float(np.quantile(b, .025)), ci_hi=float(np.quantile(b, .975)),
                                             sessions=len(vals), fills=int(sum(s["quality"][(k, lo_, hi_)]["n"] for s in sess))))
        for dl in ("all", "d0", "d1", "d2+"):
            pv = [r for r in bin_rows if r["env"] == env and r["dt_s"] == 1.0 and r["k"] == dl]
            print(f"   P(fill<=1s | Q bin) [{dl:>3}]: " + "  ".join(f"{r['q_bin']}:{r['p_fill']:.2f}" for r in pv))
        print("   AUC (held-out sessions): " + "  ".join(f"{r['model']}={r['auc_test']:.3f}" for r in auc_rows if r["env"] == env))
    save_csv(ctx.out_dir / "fill_prob_by_q.csv", bin_rows)
    save_csv(ctx.out_dir / "logit_coefficients.csv", logit_rows)
    save_csv(ctx.out_dir / "model_auc.csv", auc_rows)
    save_csv(ctx.out_dir / "quality_by_q0.csv", quality_rows)
    _plot(ctx, bin_rows, logit_rows, quality_rows)
    write_provenance(ctx, {"sessions": len(seeds), "horizon_s": horizon, "deltas_s": DELTAS, "dwell_s": DWELL, "q_edges": Q_EDGES,
                           "features": FEATURES, "envs": {k: v for k, v in ENVS.items()}, "probe": "QueueProbe(rate=3/s, checkpoint=0.25s, depths=(0,1,2))"},
                     files=["fill_prob_by_q.csv", "logit_coefficients.csv", "model_auc.csv", "quality_by_q0.csv", "queue.png"], elapsed_s=time.time() - t0)


def _plot(ctx: RunContext, bins: list[dict], logit: list[dict], quality: list[dict]) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    colors = {"noise": "tab:blue", "informed": "tab:red"}
    labels = [q_bin_label(i) for i in range(len(Q_EDGES) - 1)]
    x = np.arange(len(labels))
    for env, c in colors.items():
        for dt, ls in zip(DELTAS, (":", "--", "-", "-.")):
            rr = [r for r in bins if r["env"] == env and r["dt_s"] == dt and r["k"] == "all"]
            ax[0, 0 if env == "noise" else 1].errorbar(x, [r["p_fill"] for r in rr], [[r["p_fill"] - r["ci_lo"] for r in rr], [r["ci_hi"] - r["p_fill"] for r in rr]],
                                                       fmt="o" + ls, color=c, ms=4, capsize=2, alpha=0.5 + 0.15 * DELTAS.index(dt), label=f"dt={dt}s")
        for dl, ls in zip(("d0", "d1", "d2+"), ("o-", "s--", "^:")):
            rr = [r for r in bins if r["env"] == env and r["dt_s"] == 1.0 and r["k"] == dl]
            ax[0, 2].errorbar(x, [r["p_fill"] for r in rr], [[r["p_fill"] - r["ci_lo"] for r in rr], [r["ci_hi"] - r["p_fill"] for r in rr]],
                              fmt=ls, color=c, ms=4, capsize=2, label=f"{env} d_touch={dl[1:]}")
        cf = [r for r in logit if r["env"] == env]
        off = 0.2 if env == "informed" else -0.2
        y = np.arange(len(FEATURES)) + off
        ax[1, 0].errorbar([r["coef_per_sd"] for r in cf], y, xerr=[[r["coef_per_sd"] - r["ci_lo"] for r in cf], [r["ci_hi"] - r["coef_per_sd"] for r in cf]],
                          fmt="o", color=c, capsize=2, label=env)
        for mi, metric in enumerate(("M", "R")):
            qq = [r for r in quality if r["env"] == env and r["metric"] == metric and r["k"] == 0]
            ax[1, 1 + mi].errorbar(np.arange(len(qq)) + off / 2, [r["mean"] for r in qq], [[r["mean"] - r["ci_lo"] for r in qq], [r["ci_hi"] - r["mean"] for r in qq]],
                                   fmt="o-", color=c, capsize=3, label=env)
    for a, t in ((ax[0, 0], "noise: P(fill within dt | Q), ALL distances pooled\n(confounded by distance - see right panel)"), (ax[0, 1], "informed: P(fill within dt | Q), ALL distances pooled\n(confounded by distance)"), (ax[0, 2], "P(fill <= 1 s | Q) by CURRENT distance d_touch")):
        a.set(xticks=x, xticklabels=labels, xlabel="quantity ahead Q (lots)", ylabel="fill probability", title=t); a.legend(fontsize=7)
    ax[1, 0].set(yticks=np.arange(len(FEATURES)), yticklabels=FEATURES, xlabel="log-odds per +1 SD (dt = 1 s)", title="Logit coefficients (95% CI, session bootstrap)")
    ax[1, 0].axvline(0, color="k", lw=0.5); ax[1, 0].legend(fontsize=7)
    qlab = ["0", "1-5", "6-15", "16+"]
    ax[1, 1].set(xticks=range(4), xticklabels=qlab, xlabel="queue ahead at posting (lots)", ylabel="ticks", title="Signed post-fill mid move M (500 ms), posted at touch (k=0)"); ax[1, 1].axhline(0, color="k", lw=0.5)
    ax[1, 2].set(xticks=range(4), xticklabels=qlab, xlabel="queue ahead at posting (lots)", ylabel="ticks", title="Realised half-spread R = E + M (500 ms), k=0"); ax[1, 2].axhline(0, color="k", lw=0.5)
    for a in (ax[1, 1], ax[1, 2]):
        a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ctx.out_dir / "queue.png", dpi=140); plt.close(fig)
