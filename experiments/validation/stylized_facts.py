"""Experiment G: how closely does the simulator reproduce the stylized facts of limit-order markets?

Four flow variants, adding one mechanism at a time (each addresses a gap found earlier):

  noise           Poisson noise traders only (Stage 3)
  informed        + informed fundamental trader (adverse selection, price discovery)
  hawkes          + Hawkes (self-exciting) arrivals for the noise flow   -> arrival clustering
  hawkes+meta     + order-splitting metaorder trader, Pareto(1.5) sizes  -> persistent order signs

Statistics per session (600 s, post warm-up), then means with session-bootstrap CIs: return kurtosis and Hill tail index,
volatility clustering (ACF of |r|), return autocorrelation, order-sign ACF and its power-law exponent, Fano factor and
inter-arrival CV of aggressive orders, spread distribution, trade-size tail.

Qualitative checks - the *widely reported* stylized facts (e.g. Cont 2001; Bouchaud et al.): positive excess kurtosis of
short-horizon returns, positive and slowly decaying autocorrelation of |r|, persistent order signs with a power-law ACF
exponent between 0 and 1 (metaorder theory predicts alpha - 1 = 0.5 here), clustered order arrivals (Fano > 1). These are
sign/shape checks, not a quantitative match to any asset; the quantitative comparison needs real data (not available in this
run - no dataset has been approved for download), and nothing here should be read as a calibration.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backtest.statistics import summarize
from experiments.common import RunContext, pmap, save_csv, save_json, write_provenance
from research.stylized_facts import acf, fano_factor, hill_tail_index, log_returns, mid_on_grid, moments, power_law_exponent, spread_stats
from simulator.market_state.fundamental import FundamentalConfig
from simulator.order_flow.arrivals import NS, HawkesArrival
from simulator.order_flow.informed import InformedTrader
from simulator.order_flow.metaorder import MetaOrderConfig, MetaOrderTrader
from simulator.order_flow.noise import NoiseTrader
from simulator.simulator import SimConfig, Simulator

DESCRIPTION = "Stylized facts of the simulator across four flow variants (Experiment G)"

VARIANTS = ("noise", "informed", "hawkes", "hawkes+meta")
FANO_WINDOWS = (0.1, 0.3, 1.0, 3.0, 10.0)
ACF_LAGS = 50
SIGN_LAGS = 100
META = MetaOrderConfig()  # Pareto(xm=10, alpha=1.5) parents, child ~ LogNormal(median 3)


def participants(variant: str):
    hk = HawkesArrival.with_mean_rate(65.0, branching_ratio=0.6, beta=8.0)
    parts = [NoiseTrader(arrivals=hk) if variant.startswith("hawkes") else NoiseTrader()]
    if variant != "noise":
        parts.append(InformedTrader())
    if variant == "hawkes+meta":
        parts.append(MetaOrderTrader(META))
    return parts


@dataclass(frozen=True)
class Job:
    variant: str
    seed: int
    horizon_s: float


def session(job: Job) -> dict:
    cfg = SimConfig(seed=job.seed, horizon_s=job.horizon_s, fundamental=FundamentalConfig(sigma=0.03), record_events=False,
                    record_commands=False)
    res = Simulator(cfg, participants(job.variant)).run()
    w = cfg.warmup_s
    out: dict = {}
    for h in (0.1, 1.0):
        r = log_returns(mid_on_grid(res, h, w))
        mo = moments(r)
        out[f"kurt_{h:g}s"] = mo["excess_kurtosis"]
        out[f"zero_share_{h:g}s"] = float(np.mean(r == 0))
        out[f"hill5_{h:g}s"] = hill_tail_index(r, frac=0.05)["alpha"]
        out[f"hill1_{h:g}s"] = hill_tail_index(r, frac=0.01)["alpha"]
        a_abs, a_ret = acf(np.abs(r), ACF_LAGS), acf(r, 5)
        out[f"acf_ret1_{h:g}s"] = float(a_ret[0])
        out[f"acf_abs1_{h:g}s"] = float(a_abs[0]); out[f"acf_abs10_{h:g}s"] = float(a_abs[9]); out[f"acf_abs50_{h:g}s"] = float(a_abs[-1])
        if h == 1.0:
            out["_acf_abs_curve"] = a_abs.tolist()
            z = np.abs(r - r.mean()) / (r.std() or 1.0)
            out["_ccdf_q"] = np.quantile(z, [0.5, 0.9, 0.99, 0.999]).tolist()
    tp = res.trades
    m = tp["t_ns"] >= int(w * NS)
    first = np.r_[True, tp["taker_id"][m][1:] != tp["taker_id"][m][:-1]]
    sg = tp["aggressor"][m][first].astype(float)
    arr = tp["t_ns"][m][first]
    sacf = acf(sg, SIGN_LAGS)
    fit = power_law_exponent(sacf, lo=1, hi=SIGN_LAGS)
    out.update(sign_acf1=float(sacf[0]), sign_acf10=float(sacf[9]), sign_gamma=fit["gamma"], sign_r2=fit["r2"], _sign_curve=sacf.tolist())
    end = int(cfg.horizon_s * NS)
    for win in FANO_WINDOWS:
        out[f"fano_{win:g}s"] = fano_factor(arr, win, int(w * NS), end)
    gaps = np.diff(arr.astype(float)); gaps = gaps[gaps > 0]
    out["arrival_cv"] = float(gaps.std() / gaps.mean())
    sp = spread_stats(res.steady()[0]["spread"])
    out.update(spread_mean=sp["mean"], spread_median=sp["median"], spread_q99=sp["q99"])
    qty = tp["qty"][m][first]
    out["size_hill5"] = hill_tail_index(qty, frac=0.05)["alpha"]
    out["n_aggressive"] = float(len(arr))
    return out


CHECKS = (  # (label, metric, predicate on (lo, mean, hi), description)
    ("heavy tails: excess kurtosis (1 s returns) > 0", "kurt_1s", lambda lo, m, hi: lo > 0),
    ("vol clustering: ACF(|r|, lag 1 @1 s) > 0", "acf_abs1_1s", lambda lo, m, hi: lo > 0),
    ("vol clustering persists: ACF(|r|, lag 10 @1 s) > 0", "acf_abs10_1s", lambda lo, m, hi: lo > 0),
    ("persistent order signs: ACF(sign, lag 1) > 0", "sign_acf1", lambda lo, m, hi: lo > 0),
    ("long memory: sign-ACF exponent in (0.1, 1)", "sign_gamma", lambda lo, m, hi: lo > 0.1 and hi < 1.0),
    ("arrival clustering: Fano(1 s) > 1.1", "fano_1s", lambda lo, m, hi: lo > 1.1),
    ("spread varies: 99th pct > median", "_spread_var", lambda lo, m, hi: lo > 0),
)


def run(ctx: RunContext) -> None:
    t0 = time.time()
    seeds = ctx.seeds(default=16, quick=3)
    horizon = 60.0 if ctx.quick else 600.0
    rows, curves = [], {}
    sessions = {}
    for v in VARIANTS:
        sess = pmap(session, [Job(v, s, horizon) for s in seeds], ctx.workers)
        for s_ in sess:
            s_["_spread_var"] = s_["spread_q99"] - s_["spread_median"]
        sessions[v] = sess
        keys = sorted(k for k in sess[0] if not k.startswith("_"))
        for k in keys:
            s = summarize([x[k] for x in sess], seed=3)
            rows.append(dict(variant=v, metric=k, mean=s["mean"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"], std=s["std"], sessions=s["n"]))
        s = summarize([x["_spread_var"] for x in sess], seed=3)
        rows.append(dict(variant=v, metric="spread_q99_minus_median", mean=s["mean"], ci_lo=s["ci_lo"], ci_hi=s["ci_hi"], std=s["std"], sessions=s["n"]))
        curves[v] = {"abs_acf": np.nanmean([x["_acf_abs_curve"] for x in sess], 0).tolist(), "sign_acf": np.nanmean([x["_sign_curve"] for x in sess], 0).tolist(),
                     "ccdf_q": np.nanmean([x["_ccdf_q"] for x in sess], 0).tolist()}
    save_csv(ctx.out_dir / "stylized_facts.csv", rows)
    save_json(ctx.out_dir / "curves.json", curves)
    # ---- qualitative scoreboard
    board = []
    for label, metric, pred in CHECKS:
        rec = {"check": label}
        for v in VARIANTS:
            mkey = "spread_q99_minus_median" if metric == "_spread_var" else metric
            r = next(x for x in rows if x["variant"] == v and x["metric"] == mkey)
            rec[v] = bool(pred(r["ci_lo"], r["mean"], r["ci_hi"]))
            rec[f"{v}_mean"] = r["mean"]
        board.append(rec)
    save_csv(ctx.out_dir / "scoreboard.csv", board)
    print(f"{'qualitative check':58s} " + " ".join(f"{v:>12s}" for v in VARIANTS))
    for b in board:
        print(f"{b['check']:58s} " + " ".join(f"{('PASS' if b[v] else 'fail'):>5s} {b[v + '_mean']:+6.2f}" for v in VARIANTS))
    _plot(ctx, rows, curves)
    write_provenance(ctx, {"sessions": len(seeds), "horizon_s": horizon, "variants": VARIANTS, "hawkes": "mean rate 65/s, branching 0.6, beta 8/s",
                           "metaorder": META, "fano_windows_s": FANO_WINDOWS, "checks": [c[0] for c in CHECKS]},
                     files=["stylized_facts.csv", "scoreboard.csv", "curves.json", "stylized_facts.png"], elapsed_s=time.time() - t0)


def _plot(ctx: RunContext, rows: list[dict], curves: dict) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    colors = dict(zip(VARIANTS, ("tab:gray", "tab:blue", "tab:green", "tab:red")))
    for v in VARIANTS:
        c = colors[v]
        ax[0, 0].plot(range(1, ACF_LAGS + 1), curves[v]["abs_acf"], color=c, label=v)
        s = np.array(curves[v]["sign_acf"]); lag = np.arange(1, len(s) + 1)
        ax[0, 1].loglog(lag[s > 0], s[s > 0], "o-", ms=2, color=c, label=v)
        ax[0, 2].plot(FANO_WINDOWS, [next(r["mean"] for r in rows if r["variant"] == v and r["metric"] == f"fano_{w:g}s") for w in FANO_WINDOWS], "o-", color=c, label=v)
        ax[1, 0].plot([0.5, 0.9, 0.99, 0.999], curves[v]["ccdf_q"], "o-", color=c, label=v)
    lag = np.arange(1, SIGN_LAGS + 1)
    ax[0, 1].loglog(lag, 0.35 * lag ** -0.5, "k--", lw=1, label="slope -(alpha-1) = -0.5")
    ax[0, 0].set(title="ACF of |1 s return| (vol clustering)", xlabel="lag (s)", ylabel="autocorrelation"); ax[0, 0].axhline(0, color="k", lw=0.5)
    ax[0, 1].set(title="ACF of aggressor signs (log-log)", xlabel="lag (orders)", ylabel="autocorrelation")
    ax[0, 2].set(xscale="log", title="Fano factor of aggressive-order counts", xlabel="window (s)", ylabel="var / mean (Poisson = 1)"); ax[0, 2].axhline(1, color="k", lw=0.5)
    ax[1, 0].set(yscale="log", title="1 s return tail: quantiles of |z|", xlabel="probability level", ylabel="|z| quantile (Gaussian: 0.67, 1.64, 2.58, 3.29)")
    metrics = [("kurt_1s", "excess kurtosis (1 s)"), ("hill5_1s", "Hill index (top 5%, 1 s)")]
    for a, (m, ttl) in zip(ax[1, 1:], metrics):
        for i, v in enumerate(VARIANTS):
            r = next(x for x in rows if x["variant"] == v and x["metric"] == m)
            a.bar(i, r["mean"], color=colors[v], yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]], capsize=4)
        a.set(xticks=range(len(VARIANTS)), xticklabels=VARIANTS, title=ttl)
    for a in (ax[0, 0], ax[0, 1], ax[0, 2], ax[1, 0]):
        a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ctx.out_dir / "stylized_facts.png", dpi=140); plt.close(fig)
