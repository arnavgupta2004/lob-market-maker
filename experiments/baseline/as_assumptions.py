"""Experiment: do the Avellaneda-Stoikov assumptions hold in this simulator?

Hypotheses (stated before looking at results)
    H1  Fill intensity of a passive order decays exponentially with distance from the mid,
        ``lambda(delta) = A exp(-k delta)`` (AS assumption A2).
    H2  The intensity is constant over an order's life (Poisson fills), i.e. first-half and
        second-half hazards of the dwell window are equal.
    H3  The mid-price is a random walk with constant volatility (A1): variance ratios ~ 1 and a
        flat volatility signature.
Method: passive 1-lot probes at randomised distances (see ``research.as_calibration``), censored
MLE of intensity, session-level bootstrap (resampling whole sessions) for CIs. Two market
environments: noise flow only, and noise + informed (adverse-selection) flow.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
import numpy as np

from backtest.statistics import bootstrap_ci
from experiments.common import RunContext, save_csv, save_json, write_provenance
from research.as_calibration import (
    ProbeQuoter, add_stats, fit_exponential_intensity, mid_diagnostics, rows_from_stats, session_bin_stats,
)
from simulator.market_state.fundamental import FundamentalConfig
from simulator.order_flow.informed import InformedTrader, InformedTraderConfig
from simulator.order_flow.noise import NoiseTrader, NoiseTraderConfig
from simulator.simulator import SimConfig, Simulator

DESCRIPTION = "Estimate A, k of the fill-intensity curve and test AS assumptions (Brownian mid, Poisson fills)"

DWELL_S = 0.5
DELTAS = (0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8)
DEFAULT_FUNDAMENTAL = FundamentalConfig(sigma=0.03)
ENVS = {  # the two original environments; every Stage 4-8 experiment iterates over exactly these
    "noise": dict(informed=None),
    "informed": dict(informed=InformedTraderConfig(rate=10.0)),
}
# Environments including the jump/news stress case, used only by the Stage 9 experiments that ask for it (latency, ablation, sweeps).
# Its parameters (jumps of ~5 ticks std every ~5 s on top of the diffusion) were not tuned to any result. Kept separate from ENVS so
# that earlier experiments neither change nor break when it is added.
ALL_ENVS = {**ENVS, "informed_jump": dict(informed=InformedTraderConfig(rate=10.0),
                                          fundamental=FundamentalConfig(sigma=0.03, jump_rate=0.2, jump_sigma=0.05))}


def env_fundamental(env: str) -> FundamentalConfig:
    """Latent-value process of a named environment."""
    return ALL_ENVS[env].get("fundamental", DEFAULT_FUNDAMENTAL)
LAGS = (1, 2, 5, 10, 50)


@dataclass(frozen=True)
class CalibArgs:
    env: str
    seed: int
    horizon_s: float


def probe_session(a: CalibArgs) -> dict:
    """One session: noise (+informed) flow with a passive probe. Picklable, top-level."""
    cfg = SimConfig(seed=a.seed, horizon_s=a.horizon_s, fundamental=env_fundamental(a.env), sample_interval_s=0.1)
    parts = [NoiseTrader(NoiseTraderConfig())]
    if ALL_ENVS[a.env]["informed"] is not None:
        parts.append(InformedTrader(ALL_ENVS[a.env]["informed"]))
    probe = ProbeQuoter(deltas=DELTAS, dwell_s=DWELL_S, start_s=cfg.warmup_s)
    res = Simulator(cfg, parts + [probe]).run()
    s, _ = res.steady()
    return {"stats": session_bin_stats(probe.records, DWELL_S), "mid": mid_diagnostics(s["mid"], 0.1, LAGS),
            "n_probes": len(probe.records), "mean_spread": float(np.nanmean(s["spread"]))}


def collect(env: str, seeds: list[int], horizon_s: float, workers: int) -> list[dict]:
    args = [CalibArgs(env, s, horizon_s) for s in seeds]
    if workers > 1 and len(args) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(probe_session, args))
    return [probe_session(a) for a in args]


def analyse(sessions: list[dict], n_boot: int = 400, seed: int = 0) -> dict:
    """Pool sessions, fit ``lambda(delta)``, bootstrap over sessions."""
    total: dict = {}
    for s in sessions:
        total = add_stats(total, s["stats"])
    rows = rows_from_stats(total)
    fit = fit_exponential_intensity(rows)
    rng = np.random.default_rng(seed)
    ks, As = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(sessions), len(sessions))
        agg: dict = {}
        for i in pick:
            agg = add_stats(agg, sessions[i]["stats"])
        f = fit_exponential_intensity(rows_from_stats(agg))
        ks.append(f["k"]); As.append(f["A"])
    fit["k_ci"] = [float(np.nanquantile(ks, 0.025)), float(np.nanquantile(ks, 0.975))]
    fit["A_ci"] = [float(np.nanquantile(As, 0.025)), float(np.nanquantile(As, 0.975))]
    mid = {}
    keys = sorted({k for s in sessions for k in s["mid"]})
    for k in keys:
        vals = np.array([s["mid"].get(k, np.nan) for s in sessions])
        lo, hi = bootstrap_ci(vals, n_boot=1000)
        mid[k] = {"mean": float(np.nanmean(vals)), "ci_lo": lo, "ci_hi": hi}
    return {"intensity_rows": rows, "fit": fit, "mid": mid,
            "mean_spread": float(np.mean([s["mean_spread"] for s in sessions])), "n_sessions": len(sessions)}


def calibrate(env: str, seeds: list[int], horizon_s: float = 120.0, workers: int = 1) -> dict:
    """Public helper (used by other experiments): fit ``k`` for an environment."""
    return analyse(collect(env, seeds, horizon_s, workers), n_boot=200)


def run(ctx: RunContext) -> None:
    t0 = time.time()
    seeds = ctx.seeds(default=24, quick=3)
    horizon = 40.0 if ctx.quick else 300.0
    results = {}
    for env in ENVS:
        results[env] = analyse(collect(env, seeds, horizon, ctx.workers))
        f = results[env]["fit"]
        print(f"[{env}] k={f['k']:.3f} (95% CI {f['k_ci'][0]:.3f}..{f['k_ci'][1]:.3f})  A={f['A']:.2f}/s  "
              f"R2={f['r2']:.3f}  chi2/dof={f['chi2_per_dof']:.1f}  mean spread={results[env]['mean_spread']:.2f}")
        for q in LAGS[1:]:
            v = results[env]["mid"].get(f"vr_{q}")
            if v:
                print(f"    VR({q:>2}) = {v['mean']:.3f}  [{v['ci_lo']:.3f}, {v['ci_hi']:.3f}]")
    save_json(ctx.out_dir / "results.json", results)
    save_csv(ctx.out_dir / "intensity.csv", [dict(env=e, **r) for e, res in results.items() for r in res["intensity_rows"]])
    _plot(ctx, results)
    write_provenance(ctx, {"sessions": len(seeds), "horizon_s": horizon, "dwell_s": DWELL_S, "deltas": DELTAS,
                           "envs": {k: v for k, v in ENVS.items()}, "noise": NoiseTraderConfig(),
                           "fundamental": FundamentalConfig(sigma=0.03), "lags": LAGS},
                     files=["results.json", "intensity.csv", "as_assumptions.png"], elapsed_s=time.time() - t0)


def _plot(ctx: RunContext, results: dict) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    colors = {"noise": "tab:blue", "informed": "tab:red"}
    for env, res in results.items():
        c = colors[env]
        rows = [r for r in res["intensity_rows"] if r["fills"] > 0]
        d = np.array([r["delta"] for r in rows]); lam = np.array([r["lam"] for r in rows])
        se = np.array([r["lam_se"] for r in rows])
        ax[0, 0].errorbar(d, lam, 1.96 * se, fmt="o", color=c, ms=4, capsize=2, label=f"{env}: data")
        f = res["fit"]
        xx = np.linspace(d.min(), d.max(), 50)
        ax[0, 0].plot(xx, f["A"] * np.exp(-f["k"] * xx), "-", color=c, lw=1,
                      label=f"fit k={f['k']:.2f} [{f['k_ci'][0]:.2f},{f['k_ci'][1]:.2f}], R²={f['r2']:.2f}")
        h1 = np.array([r["lam_first_half"] for r in rows]); h2 = np.array([r["lam_second_half"] for r in rows])
        with np.errstate(divide="ignore", invalid="ignore"):
            ax[0, 1].plot(d, h2 / h1, "o-", color=c, ms=4, label=env)
        lags = [q for q in LAGS if f"vr_{q}" in res["mid"]]
        m = [res["mid"][f"vr_{q}"] for q in lags]
        ax[1, 0].errorbar(np.array(lags) * 0.1, [x["mean"] for x in m],
                          [[x["mean"] - x["ci_lo"] for x in m], [x["ci_hi"] - x["mean"] for x in m]],
                          fmt="o-", color=c, capsize=3, label=env)
        sig = [(q * 0.1, res["mid"][f"sigma_{q * 0.1:g}s"]) for q in LAGS if f"sigma_{q * 0.1:g}s" in res["mid"]]
        ax[1, 1].errorbar([s[0] for s in sig], [s[1]["mean"] for s in sig],
                          [[s[1]["mean"] - s[1]["ci_lo"] for s in sig], [s[1]["ci_hi"] - s[1]["mean"] for s in sig]],
                          fmt="o-", color=c, capsize=3, label=env)
    ax[0, 0].set(yscale="log", xlabel="distance from mid δ (ticks)", ylabel="fill intensity λ(δ) (1/s)",
                 title="H1: fill intensity vs distance (censored-MLE, 95% CI)"); ax[0, 0].legend(fontsize=7)
    ax[0, 1].axhline(1, color="k", lw=0.8, ls="--")
    ax[0, 1].set(xlabel="δ (ticks)", ylabel="hazard 2nd half / 1st half of dwell", title="H2: constant hazard? (1 = Poisson)")
    ax[0, 1].legend()
    ax[1, 0].axhline(1, color="k", lw=0.8, ls="--")
    ax[1, 0].set(xscale="log", xlabel="horizon (s)", ylabel="variance ratio VR", title="H3: mid variance ratio (1 = random walk)")
    ax[1, 0].legend()
    ax[1, 1].set(xscale="log", xlabel="sampling interval (s)", ylabel="σ̂ (ticks/√s)", title="H3: volatility signature (flat = Brownian)")
    ax[1, 1].legend()
    for a in (ax[1, 0], ax[1, 1]):  # readable log axes
        a.set_xticks([0.1, 0.2, 0.5, 1, 2, 5])
        a.xaxis.set_major_formatter(ScalarFormatter())
        a.xaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    fig.savefig(ctx.out_dir / "as_assumptions.png", dpi=140)
    plt.close(fig)
