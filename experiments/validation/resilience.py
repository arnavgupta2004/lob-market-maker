"""Book resilience: response of spread, depth and mid to a controlled aggressive shock.

Hypotheses (stated before running)
    H1  A larger shock displaces the book more (mid impact, spread excess and depth deficit grow with size).
    H2  Spread and depth recover (half-life finite, residual small) - liquidity is replenished by the noise flow.
    H3  The mid does NOT recover in the noise-only market (Stage 5 found permanent impact: noise traders re-quote around the
        moved mid), but partially reverts once informed flow pulls the price back toward the fundamental.
Method: one market order of 20 or 60 lots every 30 s (alternating buy/sell, first at t = 10 s) in 600 s sessions; per session,
mean recovery curves over 5 s pre-shock references; session-bootstrap CIs; half-life = time to halve from the first observed level.
Environments: noise only; noise + informed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.baseline.as_assumptions import ENVS
from experiments.common import RunContext, pmap, save_csv, write_provenance
from research.resilience import ShockInjector, half_life, resilience_curves
from simulator.market_state.fundamental import FundamentalConfig
from simulator.order_flow.informed import InformedTrader
from simulator.order_flow.noise import NoiseTrader
from simulator.simulator import SimConfig, Simulator

DESCRIPTION = "Book resilience after controlled aggressive shocks (spread, depth, mid)"

SIZES = (20, 60)
DT = 0.05
WINDOW = 10.0
# size 0 = the "no shock" control: same seeds and schedule of *virtual* shocks, no order sent. Effects are paired differences vs it.


@dataclass(frozen=True)
class Job:
    env: str
    seed: int
    size: int
    horizon_s: float


def session(job: Job) -> dict:
    times = list(np.arange(10.0, job.horizon_s - WINDOW - 2, 30.0))
    inj = ShockInjector(times, size=max(job.size, 1))
    cfg = SimConfig(seed=job.seed, horizon_s=job.horizon_s, sample_interval_s=DT, fundamental=FundamentalConfig(sigma=0.03), record_events=False,
                    record_commands=False)
    parts = [NoiseTrader()] + ([InformedTrader(ENVS[job.env]["informed"])] if ENVS[job.env]["informed"] else [])
    if job.size:
        parts.append(inj)
        res = Simulator(cfg, parts).run()
        shocks = inj.log
    else:  # control: identical schedule of *virtual* shocks on the undisturbed market
        res = Simulator(cfg, parts).run()
        shocks = [(int(t * 1e9), 1 if i % 2 == 0 else -1) for i, t in enumerate(times)]
    return resilience_curves(res.samples, DT, shocks, horizon_s=WINDOW, pre_s=1.0)


def run(ctx: RunContext) -> None:
    t0 = time.time()
    seeds = ctx.seeds(default=24, quick=3)
    horizon = 100.0 if ctx.quick else 600.0
    rng = np.random.default_rng(ctx.seed)
    rows, curve_rows, store = [], [], {}
    for env in ENVS:
        sess = {size: pmap(session, [Job(env, s, size, horizon) for s in seeds], ctx.workers) for size in (0, *SIZES)}
        tau = sess[0][0]["tau"]
        for size in SIZES:
            store[(env, size)] = {"tau": tau}
            for name in ("mid", "spread", "depth"):
                # PAIRED causal effect: same seed => identical market path until the shock; subtract the undisturbed control
                eff = np.array([a[name] - b[name] for a, b in zip(sess[size], sess[0])])
                mean = np.nanmean(eff, 0)
                boots = np.array([np.nanmean(eff[rng.integers(0, len(eff), len(eff))], 0) for _ in range(400)])
                lo, hi = np.nanquantile(boots, 0.025, 0), np.nanquantile(boots, 0.975, 0)
                store[(env, size)][name] = (mean, lo, hi)
                hl = half_life(mean, tau)
                hls = [h for h in (half_life(b, tau)["half_life_s"] for b in boots) if h is not None]
                # is the effect distinguishable from 0 at the end of the window / right after the shock?
                rows.append(dict(env=env, shock_lots=size, curve=name, initial_effect=hl["initial"], half_life_s=hl["half_life_s"],
                                 half_life_ci_lo=float(np.quantile(hls, .025)) if hls else None, half_life_ci_hi=float(np.quantile(hls, .975)) if hls else None,
                                 frac_boot_halving=len(hls) / len(boots), effect_at_10s=float(mean[-1]), effect_at_10s_ci_lo=float(lo[-1]),
                                 effect_at_10s_ci_hi=float(hi[-1]), effect_at_1s=float(mean[19]), effect_at_1s_ci_lo=float(lo[19]), effect_at_1s_ci_hi=float(hi[19]),
                                 n_shocks_per_session=float(np.mean([x["n_shocks"] for x in sess[size]]))))
                for k in range(len(tau)):
                    curve_rows.append(dict(env=env, shock_lots=size, curve=name, tau_s=float(tau[k]), effect=float(mean[k]), ci_lo=float(lo[k]), ci_hi=float(hi[k])))
        for r in [x for x in rows if x["env"] == env]:
            hl = "none" if r["half_life_s"] is None else f"{r['half_life_s']:.2f}s"
            print(f"[{env:8s}] shock {r['shock_lots']:>2} lots {r['curve']:>6s}: initial={r['initial_effect']:+.2f}  half-life={hl:>6s}  "
                  f"effect@1s={r['effect_at_1s']:+.2f} [{r['effect_at_1s_ci_lo']:+.2f},{r['effect_at_1s_ci_hi']:+.2f}]  effect@10s={r['effect_at_10s']:+.2f} [{r['effect_at_10s_ci_lo']:+.2f},{r['effect_at_10s_ci_hi']:+.2f}]")
    save_csv(ctx.out_dir / "resilience.csv", rows)
    save_csv(ctx.out_dir / "resilience_curves.csv", curve_rows)
    _plot(ctx, store)
    write_provenance(ctx, {"sessions": len(seeds), "horizon_s": horizon, "shock_sizes": SIZES, "shock_spacing_s": 30, "sample_dt_s": DT, "window_s": WINDOW,
                           "envs": {k: v for k, v in ENVS.items()}, "analysis": "paired (shock - no-shock control, same seeds) effect curves"},
                     files=["resilience.csv", "resilience_curves.csv", "resilience.png"], elapsed_s=time.time() - t0)


def _plot(ctx: RunContext, store: dict) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    titles = {"mid": "mid impact effect (ticks)", "spread": "spread excess effect", "depth": "depth deficit effect"}
    col = {20: "tab:orange", 60: "tab:red"}
    for i, env in enumerate(ENVS):
        for j, name in enumerate(("mid", "spread", "depth")):
            a = ax[i, j]
            for size in SIZES:
                d = store[(env, size)]
                mean, lo, hi = d[name]
                a.plot(d["tau"], mean, color=col[size], label=f"{size} lots")
                a.fill_between(d["tau"], lo, hi, color=col[size], alpha=0.18)
            a.axhline(0, color="k", lw=0.5)
            a.set(title=f"{env}: {titles[name]}", xlabel="seconds after shock"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ctx.out_dir / "resilience.png", dpi=140); plt.close(fig)
