"""The same market-making strategies on synthetic flow (Stage 4/7/9) and on REAL historical flow (Kraken BTC/USD replay).

Design
  * calibration window: the first 15 minutes of the recording. k (fill-intensity decay), beta_obi and the queue fill model are MEASURED there
    with probe participants running inside the replay - the same procedure as in the synthetic environments;
  * evaluation windows: consecutive 5-minute windows AFTER the calibration window (disjoint), each replayed from the feed state at its start;
    the strategy starts 30 s into each window;
  * arms: AS baseline, adaptive with all components off (symmetric quotes at mid +- 1/k), + inventory, full - at 0 and 10 ms one-way latency.

Rescaling (the real market is ~10x more volatile in ticks than the simulator: ~22 vs ~2 ticks/sqrt(s), with a ~1.2 tick spread): quote size 0.001 BTC,
soft inventory limit 0.01 BTC (kill at 0.02), the volatility bounds (1, 200) and quote-offset cap 200 ticks, sigma0 = 20, adverse cap 100 ticks. Skew coefficients
are chosen so the skew per QUOTE at the position limit equals the synthetic setting (lambda_I = 0.5 tick per quote; AS gamma set so the AS skew per quote is ~1 tick at
sigma = 22 ticks/sqrt(s), tau = 5 s). These scalings are design choices, not tuned on outcomes.

Caveats that matter more here than in the synthetic study: (1) replay is not counterfactual-exact - the strategy's orders displace historical liquidity, and historical
participants do not react to it; (2) queue position relative to historical orders is an approximation (L2 feed: back-of-queue reconciliation); (3) ONE recording, ~9
windows: the uncertainty is large and windows are correlated; (4) P&L is in USD for a 0.001 BTC quote, so absolute numbers are tiny - read them in bps of traded notional.
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from backtest.costs import FeeModel
from backtest.metrics import session_metrics
from backtest.statistics import paired_diff_ci, summarize
from experiments.common import RunContext, save_csv, save_json, write_provenance
from experiments.real_data.common import DEFAULT_PATH, blocks, dataset_id, load_dataset, replay, state_at, unit_value, window
from research.adverse_selection import passive_fills, summarize_fills
from research.as_calibration import ProbeQuoter, add_stats, fit_exponential_intensity, rows_from_stats, session_bin_stats
from research.imbalance import session_stats
from research.queue_position import QueueProbe, fit_fill_model
from simulator.latency.models import LatencyConfig
from simulator.order_flow.market_data import MD
from simulator.simulator import NS
from strategies.adaptive_mm import AdaptiveConfig
from strategies.baseline_mm import ASConfig

DESCRIPTION = "Market makers on replayed REAL flow: calibrate on 15 min, evaluate on later windows (Kraken BTC/USD)"

LOT = 1e-8
QUOTE = 100_000  # 0.001 BTC
LIMIT, KILL = 1_000_000, 2_000_000  # 0.01 / 0.02 BTC
ARMS = ("AS_scaled", "adaptive_off", "adaptive_inventory", "adaptive_full")
FEES = {"A_0bps": (0.0, 0.0), "B_1/2.5bps": (1.0, 2.5)}
OFF = dict(use_inventory_skew=False, use_inventory_size=False, use_obi=False, use_adverse=False, use_queue=False, use_latency=False)
DELTAS = (1, 2, 4, 8, 16, 32, 64)


def calibrate(arr, meta, cal_s: float, workers: int) -> dict:
    """k, beta_obi and the fill model from the calibration window (probe participants run inside the replay)."""
    t0 = int(arr[0, 0])
    w = window(arr, t0, t0 + int(cal_s * NS))
    probe = ProbeQuoter(deltas=DELTAS, rate=5.0, dwell_s=3.0, start_s=20.0)
    res, _ = replay(w, meta, [probe], sample_s=0.05)
    stats = session_bin_stats(probe.records, 3.0)
    rows = rows_from_stats(stats)
    fit = fit_exponential_intensity(rows, min_fills=5)
    slope = session_stats({k: v for k, v in res.samples.items()}, 0.05, [0.25], levels=1, start_s=20.0)[0.25]["slope"]
    qp = QueueProbe(dwell_s=3.0, rate=3.0, start_s=20.0)
    replay(w, meta, [qp], sample_s=0.1)
    fm = fit_fill_model([qp.dataset()], delta_s=1.0, dwell_s=3.0)
    return {"k": fit["k"], "k_r2": fit["r2"], "A": fit["A"], "beta_obi": slope, "fill_model": fm, "n_probes": len(probe.records), "intensity_rows": rows,
            "sigma_ticks_per_sqrt_s": float(np.nanstd(np.diff(res.samples["mid"][200:])) / np.sqrt(0.05))}


def arm_config(arm: str, cal: dict, lat: LatencyConfig):
    common = dict(k=cal["k"], latency=lat, quote_size=QUOTE, inventory_limit=LIMIT, kill_inventory=KILL, start_s=30.0, sigma0=20.0,
                  sigma_bounds=(1.0, 200.0), max_quote_offset_ticks=200)
    if arm == "AS_scaled":
        return ASConfig(gamma=4e-9, horizon_s=5.0, **common)
    a = dict(common, beta_obi=cal["beta_obi"], fill_model=cal["fill_model"], lambda_inv=5e-6, adverse_cap=100.0)
    flags = {"adaptive_off": OFF, "adaptive_inventory": {**OFF, "use_inventory_skew": True, "use_inventory_size": True},
             "adaptive_full": {k: True for k in OFF}}[arm]
    if not flags["use_queue"]:
        a["fill_model"] = None
    return AdaptiveConfig(**a, **flags)


def run(ctx: RunContext) -> None:
    t0 = time.time()
    arr, meta = load_dataset()
    if ctx.quick:
        arr = arr[arr[:, 0] <= arr[0, 0] + 240 * NS]
    span = int(arr[-1, 0] - arr[0, 0])
    cal_s = 60.0 if ctx.quick else 900.0
    win_s = 60.0 if ctx.quick else 300.0
    cal = calibrate(arr, meta, cal_s, ctx.workers)
    fm = cal["fill_model"]
    print(f"[{dataset_id(meta)}]\ncalibration ({cal_s:.0f}s): k={cal['k']:.4f}/tick (R2 {cal['k_r2']:.2f}, A={cal['A']:.2f}/s, {cal['n_probes']} probes)  beta_obi={cal['beta_obi']:.2f} ticks/OBI  "
          f"sigma~{cal['sigma_ticks_per_sqrt_s']:.1f} ticks/sqrt(s)  fill model a={fm.intercept:.2f} b_lnQ={fm.b_q:.3f} b_d={fm.b_d:.3f}")
    wins = blocks(span, win_s, skip_s=cal_s)
    lats = (0.0,) if ctx.quick else (0.0, 10.0)
    t_first = int(arr[0, 0])
    rows = []
    for i, (a, b) in enumerate(wins):
        w = window(arr, t_first + a, t_first + b)
        for lat_ms in lats:
            for arm in ARMS:
                mm = arm_config(arm, cal, LatencyConfig.symmetric(lat_ms)).build()
                res, rp = replay(w, meta, [mm], sample_s=0.1, record_events=True)
                m = session_metrics(res, mm.owner_id, FeeModel(), start_s=mm.cfg.start_s)
                af = summarize_fills(res, passive_fills(res, mm.owner_id))
                notional = m["notional_maker"] + m["notional_taker"]
                rec = dict(window=i, latency_ms=lat_ms, arm=arm, n_fills=m["n_fills"], pnl_gross_usd=m["pnl_gross"], notional_usd=notional, killed=float(mm.killed),
                           pnl_edge_usd=m["pnl_edge"], pnl_inventory_usd=m["pnl_inventory"], inv_abs_mean_btc=m["inv_abs_mean"] * LOT, inv_max_abs_btc=m["inv_max_abs"] * LOT,
                           eff_half_spread_ticks=af["E"], postfill_500ms_ticks=af["M_500ms"], postfill_100ms_ticks=af["M_100ms"], fill_rate_orders=m["fill_rate_orders"],
                           quote_lifetime_s=m["quote_lifetime_mean_s"], sharpe_1s=m["sharpe_1s"], max_drawdown_usd=m["max_drawdown"], touch_fidelity=rp.fidelity)
                for f, (mb, tb) in FEES.items():
                    net = m["pnl_gross"] - 1e-4 * (mb * m["notional_maker"] + tb * m["notional_taker"])
                    rec[f"pnl_net_{f}_usd"] = net
                    rec[f"pnl_net_{f}_bps"] = 1e4 * net / notional if notional > 0 else float("nan")
                rec["pnl_gross_bps"] = 1e4 * m["pnl_gross"] / notional if notional > 0 else float("nan")
                rows.append(rec)
        print(f"  window {i + 1}/{len(wins)} done")
    save_csv(ctx.out_dir / "windows.csv", rows)
    summ = []
    for lat_ms in lats:
        for arm in ARMS:
            rr = [r for r in rows if r["arm"] == arm and r["latency_ms"] == lat_ms]
            rec = dict(latency_ms=lat_ms, arm=arm, windows=len(rr))
            for k in ("pnl_gross_bps", "pnl_net_B_1/2.5bps_bps", "pnl_gross_usd", "n_fills", "inv_abs_mean_btc", "eff_half_spread_ticks", "postfill_500ms_ticks", "sharpe_1s", "killed"):
                s = summarize([r[k] for r in rr], seed=4)
                rec.update({f"{k}_mean": s["mean"], f"{k}_ci_lo": s["ci_lo"], f"{k}_ci_hi": s["ci_hi"]})
            summ.append(rec)
    save_csv(ctx.out_dir / "summary.csv", summ)
    print(f"\n{'lat':>4} {'arm':19s} {'gross bps':>10} {'net(B) bps':>11} {'fills':>6} {'|inv| BTC':>10} {'E ticks':>8} {'M500ms':>8} {'killed':>6}")
    for r in summ:
        print(f"{r['latency_ms']:>4g} {r['arm']:19s} {r['pnl_gross_bps_mean']:+10.2f} {r['pnl_net_B_1/2.5bps_bps_mean']:+11.2f} {r['n_fills_mean']:6.0f} {r['inv_abs_mean_btc_mean']:10.4f} "
              f"{r['eff_half_spread_ticks_mean']:+8.2f} {r['postfill_500ms_ticks_mean']:+8.2f} {r['killed_mean']:6.2f}")
    diffs = []
    for lat_ms in lats:
        for a_, b_ in (("adaptive_inventory", "adaptive_off"), ("adaptive_full", "adaptive_inventory"), ("adaptive_full", "AS_scaled")):
            x = [r for r in rows if r["arm"] == a_ and r["latency_ms"] == lat_ms]; y = [r for r in rows if r["arm"] == b_ and r["latency_ms"] == lat_ms]
            for k in ("pnl_gross_usd", "pnl_net_B_1/2.5bps_usd", "inv_abs_mean_btc", "n_fills"):
                d = paired_diff_ci([r[k] for r in x], [r[k] for r in y], seed=6)
                diffs.append(dict(latency_ms=lat_ms, a=a_, b=b_, metric=k, mean_diff=d["mean_diff"], ci_lo=d["ci_lo"], ci_hi=d["ci_hi"], n_windows=d["n"]))
    save_csv(ctx.out_dir / "paired_differences.csv", diffs)
    save_json(ctx.out_dir / "calibration.json", {k: v for k, v in cal.items() if k != "intensity_rows"} | {"fill_model": fm})
    _plot(ctx, summ, cal, lats)
    write_provenance(ctx, {"calibration_s": cal_s, "window_s": win_s, "windows": len(wins), "arms": ARMS, "fees_bps(maker,taker)": FEES, "quote_lots": QUOTE, "inventory_limit_lots": LIMIT,
                           "calibration": {k: v for k, v in cal.items() if k != "intensity_rows"}, "latencies_ms": lats},
                     dataset=dataset_id(meta), files=["windows.csv", "summary.csv", "paired_differences.csv", "calibration.json", "real_mm.png"], elapsed_s=time.time() - t0)


def _plot(ctx, summ, cal, lats) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    colors = dict(zip(ARMS, ("tab:gray", "tab:olive", "tab:orange", "tab:green")))
    rows = [r for r in cal["intensity_rows"] if r["fills"] > 0]
    d = np.array([r["delta"] for r in rows]); lam = np.array([r["lam"] for r in rows])
    ax[0].semilogy(d, lam, "ko"); xx = np.linspace(d.min(), d.max(), 30)
    ax[0].semilogy(xx, cal["A"] * np.exp(-cal["k"] * xx), "r-", label=f"fit k={cal['k']:.3f}/tick (R2 {cal['k_r2']:.2f})")
    ax[0].set(title="real feed: fill intensity vs distance from mid", xlabel="distance (ticks)", ylabel="lambda (1/s)"); ax[0].legend(fontsize=8)
    for j, (m, ttl) in enumerate((("pnl_gross_bps", "gross P&L (bps of traded notional)"), ("pnl_net_B_1/2.5bps_bps", "net P&L, maker 1 bp / taker 2.5 bp"))):
        a = ax[1 + j]
        for i, arm in enumerate(ARMS):
            for k, lat in enumerate(lats):
                r = next(x for x in summ if x["arm"] == arm and x["latency_ms"] == lat)
                y = r[f"{m}_mean"]
                a.bar(i * (len(lats) + 0.5) + k, y, 0.9, color=colors[arm], alpha=1.0 if lat == 0 else 0.55, yerr=[[y - r[f"{m}_ci_lo"]], [r[f"{m}_ci_hi"] - y]], capsize=3,
                      label=f"{arm}" if k == 0 else None)
        a.axhline(0, color="k", lw=0.5); a.set(title=ttl + " (solid 0 ms, faded 10 ms)", xticks=[i * (len(lats) + 0.5) + 0.5 * (len(lats) - 1) for i in range(len(ARMS))], xticklabels=ARMS)
        a.tick_params(axis="x", labelrotation=15)
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ctx.out_dir / "real_mm.png", dpi=140); plt.close(fig)
