"""Regenerate the README figures (docs/figures/*.png) from the committed result files and one short simulation.

    python scripts/make_readme_figures.py

Every figure reads results/<experiment>/*.csv written by ``python -m experiments.run ...`` (nothing is hand-entered), except `hero_market.png`, which runs a
60 s seeded simulation. Real-data figures use only the committed aggregate statistics (the raw feed is not in the repository).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("results")
OUT = Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False, "axes.axisbelow": True,
})
C = {"noise": "#4C78A8", "informed": "#E45756", "informed_jump": "#B279A2", "real": "#222222", "as": "#7F7F7F", "adaptive": "#54A24B", "ci": 0.18}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ------------------------------------------------------------------------------------------------------- 1. hero: a simulated market
def hero_market():
    from simulator.market_state.fundamental import FundamentalConfig
    from simulator.order_flow.informed import InformedTrader
    from simulator.order_flow.noise import NoiseTrader
    from simulator.simulator import NS, SimConfig, Simulator
    cfg = SimConfig(seed=7, horizon_s=90, fundamental=FundamentalConfig(sigma=0.03), sample_interval_s=0.1)
    res = Simulator(cfg, [NoiseTrader(), InformedTrader()]).run()
    s = res.samples
    t = s["t_ns"] / NS
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.9), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    ax[0].plot(t, s["fundamental"], color="k", lw=1.2, ls="--", label="latent fundamental value")
    ax[0].plot(t, s["mid"], color=C["informed"], lw=1.1, label="mid-price")
    ax[0].fill_between(t, s["best_bid"], s["best_ask"], color=C["informed"], alpha=0.25, label="bid-ask spread")
    ax[0].set(title="Simulated market: price discovery", xlabel="time (s)", ylabel="price (ticks)"); ax[0].legend(fontsize=8, loc="upper right")
    bids, asks = res.book.depth(12)
    ax[1].barh([p for p, _, _ in bids], [q for _, q, _ in bids], color=C["noise"], label="bids")
    ax[1].barh([p for p, _, _ in asks], [q for _, q, _ in asks], color=C["informed"], label="asks")
    ax[1].set(title="L2 depth snapshot", xlabel="resting quantity (lots)", ylabel="price (ticks)"); ax[1].legend(fontsize=8)
    sp = s["spread"][~np.isnan(s["spread"])]
    v, c = np.unique(sp, return_counts=True)
    ax[2].bar(v[:8], c[:8] / c.sum(), color=C["noise"])
    ax[2].set(title="Spread distribution", xlabel="spread (ticks)", ylabel="share of time")
    save(fig, "hero_market.png")


# ------------------------------------------------------------------------------------------------------- 2. engine benchmark
def engine_speedup():
    d = pd.read_csv(R / "benchmark_engine/speedup.csv")
    order = ["low_flow", "high_flow", "many_levels", "few_levels", "high_cancel", "high_match"]
    impls = [("cpp_batch", "C++ batch (lean)", "#2E5E8C"), ("cpp_batch_ev", "C++ batch (events built)", "#4C78A8"), ("cpp_percall", "C++ per call (raw)", "#9ECAE9"),
             ("cpp_wrapper", "C++ per call via wrapper", "#F58518")]
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.2
    for i, (k, lab, col) in enumerate(impls):
        rows = [d[(d.workload == wl) & (d.impl == k)].iloc[0] for wl in order]
        y = np.array([r.speedup_median for r in rows])
        ax.bar(np.arange(len(order)) + (i - 1.5) * w, y, w, color=col, label=lab,
               yerr=[y - [r.speedup_min for r in rows], [r.speedup_max for r in rows] - y], capsize=2, error_kw={"lw": 0.8})
    ax.axhline(1, color="k", lw=1)
    ax.set(yscale="log", xticks=range(len(order)), xticklabels=[o.replace("_", "\n") for o in order], ylabel="speedup over the Python engine (x)",
           title="Python vs C++ order book: speed-up depends on how it is driven")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_ylim(0.6, 80)
    save(fig, "engine_speedup.png")


# ------------------------------------------------------------------------------------------------------- 3. queue position
def queue_fill():
    d = pd.read_csv(R / "queue/fill_prob_by_q.csv")
    labels = ["0", "1-2", "3-5", "6-10", "11-20", "21-40", "41+"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    styles = [("all", "pooled over all distances (confounded)", "#BBBBBB", "o--"), ("d0", "at the touch", "#2E5E8C", "o-"), ("d1", "1 tick behind", "#4C78A8", "s-"), ("d2+", ">= 2 ticks behind", "#9ECAE9", "^-")]
    for a, env in zip(ax, ("noise", "informed")):
        for k, lab, col, ls in styles:
            r = d[(d.env == env) & (d.dt_s == 1.0) & (d.k == k)].set_index("q_bin").reindex(labels)
            if r.p_fill.isna().all():
                continue
            a.errorbar(range(len(labels)), r.p_fill, [r.p_fill - r.ci_lo, r.ci_hi - r.p_fill], fmt=ls, color=col, ms=4, capsize=2, label=lab)
        a.set(title=f"{env} flow", xticks=range(len(labels)), xticklabels=labels, xlabel="quantity ahead in queue (lots)")
    ax[0].set_ylabel("P(fill within 1 s)")
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Queue position matters - but only after conditioning on distance from the touch", y=1.03, fontweight="bold")
    save(fig, "queue_fill_probability.png")


# ------------------------------------------------------------------------------------------------------- 4. adverse selection
def adverse_selection():
    m = pd.read_csv(R / "adverse_selection/post_fill_markouts.csv")
    real = pd.read_csv(R / "real_data/markouts_real.csv").set_index("metric")
    hs = ["10ms", "50ms", "100ms", "500ms", "1s"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
    for env in ("noise", "informed"):
        r = [m[(m.env == env) & (m.scope == "mm_profile") & (m.metric == "M") & (m.horizon == h)].iloc[0] for h in hs]
        y = np.array([x["mean"] for x in r])
        ax[0].errorbar(range(len(hs)), y, [y - [x.ci_lo for x in r], [x.ci_hi for x in r] - y], fmt="o-", color=C[env], capsize=3, label=f"{env}: market-maker fills")
    nl = [m[(m.env == "informed") & (m.scope == "null_random_times") & (m.horizon == h)].iloc[0]["mean"] for h in hs]
    ax[0].plot(range(len(hs)), nl, "s:", color="gray", label="null control (random times)")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set(xticks=range(len(hs)), xticklabels=hs, ylabel="signed post-fill mid move (ticks; < 0 = adverse)", title="Simulator: fills are followed by adverse moves")
    ax[0].legend(fontsize=8)
    rh = ["100ms", "500ms", "1s", "5s", "10s"]
    y = np.array([real.loc[f"M_{h}", "mean_ticks"] for h in rh]); lo = np.array([real.loc[f"M_{h}", "ci_lo"] for h in rh]); hi = np.array([real.loc[f"M_{h}", "ci_hi"] for h in rh])
    ax[1].errorbar(range(len(rh)), y, [y - lo, hi - y], fmt="o-", color=C["real"], capsize=3, label="historical maker fills (Kraken BTC/USD)")
    ax[1].plot(range(len(rh)), [real.loc[f"null_{h}", "mean_ticks"] for h in rh], "s:", color="gray", label="null control")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(xticks=range(len(rh)), xticklabels=rh, title="Real data: same estimator, far larger effect (1 tick = 0.1 USD)", ylabel="ticks")
    ax[1].legend(fontsize=8)
    save(fig, "adverse_selection.png")


# ------------------------------------------------------------------------------------------------------- 5. inventory / risk aversion
def inventory():
    d = pd.read_csv(R / "inventory/summary.csv")
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 3.8))
    x_of = lambda g: g if g > 0 else 0.002
    for env in ("noise", "informed"):
        r = d[d.env == env].sort_values("gamma")
        xs = [x_of(g) for g in r.gamma]
        ax[0].errorbar(xs, r.pnl_net_mean, [r.pnl_net_mean - r.pnl_net_ci_lo, r.pnl_net_ci_hi - r.pnl_net_mean], fmt="o-", color=C[env], capsize=3, label=env)
        ax[1].errorbar(xs, r.inv_abs_mean_mean, [r.inv_abs_mean_mean - r.inv_abs_mean_ci_lo, r.inv_abs_mean_ci_hi - r.inv_abs_mean_mean], fmt="o-", color=C[env], capsize=3, label=env)
        ax[2].plot(xs, r.sharpe_1s_mean, "o-", color=C[env], label=env)
    for a, t in zip(ax, ("Mean net P&L per session", "Mean |inventory| (lots)", "Per-second Sharpe")):
        a.set(xscale="log", xlabel="risk aversion  gamma  (0 plotted at 0.002)", title=t); a.legend(fontsize=8)
    ax[0].axhline(0, color="k", lw=0.6)
    fig.suptitle("Avellaneda-Stoikov baseline: inventory skew is a risk control with a P&L price", y=1.04, fontweight="bold")
    save(fig, "inventory_gamma.png")


# ------------------------------------------------------------------------------------------------------- 6. latency
def latency():
    d = pd.read_csv(R / "latency/summary.csv")
    arms = [("as_gamma0.01", "AS baseline (gamma = .01)", "#7F7F7F", "s--"), ("adaptive_no_latency_term", "adaptive, no latency term", "#B8D98B", "^-"), ("adaptive_full", "adaptive, full", "#54A24B", "o-")]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9), sharey=True)
    for a, env in zip(ax, ("informed", "informed_jump")):
        for arm, lab, col, ls in arms:
            r = d[(d.env == env) & (d.arm == arm)].sort_values("latency_ms")
            a.errorbar(r.latency_ms, r.pnl_net_mean, [r.pnl_net_mean - r.pnl_net_ci_lo, r.pnl_net_ci_hi - r.pnl_net_mean], fmt=ls, color=col, capsize=2, label=lab)
        a.axhline(0, color="k", lw=0.6)
        a.set(xscale="symlog", xticks=[0, 1, 5, 10, 25, 50], xticklabels=["0", "1", "5", "10", "25", "50"], xlabel="one-way latency (ms)", xlim=(-0.3, 58))
    ax[0].set_ylabel("mean net P&L per session"); ax[0].legend(fontsize=8)
    ax[0].set_title("informed flow (diffusion)"); ax[1].set_title("informed flow + news jumps")
    fig.suptitle("Latency erodes the adaptive maker's P&L; the AS baseline's apparent gain is damping of its over-reactive skew", y=1.03, fontweight="bold")
    save(fig, "latency.png")


# ------------------------------------------------------------------------------------------------------- 7. ablation
def ablation():
    d = pd.read_csv(R / "ablation/incremental.csv")
    d = d[d.versus != "AS_gamma0.01"]
    k = "d_pnl_net_B_0.5/2bps"
    steps = ["L1_inventory", "L2_obi", "L3_adverse", "L4_queue", "L5_latency(full)"]
    names = ["+ inventory", "+ OBI", "+ adverse\nselection", "+ queue\nrule", "+ latency\nterm"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    w = 0.26
    for i, env in enumerate(("noise", "informed", "informed_jump")):
        r = [d[(d.env == env) & (d.step == s)].iloc[0] for s in steps]
        y = np.array([x[k] for x in r])
        ax[0].bar(np.arange(5) + (i - 1) * w, y, w, color=C[env], label=env,
                  yerr=[y - [x[k + "_lo"] for x in r], [x[k + "_hi"] for x in r] - y], capsize=2, error_kw={"lw": 0.8})
        ax[1].bar(np.arange(5) + (i - 1) * w, [x["d_sharpe_1s"] for x in r], w, color=C[env])
        ax[2].bar(np.arange(5) + (i - 1) * w, [x["d_inv_abs_mean"] for x in r], w, color=C[env])
    for a, t, yl in zip(ax, ("Incremental net P&L (after fees), paired 95% CI", "Incremental per-second Sharpe", "Incremental mean |inventory| (lots)"), ("currency / session", "", "lots")):
        a.axhline(0, color="k", lw=0.6)
        a.set(xticks=range(5), xticklabels=names, title=t, ylabel=yl)
    ax[0].legend(fontsize=8)
    fig.suptitle("Ablation ladder: what each microstructure component adds", y=1.03, fontweight="bold")
    save(fig, "ablation_ladder.png")


# ------------------------------------------------------------------------------------------------------- 8. real vs simulator
def real_vs_sim():
    o = pd.read_csv(R / "real_data/obi_real.csv")
    mm = pd.read_csv(R / "real_mm/summary.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].errorbar(o.horizon_s, o["corr"], [o["corr"] - o.corr_lo, o.corr_hi - o["corr"]], fmt="o-", color=C["real"], capsize=3, lw=2, label="real (Kraken BTC/USD, block CI)")
    ax[0].plot(o.horizon_s, o.sim_noise_corr, "s--", color=C["noise"], label="simulator: noise flow")
    ax[0].plot(o.horizon_s, o.sim_informed_corr, "^--", color=C["informed"], label="simulator: informed flow")
    ax[0].set(xscale="log", xlabel="forecast horizon (s)", ylabel="correlation of OBI with forward mid change", title="Order-book imbalance: real vs simulated predictability")
    ax[0].legend(fontsize=8)
    arms = ["AS_scaled", "adaptive_off", "adaptive_inventory", "adaptive_full"]
    labs = ["AS", "symmetric", "+ inventory", "full"]
    cols = ["#7F7F7F", "#B8D98B", "#8CC084", "#54A24B"]
    for j, lat in enumerate((0.0, 10.0)):
        r = [mm[(mm.arm == a) & (mm.latency_ms == lat)].iloc[0] for a in arms]
        y = np.array([x.pnl_gross_bps_mean for x in r])
        ax[1].bar(np.arange(4) + (j - 0.5) * 0.38, y, 0.36, color=cols, alpha=1.0 if lat == 0 else 0.55,
                  yerr=[y - [x.pnl_gross_bps_ci_lo for x in r], [x.pnl_gross_bps_ci_hi for x in r] - y], capsize=3)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(xticks=range(4), xticklabels=labs, ylabel="gross P&L (bp of traded notional)", title="Market makers on replayed real flow\n(solid: 0 ms latency, faded: 10 ms)")
    save(fig, "real_vs_simulated.png")


if __name__ == "__main__":
    for fn in (hero_market, engine_speedup, queue_fill, adverse_selection, inventory, latency, ablation, real_vs_sim):
        fn()
