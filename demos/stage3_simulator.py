"""Stage 3 demonstration: discrete-event simulator with noise flow, then informed flow.

    python demos/stage3_simulator.py            # writes results/stage3_simulator.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine.events import stream_hash
from simulator.market_state.fundamental import FundamentalConfig
from simulator.order_flow.informed import InformedTrader, InformedTraderConfig
from simulator.order_flow.noise import NoiseTrader
from simulator.simulator import NS, SimConfig, Simulator

OUT = Path("results")


def summarize(label: str, r) -> None:
    s, tr = r.steady()
    ok = ~np.isnan(s["mid"])
    dm = np.diff(s["mid"][ok])
    print(f"\n[{label}]  commands={r.n_commands}  events={r.n_events}  trades={len(tr['price'])}")
    print(f"  spread (ticks): mean={np.nanmean(s['spread']):.2f}  median={np.nanmedian(s['spread']):.0f}  "
          f"p95={np.nanpercentile(s['spread'], 95):.0f}   empty-side samples={int((~ok).sum())}")
    print(f"  resting orders: mean={s['n_orders'].mean():.0f}   depth(5 lvls) bid/ask: "
          f"{s['bid_depth5'].mean():.0f}/{s['ask_depth5'].mean():.0f} lots")
    print(f"  mid change per {r.config.sample_interval_s}s: std={dm.std():.2f} ticks")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fund = FundamentalConfig(sigma=0.03)  # price units / sqrt(s); tick = 0.01
    cfg = SimConfig(seed=7, horizon_s=120, fundamental=fund)

    a = Simulator(cfg, [NoiseTrader()]).run()
    a2 = Simulator(cfg, [NoiseTrader()]).run()
    summarize("noise only", a)
    print(f"  determinism: hash(run1)==hash(run2): {stream_hash(a.events) == stream_hash(a2.events)}")

    b = Simulator(cfg, [NoiseTrader(), InformedTrader(InformedTraderConfig(rate=10.0))]).run()
    summarize("noise + informed (fundamental) traders", b)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for r, lab, c in ((a, "noise only", "tab:blue"), (b, "noise + informed", "tab:red")):
        s, _ = r.steady()
        t = s["t_ns"] / NS
        ax[0].plot(t, s["mid"], color=c, lw=0.8, label=f"mid: {lab}")
        err = np.abs(s["mid"] - s["fundamental"])
        print(f"  mean |mid - fundamental| [{lab}]: {np.nanmean(err):.2f} ticks")
    ax[0].plot(t, s["fundamental"], "k--", lw=1, label="fundamental")
    ax[0].set(title="Price discovery", xlabel="time (s)", ylabel="ticks"); ax[0].legend(fontsize=7)
    s, _ = a.steady()
    vals, cnt = np.unique(s["spread"][~np.isnan(s["spread"])], return_counts=True)
    ax[1].bar(vals, cnt / cnt.sum()); ax[1].set(title="Spread distribution (noise only)", xlabel="ticks", ylabel="frequency")
    ax[2].hist(np.diff(s["mid"][~np.isnan(s["mid"])]), bins=40, color="gray")
    ax[2].set(title="Mid-price change per 100 ms (noise only)", xlabel="ticks"); ax[2].set_yscale("log")
    fig.tight_layout(); fig.savefig(OUT / "stage3_simulator.png", dpi=130)
    print(f"\nfigure -> {OUT / 'stage3_simulator.png'}")


if __name__ == "__main__":
    main()
