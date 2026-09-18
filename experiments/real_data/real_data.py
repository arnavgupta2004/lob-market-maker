"""Real-data analysis: the recorded Kraken BTC/USD feed replayed through the simulator's engine, compared with the simulator's own results.

Because replay drives the *same* order-book engine, every estimator used on synthetic markets (stylized facts, OBI predictiveness,
price impact, adverse selection) is applied to the real feed unchanged. Sections:

  1. data quality and replay fidelity (checksum match, touch fidelity, trade-price mismatches, timestamp inversions)
  2. stylized facts (returns, volatility clustering, order-sign memory, arrival clustering, spread, trade sizes)
  3. order-book imbalance: does it predict the next mid move? (Experiment A on real data)
  4. price impact I(Q) ~ Q^alpha of aggressive orders (bursts of same-instant prints merged into one parent order)
  5. adverse selection of passive (historical maker) fills: signed post-fill mid move (Experiment C on real data)

Uncertainty: the recording is ONE contiguous session, so intervals bootstrap consecutive 5-minute BLOCKS. Blocks are not independent
(volatility and liquidity regimes persist), so these intervals are optimistic; with ~5-12 blocks they are also wide. Results describe this
recording (one instrument, one time window) and are not general statements about crypto or equity markets.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import RunContext, save_csv, save_json, write_provenance
from experiments.real_data.common import DEFAULT_PATH, block_ci, blocks, dataset_id, load_dataset, replay
from research.adverse_selection import passive_fills
from research.common import mid_at
from research.imbalance import session_stats
from research.price_impact import add_stats, aggressive_orders, fit_power_law, merge_bursts, size_bin_stats
from research.stylized_facts import summarize_window
from simulator.simulator import NS

DESCRIPTION = "Real Kraken BTC/USD feed: replay fidelity, stylized facts, OBI, price impact, adverse selection vs the simulator"

OBI_H = (0.1, 0.5, 1.0, 5.0, 10.0)
MARK_H = {"100ms": 100_000_000, "500ms": 500_000_000, "1s": 1_000_000_000, "5s": 5_000_000_000, "10s": 10_000_000_000}
BTC_EDGES = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 1e4])
SIM_DIR = Path("results")


def _sim_table(name: str) -> pd.DataFrame | None:
    p = SIM_DIR / name
    return pd.read_csv(p) if p.exists() else None


def run(ctx: RunContext) -> None:
    t0 = time.time()
    arr, meta = load_dataset()
    lot = meta["lot_size"]
    edges = BTC_EDGES / lot  # size bins in lots
    if ctx.quick:  # smoke test: first two minutes only
        arr = arr[arr[:, 0] <= arr[0, 0] + 120 * NS]
    res, rp = replay(arr, meta, track=True, sample_s=0.1)
    span_ns = int(arr[-1, 0] - arr[0, 0])
    block_s = 60.0 if ctx.quick else 300.0
    B = blocks(span_ns, block_s, skip_s=10.0 if ctx.quick else 60.0)
    q = meta["quality"]
    quality = {"dataset": dataset_id(meta), "span_s": span_ns / NS, "feed_events": int(len(arr)), "feed_trades": rp.n_trades, "feed_level_updates": rp.n_levels,
               "exchange_checksum_match_rate": q["checksum_match_rate"], "exchange_checksum_checked": q["checksum_checked"], "timestamp_inversions": q["ts_inversions"],
               "reconnects": q["reconnects"], "replay_touch_fidelity": rp.fidelity, "replay_trade_price_mismatches": rp.rec.n_trade_price_mismatch,
               "replay_crossing_adds": rp.rec.n_crossing_adds, "blocks": len(B), "block_s": block_s,
               "trade_rate_per_s": rp.n_trades / (span_ns / NS), "spread_ticks_mean": float(np.nanmean(res.samples["spread"])),
               "spread_ticks_median": float(np.nanmedian(res.samples["spread"]))}
    save_json(ctx.out_dir / "quality.json", quality)
    print(f"[data] {quality['dataset']}\n  checksum match {quality['exchange_checksum_match_rate']:.4f} over {quality['exchange_checksum_checked']} updates; replay touch fidelity "
          f"{quality['replay_touch_fidelity']:.4f}; trade-price mismatches {quality['replay_trade_price_mismatches']}/{rp.n_trades}; ts inversions {quality['timestamp_inversions']}; "
          f"{len(B)} blocks of {block_s:.0f}s; trades/s {quality['trade_rate_per_s']:.2f}; spread mean {quality['spread_ticks_mean']:.2f} ticks")

    # ---------------------------------------------------------------- 2. stylized facts
    per = [summarize_window(res, a, b, merge_bursts=True) for a, b in B]
    keys = sorted(k for k in per[0] if not k.startswith("_") and not k.startswith("n_returns"))
    sim = _sim_table("stylized_facts/stylized_facts.csv")
    rows = []
    for k in keys:
        m, lo, hi = block_ci([p.get(k, np.nan) for p in per])
        rec = {"metric": k, "real_mean": m, "real_ci_lo": lo, "real_ci_hi": hi, "blocks": len(per)}
        if sim is not None:
            for v in ("noise", "informed", "hawkes", "hawkes+meta"):
                r = sim[(sim.variant == v) & (sim.metric == k)]
                rec[f"sim_{v}"] = float(r["mean"].iloc[0]) if len(r) else np.nan
        rows.append(rec)
    # tails / kurtosis at horizons where returns are not dominated by tick discreteness: pooled over the whole post-warm-up period
    from research.stylized_facts import hill_tail_index, log_returns, moments
    from research.as_calibration import mid_diagnostics
    pooled = {}
    for h in (1.0, 10.0, 60.0):
        blocks_r = [log_returns(np.array(mid_at(res, np.arange(a, b + 1, int(h * NS))))) for a, b in B]
        rng_b = np.random.default_rng(5)
        def stat(rs, f):
            return f(np.concatenate(rs))
        for name, fn in (("kurt", lambda r: moments(r)["excess_kurtosis"]), ("hill5", lambda r: hill_tail_index(r, frac=0.05)["alpha"]),
                         ("hill10", lambda r: hill_tail_index(r, frac=0.10)["alpha"]), ("zero_share", lambda r: float(np.mean(r == 0)))):
            v = stat(blocks_r, fn)
            boots = [stat([blocks_r[i] for i in rng_b.integers(0, len(blocks_r), len(blocks_r))], fn) for _ in range(300)]
            lo, hi = np.nanquantile(boots, [0.025, 0.975])
            pooled[f"{name}_{h:g}s_pooled"] = (v, float(lo), float(hi), int(sum(len(r) for r in blocks_r)))
    for k, (v, lo, hi, n) in pooled.items():
        rows.append({"metric": k, "real_mean": v, "real_ci_lo": lo, "real_ci_hi": hi, "blocks": len(B), "n_returns": n})
    # volatility signature and variance ratios of the mid (real): sigma(h) in ticks/sqrt(s); VR(q) > 1 = trending, < 1 = mean reverting
    md = mid_diagnostics(np.array(mid_at(res, np.arange(B[0][0], B[-1][1] + 1, NS))), 1.0, lags=(1, 2, 5, 10, 60))
    for k, v in md.items():
        if k.startswith(("vr_", "sigma_")):
            rows.append({"metric": f"real_{k}", "real_mean": v, "real_ci_lo": np.nan, "real_ci_hi": np.nan, "blocks": 1})
    save_csv(ctx.out_dir / "stylized_real_vs_sim.csv", rows)
    show = ["zero_share_1s", "acf_ret1_0.1s", "acf_abs1_1s", "acf_abs10_1s", "sign_acf1", "sign_gamma", "fano_1s", "fano_10s", "arrival_cv", "spread_median", "spread_mean", "size_hill5",
            "kurt_1s_pooled", "kurt_10s_pooled", "kurt_60s_pooled", "hill5_10s_pooled", "hill10_10s_pooled", "hill5_60s_pooled", "real_sigma_1s", "real_sigma_10s", "real_sigma_60s", "real_vr_10", "real_vr_60"]
    print("[stylized facts]  metric: real [block CI] | sim noise / informed / hawkes / hawkes+meta")
    for r in rows:
        if r["metric"] in show:
            sv = " / ".join(f"{r.get('sim_' + v, np.nan):.2f}" for v in ("noise", "informed", "hawkes", "hawkes+meta"))
            print(f"  {r['metric']:17s} {r['real_mean']:+8.3f} [{r['real_ci_lo']:+.3f},{r['real_ci_hi']:+.3f}] | {sv}")

    # ---------------------------------------------------------------- 3. OBI predictiveness
    samples = res.samples
    obi_rows = []
    obi_blocks = []
    rng = np.random.default_rng(1)
    for a, b in B:
        m = (samples["t_ns"] >= a) & (samples["t_ns"] < b)
        sl = {k: v[m] for k, v in samples.items()}
        obi_blocks.append(session_stats(sl, 0.1, OBI_H, levels=1, start_s=0.0, rng=rng))
    sim_obi = _sim_table("imbalance/summary.csv")
    for h in OBI_H:
        rec = {"horizon_s": h}
        for st in ("corr", "slope", "dir_acc_excess", "corr_backward", "corr_shuffled"):
            m, lo, hi = block_ci([x[h][st] for x in obi_blocks])
            rec.update({st: m, f"{st}_lo": lo, f"{st}_hi": hi})
        rec["mean_spread_ticks"] = quality["spread_ticks_mean"]
        if sim_obi is not None:
            for env in ("noise", "informed"):
                r = sim_obi[(sim_obi.env == env) & (sim_obi.variant == "L1") & (np.isclose(sim_obi.horizon_s, h))]
                rec[f"sim_{env}_corr"] = float(r["corr"].iloc[0]) if len(r) else np.nan
                rec[f"sim_{env}_slope"] = float(r["slope"].iloc[0]) if len(r) else np.nan
        obi_rows.append(rec)
    # economic size of the OBI signal: conditional mean forward move by OBI bin, in ticks and in basis points of the price, next to the half-spread and fees
    from research.imbalance import OBI_EDGES
    mid_px = float(np.nanmean(samples["mid"]) * meta["tick_size"])  # USD
    bin_rows = []
    for h in (1.0, 5.0):
        for i in range(len(OBI_EDGES) - 1):
            v = [x[h]["bin_mean"][i] for x in obi_blocks]
            m_, lo_, hi_ = block_ci(v)
            n_ = float(np.mean([x[h]["bin_n"][i] for x in obi_blocks]))
            bin_rows.append({"horizon_s": h, "obi_lo": OBI_EDGES[i], "obi_hi": OBI_EDGES[i + 1], "mean_fwd_move_ticks": m_, "ci_lo": lo_, "ci_hi": hi_,
                             "mean_fwd_move_bps": 1e4 * m_ * meta["tick_size"] / mid_px, "mean_samples_per_block": n_,
                             "half_spread_bps": 1e4 * 0.5 * quality["spread_ticks_mean"] * meta["tick_size"] / mid_px})
    save_csv(ctx.out_dir / "obi_bins_real.csv", bin_rows)
    top = [r for r in bin_rows if r["obi_lo"] >= 0.79]
    for r in top:
        print(f"[OBI economics] h={r['horizon_s']:g}s, OBI in (0.8,1]: mean forward move {r['mean_fwd_move_ticks']:+.2f} ticks [{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] = {r['mean_fwd_move_bps']:+.3f} bp   "
              f"(half-spread {r['half_spread_bps']:.3f} bp; 1 bp fee = {1e-4 * mid_px / meta['tick_size']:.0f} ticks)")
    save_csv(ctx.out_dir / "obi_real.csv", obi_rows)
    print("[OBI] horizon: real corr [CI] slope(ticks/OBI) | sim corr noise/informed")
    for r in obi_rows:
        print(f"  {r['horizon_s']:>5g}s  {r['corr']:+.3f} [{r['corr_lo']:+.3f},{r['corr_hi']:+.3f}]  slope {r['slope']:+.3f}  back {r['corr_backward']:+.3f} | {r.get('sim_noise_corr', np.nan):+.3f} / {r.get('sim_informed_corr', np.nan):+.3f}")

    # ---------------------------------------------------------------- 4. price impact (bursts merged)
    ao_all = merge_bursts(aggressive_orders(res))
    keys_imp = ("I_0", "I_1s", "I_5s")
    imp_stats: list[dict] = []
    from research.price_impact import impacts
    for a, b in B:
        m = (ao_all.t >= a) & (ao_all.t < b)
        ao = type(ao_all)(*[x[m] for x in (ao_all.t, ao_all.side, ao_all.qty, ao_all.mid_before, ao_all.vwap, ao_all.taker_owner)])
        if len(ao.t) < 10:
            imp_stats.append({k: {} for k in keys_imp})
            continue
        im = impacts(res, ao, horizons_s=(1.0, 5.0))
        imp_stats.append({"I_0": size_bin_stats(ao.qty, im["I_0"], edges), "I_1s": size_bin_stats(ao.qty, im["I_1.0s"] if "I_1.0s" in im else im["I_1s"], edges),
                          "I_5s": size_bin_stats(ao.qty, im["I_5s"], edges), "n": len(ao.t)})
    imp_rows, fits = [], {}
    rng = np.random.default_rng(2)
    for k in keys_imp:
        valid = [s[k] for s in imp_stats if s.get(k)]
        tot: dict = {}
        for s in valid:
            tot = add_stats(tot, s)
        fit = fit_power_law(tot, min_n=10)
        boots = []
        for _ in range(300):
            pick = rng.integers(0, len(valid), len(valid))
            agg: dict = {}
            for i in pick:
                agg = add_stats(agg, valid[i])
            boots.append(fit_power_law(agg, min_n=10)["alpha"])
        # step-aware second fit: only sizes >= 5e-4 BTC (below that an order barely exceeds the touch and impact is ~ the spread, a constant)
        big = {i: v for i, v in tot.items() if i >= 3}  # bins 3.. = sizes >= 1e-3 BTC after the below-first-edge bin (-1)
        fit_big = fit_power_law(big, min_n=10)
        fits[k] = {"alpha": fit["alpha"], "lo": float(np.nanquantile(boots, 0.025)), "hi": float(np.nanquantile(boots, 0.975)), "r2": fit["r2"], "bins_used": fit["n_bins"],
                   "alpha_ge_1e-3btc": fit_big["alpha"], "r2_ge_1e-3btc": fit_big["r2"]}
        for i in sorted(tot):
            n_, s1, s2, sq = tot[i]
            imp_rows.append({"impact": k, "size_btc_lo": (BTC_EDGES[i] if i >= 0 else 0.0), "n_orders": int(n_), "mean_size_btc": sq / n_ * lot, "mean_impact_ticks": s1 / n_, "se": float(np.sqrt(max(s2 / n_ - (s1 / n_) ** 2, 0) / n_))})
        print(f"[impact] {k}: alpha = {fit['alpha']:.2f} [{fits[k]['lo']:.2f}, {fits[k]['hi']:.2f}]  (bins {fit['n_bins']}, R2 {fit['r2']:.2f}); sizes >= 1e-3 BTC only: alpha = {fit_big['alpha']:.2f} (R2 {fit_big['r2']:.2f})  parent orders {sum(int(v[..., 0].sum()) if False else int(sum(x[0] for x in s.values())) for s in valid)}")
    save_csv(ctx.out_dir / "impact_real.csv", imp_rows)
    save_json(ctx.out_dir / "impact_fits.json", fits)

    # ---------------------------------------------------------------- 5. adverse selection of historical maker fills
    f_all = passive_fills(res)
    mark_rows = []
    per_block = []
    rng = np.random.default_rng(3)
    for a, b in B:
        m = (f_all.t >= a) & (f_all.t < b)
        f = f_all.select(m)
        rec = {}
        if len(f) > 20:
            E = f.side * (f.mid_before - f.price)
            rec["E_mean"], rec["E_median"], rec["E_qtyw"] = float(E.mean()), float(np.median(E)), float(np.sum(E * f.qty) / np.sum(f.qty))
            rec["n_fills"] = len(f)
            for lab, h in MARK_H.items():
                ok = f.t + h < b
                M = f.side * (mid_at(res, f.t + h) - f.mid_before)
                if ok.any():
                    rec[f"M_{lab}"], rec[f"Mmed_{lab}"] = float(M[ok].mean()), float(np.median(M[ok]))
                    rec[f"Mqtyw_{lab}"] = float(np.sum(M[ok] * f.qty[ok]) / np.sum(f.qty[ok]))
        # null control: random times, random side within the block
        tt = rng.integers(a, b - max(MARK_H.values()), 5000); ss = rng.choice([-1, 1], 5000); m0 = mid_at(res, tt)
        for lab, h in MARK_H.items():
            rec[f"null_{lab}"] = float(np.mean(ss * (mid_at(res, tt + h) - m0)))
        per_block.append(rec)
    names = ["E_mean", "E_median", "E_qtyw"] + [f"{p}_{h}" for p in ("M", "Mmed", "Mqtyw", "null") for h in MARK_H]
    for lab in names:
        m, lo, hi = block_ci([p.get(lab, np.nan) for p in per_block])
        mark_rows.append({"metric": lab, "mean_ticks": m, "ci_lo": lo, "ci_hi": hi})
    save_csv(ctx.out_dir / "markouts_real.csv", mark_rows)
    g = lambda n: next(r for r in mark_rows if r["metric"] == n)
    print(f"[adverse selection of historical maker fills; ticks (1 tick = 0.1 USD); per fill unless noted]  E mean {g('E_mean')['mean_ticks']:+.2f} / median {g('E_median')['mean_ticks']:+.2f} / qty-weighted {g('E_qtyw')['mean_ticks']:+.2f}")
    for h in MARK_H:
        print(f"    M_{h:>5s}: mean {g('M_' + h)['mean_ticks']:+7.2f} [{g('M_' + h)['ci_lo']:+.2f},{g('M_' + h)['ci_hi']:+.2f}]   median {g('Mmed_' + h)['mean_ticks']:+6.2f}   qty-weighted {g('Mqtyw_' + h)['mean_ticks']:+7.2f}   null {g('null_' + h)['mean_ticks']:+.3f}")
    _plot(ctx, per, rows, obi_rows, imp_rows, fits, mark_rows)
    write_provenance(ctx, {"block_s": block_s, "blocks": len(B), "obi_horizons": OBI_H, "markout_horizons": list(MARK_H), "size_edges_btc": BTC_EDGES.tolist(),
                           "quality": quality, "uncertainty": "block bootstrap over consecutive blocks of one contiguous recording (optimistic: blocks are correlated)"},
                     dataset=dataset_id(meta), files=["quality.json", "stylized_real_vs_sim.csv", "obi_real.csv", "obi_bins_real.csv", "impact_real.csv", "impact_fits.json", "markouts_real.csv", "real_data.png"],
                     elapsed_s=time.time() - t0)


def _plot(ctx, per, rows, obi_rows, imp_rows, fits, mark_rows) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    curves = json.loads(Path("results/stylized_facts/curves.json").read_text()) if Path("results/stylized_facts/curves.json").exists() else {}
    def mean_curve(seqs, n=30):
        """Average of per-block curves truncated to their common length (<= n); empty if none is long enough."""
        seqs = [x for x in seqs if len(x) >= 5]
        if not seqs:
            return np.array([])
        L = min(n, min(len(x) for x in seqs))
        return np.nanmean([x[:L] for x in seqs], 0)

    a1 = mean_curve([p.get("_acf_abs_1s", []) for p in per])
    ax[0, 0].plot(range(1, len(a1) + 1), a1, "k-", lw=2, label="real")
    for v, c in (("noise", "tab:gray"), ("hawkes+meta", "tab:red")):
        if v in curves:
            ax[0, 0].plot(range(1, 31), curves[v]["abs_acf"][:30], color=c, label=f"sim {v}")
    ax[0, 0].axhline(0, color="k", lw=0.4); ax[0, 0].set(title="ACF of |1 s return|", xlabel="lag (s)"); ax[0, 0].legend(fontsize=7)
    s_ = mean_curve([p.get("_sign_acf", []) for p in per])
    ax[0, 1].loglog(range(1, len(s_) + 1), np.where(s_ > 0, s_, np.nan), "ko-", ms=3, label="real")
    for v, c in (("informed", "tab:blue"), ("hawkes+meta", "tab:red")):
        if v in curves:
            x = np.array(curves[v]["sign_acf"][:30]); ax[0, 1].loglog(range(1, 31), np.where(x > 0, x, np.nan), color=c, label=f"sim {v}")
    ax[0, 1].set(title="ACF of aggressor signs", xlabel="lag (orders)"); ax[0, 1].legend(fontsize=7)
    h = [r["horizon_s"] for r in obi_rows]
    ax[0, 2].errorbar(h, [r["corr"] for r in obi_rows], [[r["corr"] - r["corr_lo"] for r in obi_rows], [r["corr_hi"] - r["corr"] for r in obi_rows]], fmt="ko-", capsize=3, label="real (block CI)")
    for env, c in (("noise", "tab:gray"), ("informed", "tab:blue")):
        ax[0, 2].plot(h, [r.get(f"sim_{env}_corr", np.nan) for r in obi_rows], "s--", color=c, label=f"sim {env}")
    ax[0, 2].set(xscale="log", title="OBI vs forward mid change: correlation", xlabel="horizon (s)"); ax[0, 2].axhline(0, color="k", lw=0.4); ax[0, 2].legend(fontsize=7)
    for k, c in (("I_0", "tab:gray"), ("I_1s", "tab:blue"), ("I_5s", "tab:red")):
        rr = [r for r in imp_rows if r["impact"] == k and r["n_orders"] >= 10 and r["mean_impact_ticks"] > 0]
        if rr:
            q = np.array([r["mean_size_btc"] for r in rr]); y = np.array([r["mean_impact_ticks"] for r in rr]); se = np.array([r["se"] for r in rr])
            ax[1, 0].errorbar(q, y, 1.96 * se, fmt="o", color=c, capsize=2, ms=4, label=f"{k}: alpha={fits[k]['alpha']:.2f} [{fits[k]['lo']:.2f},{fits[k]['hi']:.2f}]")
    ax[1, 0].set(xscale="log", yscale="log", title="price impact of parent orders (real)", xlabel="order size (BTC)", ylabel="mean impact (ticks)"); ax[1, 0].legend(fontsize=7)
    labs = ["100ms", "500ms", "1s", "5s", "10s"]
    y = np.array([next(r for r in mark_rows if r["metric"] == f"M_{l}")["mean_ticks"] for l in labs])  # per-fill mean
    lo = np.array([next(r for r in mark_rows if r["metric"] == f"M_{l}")["ci_lo"] for l in labs]); hi = np.array([next(r for r in mark_rows if r["metric"] == f"M_{l}")["ci_hi"] for l in labs])
    ax[1, 1].errorbar(range(5), y, [y - lo, hi - y], fmt="ko-", capsize=3, label="historical maker fills (per-fill mean)")
    ax[1, 1].plot(range(5), [next(r for r in mark_rows if r["metric"] == f"null_{l}")["mean_ticks"] for l in labs], "s:", color="gray", label="null (random times)")
    ax[1, 1].set(xticks=range(5), xticklabels=labs, title="signed post-fill mid move M (ticks; <0 adverse)"); ax[1, 1].axhline(0, color="k", lw=0.4); ax[1, 1].legend(fontsize=7)
    ks = [r for r in rows if r["metric"] in ("kurt_10s_pooled", "hill5_10s_pooled", "hill5_60s_pooled")]
    ax[1, 2].bar(range(len(ks)), [r["real_mean"] for r in ks], color="k", alpha=0.7, yerr=[[r["real_mean"] - r["real_ci_lo"] for r in ks], [r["real_ci_hi"] - r["real_mean"] for r in ks]], capsize=3, label="real")
    ax[1, 2].set(xticks=range(len(ks)), xticklabels=[r["metric"].replace("_pooled", "") for r in ks], title="return tails at 10 s / 60 s (pooled, block CI)"); ax[1, 2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ctx.out_dir / "real_data.png", dpi=140); plt.close(fig)
