"""Empirical check of the engines' asymptotic behaviour (spec: add O(log P), cancel O(1), match O(k)).

Three scaling experiments, each run on both engines; the log-log slope of time against the scaled quantity
estimates the exponent (0 = flat/constant, ~1 = linear; logarithmic growth shows as a small slope):

* ``add``    insert + cancel a *new interior* price level in a book that already has P levels (bids at even ticks,
             new orders at odd ticks) - the operation whose cost depends on P;
* ``cancel`` cancel a random resting order in a book of N orders spread over 100 levels;
* ``match``  a market order consuming exactly k resting orders of one level.
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine import cpp_engine
from engine.commands import Cancel, NewLimit, NewMarket
from engine.common import Side
from engine.python.order_book import OrderBook
from experiments.common import RunContext, save_csv, write_provenance

DESCRIPTION = "Empirical scaling of add / cancel / match cost with book size (Python vs C++)"


def _arr(rows):
    return np.array(rows, dtype=np.int64).reshape(-1, 8)


def bench_add(P: int, M: int, seed: int) -> dict:
    """Preload P bid levels at even ticks; time M x (add at a random odd tick, cancel it)."""
    rng = np.random.default_rng(seed)
    pre = [NewLimit(i + 1, Side.BUY, 2 * (i + 1), 1) for i in range(P)]
    odd = 2 * rng.integers(0, P, M) + 1
    ops, nid = [], P + 1
    for px in odd:
        ops.append(NewLimit(nid, Side.BUY, int(px), 1))
        ops.append(Cancel(nid))
        nid += 1
    py = OrderBook()
    py.process_all(pre)
    t = time.perf_counter()
    for c in ops:
        py.process(c)
    t_py = (time.perf_counter() - t) / M
    cp = cpp_engine.CppOrderBook()
    cp.run_batch(cpp_engine.commands_to_array(pre))
    arr = cpp_engine.commands_to_array(ops)
    t = time.perf_counter()
    cp.run_batch(arr)
    t_cpp = (time.perf_counter() - t) / M
    return {"python_us": t_py * 1e6, "cpp_us": t_cpp * 1e6}


def bench_cancel(N: int, M: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    pre = [NewLimit(i + 1, Side.BUY, 1 + (i % 100), 1) for i in range(N)]
    victims = rng.choice(N, M, replace=False) + 1
    ops = [Cancel(int(v)) for v in victims]
    py = OrderBook()
    py.process_all(pre)
    t = time.perf_counter()
    for c in ops:
        py.process(c)
    t_py = (time.perf_counter() - t) / M
    cp = cpp_engine.CppOrderBook()
    cp.run_batch(cpp_engine.commands_to_array(pre))
    arr = cpp_engine.commands_to_array(ops)
    t = time.perf_counter()
    cp.run_batch(arr)
    t_cpp = (time.perf_counter() - t) / M
    return {"python_us": t_py * 1e6, "cpp_us": t_cpp * 1e6}


def bench_match(k: int, trials: int) -> dict:
    """Median time of a market buy that consumes exactly k one-lot orders of a single level."""
    nid = 1
    py = OrderBook()
    cp = cpp_engine.CppOrderBook()
    rows, py_t = [], []
    pc = time.perf_counter_ns
    for _ in range(trials):
        for _ in range(k):
            py.submit_limit(nid, Side.SELL, 100, 1)
            rows.append((0, nid, -1, 100, 1, 0, 0, 0))
            nid += 1
        cmd = NewMarket(nid, Side.BUY, k)
        t0 = pc()
        py.process(cmd)
        py_t.append(pc() - t0)
        rows.append((1, nid, 1, 0, k, 0, 0, 0))
        nid += 1
    res = cp.run_batch(_arr(rows), timed=True)
    cpp_t = res["latency_ns"][res["class"] == 1]
    assert len(cpp_t) == trials
    return {"python_us": float(np.median(py_t)) / 1e3, "cpp_us": float(np.median(cpp_t)) / 1e3}


def slope(x, y) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def run(ctx: RunContext) -> None:
    if not cpp_engine.available():
        raise SystemExit("C++ extension not built - run scripts/build_cpp.sh first")
    t0 = time.time()
    q = ctx.quick
    rows = []
    for P in (10, 100, 1_000, 10_000, 100_000) if not q else (10, 1_000):
        r = bench_add(P, 20_000 if not q else 2_000, ctx.seed)
        rows.append({"experiment": "add_new_level_and_cancel", "size": P, **r})
    for N in (1_000, 10_000, 100_000, 400_000) if not q else (1_000, 10_000):
        r = bench_cancel(N, min(20_000, N // 2), ctx.seed)
        rows.append({"experiment": "cancel", "size": N, **r})
    for k in (1, 10, 100, 1_000, 5_000) if not q else (1, 100):
        r = bench_match(k, max(20, 20_000 // max(k, 1)) if not q else 10)
        rows.append({"experiment": "match_k_orders", "size": k, **r})
    for r in rows:
        r["speedup"] = r["python_us"] / r["cpp_us"]
    fits = []
    for exp in ("add_new_level_and_cancel", "cancel", "match_k_orders"):
        sub = [r for r in rows if r["experiment"] == exp]
        if len(sub) >= 3:
            for impl in ("python_us", "cpp_us"):
                fits.append({"experiment": exp, "impl": impl, "loglog_slope": slope([r["size"] for r in sub], [r[impl] for r in sub]),
                             "loglog_slope_upper_half": slope([r["size"] for r in sub[len(sub) // 2:]], [r[impl] for r in sub[len(sub) // 2:]])})
    for r in rows:
        print(f"{r['experiment']:26s} size={r['size']:>7}  python {r['python_us']:9.3f} us   cpp {r['cpp_us']:9.3f} us   speedup {r['speedup']:6.1f}x")
    for f in fits:
        print(f"  slope {f['experiment']:26s} {f['impl']:9s} all={f['loglog_slope']:+.2f}  upper-half={f['loglog_slope_upper_half']:+.2f}")
    save_csv(ctx.out_dir / "complexity.csv", rows)
    save_csv(ctx.out_dir / "complexity_fits.csv", fits)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = {"add_new_level_and_cancel": ("price levels P", "add+cancel of a new interior level"), "cancel": ("resting orders N", "cancel a random order"),
              "match_k_orders": ("orders consumed k", "market order consuming k orders")}
    for a, exp in zip(ax, titles):
        sub = [r for r in rows if r["experiment"] == exp]
        a.plot([r["size"] for r in sub], [r["python_us"] for r in sub], "o-", color="tab:gray", label="python")
        a.plot([r["size"] for r in sub], [r["cpp_us"] for r in sub], "s-", color="tab:blue", label="C++")
        a.set(xscale="log", yscale="log", xlabel=titles[exp][0], ylabel="time per operation (us)", title=titles[exp][1]); a.legend()
    fig.tight_layout(); fig.savefig(ctx.out_dir / "complexity.png", dpi=140); plt.close(fig)
    write_provenance(ctx, {"note": "timings are hardware dependent", "seed": ctx.seed}, dataset="synthetic books",
                     files=["complexity.csv", "complexity_fits.csv", "complexity.png"], elapsed_s=time.time() - t0)
