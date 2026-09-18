"""Python vs C++ order-book benchmark on identical command streams.

For every workload the harness first *verifies* that both engines emit the identical event stream (so the timings
compare identical work), then measures

* throughput - commands/s (add + cancel + modify + market), new orders/s, trades/s, events/s;
* latency - per-command wall time by class (add that rests, match, cancel, modify), median / mean / p99;
* memory - RSS growth after building a book, in a fresh process per implementation;
* speedup - Python time / C++ time, paired by repetition (median and min-max over repetitions).

Four ways of running the engines are reported, because the honest speedup depends on how the C++ is driven:

  python          reference engine, ``OrderBook.process`` per command (events are constructed, not stored)
  cpp_percall     C++ engine, one pybind11 call per command returning the events as tuples
  cpp_wrapper     C++ engine through ``CppOrderBook`` (events converted to ``Event`` objects - the drop-in API)
  cpp_batch       C++ engine, whole command array in one call, events counted only ("lean")
  cpp_batch_ev    same but every event materialised into a C++ vector (comparable to Python constructing Events)

Workloads (``engine.workload`` parameters; the *measured* properties - command mix, mean/max resting orders, mean
active price levels, trades per command - are written next to the results in ``workloads.csv``):
low_flow / high_flow = small (~10 orders) vs large (~25k orders) resting book (engine cost does not depend on the
wall-clock arrival rate, so book size is the engine-relevant meaning of "flow"), many_levels / few_levels = 5000 vs 2
tick price range, high_cancel = 44% of commands are cancels, high_match = 50% marketable limits + 20% market orders.

Timings depend on the machine and are not bit-reproducible; workloads, seeds and the correctness verification are.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine import cpp_engine
from engine.commands import Cancel, Modify, NewLimit, NewMarket
from engine.events import Event
from engine.python.order_book import OrderBook
from engine.workload import WorkloadParams, generate_workload
from experiments.common import REPO, RunContext, save_csv, save_json, write_provenance

DESCRIPTION = "Benchmark the Python vs C++ order book (throughput, latency, memory, speedup)"

WORKLOADS: dict[str, WorkloadParams] = {
    "low_flow": WorkloadParams(price_range=20, p_limit=0.49, p_market=0.03, p_cancel=0.44, p_modify=0.04),
    "high_flow": WorkloadParams(price_range=200, p_limit=0.80, p_market=0.05, p_cancel=0.10, p_modify=0.05),
    "many_levels": WorkloadParams(price_range=5000),
    "few_levels": WorkloadParams(price_range=2),
    "high_cancel": WorkloadParams(price_range=50, p_limit=0.50, p_market=0.02, p_cancel=0.44, p_modify=0.04),
    "high_match": WorkloadParams(p_limit=0.60, p_market=0.20, p_cancel=0.15, p_modify=0.05, p_aggressive=0.50, max_qty=20),
}
CLASS_NAMES = {0: "add", 1: "match", 2: "cancel", 3: "modify"}


def _cpu() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or platform.processor()
    except Exception:  # pragma: no cover
        return platform.processor()


def build_info() -> dict:
    info = {"cpu": _cpu(), "platform": platform.platform(), "python": platform.python_version()}
    cache = REPO / "build" / "CMakeCache.txt"
    if cache.exists():
        wanted = {"CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_CXX_FLAGS_RELEASE"}
        for line in cache.read_text().splitlines():
            if "=" in line and not line.startswith("//"):
                name = line.split("=", 1)[0].split(":")[0]
                if name in wanted:  # exact match (a prefix match also hits *_ADVANCED / *_INIT entries)
                    info[name] = line.split("=", 1)[1]
    return info


def _timeit(fn, reps: int) -> list[float]:
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return out


def workload_stats(cmds, py_events: list[Event], book: OrderBook, sizes: list[int], levels: list[int]) -> dict:
    n = len(cmds)
    kinds = [type(c) for c in cmds]
    trades = sum(1 for e in py_events if e.type.value == "TRADE")
    cmd_with_trade = 0
    return {
        "commands": n,
        "frac_limit": kinds.count(NewLimit) / n, "frac_market": kinds.count(NewMarket) / n,
        "frac_cancel": kinds.count(Cancel) / n, "frac_modify": kinds.count(Modify) / n,
        "events": len(py_events), "trades": trades, "trades_per_command": trades / n,
        "final_resting_orders": len(book), "mean_resting_orders": float(np.mean(sizes)), "max_resting_orders": int(max(sizes)),
        "mean_active_levels": float(np.mean(levels)),
    }


def run_workload(name: str, params: WorkloadParams, seed: int, reps: int, latency: bool) -> dict:
    cmds = generate_workload(seed, params)
    arr = cpp_engine.commands_to_array(cmds)
    tuples = [tuple(r) for r in arr.tolist()]
    # ---- correctness: identical event streams (this is what makes the timings comparable)
    ref = OrderBook(record_events=True)
    sizes, levels = [], []
    every = max(1, len(cmds) // 400)
    for i, c in enumerate(cmds):
        ref.process(c)
        if i % every == 0:
            sizes.append(len(ref))
            levels.append(len(ref.bids) + len(ref.asks))
    ref_arr = cpp_engine.events_to_array(ref.events)
    res = cpp_engine.CppOrderBook().run_batch(arr, collect_events=True, checksum=True)
    assert np.array_equal(res["events"], ref_arr), f"{name}: C++ and Python event streams differ"
    assert res["checksum"] == cpp_engine.events_checksum(ref_arr)
    stats = workload_stats(cmds, ref.events, ref, sizes, levels)
    n = len(cmds)
    n_new = sum(1 for c in cmds if isinstance(c, (NewLimit, NewMarket)))

    def py_run():
        b = OrderBook(record_events=False)
        for c in cmds:
            b.process(c)

    def cpp_percall():
        b = cpp_engine._lob_cpp.OrderBook(0)
        pr = b.process_raw
        for t in tuples:
            pr(*t)

    def cpp_wrapper():
        b = cpp_engine.CppOrderBook()
        for c in cmds:
            b.process(c)

    def cpp_batch():
        cpp_engine.CppOrderBook().run_batch(arr)

    def cpp_batch_ev():
        cpp_engine.CppOrderBook().run_batch(arr, collect_events=True)

    impls = {"python": py_run, "cpp_percall": cpp_percall, "cpp_wrapper": cpp_wrapper, "cpp_batch": cpp_batch, "cpp_batch_ev": cpp_batch_ev}
    times: dict[str, list[float]] = {}
    for k, fn in impls.items():
        fn()  # warm-up (allocator, caches)
    for rep in range(reps):  # interleave implementations so drift affects all equally
        for k, fn in impls.items():
            t = time.perf_counter()
            fn()
            times.setdefault(k, []).append(time.perf_counter() - t)
    out = {"name": name, "stats": stats, "times": times, "n": n, "n_new": n_new, "trades": stats["trades"], "events": stats["events"]}
    if latency:
        out["latency"] = latency_profile(cmds, arr)
    return out


def latency_profile(cmds, arr) -> dict:
    """Per-command latency (ns) by class for Python and C++ (native timing inside the batch loop)."""
    cls_py, lat_py = [], []
    pc = time.perf_counter_ns
    b = OrderBook(record_events=False)
    for c in cmds:
        t0 = pc()
        ev = b.process(c)
        dt = pc() - t0
        if any(e.type.value == "TRADE" for e in ev):
            k = 1
        elif ev and ev[-1].type.value == "REJECT":
            k = 5
        elif isinstance(c, NewLimit):
            k = 0
        elif isinstance(c, Cancel):
            k = 2
        elif isinstance(c, Modify):
            k = 3
        else:
            k = 4
        cls_py.append(k)
        lat_py.append(dt)
    t = cpp_engine.CppOrderBook().run_batch(arr, timed=True)
    py_over = float(np.median([(pc() - pc()) * -1 for _ in range(20000)]))
    return {"python": (np.array(cls_py), np.array(lat_py)), "cpp": (t["class"], t["latency_ns"]),
            "overhead_ns": {"python": py_over, "cpp": cpp_engine._lob_cpp.timer_overhead_ns(200000)}}


def memory_probe(arr: np.ndarray, tmp: Path) -> list[dict]:
    path = tmp / "mem_cmds.npy"
    np.save(path, arr)
    rows = []
    for impl in ("python", "cpp"):
        r = subprocess.run([sys.executable, "-m", "benchmarks._memory_probe", impl, str(path)], cwd=REPO, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr)
        rows.append(json.loads(r.stdout.strip().splitlines()[-1]))
    return rows


def run(ctx: RunContext) -> None:
    if not cpp_engine.available():
        raise SystemExit("C++ extension not built - run scripts/build_cpp.sh first")
    t0 = time.time()
    n_cmds = 20_000 if ctx.quick else 200_000
    reps = 1 if ctx.quick else 5
    seed = ctx.seed
    results = []
    for name, p in WORKLOADS.items():
        r = run_workload(name, WorkloadParams(**{**p.__dict__, "n_commands": n_cmds}), seed, reps, latency=True)
        results.append(r)
        m = {k: statistics.median(v) for k, v in r["times"].items()}
        print(f"{name:12s} resting(mean)={r['stats']['mean_resting_orders']:>7.0f}  python {r['n'] / m['python'] / 1e3:8.0f}k/s   cpp_percall {m['python'] / m['cpp_percall']:5.1f}x"
              f"   cpp_wrapper {m['python'] / m['cpp_wrapper']:5.2f}x   cpp_batch_ev {m['python'] / m['cpp_batch_ev']:6.1f}x   cpp_batch {m['python'] / m['cpp_batch']:6.1f}x")
    # ---- flat tables
    thr, sp, lat_rows, wl = [], [], [], []
    for r in results:
        wl.append({"workload": r["name"], **r["stats"]})
        med = {k: statistics.median(v) for k, v in r["times"].items()}
        for k, v in r["times"].items():
            thr.append({"workload": r["name"], "impl": k, "median_s": med[k], "min_s": min(v), "max_s": max(v), "reps": len(v),
                        "commands_per_s": r["n"] / med[k], "new_orders_per_s": r["n_new"] / med[k], "trades_per_s": r["trades"] / med[k],
                        "events_per_s": r["events"] / med[k]})
        for k in ("cpp_percall", "cpp_wrapper", "cpp_batch", "cpp_batch_ev"):
            ratios = [a / b for a, b in zip(r["times"]["python"], r["times"][k])]
            sp.append({"workload": r["name"], "impl": k, "speedup_median": statistics.median(ratios), "speedup_min": min(ratios),
                       "speedup_max": max(ratios), "speedup_ratio_of_medians": med["python"] / med[k]})
        lat = r["latency"]
        for impl, (cls, ns) in lat.items():
            if impl == "overhead_ns":
                continue
            ov = lat["overhead_ns"]["python" if impl == "python" else "cpp"]
            for code, label in CLASS_NAMES.items():
                x = ns[cls == code].astype(float)
                if len(x) < 20:
                    continue
                lat_rows.append({"workload": r["name"], "impl": impl, "class": label, "n": len(x), "median_ns": float(np.median(x)),
                                 "mean_ns": float(x.mean()), "p99_ns": float(np.quantile(x, 0.99)), "timer_overhead_ns": ov,
                                 "median_ns_minus_overhead": max(0.0, float(np.median(x)) - ov)})
    # ---- memory (resting-book size: the high_flow workload accumulates the largest book)
    with tempfile.TemporaryDirectory() as td:
        arr = cpp_engine.commands_to_array(generate_workload(seed, WorkloadParams(**{**WORKLOADS["high_flow"].__dict__, "n_commands": n_cmds})))
        mem = memory_probe(arr, Path(td))
    print("memory (high_flow): " + "  ".join(f"{m['impl']}: {m['resting_orders']} orders, {m['bytes_per_resting_order']:.0f} B/order" for m in mem))
    save_csv(ctx.out_dir / "throughput.csv", thr)
    save_csv(ctx.out_dir / "speedup.csv", sp)
    save_csv(ctx.out_dir / "latency.csv", lat_rows)
    save_csv(ctx.out_dir / "memory.csv", mem)
    save_csv(ctx.out_dir / "workloads.csv", wl)
    save_json(ctx.out_dir / "summary.json", {"build": build_info(), "n_commands": n_cmds, "reps": reps})
    _plot(ctx, thr, sp, lat_rows)
    write_provenance(ctx, {"n_commands": n_cmds, "reps": reps, "workloads": {k: v for k, v in WORKLOADS.items()}, "build": build_info(),
                           "note": "timings are hardware dependent and not bit-reproducible; workloads, seeds and correctness checks are"},
                     dataset="synthetic command streams (engine.workload)",
                     files=["throughput.csv", "speedup.csv", "latency.csv", "memory.csv", "workloads.csv", "benchmark.png"],
                     elapsed_s=time.time() - t0)


def _plot(ctx: RunContext, thr: list[dict], sp: list[dict], lat: list[dict]) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    names = list(WORKLOADS)
    impls = ["python", "cpp_wrapper", "cpp_percall", "cpp_batch_ev", "cpp_batch"]
    colors = dict(zip(impls, ("tab:gray", "tab:orange", "tab:olive", "tab:green", "tab:blue")))
    w = 0.16
    x = np.arange(len(names))
    for i, impl in enumerate(impls):
        y = [next(r["commands_per_s"] for r in thr if r["workload"] == n and r["impl"] == impl) for n in names]
        ax[0].bar(x + (i - 2) * w, y, w, color=colors[impl], label=impl)
    ax[0].set(yscale="log", xticks=x, xticklabels=names, ylabel="commands / s", title="Throughput")
    ax[0].set_ylim(top=ax[0].get_ylim()[1] * 6); ax[0].legend(fontsize=7, ncol=3, loc="upper left")
    ax[0].tick_params(axis="x", labelrotation=25)
    for i, impl in enumerate(impls[1:]):
        rows = [next(r for r in sp if r["workload"] == n and r["impl"] == impl) for n in names]
        y = np.array([r["speedup_median"] for r in rows])
        ax[1].bar(x + (i - 1.5) * 0.2, y, 0.2, color=colors[impl], label=impl,
                  yerr=[y - [r["speedup_min"] for r in rows], [r["speedup_max"] for r in rows] - y], capsize=2)
    ax[1].axhline(1, color="k", lw=0.8); ax[1].set(yscale="log", xticks=x, xticklabels=names, ylabel="speedup over Python (x)", title="Speedup (median, min-max over reps)")
    ax[1].tick_params(axis="x", labelrotation=25)
    ax[1].set_ylim(top=ax[1].get_ylim()[1] * 6); ax[1].legend(fontsize=7, ncol=2, loc="upper left")
    classes = ["add", "match", "cancel", "modify"]
    for j, impl in enumerate(("python", "cpp")):
        y = [np.mean([r["median_ns"] for r in lat if r["impl"] == impl and r["class"] == c]) for c in classes]
        ax[2].bar(np.arange(4) + (j - 0.5) * 0.35, y, 0.35, color="tab:gray" if impl == "python" else "tab:blue", label=impl)
    ax[2].set(yscale="log", xticks=range(4), xticklabels=classes, ylabel="median latency (ns), mean over workloads", title="Per-command latency (timer overhead included)")
    ax[2].legend()
    fig.tight_layout(); fig.savefig(ctx.out_dir / "benchmark.png", dpi=140); plt.close(fig)
