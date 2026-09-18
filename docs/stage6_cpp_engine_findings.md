# Stage 6 - C++ engine, pybind11 bindings, differential testing, benchmarks

## Build and run

```bash
pip install -e ".[dev]"                 # numpy/scipy/pandas/matplotlib + pytest, hypothesis, pybind11, cmake, ninja
scripts/build_cpp.sh                    # -> build/ (lobcore, lob_tests) and engine/_lob_cpp*.so
.venv/bin/ctest --test-dir build        # C++ unit + fuzz tests (20 tests)
pytest tests/integration/test_differential.py       # Python vs C++ differential tests
python -m experiments.run benchmark_engine --seed 42
python -m experiments.run benchmark_complexity --seed 42
```

## What was built (`engine/cpp/`)

| file | role |
|---|---|
| `types.hpp` | `Event`, reason/event codes, snapshot payload (mirror of `engine/events.py`) |
| `orders.hpp/.cpp` | `Order` node (intrusive FIFO links, level back-pointer) and a free-list `OrderPool` (no `malloc` on add/cancel) |
| `price_levels.hpp` | `PriceLevel` (O(1) push/unlink/reduce) and `SideMap = std::map<i64, PriceLevel>` keyed by a *priority key* so `begin()` is always the best level |
| `order_book.hpp/.cpp` | commands (limit GTC/IOC/post-only, market, cancel, cancel-replace, snapshot), queries, `validate()` |
| `matching.cpp` | the price-time matcher |
| `bindings.cpp` | pybind11 module `engine._lob_cpp` |
| `tests/test_engine.cpp` | 20 dependency-free unit tests + a 300-seed fuzz that runs `validate()` after every command |

Semantics are a line-by-line port of the Python reference: same event grammar and reason codes, size-decrease keeps priority
while reprice/size-increase requeues, a marketable modify executes as `CANCEL(REPLACE)` + fresh arrival, lifetime-unique ids,
no self-trade prevention. Order-id index: `unordered_map<id, Order*>`; lifetime-uniqueness set: `unordered_set<id>`.
Events are optional: with no sink and no checksum they are *counted but never constructed* (the "lean" mode used to measure
raw matching speed); with a sink they are materialised; an order-sensitive rolling checksum can be maintained independently.

`engine/cpp_engine.py` wraps the extension as `CppOrderBook`, a drop-in for the Python `OrderBook` (events come back as
`engine.events.Event`), and adds a batch interface (`run_batch`) over a packed `(n, 8)` int64 command array. `Simulator` now
accepts `book_factory=`, so a whole simulation can run on either engine.

## Correctness evidence (`tests/integration/test_differential.py`, 23 tests)

Identical command sequence -> Python engine / C++ engine -> compare state, trades, executions, remaining quantities, events.

* **Batch:** 9 regimes x 300 seeds = 2,700 randomized sequences x 400 commands; the full event array, its checksum, the final
  L3 book, trade counts and sequence numbers must all match. (Regimes: default, few/many levels, cancel-, match- and
  modify-heavy, noisy with invalid commands/snapshots/post-only/IOC, huge book, dense crossing.)
* **Stepwise:** 9 regimes x 60 seeds x 200 commands: after *every* command compare events, L3 state, best bid/ask/spread/mid,
  L2 depth, imbalance, volume, `queue_position`, `get_order`, `level_qty`, and run the C++ invariant validator.
* **Adversarial:** 250 hypothesis-generated sequences in a tiny id/price universe (dense collisions, crossings, duplicates).
* **Cross-replay:** the *Python* replayer applies the *C++* audit trail (including SNAPSHOT checkpoints) and rebuilds the C++ book.
* **End-to-end:** a 30 s simulation (noise + informed flow + Avellaneda-Stoikov maker) run with each engine as the true book
  gives a byte-identical event stream, identical final book, samples and trade tape.
* **Do the tests bite?** I injected 9 deliberate bugs into the C++ (limit-price and post-only boundary off-by-ones, LIFO instead
  of FIFO, modify never keeping priority, market duplicate-id acceptance, trade reporting the wrong remaining quantity, empty
  levels not removed, post-only check skipped on crossing modify, reversed FIFO in snapshots). All 9 were detected, 8 within
  ~0.5 s; the slowest (market-order duplicate ids) needs the "noisy" regime (9.5 s). One earlier attempt hung because my mutant
  itself created a cyclic list - that was a badly written mutant, not a test-suite gap; it was redone correctly.

## Benchmark (Apple M4, macOS, clang 17, `-O3 -DNDEBUG`, Python 3.13; 200,000 commands per workload, 5 interleaved repetitions)

Before timing, every workload is verified to produce the *identical event stream* on both engines. Workloads are measured, not
just named (`results/benchmark_engine/workloads.csv`): mean resting orders / active price levels are ~12 / 10 (`low_flow`),
~52,600 / 437 (`high_flow`), ~9,700 / 5,300 (`many_levels`), ~810 / 11 (`few_levels`; the mid drifts, so it is not literally 2 levels),
~35 / 28 with 44% cancels (`high_cancel`), and 0.52 trades per command (`high_match`). "Flow" means book regime: engine cost does not
depend on the wall-clock arrival rate.

**Throughput (commands/s)** - Python: 417k-548k; C++ per-call with raw tuples: 2.3M-3.0M; C++ batch with events materialised:
4.7M-9.1M; C++ batch lean: 6.0M-14.5M. At `high_match`: Python 417k commands/s and 215k trades/s vs C++ batch 9.0M / 4.7M.

**Speedup over Python** (median of paired repetitions; the min-max over the 5 repetitions is within about +-5% for most workloads, but `high_flow` shows a few high outlier repetitions of up to ~+25%, so treat its exact ratio as +-10%):

| how C++ is driven | speedup range over the 6 workloads |
|---|---|
| batch, lean (no events built) | 14x-29x |
| batch, events materialised (fairest engine-to-engine: Python also builds `Event`s) | 11x-18x |
| one pybind call per command, raw tuples | 5x-6x |
| one call per command through the `CppOrderBook` wrapper (builds `Event` objects) | **0.9x-1.0x** |

**The honest reading:** the C++ core is 11-18x faster than the Python core doing the same work, but *from Python* that gain is
only realised if the interface is coarse-grained. Converting each command's events into Python objects costs as much as the
Python engine's whole matching step, so the drop-in wrapper is no faster than pure Python. For the simulator, whose loop is
per-command Python, the C++ book therefore gives correctness parity but no speedup; a speedup would require moving the
participant logic or batching commands.

**Speedup depends on the regime:** it is smallest with many price levels (13.6x lean) and largest with few (28.8x). Plausible cause
(not verified with counters): `std::map` node chasing/cache misses at ~5,000 levels vs Python's cheap contiguous `list.insert`.

**Latency per command** (native timer in C++, `perf_counter_ns` in Python; timer overhead 13-14 ns C++ / 41 ns Python, included).
Python medians: add 1.9-2.2 us, cancel 1.5-1.8 us, modify 2.1-2.4 us, match 3.6-6.5 us (p99 up to 15 us). C++ medians: add 42-125 ns,
cancel 42-83 ns, modify 42-166 ns, match 83-208 ns (p99 84-625 ns). **Caveat:** the macOS clock ticks in ~42 ns steps, so C++ medians
are quantised (42, 83, 125 ns...); read them as "under ~0.2 us", not to the nanosecond.

**Memory** (RSS growth after building a 105,547-order book from the `high_flow` stream, fresh process per engine, 3 runs):
Python 227-243 B per resting order, C++ 199-207 B. Only ~15% smaller: both carry hash containers (order index and the lifetime
id set for *all* 200,000 ids) and C++ adds `std::map` nodes. RSS via `ps` is coarse (+-7% run to run).

## Complexity, checked empirically (`benchmark_complexity`)

| operation | Python | C++ |
|---|---|---|
| match k orders | linear: slope 0.96 (1.00 upper half) | linear: 0.91 (0.99) |
| cancel by id, N = 1k -> 400k | flat: 1.4 -> 1.7 us (slope 0.03) | 0.039 -> 0.28 us (slope 0.35) |
| add+cancel of a new interior level, P = 10 -> 100k | 3.4 -> 32 us (slope 0.22; 0.46 in the upper half) | 0.10 -> 0.43 us (slope 0.15) |

* Match is O(k) in both. Cancel is algorithmically O(1) in both (hash lookup + pointer unlink), flat in Python; the C++ time
  grows with N, which I attribute to cache misses as the index outgrows the cache (a hypothesis - not measured with hardware counters).
* Add is ~O(log P) in C++ (0.15 slope over four decades). In Python the documented O(P) `list.insert` memmove is *measurable only at
  very large P*: negligible up to ~10^3-10^4 levels, then 5x the cost from 10^4 to 10^5 levels.

## Limitations

* One machine, one compiler, one Python version; timings are not bit-reproducible (workloads, seeds and equivalence checks are).
* The C++ engine has no event *replay* (`apply_event`); replay lives in the Python reference and is cross-checked against C++ events.
* Synthetic command streams: real feeds have bursts and price-time structure that may change cache behaviour and the speedups.
* The lifetime-unique-id set makes memory grow with every id ever seen (same in both engines).
