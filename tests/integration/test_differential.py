"""Differential testing: the C++ engine must behave *identically* to the Python reference.

    identical command sequence -> Python engine / C++ engine -> compare
        book state, trades, executions, remaining quantities, and the full event stream

Layers (from cheapest/widest to most detailed):
  * batch: thousands of long seeded sequences, full event array + checksum + final state;
  * stepwise: after EVERY command compare events, L3 state, top-of-book queries and queue positions, and run the
    C++ invariant validator;
  * adversarial: hypothesis-generated sequences in a tiny id/price universe (dense collisions/crossings);
  * cross-replay: events produced by C++ are applied by the *Python* replayer and must rebuild the C++ book;
  * end-to-end: an entire simulation (noise + informed flow + a market maker) run on each engine gives a
    byte-identical event stream.
"""
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from engine import cpp_engine
from engine.commands import NewLimit
from engine.common import Side
from engine.events import stream_hash
from engine.python.order_book import OrderBook
from engine.replay import replay_events
from engine.workload import WorkloadParams, generate_workload
from tests.integration.test_random_oracle import REGIMES, commands

pytestmark = pytest.mark.skipif(not cpp_engine.available(), reason="C++ extension not built (scripts/build_cpp.sh)")
CppOrderBook = cpp_engine.CppOrderBook

BATCH_REGIMES = {
    **REGIMES,
    "huge_book": WorkloadParams(n_commands=400, price_range=2000, p_limit=0.8, p_cancel=0.1, p_modify=0.05, p_market=0.02),
    "dense_cross": WorkloadParams(n_commands=400, price_range=1, p_aggressive=0.6, p_market=0.15, max_qty=3),
}


# --------------------------------------------------------------------------------------------- batch
@pytest.mark.parametrize("regime", BATCH_REGIMES)
def test_batch_thousands_of_sequences_identical(regime):
    """~300 sequences per regime x 9 regimes = ~2,700 randomized sequences, each compared in full."""
    params = BATCH_REGIMES[regime]
    n_seq = 300
    for seed in range(1000, 1000 + n_seq):
        cmds = generate_workload(seed, params)
        py = OrderBook(record_events=True)
        py.process_all(cmds)
        cpp = CppOrderBook()
        res = cpp.run_batch(cpp_engine.commands_to_array(cmds), collect_events=True, checksum=True)
        py_events = cpp_engine.events_to_array(py.events)
        assert np.array_equal(res["events"], py_events), f"{regime} seed {seed}: event stream differs"
        assert res["checksum"] == cpp_engine.events_checksum(py_events)
        assert cpp.state() == py.state(), f"{regime} seed {seed}: final book differs"
        assert res["n_trades"] == sum(1 for e in py.events if e.type.value == "TRADE")
        assert res["final_seq"] == py.seq and len(cpp) == len(py)


def test_batch_is_resumable_and_counters_are_consistent():
    cmds = generate_workload(7, WorkloadParams(n_commands=2000, p_invalid=0.03))
    arr = cpp_engine.commands_to_array(cmds)
    whole, split = CppOrderBook(), CppOrderBook()
    a = whole.run_batch(arr, collect_events=True)
    b1 = split.run_batch(arr[:900], collect_events=True)
    b2 = split.run_batch(arr[900:], collect_events=True)
    assert np.array_equal(a["events"], np.vstack([b1["events"], b2["events"]]))
    assert a["n_events"] == b1["n_events"] + b2["n_events"] and b2["first_seq"] == b1["final_seq"] + 1
    assert whole.state() == split.state()
    # counters do not depend on whether events are materialised
    lean = CppOrderBook().run_batch(arr, collect_events=False, checksum=False)
    assert (lean["n_events"], lean["n_trades"], lean["traded_qty"], lean["n_rejects"]) == (
        a["n_events"], a["n_trades"], a["traded_qty"], a["n_rejects"])
    assert lean["events"] is None and lean["checksum"] is None


def test_timed_batch_classifies_commands_and_matches_untimed_result():
    cmds = generate_workload(3, WorkloadParams(n_commands=3000, p_invalid=0.02, p_aggressive=0.2))
    arr = cpp_engine.commands_to_array(cmds)
    t = CppOrderBook().run_batch(arr, collect_events=True, timed=True)
    u = CppOrderBook().run_batch(arr, collect_events=True)
    assert np.array_equal(t["events"], u["events"])  # timing does not perturb behaviour
    assert t["latency_ns"].shape == (3000,) and (t["latency_ns"] >= 0).all()
    cls = t["class"]
    ops = arr[:, 0]
    assert set(np.unique(cls)) <= {0, 1, 2, 3, 4, 5}
    assert (cls[(ops == cpp_engine.OP_CANCEL) & (cls != 5)] == 2).all()
    assert (cls[(ops == cpp_engine.OP_MODIFY) & (~np.isin(cls, [1, 5]))] == 3).all()
    assert t["n_trades"] > 0 and (cls == 1).sum() > 0 and (cls == 0).sum() > 0


def test_batch_rejects_malformed_input():
    with pytest.raises(Exception):
        CppOrderBook().run_batch(np.zeros((3, 5), dtype=np.int64))
    bad = np.zeros((1, 8), dtype=np.int64)
    bad[0, 0] = 99
    with pytest.raises(Exception):
        CppOrderBook().run_batch(bad)


# ----------------------------------------------------------------------------------------- stepwise
def check_stepwise(cmds):
    py, cpp = OrderBook(record_events=True), CppOrderBook(record_events=True)
    for c in cmds:
        pe, ce = py.process(c), cpp.process(c)
        assert pe == ce, f"events differ on {c}:\n py={pe}\ncpp={ce}"
        cpp.validate()
        assert py.state() == cpp.state(), f"state differs after {c}"
        assert (py.best_bid(), py.best_ask(), py.spread(), py.mid_price()) == (
            cpp.best_bid(), cpp.best_ask(), cpp.spread(), cpp.mid_price())
        assert py.depth(3) == cpp.depth(3) and py.depth() == cpp.depth()
        assert py.imbalance(1) == cpp.imbalance(1) and py.imbalance(5) == cpp.imbalance(5)
        assert py.volume(Side.BUY, 2) == cpp.volume(Side.BUY, 2) and len(py) == len(cpp)
        for oid in list(py._index)[:4]:
            assert py.queue_position(oid) == cpp.queue_position(oid)
            n, v = py.get_order(oid), cpp.get_order(oid)
            assert (n.order_id, n.side, n.price, n.qty, n.owner, n.post_only) == tuple(v)
            assert oid in cpp
        if isinstance(c, NewLimit):
            assert py.level_qty(c.side, c.price) == cpp.level_qty(c.side, c.price)
            for owner in (0, 1, 2, 3):
                assert py.owner_orders_at(c.side, c.price, owner) == cpp.owner_orders_at(c.side, c.price, owner)
    assert py.events == cpp.events and py.seq == cpp.seq
    return py, cpp


@pytest.mark.parametrize("regime", REGIMES)
def test_stepwise_identical_after_every_command(regime):
    params = WorkloadParams(**{**REGIMES[regime].__dict__, "n_commands": 200})
    for seed in range(60):
        check_stepwise(generate_workload(seed, params))


@settings(max_examples=250, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(__import__("hypothesis").strategies.lists(commands, max_size=120))
def test_stepwise_adversarial_hypothesis_sequences(cmds):
    check_stepwise(cmds)


# ---------------------------------------------------------------------------------------- cross replay
def test_python_replay_of_cpp_events_rebuilds_the_cpp_book():
    cmds = generate_workload(11, WorkloadParams(n_commands=3000, p_invalid=0.03, p_snapshot=0.005))
    cpp = CppOrderBook(record_events=True)
    cpp.process_all(cmds)
    rebuilt = replay_events(cpp.events)  # reference implementation applies the C++ audit trail (incl. snapshots)
    assert rebuilt.state() == cpp.state() and rebuilt.seq == cpp.seq
    py = OrderBook(record_events=True)
    py.process_all(cmds)
    assert stream_hash(py.events) == stream_hash(cpp.events)


def test_listeners_and_recording_behave_like_the_reference():
    got_py, got_cpp = [], []
    py, cpp = OrderBook(), CppOrderBook()
    py.subscribe(got_py.append)
    cpp.subscribe(got_cpp.append)
    for c in generate_workload(5, WorkloadParams(n_commands=500)):
        py.process(c)
        cpp.process(c)
    assert got_py == got_cpp and cpp.events == [] and len(got_cpp) > 100


# ------------------------------------------------------------------------------------- end-to-end
def test_full_simulation_is_byte_identical_on_both_engines():
    """Noise + informed flow + an Avellaneda-Stoikov maker; the true book is Python in one run, C++ in the other."""
    from simulator.market_state.fundamental import FundamentalConfig
    from simulator.order_flow.informed import InformedTrader
    from simulator.order_flow.noise import NoiseTrader
    from simulator.simulator import SimConfig, Simulator
    from strategies.baseline_mm import ASConfig

    def run(factory):
        cfg = SimConfig(seed=13, horizon_s=30, fundamental=FundamentalConfig(sigma=0.03))
        parts = [NoiseTrader(), InformedTrader(), ASConfig(gamma=0.02).build()]
        return Simulator(cfg, parts, book_factory=factory).run()

    a, b = run(None), run(CppOrderBook)
    assert a.n_events == b.n_events > 2000
    assert stream_hash(a.events) == stream_hash(b.events)
    assert a.book.state() == b.book.state()
    assert np.array_equal(a.samples["mid"], b.samples["mid"], equal_nan=True)
    assert all(np.array_equal(a.trades[k], b.trades[k], equal_nan=True) for k in a.trades)
    b.book.validate()
