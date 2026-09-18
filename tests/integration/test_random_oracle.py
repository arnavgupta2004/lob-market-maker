"""Randomized / property-based verification of the Python engine.

After *every* command we check (1) structural invariants, (2) exact agreement with the naive
oracle on emitted events and resulting L3 state, and (3) that applying the emitted events to
an independent book reproduces the state.
"""
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from engine.commands import Cancel, Modify, NewLimit, NewMarket, SnapshotRequest
from engine.common import EventType, Side
from engine.python.order_book import OrderBook
from engine.workload import WorkloadParams, generate_workload
from tests.oracle import NaiveBook


def check_sequence(cmds):
    book = OrderBook(record_events=True)
    mirror = OrderBook()
    oracle = NaiveBook()
    for c in cmds:
        events = book.process(c)
        expected = oracle.process(c)
        got = [("SNAPSHOT",) if e.type is EventType.SNAPSHOT else e.core() for e in events]
        assert got == expected, f"event mismatch on {c}"
        book.validate()
        assert book.state() == oracle.state(), f"state mismatch after {c}"
        for e in events:
            mirror.apply_event(e)
        assert mirror.state() == book.state()
        # derived quantities are consistent with the state
        bids, asks = book.depth()
        assert book.best_bid() == (bids[0][0] if bids else None)
        assert book.best_ask() == (asks[0][0] if asks else None)
        assert sum(q for _, q, _ in bids) + sum(q for _, q, _ in asks) == sum(
            n[1] for side in book.state().values() for _, orders in side for n in orders)
    return book


# --- seeded workloads at several regimes (few/many levels, cancel-heavy, match-heavy, noisy)
REGIMES = {
    "default": WorkloadParams(n_commands=600),
    "few_levels": WorkloadParams(n_commands=600, price_range=2, p_aggressive=0.2),
    "many_levels": WorkloadParams(n_commands=600, price_range=500),
    "cancel_heavy": WorkloadParams(n_commands=600, p_cancel=0.6, p_limit=0.3, p_modify=0.05),
    "match_heavy": WorkloadParams(n_commands=600, p_aggressive=0.4, p_market=0.2, max_qty=20),
    "modify_heavy": WorkloadParams(n_commands=600, p_modify=0.4, p_cancel=0.1),
    "noisy": WorkloadParams(n_commands=600, p_invalid=0.15, p_snapshot=0.02, p_post_only=0.2, p_ioc=0.2),
}


@pytest.mark.parametrize("regime", REGIMES)
@pytest.mark.parametrize("seed", range(12))
def test_seeded_workloads_match_oracle(regime, seed):
    check_sequence(generate_workload(seed, REGIMES[regime]))


# --- hypothesis: tiny id/price universe => dense collisions, crossings, duplicates, unknowns
ids = st.integers(1, 60)
prices = st.integers(0, 8)
qtys = st.integers(0, 6)
sides = st.sampled_from([Side.BUY, Side.SELL])
commands = st.one_of(
    st.builds(NewLimit, order_id=ids, side=sides, price=prices, qty=qtys, ts=st.just(0),
              owner=st.integers(0, 3), post_only=st.booleans(), ioc=st.booleans()),
    st.builds(NewMarket, order_id=ids, side=sides, qty=qtys, ts=st.just(0), owner=st.integers(0, 3)),
    st.builds(Cancel, order_id=ids),
    st.builds(Modify, order_id=ids, new_price=prices, new_qty=qtys),
    st.just(SnapshotRequest()),
)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(commands, max_size=150))
def test_hypothesis_sequences_match_oracle(cmds):
    check_sequence(cmds)


@settings(max_examples=100, deadline=None)
@given(st.lists(commands, max_size=150))
def test_hypothesis_book_never_crossed_and_ids_never_reused(cmds):
    book = OrderBook(record_events=True)
    for c in cmds:
        book.process(c)
        b, a = book.best_bid(), book.best_ask()
        assert b is None or a is None or b < a
    adds = [e.order_id for e in book.events if e.type is EventType.ADD]
    # an id may ADD at most once (modify-crossing re-ADDs the same id only after CANCEL(REPLACE))
    replaced = {e.order_id for e in book.events if e.type is EventType.CANCEL and e.reason == "REPLACE"}
    dup = {i for i in adds if adds.count(i) > 1}
    assert dup <= replaced


@settings(max_examples=150, deadline=None)
@given(st.lists(commands, max_size=150))
def test_hypothesis_quantity_conservation(cmds):
    """For every accepted order: qty = executed + (rested | expired). Trades net to zero
    resting quantity: each TRADE removes exactly its qty from a maker."""
    book = OrderBook(record_events=True)
    for c in cmds:
        resting_before = sum(n.qty for n in book._index.values())
        events = book.process(c)
        traded = sum(e.qty for e in events if e.type is EventType.TRADE)
        if isinstance(c, (NewLimit, NewMarket)) and events[-1].type is not EventType.REJECT:
            tail = sum(e.qty for e in events if e.type in (EventType.ADD, EventType.EXPIRE))
            assert traded + tail == c.qty
            rested = sum(e.qty for e in events if e.type is EventType.ADD)
            assert sum(n.qty for n in book._index.values()) == resting_before - traded + rested
