"""Event serialization, deterministic replay, and audit-trail sufficiency."""
import dataclasses

import pytest

from engine.commands import (
    Cancel, Modify, NewLimit, NewMarket, SnapshotRequest, command_from_dict, command_to_dict,
    read_commands, write_commands,
)
from engine.common import EventType as ET, Side
from engine.events import Event, dumps, read_jsonl, stream_hash, write_jsonl
from engine.python.order_book import OrderBook, ReplayError
from engine.replay import replay_commands, replay_events
from engine.workload import WorkloadParams, generate_workload

PARAMS = WorkloadParams(n_commands=1500, p_invalid=0.05, p_snapshot=0.01)


def test_event_dict_roundtrip_all_types():
    b = OrderBook(record_events=True)
    for c in generate_workload(3, PARAMS):
        b.process(c)
    types = {e.type for e in b.events}
    assert types == set(ET), "workload should exercise every event type"
    for e in b.events:
        assert Event.from_dict(e.to_dict()) == e


def test_command_roundtrip_and_file_io(tmp_path):
    cmds = generate_workload(5, PARAMS)
    for c in cmds:
        assert command_from_dict(command_to_dict(c)) == c
    p = tmp_path / "cmds.jsonl"
    write_commands(p, cmds)
    assert list(read_commands(p)) == cmds


def test_event_file_roundtrip_and_hash(tmp_path):
    b = replay_commands(generate_workload(7, PARAMS))
    p = tmp_path / "events.jsonl"
    write_jsonl(p, b.events)
    back = list(read_jsonl(p))
    assert back == b.events
    assert stream_hash(back) == stream_hash(b.events)


def test_canonical_serialization_is_sparse_and_sorted():
    e = Event(1, 5, ET.CANCEL, order_id=3, side=Side.BUY, price=10, qty=2, reason="USER")
    assert dumps(e) == '{"order_id":3,"price":10,"qty":2,"reason":"USER","seq":1,"side":"B","ts":5,"type":"CANCEL"}'


def test_same_commands_give_byte_identical_event_streams():
    a = replay_commands(generate_workload(11, PARAMS))
    b = replay_commands(generate_workload(11, PARAMS))
    assert stream_hash(a.events) == stream_hash(b.events)
    c = replay_commands(generate_workload(12, PARAMS))
    assert stream_hash(a.events) != stream_hash(c.events)


def test_generator_is_deterministic_given_seed():
    assert generate_workload(1, PARAMS) == generate_workload(1, PARAMS)
    assert generate_workload(1, PARAMS) != generate_workload(2, PARAMS)


@pytest.mark.parametrize("seed", range(8))
def test_replaying_events_reconstructs_identical_book(seed):
    """Stage-2 acceptance test: the event stream alone reproduces the book exactly."""
    live = replay_commands(generate_workload(seed, PARAMS))
    rebuilt = replay_events(live.events)
    assert rebuilt.state() == live.state()
    assert rebuilt.seq == live.seq
    rebuilt.validate()


def test_replay_is_stepwise_identical_not_just_at_the_end():
    live = OrderBook(record_events=True)
    mirror = OrderBook()
    for c in generate_workload(21, PARAMS):
        for ev in live.process(c):
            mirror.apply_event(ev)
        assert mirror.state() == live.state()


def test_replay_from_snapshot_checkpoint():
    cmds = generate_workload(4, WorkloadParams(n_commands=800))
    live = OrderBook(record_events=True)
    live.process_all(cmds[:400])
    snap = live.snapshot(ts=1)
    live.process_all(cmds[400:])
    tail = live.events[live.events.index(snap) + 1:]
    fresh = OrderBook()
    fresh.load_state(snap.book)
    fresh._seq = snap.seq
    replay_events(tail, fresh)
    assert fresh.state() == live.state()


def test_replay_detects_tampered_stream():
    live = replay_commands(generate_workload(2, WorkloadParams(n_commands=400)))
    evs = list(live.events)
    i = next(k for k, e in enumerate(evs) if e.type is ET.TRADE)
    evs[i] = dataclasses.replace(evs[i], qty=evs[i].qty + 1)
    with pytest.raises(ReplayError):
        replay_events(evs)


def test_replay_detects_dropped_event():
    live = replay_commands(generate_workload(2, WorkloadParams(n_commands=400)))
    evs = list(live.events)
    del evs[len(evs) // 2]
    with pytest.raises(ReplayError):
        replay_events(evs)


def test_snapshot_mismatch_is_detected():
    b = OrderBook(record_events=True)
    b.submit_limit(1, Side.BUY, 99, 5)
    snap = b.snapshot()
    other = OrderBook()
    other.apply_event(dataclasses.replace(b.events[0], qty=6))
    with pytest.raises(ReplayError):
        other.apply_event(snap)


def test_explicit_small_scenario_events_are_complete():
    b = OrderBook(record_events=True)
    b.process(NewLimit(1, Side.SELL, 100, 5, ts=1, owner=1))
    b.process(NewLimit(2, Side.SELL, 100, 5, ts=2, owner=2))
    b.process(NewMarket(3, Side.BUY, 7, ts=3, owner=3))
    b.process(Modify(2, 101, 3, ts=4))
    b.process(Cancel(2, ts=5))
    b.process(SnapshotRequest(ts=6))
    assert [e.type for e in b.events] == [ET.ADD, ET.ADD, ET.TRADE, ET.TRADE, ET.MODIFY, ET.CANCEL, ET.SNAPSHOT]
    assert replay_events(b.events).state() == {"bids": [], "asks": []}
