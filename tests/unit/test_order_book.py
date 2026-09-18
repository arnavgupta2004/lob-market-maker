"""Unit tests of the Python reference order book, one behaviour per test."""
import pytest

from engine.common import EventType as ET, Reason, Side
from engine.python.order_book import OrderBook

B, S = Side.BUY, Side.SELL


def make(record=True) -> OrderBook:
    return OrderBook(record_events=record)


def trades(events):
    return [e for e in events if e.type is ET.TRADE]


# ------------------------------------------------------------------- priority
def test_fifo_within_price_level():
    b = make()
    b.submit_limit(1, S, 100, 5)
    b.submit_limit(2, S, 100, 5)
    b.submit_limit(3, S, 100, 5)
    ev = b.submit_market(10, B, 7)
    t = trades(ev)
    assert [(x.maker_id, x.qty, x.maker_remaining) for x in t] == [(1, 5, 0), (2, 2, 3)]
    assert b.get_order(2).qty == 3 and b.get_order(3).qty == 5
    b.validate()


def test_price_priority_beats_time_priority():
    b = make()
    b.submit_limit(1, S, 102, 5)  # earliest but worst price
    b.submit_limit(2, S, 100, 5)
    b.submit_limit(3, S, 101, 5)
    t = trades(b.submit_market(10, B, 12))
    assert [(x.maker_id, x.price, x.qty) for x in t] == [(2, 100, 5), (3, 101, 5), (1, 102, 2)]


def test_bid_side_price_priority():
    b = make()
    b.submit_limit(1, B, 98, 5)
    b.submit_limit(2, B, 100, 5)
    b.submit_limit(3, B, 99, 5)
    t = trades(b.submit_market(10, S, 11))
    assert [x.maker_id for x in t] == [2, 3, 1]


def test_multiple_orders_same_price():
    b = make()
    for i in range(1, 6):
        b.submit_limit(i, B, 100, 2)
    assert b.depth()[0] == [(100, 10, 5)]
    assert b.best_bid() == 100 and len(b) == 5


# ------------------------------------------------------------------ execution
def test_partial_fill_of_resting_order_keeps_queue_position():
    b = make()
    b.submit_limit(1, S, 100, 10)
    b.submit_limit(2, S, 100, 10)
    b.submit_market(9, B, 4)
    assert b.get_order(1).qty == 6
    assert b.queue_position(1) == (0, 0)  # still at the front
    assert b.queue_position(2) == (6, 1)


def test_partial_fill_of_incoming_limit_rests_remainder():
    b = make()
    b.submit_limit(1, S, 100, 4)
    ev = b.submit_limit(2, B, 100, 10)
    assert [e.type for e in ev] == [ET.TRADE, ET.ADD]
    assert ev[1].qty == 6 and b.best_bid() == 100 and b.best_ask() is None
    assert b.get_order(2).qty == 6


def test_full_fill_removes_orders_and_levels():
    b = make()
    b.submit_limit(1, S, 100, 5)
    b.submit_limit(2, B, 100, 5)
    assert len(b) == 0 and b.best_bid() is None and b.best_ask() is None
    assert b.depth() == ([], [])


def test_multiple_fills_from_one_market_order_across_levels():
    b = make()
    b.submit_limit(1, S, 100, 3)
    b.submit_limit(2, S, 101, 3)
    b.submit_limit(3, S, 102, 3)
    ev = b.submit_market(9, B, 8)
    assert [(e.price, e.qty) for e in trades(ev)] == [(100, 3), (101, 3), (102, 2)]
    assert b.best_ask() == 102 and b.depth()[1] == [(102, 1, 1)]
    assert all(e.order_id == 9 and e.side is B for e in trades(ev))


def test_execution_at_resting_price_gives_taker_price_improvement():
    b = make()
    b.submit_limit(1, S, 100, 5)
    ev = b.submit_limit(2, B, 105, 5)  # willing to pay 105, executes at 100
    assert trades(ev)[0].price == 100


def test_limit_order_only_matches_up_to_its_limit():
    b = make()
    b.submit_limit(1, S, 100, 5)
    b.submit_limit(2, S, 103, 5)
    b.submit_limit(3, B, 101, 20)
    assert b.best_ask() == 103 and b.best_bid() == 101
    assert b.get_order(3).qty == 15


def test_market_order_exhaustion_discards_remainder():
    b = make()
    b.submit_limit(1, S, 100, 3)
    ev = b.submit_market(2, B, 10)
    assert [e.type for e in ev] == [ET.TRADE, ET.EXPIRE]
    assert ev[1].qty == 7 and ev[1].reason == Reason.MARKET_EXHAUSTED
    assert len(b) == 0  # remainder never rests


def test_market_order_on_empty_book():
    b = make()
    ev = b.submit_market(1, B, 5)
    assert [e.type for e in ev] == [ET.EXPIRE] and ev[0].qty == 5


def test_ioc_remainder_expires_and_does_not_rest():
    b = make()
    b.submit_limit(1, S, 100, 3)
    ev = b.submit_limit(2, B, 100, 10, ioc=True)
    assert [e.type for e in ev] == [ET.TRADE, ET.EXPIRE] and ev[1].reason == Reason.IOC
    assert b.best_bid() is None


# ------------------------------------------------------------- crossed book
def test_book_is_never_crossed_after_aggressive_limit():
    b = make()
    b.submit_limit(1, S, 100, 5)
    b.submit_limit(2, B, 101, 2)
    assert b.best_bid() is None or b.best_bid() < b.best_ask()
    b.validate()


def test_post_only_rejected_when_it_would_cross():
    b = make()
    b.submit_limit(1, S, 100, 5)
    ev = b.submit_limit(2, B, 100, 5, post_only=True)
    assert [e.type for e in ev] == [ET.REJECT] and ev[0].reason == Reason.POST_ONLY_WOULD_CROSS
    assert b.best_bid() is None and b.get_order(1).qty == 5
    ev = b.submit_limit(3, B, 99, 5, post_only=True)  # passive: accepted
    assert [e.type for e in ev] == [ET.ADD]


def test_post_only_and_ioc_is_invalid():
    b = make()
    ev = b.submit_limit(1, B, 100, 5, post_only=True, ioc=True)
    assert ev[0].reason == Reason.INVALID


# ------------------------------------------------------------- top of book
def test_best_bid_ask_update_as_levels_appear_and_vanish():
    b = make()
    b.submit_limit(1, B, 99, 5)
    assert b.best_bid() == 99
    b.submit_limit(2, B, 100, 5)
    assert b.best_bid() == 100
    b.submit_limit(3, S, 103, 5)
    b.submit_limit(4, S, 102, 5)
    assert b.best_ask() == 102
    b.cancel(2)
    assert b.best_bid() == 99
    b.submit_market(9, B, 5)
    assert b.best_ask() == 103


def test_spread_mid_and_none_when_one_sided():
    b = make()
    assert b.spread() is None and b.mid_price() is None
    b.submit_limit(1, B, 99, 5)
    assert b.spread() is None and b.mid_price() is None
    b.submit_limit(2, S, 102, 5)
    assert b.spread() == 3 and b.mid_price() == 100.5


def test_l2_depth_aggregates_and_orders_best_first():
    b = make()
    b.submit_limit(1, B, 99, 3)
    b.submit_limit(2, B, 99, 4)
    b.submit_limit(3, B, 97, 1)
    b.submit_limit(4, S, 101, 2)
    b.submit_limit(5, S, 103, 6)
    bids, asks = b.depth()
    assert bids == [(99, 7, 2), (97, 1, 1)] and asks == [(101, 2, 1), (103, 6, 1)]
    assert b.depth(1) == ([(99, 7, 2)], [(101, 2, 1)])


def test_l3_state_lists_orders_in_fifo_order():
    b = make()
    b.submit_limit(5, B, 99, 3, owner=7)
    b.submit_limit(2, B, 99, 4, owner=8)
    assert b.state()["bids"] == [[99, [[5, 3, 7, 0], [2, 4, 8, 0]]]]


def test_imbalance():
    b = make()
    assert b.imbalance() is None
    b.submit_limit(1, B, 99, 30)
    b.submit_limit(2, S, 101, 10)
    assert b.imbalance(1) == pytest.approx(0.5)
    b.submit_limit(3, S, 102, 20)
    assert b.imbalance(2) == pytest.approx(0.0)


# --------------------------------------------------------------- cancellation
def test_cancel_removes_order_and_reports_quantity():
    b = make()
    b.submit_limit(1, B, 99, 5)
    ev = b.cancel(1)
    assert [(e.type, e.qty, e.price, e.reason) for e in ev] == [(ET.CANCEL, 5, 99, Reason.USER)]
    assert 1 not in b and b.best_bid() is None


def test_cancel_middle_of_queue_preserves_order_of_the_rest():
    b = make()
    for i in (1, 2, 3):
        b.submit_limit(i, S, 100, 1)
    b.cancel(2)
    assert [n.order_id for n in b.asks.best_level()] == [1, 3]
    assert b.depth()[1] == [(100, 2, 2)]
    b.validate()


def test_cancel_unknown_or_filled_order_is_rejected():
    b = make()
    assert b.cancel(42)[0].reason == Reason.UNKNOWN_ORDER
    b.submit_limit(1, S, 100, 1)
    b.submit_market(2, B, 1)
    assert b.cancel(1)[0].reason == Reason.UNKNOWN_ORDER
    b.submit_limit(3, B, 90, 1)
    b.cancel(3)
    assert b.cancel(3)[0].reason == Reason.UNKNOWN_ORDER  # double cancel


def test_empty_price_level_is_removed_on_cancel_and_on_fill():
    b = make()
    b.submit_limit(1, B, 99, 5)
    b.submit_limit(2, B, 98, 5)
    b.cancel(1)
    assert list(b.bids.levels) == [98] and b.best_bid() == 98
    b.submit_limit(3, S, 98, 5)  # fills order 2 fully
    assert len(b.bids) == 0 and b.best_bid() is None
    b.validate()


# ----------------------------------------------------------------- cancel/replace
def test_modify_size_decrease_keeps_priority():
    b = make()
    b.submit_limit(1, B, 99, 10)
    b.submit_limit(2, B, 99, 10)
    ev = b.modify(1, 99, 4)
    assert ev[0].type is ET.MODIFY and ev[0].keeps_priority
    assert b.queue_position(1) == (0, 0) and b.get_order(1).qty == 4
    assert b.depth()[0] == [(99, 14, 2)]


def test_modify_size_increase_loses_priority():
    b = make()
    b.submit_limit(1, B, 99, 10)
    b.submit_limit(2, B, 99, 10)
    ev = b.modify(1, 99, 15)
    assert not ev[0].keeps_priority
    assert b.queue_position(1) == (10, 1) and b.get_order(1).qty == 15


def test_modify_price_change_moves_level_and_loses_priority():
    b = make()
    b.submit_limit(1, B, 99, 5)
    b.submit_limit(2, B, 98, 5)
    b.modify(2, 99, 5)  # joins the back of the 99 level
    assert b.queue_position(2) == (5, 1)
    assert list(b.bids.levels) == [99]  # old level removed
    b.validate()


def test_modify_to_marketable_price_executes_as_cancel_plus_aggressive_arrival():
    b = make()
    b.submit_limit(1, S, 100, 3)
    b.submit_limit(2, B, 95, 10)
    ev = b.modify(2, 100, 10)
    assert [e.type for e in ev] == [ET.CANCEL, ET.TRADE, ET.ADD]
    assert ev[0].reason == Reason.REPLACE and ev[2].qty == 7 and ev[2].price == 100
    assert b.best_bid() == 100 and b.best_ask() is None
    b.validate()


def test_modify_rejections():
    b = make()
    b.submit_limit(1, B, 99, 5)
    assert b.modify(9, 99, 1)[0].reason == Reason.UNKNOWN_ORDER
    assert b.modify(1, 99, 0)[0].reason == Reason.INVALID
    assert b.modify(1, 0, 5)[0].reason == Reason.INVALID
    assert b.modify(1, 99, 5)[0].reason == Reason.NO_CHANGE
    b.submit_limit(2, S, 105, 5)
    b.submit_limit(3, B, 100, 5, post_only=True)
    ev = b.modify(3, 105, 5)  # post-only may not cross via modify either
    assert ev[0].reason == Reason.POST_ONLY_WOULD_CROSS and b.get_order(3).price == 100


# ------------------------------------------------------------------- ids
def test_order_id_uniqueness_is_lifetime():
    b = make()
    b.submit_limit(1, B, 99, 5)
    assert b.submit_limit(1, B, 98, 5)[0].reason == Reason.DUPLICATE_ID
    b.cancel(1)
    assert b.submit_limit(1, B, 98, 5)[0].reason == Reason.DUPLICATE_ID  # even after cancel
    b.submit_limit(2, S, 100, 1)
    b.submit_market(3, B, 1)  # id 3 taker fully filled, never rests
    assert b.submit_market(3, B, 1)[0].reason == Reason.DUPLICATE_ID
    assert len(b) == 0


def test_invalid_orders_are_rejected_without_state_change():
    b = make()
    assert b.submit_limit(1, B, 100, 0)[0].reason == Reason.INVALID
    assert b.submit_limit(2, B, 0, 5)[0].reason == Reason.INVALID
    assert b.submit_market(3, B, 0)[0].reason == Reason.INVALID
    assert len(b) == 0 and b.state() == {"bids": [], "asks": []}


# --------------------------------------------------------------- events
def test_event_sequence_numbers_are_contiguous_and_timestamps_pass_through():
    b = make()
    b.submit_limit(1, S, 100, 5, ts=10)
    b.submit_limit(2, B, 100, 2, ts=20)
    b.cancel(1, ts=30)
    assert [e.seq for e in b.events] == list(range(1, len(b.events) + 1))
    assert [e.ts for e in b.events] == [10, 20, 30]
    assert [e.type for e in b.events] == [ET.ADD, ET.TRADE, ET.CANCEL]


def test_trade_event_carries_full_attribution():
    b = make()
    b.submit_limit(1, S, 100, 5, owner=11)
    (t,) = trades(b.submit_limit(2, B, 101, 3, owner=22))
    assert (t.order_id, t.side, t.owner) == (2, B, 22)
    assert (t.maker_id, t.maker_owner, t.maker_remaining) == (1, 11, 2)
    assert (t.price, t.qty) == (100, 3)


def test_listener_receives_every_event():
    b = make(record=False)
    got = []
    b.subscribe(got.append)
    b.submit_limit(1, S, 100, 1)
    b.submit_market(2, B, 1)
    assert [e.type for e in got] == [ET.ADD, ET.TRADE] and b.events == []


def test_queue_position_unknown_order():
    assert make().queue_position(5) is None


def test_owner_orders_at_returns_only_that_owners_orders_in_fifo_order():
    b = make()
    b.submit_limit(1, B, 99, 3, owner=7)
    b.submit_limit(2, B, 99, 4, owner=8)
    b.submit_limit(3, B, 99, 5, owner=7)
    b.submit_limit(4, B, 98, 1, owner=7)
    assert b.owner_orders_at(B, 99, 7) == [(1, 3), (3, 5)]
    assert b.owner_orders_at(B, 99, 8) == [(2, 4)]
    assert b.owner_orders_at(B, 99, 9) == [] and b.owner_orders_at(B, 50, 7) == [] and b.owner_orders_at(S, 99, 7) == []
    b.submit_market(9, S, 3)  # partial: order 1 (front) filled
    assert b.owner_orders_at(B, 99, 7) == [(3, 5)]
