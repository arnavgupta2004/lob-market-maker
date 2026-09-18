// Price-time-priority limit order book (C++ implementation of the Python reference in
// engine/python/order_book.py). Semantics - event grammar, reason codes, cancel/replace priority
// rules, lifetime-unique ids, no self-trade prevention - are identical; tests/integration/
// test_differential.py enforces that.
//
// Complexity (P = active price levels on a side, k = orders/levels consumed by a match):
//   add (rest)   O(log P)    cancel  O(1) after id lookup    match  O(k)
#pragma once
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "orders.hpp"
#include "price_levels.hpp"
#include "types.hpp"

namespace lob {

struct Counters {
    u64 events = 0, trades = 0, rejects = 0;
    i64 traded_qty = 0;
};

class OrderBook {
public:
    explicit OrderBook(std::size_t reserve_orders = 0);

    // ---- commands (all timestamps in ns are passed through to events) ----
    void submit_limit(i64 id, int side, i64 price, i64 qty, i64 ts = 0, i64 owner = 0,
                      bool post_only = false, bool ioc = false);
    void submit_market(i64 id, int side, i64 qty, i64 ts = 0, i64 owner = 0);
    void cancel(i64 id, i64 ts = 0);
    void modify(i64 id, i64 new_price, i64 new_qty, i64 ts = 0);
    void snapshot(i64 ts = 0);

    // ---- event plumbing ----
    // Events are appended to `sink` if non-null. Independently, an order-sensitive checksum over every
    // event can be maintained. With neither, events are counted but never constructed (max throughput).
    void set_sink(std::vector<Event>* sink) noexcept { sink_ = sink; }
    void set_checksum(bool on) noexcept { checksum_on_ = on; }
    u64 checksum() const noexcept { return checksum_; }
    const Counters& counters() const noexcept { return counters_; }
    i64 seq() const noexcept { return seq_; }

    // ---- queries ----
    std::size_t size() const noexcept { return index_.size(); }
    bool contains(i64 id) const { return index_.count(id) != 0; }
    std::optional<i64> best_bid() const;
    std::optional<i64> best_ask() const;
    std::size_t num_levels(int side) const noexcept { return (side == BUY ? bids_ : asks_).size(); }
    // (price, total_qty, n_orders) best first, at most `levels` (0 = all)
    std::vector<std::tuple<i64, i64, i64>> depth(int side, std::size_t levels = 0) const;
    i64 volume(int side, std::size_t levels) const;
    i64 level_qty(int side, i64 price) const;
    // (id, qty) of `owner`'s orders at a price, FIFO order
    std::vector<std::pair<i64, i64>> owner_orders_at(int side, i64 price, i64 owner) const;
    // (quantity ahead, orders ahead) in the order's FIFO; nullopt if unknown id
    std::optional<std::pair<i64, i64>> queue_position(i64 id) const;
    struct OrderInfo { int side; i64 price, qty, owner; bool post_only; };
    std::optional<OrderInfo> order_info(i64 id) const;
    Snapshot state() const;

    // Throws std::logic_error on the first violated structural invariant.
    void validate() const;

private:
    template <class Fill> void emit(EventType type, i64 ts, Fill&& fill);
    void reject(i64 ts, i64 id, Reason r, int side, i64 price, i64 qty, i64 owner);
    bool would_cross(int side, i64 price) const;
    // Match up to `qty` against the opposite side; returns the unfilled remainder. (matching.cpp)
    i64 execute(int side, bool has_limit, i64 limit, i64 qty, i64 taker_id, i64 taker_owner, i64 ts);
    void rest(Order* o);
    void unrest(Order* o);
    SideMap& side_map(int side) { return side == BUY ? bids_ : asks_; }
    const SideMap& side_map(int side) const { return side == BUY ? bids_ : asks_; }
    void hash_event(const Event& e);

    SideMap bids_, asks_;
    std::unordered_map<i64, Order*> index_;
    std::unordered_set<i64> seen_;  // lifetime id uniqueness
    OrderPool pool_;
    i64 seq_ = 0;
    std::vector<Event>* sink_ = nullptr;
    bool checksum_on_ = false;
    u64 checksum_ = 1469598103934665603ULL;
    Event scratch_;
    Counters counters_;
};

// Defined in the header so both translation units (order_book.cpp, matching.cpp) inline it.
template <class Fill>
inline void OrderBook::emit(EventType type, i64 ts, Fill&& fill) {
    ++seq_;
    ++counters_.events;
    if (!sink_ && !checksum_on_) return;  // max-throughput mode: the event is counted, never built
    Event* e;
    if (sink_) {
        sink_->emplace_back();
        e = &sink_->back();
    } else {
        scratch_ = Event{};
        e = &scratch_;
    }
    e->seq = seq_;
    e->ts = ts;
    e->type = type;
    fill(*e);
    if (checksum_on_) hash_event(*e);
}

}  // namespace lob
