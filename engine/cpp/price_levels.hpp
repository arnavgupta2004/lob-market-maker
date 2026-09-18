// Price level (FIFO of orders) and the per-side price -> level map.
#pragma once
#include <map>

#include "orders.hpp"

namespace lob {

struct PriceLevel {
    i64 price = 0;
    Order* head = nullptr;
    Order* tail = nullptr;
    i64 total_qty = 0;
    i64 count = 0;

    // Append at the back of the queue (lowest time priority). O(1).
    void push_back(Order* o) noexcept {
        o->level = this;
        o->prev = tail;
        o->next = nullptr;
        if (tail) tail->next = o; else head = o;
        tail = o;
        total_qty += o->qty;
        ++count;
    }
    // Unlink from anywhere in the queue. O(1).
    void remove(Order* o) noexcept {
        if (o->prev) o->prev->next = o->next; else head = o->next;
        if (o->next) o->next->prev = o->prev; else tail = o->prev;
        total_qty -= o->qty;
        --count;
        o->prev = o->next = nullptr;
        o->level = nullptr;
    }
    // Reduce quantity in place (keeps queue position). O(1).
    void reduce(Order* o, i64 delta) noexcept {
        o->qty -= delta;
        total_qty -= delta;
    }
};

// Keyed by a *priority key* so begin() is always the best level:
//   bids: key = -price (highest price first), asks: key = price (lowest price first).
// std::map is node based, so PriceLevel addresses stay valid until the level is erased.
// insert O(log P); erase best O(1) amortised; erase by key O(log P).
using SideMap = std::map<i64, PriceLevel>;

inline i64 side_key(int side, i64 price) noexcept { return side == BUY ? -price : price; }

}  // namespace lob
