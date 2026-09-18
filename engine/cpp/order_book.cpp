#include "order_book.hpp"

#include <sstream>

namespace lob {

OrderBook::OrderBook(std::size_t reserve_orders) {
    if (reserve_orders) {
        index_.reserve(reserve_orders);
        seen_.reserve(reserve_orders * 2);
    }
}

void OrderBook::hash_event(const Event& e) {
    // Order-sensitive Horner hash over the 15 words of the event core (see engine/cpp_engine.py).
    constexpr u64 P = 0x100000001b3ULL;
    const i64 words[15] = {e.seq, e.ts, static_cast<i64>(e.type), e.order_id, e.side, e.price, e.qty, e.owner,
                           e.maker_id, e.maker_owner, e.maker_remaining, e.new_price, e.new_qty,
                           static_cast<i64>(e.keeps_priority) | (static_cast<i64>(e.post_only) << 1),
                           static_cast<i64>(e.reason)};
    u64 h = checksum_;
    for (i64 w : words) h = h * P + static_cast<u64>(w);
    checksum_ = h;
}

void OrderBook::reject(i64 ts, i64 id, Reason r, int side, i64 price, i64 qty, i64 owner) {
    ++counters_.rejects;
    emit(EventType::REJECT, ts, [&](Event& e) {
        e.order_id = id; e.side = side; e.price = price; e.qty = qty; e.owner = owner; e.reason = r;
    });
}

bool OrderBook::would_cross(int side, i64 price) const {
    if (side == BUY) return !asks_.empty() && price >= asks_.begin()->second.price;
    return !bids_.empty() && price <= bids_.begin()->second.price;
}

void OrderBook::rest(Order* o) {
    SideMap& m = side_map(o->side);
    auto it = m.try_emplace(side_key(o->side, o->price)).first;
    it->second.price = o->price;
    it->second.push_back(o);
    index_[o->id] = o;
}

void OrderBook::unrest(Order* o) {
    PriceLevel* lvl = o->level;
    const i64 key = side_key(o->side, o->price);
    lvl->remove(o);
    if (lvl->count == 0) side_map(o->side).erase(key);
    index_.erase(o->id);
}

// ------------------------------------------------------------------------------------ commands
void OrderBook::submit_limit(i64 id, int side, i64 price, i64 qty, i64 ts, i64 owner, bool post_only, bool ioc) {
    if (!seen_.insert(id).second) {
        reject(ts, id, Reason::DUPLICATE_ID, side, price, qty, owner);
        return;
    }
    if (qty < 1 || price < 1 || (post_only && ioc)) {
        reject(ts, id, Reason::INVALID, side, price, qty, owner);
        return;
    }
    if (post_only && would_cross(side, price)) {
        reject(ts, id, Reason::POST_ONLY_WOULD_CROSS, side, price, qty, owner);
        return;
    }
    const i64 remaining = execute(side, true, price, qty, id, owner, ts);
    if (remaining > 0) {
        if (ioc) {
            emit(EventType::EXPIRE, ts, [&](Event& e) {
                e.order_id = id; e.side = side; e.price = price; e.qty = remaining; e.owner = owner; e.reason = Reason::IOC;
            });
        } else {
            rest(pool_.acquire(id, side, price, remaining, owner, post_only));
            emit(EventType::ADD, ts, [&](Event& e) {
                e.order_id = id; e.side = side; e.price = price; e.qty = remaining; e.owner = owner; e.post_only = post_only;
            });
        }
    }
}

void OrderBook::submit_market(i64 id, int side, i64 qty, i64 ts, i64 owner) {
    if (!seen_.insert(id).second) {
        reject(ts, id, Reason::DUPLICATE_ID, side, 0, qty, owner);
        return;
    }
    if (qty < 1) {
        reject(ts, id, Reason::INVALID, side, 0, qty, owner);
        return;
    }
    const i64 remaining = execute(side, false, 0, qty, id, owner, ts);
    if (remaining > 0) {
        emit(EventType::EXPIRE, ts, [&](Event& e) {
            e.order_id = id; e.side = side; e.qty = remaining; e.owner = owner; e.reason = Reason::MARKET_EXHAUSTED;
        });
    }
}

void OrderBook::cancel(i64 id, i64 ts) {
    auto it = index_.find(id);
    if (it == index_.end()) {
        ++counters_.rejects;
        emit(EventType::REJECT, ts, [&](Event& e) { e.order_id = id; e.reason = Reason::UNKNOWN_ORDER; });
        return;
    }
    Order* o = it->second;
    const int side = o->side;
    const i64 price = o->price, qty = o->qty, owner = o->owner;
    unrest(o);
    pool_.release(o);
    emit(EventType::CANCEL, ts, [&](Event& e) {
        e.order_id = id; e.side = side; e.price = price; e.qty = qty; e.owner = owner; e.reason = Reason::USER;
    });
}

void OrderBook::modify(i64 id, i64 new_price, i64 new_qty, i64 ts) {
    auto it = index_.find(id);
    if (it == index_.end()) {
        ++counters_.rejects;
        emit(EventType::REJECT, ts, [&](Event& e) {
            e.order_id = id; e.price = new_price; e.qty = new_qty; e.reason = Reason::UNKNOWN_ORDER;
        });
        return;
    }
    Order* o = it->second;
    const int side = o->side;
    const i64 owner = o->owner, old_price = o->price, old_qty = o->qty;
    const bool post_only = o->post_only;
    if (new_qty < 1 || new_price < 1) {
        reject(ts, id, Reason::INVALID, side, new_price, new_qty, owner);
        return;
    }
    if (new_price == old_price && new_qty == old_qty) {
        reject(ts, id, Reason::NO_CHANGE, side, new_price, new_qty, owner);
        return;
    }
    if (new_price == old_price && new_qty < old_qty) {  // in place: keeps queue priority
        o->level->reduce(o, old_qty - new_qty);
        emit(EventType::MODIFY, ts, [&](Event& e) {
            e.order_id = id; e.side = side; e.price = old_price; e.qty = old_qty; e.owner = owner;
            e.new_price = new_price; e.new_qty = new_qty; e.keeps_priority = true; e.post_only = post_only;
        });
        return;
    }
    if (would_cross(side, new_price)) {
        if (post_only) {
            reject(ts, id, Reason::POST_ONLY_WOULD_CROSS, side, new_price, new_qty, owner);
            return;
        }
        unrest(o);
        pool_.release(o);
        emit(EventType::CANCEL, ts, [&](Event& e) {
            e.order_id = id; e.side = side; e.price = old_price; e.qty = old_qty; e.owner = owner; e.reason = Reason::REPLACE;
        });
        const i64 remaining = execute(side, true, new_price, new_qty, id, owner, ts);
        if (remaining > 0) {
            rest(pool_.acquire(id, side, new_price, remaining, owner, false));
            emit(EventType::ADD, ts, [&](Event& e) {
                e.order_id = id; e.side = side; e.price = new_price; e.qty = remaining; e.owner = owner;
            });
        }
        return;
    }
    unrest(o);
    pool_.release(o);
    rest(pool_.acquire(id, side, new_price, new_qty, owner, post_only));
    emit(EventType::MODIFY, ts, [&](Event& e) {
        e.order_id = id; e.side = side; e.price = old_price; e.qty = old_qty; e.owner = owner;
        e.new_price = new_price; e.new_qty = new_qty; e.keeps_priority = false; e.post_only = post_only;
    });
}

void OrderBook::snapshot(i64 ts) {
    // Built only if someone consumes events; the sequence number always advances.
    const bool wanted = sink_ != nullptr || checksum_on_;
    auto snap = wanted ? std::make_shared<Snapshot>(state()) : nullptr;
    emit(EventType::SNAPSHOT, ts, [&](Event& e) { e.book = snap; });
}

// ------------------------------------------------------------------------------------ queries
std::optional<i64> OrderBook::best_bid() const {
    if (bids_.empty()) return std::nullopt;
    return bids_.begin()->second.price;
}

std::optional<i64> OrderBook::best_ask() const {
    if (asks_.empty()) return std::nullopt;
    return asks_.begin()->second.price;
}

std::vector<std::tuple<i64, i64, i64>> OrderBook::depth(int side, std::size_t levels) const {
    std::vector<std::tuple<i64, i64, i64>> out;
    for (const auto& kv : side_map(side)) {
        if (levels && out.size() >= levels) break;
        out.emplace_back(kv.second.price, kv.second.total_qty, kv.second.count);
    }
    return out;
}

i64 OrderBook::volume(int side, std::size_t levels) const {
    i64 total = 0;
    std::size_t n = 0;
    for (const auto& kv : side_map(side)) {
        if (n++ >= levels) break;
        total += kv.second.total_qty;
    }
    return total;
}

i64 OrderBook::level_qty(int side, i64 price) const {
    auto it = side_map(side).find(side_key(side, price));
    return it == side_map(side).end() ? 0 : it->second.total_qty;
}

std::optional<std::pair<i64, i64>> OrderBook::queue_position(i64 id) const {
    auto it = index_.find(id);
    if (it == index_.end()) return std::nullopt;
    i64 qty = 0, n = 0;
    for (const Order* o = it->second->prev; o != nullptr; o = o->prev) {
        qty += o->qty;
        ++n;
    }
    return std::make_pair(qty, n);
}

std::optional<OrderBook::OrderInfo> OrderBook::order_info(i64 id) const {
    auto it = index_.find(id);
    if (it == index_.end()) return std::nullopt;
    const Order* o = it->second;
    return OrderInfo{o->side, o->price, o->qty, o->owner, o->post_only};
}

Snapshot OrderBook::state() const {
    Snapshot s;
    auto fill = [](const SideMap& m, std::vector<LevelSnap>& out) {
        for (const auto& kv : m) {
            LevelSnap ls{kv.second.price, {}};
            for (const Order* o = kv.second.head; o != nullptr; o = o->next)
                ls.orders.push_back({o->id, o->qty, o->owner, o->post_only});
            out.push_back(std::move(ls));
        }
    };
    fill(bids_, s.bids);
    fill(asks_, s.asks);
    return s;
}

void OrderBook::validate() const {
    auto fail = [](const std::string& m) { throw std::logic_error("invariant violated: " + m); };
    std::size_t counted = 0;
    for (int side : {BUY, SELL}) {
        for (const auto& kv : side_map(side)) {
            const PriceLevel& lvl = kv.second;
            if (kv.first != side_key(side, lvl.price)) fail("level key does not match price");
            if (lvl.count <= 0 || lvl.head == nullptr || lvl.tail == nullptr) fail("empty level present");
            i64 total = 0, n = 0;
            const Order* prev = nullptr;
            for (const Order* o = lvl.head; o != nullptr; o = o->next) {
                if (o->level != &lvl || o->price != lvl.price || o->side != side) fail("order/level mismatch");
                if (o->qty <= 0) fail("non-positive resting quantity");
                if (o->prev != prev) fail("broken prev link");
                auto it = index_.find(o->id);
                if (it == index_.end() || it->second != o) fail("index mismatch");
                total += o->qty;
                ++n;
                prev = o;
            }
            if (lvl.tail != prev) fail("bad tail");
            if (total != lvl.total_qty || n != lvl.count) fail("level aggregates stale");
            counted += static_cast<std::size_t>(n);
        }
    }
    if (counted != index_.size()) fail("index has orphans");
    for (const auto& kv : index_)
        if (!seen_.count(kv.first)) fail("live id missing from seen-set");
    if (pool_.live() != index_.size()) fail("pool live count != resting orders");
    if (!bids_.empty() && !asks_.empty() && bids_.begin()->second.price >= asks_.begin()->second.price)
        fail("crossed book");
}

}  // namespace lob
