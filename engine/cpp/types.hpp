// Shared value types for the C++ order book. Mirrors engine/common.py + engine/events.py:
// prices are integer ticks, quantities integer lots, timestamps integer nanoseconds.
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace lob {

using i64 = std::int64_t;
using u64 = std::uint64_t;

constexpr int BUY = 1;
constexpr int SELL = -1;

// Codes are part of the Python<->C++ wire format (engine/cpp_engine.py); keep in sync.
enum class EventType : std::uint8_t { ADD = 0, TRADE, CANCEL, MODIFY, EXPIRE, REJECT, SNAPSHOT };
enum class Reason : std::uint8_t {
    NONE = 0, USER, REPLACE, IOC, MARKET_EXHAUSTED, DUPLICATE_ID, UNKNOWN_ORDER, INVALID,
    POST_ONLY_WOULD_CROSS, NO_CHANGE
};

struct OrderSnap { i64 id, qty, owner; bool post_only; };
struct LevelSnap { i64 price; std::vector<OrderSnap> orders; };
struct Snapshot { std::vector<LevelSnap> bids, asks; };  // best price first, FIFO within a level

// One state transition. Field meaning per type is documented in engine/events.py.
struct Event {
    i64 seq = 0, ts = 0;
    EventType type = EventType::ADD;
    Reason reason = Reason::NONE;
    int side = 0;  // +1 / -1, 0 = none
    bool keeps_priority = false, post_only = false;
    i64 order_id = 0, price = 0, qty = 0, owner = 0;
    i64 maker_id = 0, maker_owner = 0, maker_remaining = 0;
    i64 new_price = 0, new_qty = 0;
    std::shared_ptr<Snapshot> book;  // SNAPSHOT events only
};

}  // namespace lob
