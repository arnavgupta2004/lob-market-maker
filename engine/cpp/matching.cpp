// Price-time-priority matching: walk the opposite side from the best price inwards; within a level
// consume the FIFO from the head; every execution happens at the maker's (resting) price.
#include "order_book.hpp"

namespace lob {

i64 OrderBook::execute(int side, bool has_limit, i64 limit, i64 qty, i64 taker_id, i64 taker_owner, i64 ts) {
    SideMap& opp = side == BUY ? asks_ : bids_;
    while (qty > 0) {
        auto it = opp.begin();
        if (it == opp.end()) break;
        PriceLevel& lvl = it->second;
        if (has_limit) {
            if (side == BUY && lvl.price > limit) break;
            if (side == SELL && lvl.price < limit) break;
        }
        while (qty > 0 && lvl.head != nullptr) {
            Order* maker = lvl.head;
            const i64 fill = qty < maker->qty ? qty : maker->qty;
            const i64 remaining = maker->qty - fill;
            ++counters_.trades;
            counters_.traded_qty += fill;
            emit(EventType::TRADE, ts, [&](Event& e) {
                e.order_id = taker_id; e.side = side; e.price = maker->price; e.qty = fill; e.owner = taker_owner;
                e.maker_id = maker->id; e.maker_owner = maker->owner; e.maker_remaining = remaining;
            });
            qty -= fill;
            if (remaining == 0) {
                lvl.remove(maker);
                index_.erase(maker->id);
                pool_.release(maker);
            } else {
                lvl.reduce(maker, fill);
            }
        }
        if (lvl.count == 0) opp.erase(it);
    }
    return qty;
}

}  // namespace lob
