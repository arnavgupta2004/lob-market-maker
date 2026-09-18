#include "orders.hpp"

namespace lob {

Order* OrderPool::acquire(i64 id, int side, i64 price, i64 qty, i64 owner, bool post_only) {
    if (free_.empty()) {
        blocks_.emplace_back(new Order[kBlock]);
        Order* b = blocks_.back().get();
        free_.reserve(free_.size() + kBlock);
        for (std::size_t i = kBlock; i-- > 0;) free_.push_back(b + i);
    }
    Order* o = free_.back();
    free_.pop_back();
    o->id = id; o->side = side; o->price = price; o->qty = qty; o->owner = owner;
    o->post_only = post_only; o->prev = o->next = nullptr; o->level = nullptr;
    ++live_;
    return o;
}

void OrderPool::release(Order* o) noexcept {
    free_.push_back(o);
    --live_;
}

}  // namespace lob
