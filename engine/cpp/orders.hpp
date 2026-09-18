// Order node and its pool allocator.
#pragma once
#include <memory>
#include <vector>

#include "types.hpp"

namespace lob {

struct PriceLevel;

// A resting order: simultaneously a node of its level's intrusive FIFO list and the value stored in
// the order-id index, so an arbitrary order is unlinked in O(1) once found by id.
struct Order {
    i64 id = 0, price = 0, qty = 0, owner = 0;
    Order* prev = nullptr;
    Order* next = nullptr;
    PriceLevel* level = nullptr;
    int side = 0;
    bool post_only = false;
};

// Free-list allocator: nodes are carved out of large blocks, so add/cancel never call malloc on the hot path.
class OrderPool {
public:
    OrderPool() = default;
    OrderPool(const OrderPool&) = delete;
    OrderPool& operator=(const OrderPool&) = delete;

    Order* acquire(i64 id, int side, i64 price, i64 qty, i64 owner, bool post_only);
    void release(Order* o) noexcept;
    std::size_t live() const noexcept { return live_; }

private:
    static constexpr std::size_t kBlock = 4096;
    std::vector<std::unique_ptr<Order[]>> blocks_;
    std::vector<Order*> free_;
    std::size_t live_ = 0;
};

}  // namespace lob
