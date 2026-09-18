// Unit + randomized invariant tests for the C++ order book. Dependency-free: `ctest --test-dir build`.
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "order_book.hpp"

using namespace lob;

namespace {
struct TestCase { const char* name; std::function<void()> fn; };
std::vector<TestCase>& registry() { static std::vector<TestCase> r; return r; }
struct Reg { Reg(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); } };
int g_failures = 0;
#define TEST(name) static void name(); static Reg reg_##name(#name, name); static void name()
#define CHECK(cond) do { if (!(cond)) { std::printf("    CHECK failed: %s (%s:%d)\n", #cond, __FILE__, __LINE__); ++g_failures; } } while (0)
#define CHECK_EQ(a, b) do { auto va = (a); auto vb = (b); if (!(va == vb)) { std::printf("    CHECK_EQ failed: %s == %s (%lld vs %lld) (%s:%d)\n", #a, #b, (long long)va, (long long)vb, __FILE__, __LINE__); ++g_failures; } } while (0)

struct Harness {
    OrderBook book;
    std::vector<Event> ev;
    Harness() { book.set_sink(&ev); }
    std::vector<Event> take() { auto out = std::move(ev); ev.clear(); return out; }
};
std::vector<Event> trades(const std::vector<Event>& v) {
    std::vector<Event> t;
    for (const auto& e : v) if (e.type == EventType::TRADE) t.push_back(e);
    return t;
}
}  // namespace

TEST(fifo_within_level) {
    Harness h;
    h.book.submit_limit(1, SELL, 100, 5); h.book.submit_limit(2, SELL, 100, 5); h.book.submit_limit(3, SELL, 100, 5);
    h.take();
    h.book.submit_market(10, BUY, 7);
    auto t = trades(h.take());
    CHECK_EQ(t.size(), 2u);
    CHECK_EQ(t[0].maker_id, 1); CHECK_EQ(t[0].qty, 5); CHECK_EQ(t[0].maker_remaining, 0);
    CHECK_EQ(t[1].maker_id, 2); CHECK_EQ(t[1].qty, 2); CHECK_EQ(t[1].maker_remaining, 3);
    h.book.validate();
}

TEST(price_priority_beats_time) {
    Harness h;
    h.book.submit_limit(1, SELL, 102, 5); h.book.submit_limit(2, SELL, 100, 5); h.book.submit_limit(3, SELL, 101, 5);
    h.take();
    h.book.submit_market(10, BUY, 12);
    auto t = trades(h.take());
    CHECK_EQ(t.size(), 3u);
    CHECK_EQ(t[0].maker_id, 2); CHECK_EQ(t[1].maker_id, 3); CHECK_EQ(t[2].maker_id, 1); CHECK_EQ(t[2].qty, 2);
}

TEST(bid_side_price_priority) {
    Harness h;
    h.book.submit_limit(1, BUY, 98, 5); h.book.submit_limit(2, BUY, 100, 5); h.book.submit_limit(3, BUY, 99, 5);
    h.take();
    h.book.submit_market(10, SELL, 11);
    auto t = trades(h.take());
    CHECK_EQ(t[0].maker_id, 2); CHECK_EQ(t[1].maker_id, 3); CHECK_EQ(t[2].maker_id, 1);
}

TEST(partial_fill_keeps_queue_position) {
    Harness h;
    h.book.submit_limit(1, SELL, 100, 10); h.book.submit_limit(2, SELL, 100, 10);
    h.book.submit_market(9, BUY, 4);
    CHECK_EQ(h.book.order_info(1)->qty, 6);
    CHECK_EQ(h.book.queue_position(1)->first, 0);
    CHECK_EQ(h.book.queue_position(2)->first, 6);
}

TEST(aggressive_limit_rests_remainder_and_never_crosses) {
    Harness h;
    h.book.submit_limit(1, SELL, 100, 4);
    h.take();
    h.book.submit_limit(2, BUY, 101, 10);
    auto ev = h.take();
    CHECK_EQ(ev.size(), 2u);
    CHECK(ev[0].type == EventType::TRADE && ev[1].type == EventType::ADD);
    CHECK_EQ(ev[0].price, 100);  // executes at the resting price
    CHECK_EQ(ev[1].qty, 6);
    CHECK_EQ(*h.book.best_bid(), 101);
    CHECK(!h.book.best_ask().has_value());
    h.book.validate();
}

TEST(market_order_exhaustion_expires_remainder) {
    Harness h;
    h.book.submit_limit(1, SELL, 100, 3);
    h.take();
    h.book.submit_market(2, BUY, 10);
    auto ev = h.take();
    CHECK_EQ(ev.size(), 2u);
    CHECK(ev[1].type == EventType::EXPIRE && ev[1].reason == Reason::MARKET_EXHAUSTED && ev[1].qty == 7);
    CHECK_EQ(h.book.size(), 0u);
}

TEST(ioc_and_post_only) {
    Harness h;
    h.book.submit_limit(1, SELL, 100, 3);
    h.take();
    h.book.submit_limit(2, BUY, 100, 10, 0, 0, false, true);
    auto ev = h.take();
    CHECK(ev.back().type == EventType::EXPIRE && ev.back().reason == Reason::IOC);
    CHECK_EQ(h.book.size(), 0u);  // the IOC consumed the only ask and its remainder never rests
    h.book.submit_limit(6, SELL, 100, 3);  // re-seed an ask: post-only at 100 must now be refused
    h.take();
    h.book.submit_limit(3, BUY, 100, 5, 0, 0, true, false);
    CHECK(h.take()[0].reason == Reason::POST_ONLY_WOULD_CROSS);
    h.book.submit_limit(4, BUY, 99, 5, 0, 0, true, false);
    CHECK(h.take()[0].type == EventType::ADD);
    h.book.submit_limit(5, BUY, 99, 5, 0, 0, true, true);
    CHECK(h.take()[0].reason == Reason::INVALID);
}

TEST(cancel_and_empty_level_removal) {
    Harness h;
    h.book.submit_limit(1, BUY, 99, 5); h.book.submit_limit(2, BUY, 98, 5);
    h.take();
    h.book.cancel(1);
    auto ev = h.take();
    CHECK(ev[0].type == EventType::CANCEL && ev[0].qty == 5 && ev[0].reason == Reason::USER);
    CHECK_EQ(h.book.num_levels(BUY), 1u);
    CHECK_EQ(*h.book.best_bid(), 98);
    h.book.cancel(1);
    CHECK(h.take()[0].reason == Reason::UNKNOWN_ORDER);
    h.book.validate();
}

TEST(cancel_middle_preserves_order_of_rest) {
    Harness h;
    for (int i = 1; i <= 3; ++i) h.book.submit_limit(i, SELL, 100, 1);
    h.book.cancel(2);
    CHECK_EQ(h.book.queue_position(3)->second, 1);
    CHECK_EQ(std::get<1>(h.book.depth(SELL)[0]), 2);
    h.book.validate();
}

TEST(modify_priority_rules) {
    Harness h;
    h.book.submit_limit(1, BUY, 99, 10); h.book.submit_limit(2, BUY, 99, 10);
    h.take();
    h.book.modify(1, 99, 4);
    auto ev = h.take();
    CHECK(ev[0].type == EventType::MODIFY && ev[0].keeps_priority);
    CHECK_EQ(h.book.queue_position(1)->first, 0);
    h.book.modify(1, 99, 15);
    CHECK(!h.take()[0].keeps_priority);
    CHECK_EQ(h.book.queue_position(1)->first, 10);
    h.book.modify(2, 98, 10);  // price change: level 99 keeps order 1, order 2 moves to 98
    CHECK_EQ(h.book.num_levels(BUY), 2u);
    h.book.validate();
}

TEST(modify_to_marketable_price_is_cancel_plus_arrival) {
    Harness h;
    h.book.submit_limit(1, SELL, 100, 3); h.book.submit_limit(2, BUY, 95, 10);
    h.take();
    h.book.modify(2, 100, 10);
    auto ev = h.take();
    CHECK_EQ(ev.size(), 3u);
    CHECK(ev[0].type == EventType::CANCEL && ev[0].reason == Reason::REPLACE);
    CHECK(ev[1].type == EventType::TRADE);
    CHECK(ev[2].type == EventType::ADD && ev[2].qty == 7);
    h.book.validate();
}

TEST(modify_rejections) {
    Harness h;
    h.book.submit_limit(1, BUY, 99, 5);
    h.take();
    h.book.modify(9, 99, 1); CHECK(h.take()[0].reason == Reason::UNKNOWN_ORDER);
    h.book.modify(1, 99, 0); CHECK(h.take()[0].reason == Reason::INVALID);
    h.book.modify(1, 99, 5); CHECK(h.take()[0].reason == Reason::NO_CHANGE);
    h.book.submit_limit(2, SELL, 105, 5); h.book.submit_limit(3, BUY, 100, 5, 0, 0, true);
    h.take();
    h.book.modify(3, 105, 5);
    CHECK(h.take()[0].reason == Reason::POST_ONLY_WOULD_CROSS);
    CHECK_EQ(h.book.order_info(3)->price, 100);
}

TEST(order_ids_are_unique_for_the_lifetime) {
    Harness h;
    h.book.submit_limit(1, BUY, 99, 5);
    h.take();
    h.book.submit_limit(1, BUY, 98, 5);
    CHECK(h.take()[0].reason == Reason::DUPLICATE_ID);
    h.book.cancel(1);
    h.take();
    h.book.submit_limit(1, BUY, 98, 5);
    CHECK(h.take()[0].reason == Reason::DUPLICATE_ID);
    h.book.submit_limit(2, SELL, 100, 1); h.book.submit_market(3, BUY, 1);
    h.take();
    h.book.submit_market(3, BUY, 1);
    CHECK(h.take()[0].reason == Reason::DUPLICATE_ID);
}

TEST(invalid_orders_rejected_without_state_change) {
    Harness h;
    h.book.submit_limit(1, BUY, 100, 0); CHECK(h.take()[0].reason == Reason::INVALID);
    h.book.submit_limit(2, BUY, 0, 5); CHECK(h.take()[0].reason == Reason::INVALID);
    h.book.submit_market(3, BUY, 0); CHECK(h.take()[0].reason == Reason::INVALID);
    CHECK_EQ(h.book.size(), 0u);
}

TEST(event_sequence_numbers_and_timestamps) {
    Harness h;
    h.book.submit_limit(1, SELL, 100, 5, 10);
    h.book.submit_limit(2, BUY, 100, 2, 20);
    h.book.cancel(1, 30);
    CHECK_EQ(h.ev.size(), 3u);
    for (size_t i = 0; i < h.ev.size(); ++i) CHECK_EQ(h.ev[i].seq, (i64)i + 1);
    CHECK_EQ(h.ev[0].ts, 10); CHECK_EQ(h.ev[2].ts, 30);
}

TEST(counters_advance_without_event_construction) {
    OrderBook b;  // no sink, no checksum: events counted, never built
    b.submit_limit(1, SELL, 100, 5); b.submit_market(2, BUY, 3);
    CHECK_EQ(b.counters().events, 2u); CHECK_EQ(b.counters().trades, 1u); CHECK_EQ(b.counters().traded_qty, 3);
    CHECK_EQ(b.seq(), 2);
}

TEST(checksum_is_independent_of_sink_and_order_sensitive) {
    auto run = [](bool with_sink, int variant) {
        OrderBook b; std::vector<Event> ev;
        if (with_sink) b.set_sink(&ev);
        b.set_checksum(true);
        b.submit_limit(1, SELL, 100, 5); b.submit_limit(2, SELL, 101, 5);
        if (variant == 0) b.submit_market(3, BUY, 6); else b.submit_market(3, BUY, 7);
        return b.checksum();
    };
    CHECK(run(true, 0) == run(false, 0));
    CHECK(run(true, 0) != run(true, 1));
}

TEST(snapshot_event_carries_state) {
    Harness h;
    h.book.submit_limit(1, BUY, 99, 5, 0, 7); h.book.submit_limit(2, SELL, 101, 3);
    h.take();
    h.book.snapshot(5);
    auto ev = h.take();
    CHECK(ev[0].type == EventType::SNAPSHOT && ev[0].book != nullptr);
    CHECK_EQ(ev[0].book->bids.size(), 1u); CHECK_EQ(ev[0].book->bids[0].orders[0].owner, 7);
    CHECK_EQ(ev[0].book->asks[0].price, 101);
}

TEST(depth_volume_and_level_qty) {
    Harness h;
    h.book.submit_limit(1, BUY, 99, 3); h.book.submit_limit(2, BUY, 99, 4); h.book.submit_limit(3, BUY, 97, 1);
    auto d = h.book.depth(BUY);
    CHECK_EQ(d.size(), 2u); CHECK_EQ(std::get<1>(d[0]), 7); CHECK_EQ(std::get<2>(d[0]), 2);
    CHECK_EQ(h.book.volume(BUY, 1), 7); CHECK_EQ(h.book.volume(BUY, 5), 8);
    CHECK_EQ(h.book.level_qty(BUY, 97), 1); CHECK_EQ(h.book.level_qty(BUY, 50), 0);
}

// ---- randomized: invariants after every command + per-order quantity conservation -------------------------
TEST(random_fuzz_invariants_and_conservation) {
    for (unsigned seed = 0; seed < 300; ++seed) {
        std::mt19937_64 rng(seed);
        Harness h;
        std::vector<i64> ids;
        i64 next_id = 1;
        const int range = 1 + static_cast<int>(seed % 12) * 3;
        for (int step = 0; step < 400; ++step) {
            h.ev.clear();
            const int r = static_cast<int>(rng() % 100);
            i64 qty = 0; bool orderable = false;
            if (r < 45) {
                const int side = rng() & 1 ? BUY : SELL;
                qty = 1 + static_cast<i64>(rng() % 8);
                const i64 px = 100 + (side == BUY ? -1 : 1) * static_cast<i64>(rng() % range) + static_cast<i64>(rng() % 3) - 1;
                h.book.submit_limit(next_id, side, px, qty, step, static_cast<i64>(rng() % 4), rng() % 10 == 0, rng() % 10 == 0);
                ids.push_back(next_id++); orderable = true;
            } else if (r < 55) {
                qty = 1 + static_cast<i64>(rng() % 15);
                h.book.submit_market(next_id++, rng() & 1 ? BUY : SELL, qty, step);
                orderable = true;
            } else if (r < 80 && !ids.empty()) {
                h.book.cancel(ids[rng() % ids.size()], step);
            } else if (!ids.empty()) {
                h.book.modify(ids[rng() % ids.size()], 100 + static_cast<i64>(rng() % 9) - 4, 1 + static_cast<i64>(rng() % 8), step);
            }
            h.book.validate();
            if (orderable && h.ev.back().type != EventType::REJECT) {
                i64 traded = 0, tail = 0;
                for (const auto& e : h.ev) {
                    if (e.type == EventType::TRADE) traded += e.qty;
                    if (e.type == EventType::ADD || e.type == EventType::EXPIRE) tail += e.qty;
                }
                CHECK_EQ(traded + tail, qty);
            }
        }
    }
}

int main() {
    int run = 0;
    for (const auto& t : registry()) {
        const int before = g_failures;
        t.fn();
        std::printf("[%s] %s\n", g_failures == before ? " OK " : "FAIL", t.name);
        ++run;
    }
    std::printf("%d tests, %d failed checks\n", run, g_failures);
    return g_failures == 0 ? 0 : 1;
}
