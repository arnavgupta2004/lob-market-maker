// pybind11 bindings: `engine._lob_cpp`.
//
// Two ways to drive the engine from Python:
//   * process_raw(...)  - one command per call; returns the events as tuples (used by the CppOrderBook wrapper);
//   * run_batch(array)  - a packed int64 command array executed entirely in C++ (isolates engine speed from
//                         per-call binding overhead); optionally collects events, checksums them, or times each command.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>

#include "order_book.hpp"

namespace py = pybind11;
using namespace lob;

namespace {

enum Op : i64 { OP_LIMIT = 0, OP_MARKET = 1, OP_CANCEL = 2, OP_MODIFY = 3, OP_SNAPSHOT = 4 };
constexpr int kCols = 8;         // op, id, side, price, qty, ts, owner, flags(bit0 post_only, bit1 ioc)
constexpr int kEventCols = 15;   // see hash_event / engine/cpp_engine.py

void dispatch(OrderBook& b, const i64* c) {
    const i64 flags = c[7];
    switch (c[0]) {
        case OP_LIMIT: b.submit_limit(c[1], static_cast<int>(c[2]), c[3], c[4], c[5], c[6], flags & 1, flags & 2); break;
        case OP_MARKET: b.submit_market(c[1], static_cast<int>(c[2]), c[4], c[5], c[6]); break;
        case OP_CANCEL: b.cancel(c[1], c[5]); break;
        case OP_MODIFY: b.modify(c[1], c[3], c[4], c[5]); break;
        case OP_SNAPSHOT: b.snapshot(c[5]); break;
        default: throw std::invalid_argument("unknown op code");
    }
}

// Canonical L3 state, identical in shape to OrderBook.state() of the Python engine:
//   {"bids": [(price, [(id, qty, owner, post_only_int), ...]), ...], "asks": [...]}  (best first, FIFO within level)
// Tuples/lists compare equal to the Python engine's nested lists after normalisation in cpp_engine.py.
py::object snapshot_to_py(const Snapshot& s) {
    auto side = [](const std::vector<LevelSnap>& levels) {
        py::list out;
        for (const auto& l : levels) {
            py::list orders;
            for (const auto& o : l.orders) orders.append(py::make_tuple(o.id, o.qty, o.owner, o.post_only ? 1 : 0));
            out.append(py::make_tuple(l.price, orders));
        }
        return out;
    };
    py::dict d;
    d["bids"] = side(s.bids);
    d["asks"] = side(s.asks);
    return d;
}

}  // namespace

PYBIND11_MODULE(_lob_cpp, m) {
    m.doc() = "C++ limit order book (pybind11). See engine/cpp_engine.py for the Python-facing wrapper.";
    m.attr("N_COMMAND_COLS") = kCols;
    m.attr("N_EVENT_COLS") = kEventCols;

    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<std::size_t>(), py::arg("reserve_orders") = 0)
        .def("process_raw",
             [](OrderBook& b, i64 op, i64 id, int side, i64 price, i64 qty, i64 ts, i64 owner, i64 flags) {
                 std::vector<Event> ev;
                 b.set_sink(&ev);
                 const i64 c[kCols] = {op, id, side, price, qty, ts, owner, flags};
                 dispatch(b, c);
                 b.set_sink(nullptr);
                 py::list out;
                 for (const auto& e : ev) {
                     py::object book = py::none();
                     if (e.book) book = snapshot_to_py(*e.book);
                     out.append(py::make_tuple(
                         e.seq, e.ts, static_cast<int>(e.type), e.order_id, e.side, e.price, e.qty, e.owner, e.maker_id,
                         e.maker_owner, e.maker_remaining, e.new_price, e.new_qty,
                         static_cast<int>(e.keeps_priority) | (static_cast<int>(e.post_only) << 1),
                         static_cast<int>(e.reason), book));
                 }
                 return out;
             },
             py::arg("op"), py::arg("order_id") = 0, py::arg("side") = 0, py::arg("price") = 0, py::arg("qty") = 0,
             py::arg("ts") = 0, py::arg("owner") = 0, py::arg("flags") = 0)
        .def("run_batch",
             [](OrderBook& b, py::array_t<i64, py::array::c_style | py::array::forcecast> cmds, bool collect_events,
                bool checksum, bool timed) {
                 if (cmds.ndim() != 2 || cmds.shape(1) != kCols) throw std::invalid_argument("commands must be (n, 8) int64");
                 const py::ssize_t n = cmds.shape(0);
                 const i64* data = cmds.data();
                 std::vector<Event> ev;
                 std::vector<i64> lat;
                 std::vector<std::uint8_t> cls;
                 if (timed) { lat.resize(n); cls.resize(n); }
                 if (collect_events) b.set_sink(&ev);
                 b.set_checksum(checksum);
                 const Counters c0 = b.counters();
                 const i64 seq0 = b.seq();
                 {
                     py::gil_scoped_release release;
                     if (timed) {
                         using clk = std::chrono::steady_clock;
                         for (py::ssize_t i = 0; i < n; ++i) {
                             const i64* row = data + i * kCols;
                             const u64 t0 = b.counters().trades, r0 = b.counters().rejects;
                             const auto a = clk::now();
                             dispatch(b, row);
                             const auto z = clk::now();
                             lat[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(z - a).count();
                             const bool traded = b.counters().trades > t0, rejected = b.counters().rejects > r0;
                             std::uint8_t k = 4;  // 0 add (rests, no trade) 1 match 2 cancel 3 modify 4 other 5 rejected
                             if (rejected) k = 5;
                             else if (traded) k = 1;
                             else if (row[0] == OP_LIMIT) k = 0;
                             else if (row[0] == OP_CANCEL) k = 2;
                             else if (row[0] == OP_MODIFY) k = 3;
                             cls[i] = k;
                         }
                     } else {
                         for (py::ssize_t i = 0; i < n; ++i) dispatch(b, data + i * kCols);
                     }
                 }
                 b.set_sink(nullptr);
                 b.set_checksum(false);
                 const Counters& c1 = b.counters();
                 py::dict out;
                 out["n_commands"] = static_cast<i64>(n);
                 out["n_events"] = static_cast<i64>(c1.events - c0.events);
                 out["n_trades"] = static_cast<i64>(c1.trades - c0.trades);
                 out["n_rejects"] = static_cast<i64>(c1.rejects - c0.rejects);
                 out["traded_qty"] = c1.traded_qty - c0.traded_qty;
                 out["final_seq"] = b.seq();
                 out["first_seq"] = seq0 + 1;
                 out["checksum"] = checksum ? py::cast(b.checksum()) : py::none();
                 if (collect_events) {
                     py::array_t<i64> arr({static_cast<py::ssize_t>(ev.size()), static_cast<py::ssize_t>(kEventCols)});
                     auto r = arr.mutable_unchecked<2>();
                     for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(ev.size()); ++i) {
                         const Event& e = ev[i];
                         const i64 w[kEventCols] = {e.seq, e.ts, static_cast<i64>(e.type), e.order_id, e.side, e.price, e.qty, e.owner,
                                                    e.maker_id, e.maker_owner, e.maker_remaining, e.new_price, e.new_qty,
                                                    static_cast<i64>(e.keeps_priority) | (static_cast<i64>(e.post_only) << 1),
                                                    static_cast<i64>(e.reason)};
                         for (int j = 0; j < kEventCols; ++j) r(i, j) = w[j];
                     }
                     out["events"] = arr;
                 } else {
                     out["events"] = py::none();
                 }
                 if (timed) {
                     out["latency_ns"] = py::array_t<i64>(n, lat.data());
                     out["class"] = py::array_t<std::uint8_t>(n, cls.data());
                 } else {
                     out["latency_ns"] = py::none();
                     out["class"] = py::none();
                 }
                 return out;
             },
             py::arg("commands"), py::arg("collect_events") = false, py::arg("checksum") = false, py::arg("timed") = false)
        .def("best_bid", [](const OrderBook& b) { return b.best_bid(); })
        .def("best_ask", [](const OrderBook& b) { return b.best_ask(); })
        .def("depth", [](const OrderBook& b, int side, std::size_t levels) { return b.depth(side, levels); },
             py::arg("side"), py::arg("levels") = 0)
        .def("volume", &OrderBook::volume)
        .def("level_qty", &OrderBook::level_qty)
        .def("num_levels", &OrderBook::num_levels)
        .def("queue_position", [](const OrderBook& b, i64 id) { return b.queue_position(id); })
        .def("order_info", [](const OrderBook& b, i64 id) -> py::object {
            auto o = b.order_info(id);
            if (!o) return py::none();
            return py::make_tuple(o->side, o->price, o->qty, o->owner, o->post_only);
        })
        .def("state", [](const OrderBook& b) { return snapshot_to_py(b.state()); })
        .def("validate", &OrderBook::validate)
        .def("contains", &OrderBook::contains)
        .def("__len__", &OrderBook::size)
        .def_property_readonly("seq", &OrderBook::seq)
        .def_property_readonly("checksum", &OrderBook::checksum);

    m.def("timer_overhead_ns", [](int reps) {
        using clk = std::chrono::steady_clock;
        i64 acc = 0;
        const auto s = clk::now();
        for (int i = 0; i < reps; ++i) {
            const auto a = clk::now();
            const auto z = clk::now();
            acc += std::chrono::duration_cast<std::chrono::nanoseconds>(z - a).count();
        }
        (void)s;
        return static_cast<double>(acc) / reps;
    }, py::arg("reps") = 200000, "Mean cost (ns) of one steady_clock now()-now() pair, for latency-measurement context.");
}
