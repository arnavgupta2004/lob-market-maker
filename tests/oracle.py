"""Naive, obviously-correct order book used as a test oracle.

It stores orders in a flat dict and finds the best/next order by scanning and sorting on
``(price, arrival_counter)`` every time: O(N) per operation, no linked lists, no sorted key
arrays, no shared code with the engine. It re-implements the *specification* in
``engine/python/order_book.py``; agreement between the two after every command is strong
evidence that the fast structures implement that spec.
"""
from __future__ import annotations

from engine.commands import Cancel, Command, Modify, NewLimit, NewMarket, SnapshotRequest
from engine.common import Reason, Side


def C(type_, order_id=0, side=0, price=0, qty=0, owner=0, maker_id=0, maker_owner=0,
      maker_remaining=0, new_price=0, new_qty=0, keeps_priority=False, post_only=False,
      reason=""):
    """Build a tuple in the layout of ``Event.core()``."""
    return (type_, order_id, int(side), price, qty, owner, maker_id, maker_owner,
            maker_remaining, new_price, new_qty, keeps_priority, post_only, reason)


class NaiveBook:
    def __init__(self):
        self.orders: dict[int, dict] = {}
        self.arrival = 0
        self.seen: set[int] = set()

    # -- helpers
    def _rest(self, oid, side, price, qty, owner, post_only):
        self.arrival += 1
        self.orders[oid] = dict(id=oid, side=side, price=price, qty=qty, owner=owner,
                                post_only=post_only, arr=self.arrival)

    def _best(self, side):
        opp = [o for o in self.orders.values() if o["side"] != side]
        return (min(o["price"] for o in opp) if side == Side.BUY else max(o["price"] for o in opp)) if opp else None

    def _crosses(self, side, price):
        b = self._best(side)
        if b is None:
            return False
        return price >= b if side == Side.BUY else price <= b

    def _match(self, side, limit, qty, taker, owner, out):
        while qty > 0:
            opp = [o for o in self.orders.values() if o["side"] != side]
            if limit is not None:
                opp = [o for o in opp if (o["price"] <= limit if side == Side.BUY else o["price"] >= limit)]
            if not opp:
                break
            best = min(opp, key=lambda o: (o["price"] if side == Side.BUY else -o["price"], o["arr"]))
            fill = min(qty, best["qty"])
            rem = best["qty"] - fill
            out.append(C("TRADE", taker, side, best["price"], fill, owner, best["id"], best["owner"], rem))
            qty -= fill
            if rem == 0:
                del self.orders[best["id"]]
            else:
                best["qty"] = rem
        return qty

    # -- commands (return list of core tuples)
    def process(self, cmd: Command):
        out: list = []
        if isinstance(cmd, NewLimit):
            self._limit(cmd, out)
        elif isinstance(cmd, NewMarket):
            self._market(cmd, out)
        elif isinstance(cmd, Cancel):
            o = self.orders.pop(cmd.order_id, None)
            if o is None:
                out.append(C("REJECT", cmd.order_id, reason=Reason.UNKNOWN_ORDER))
            else:
                out.append(C("CANCEL", o["id"], o["side"], o["price"], o["qty"], o["owner"], reason=Reason.USER))
        elif isinstance(cmd, Modify):
            self._modify(cmd, out)
        elif isinstance(cmd, SnapshotRequest):
            out.append(("SNAPSHOT",))
        return out

    def _limit(self, c: NewLimit, out):
        if c.order_id in self.seen:
            out.append(C("REJECT", c.order_id, c.side, c.price, c.qty, c.owner, reason=Reason.DUPLICATE_ID))
            return
        self.seen.add(c.order_id)
        if c.qty < 1 or c.price < 1 or (c.post_only and c.ioc):
            out.append(C("REJECT", c.order_id, c.side, c.price, c.qty, c.owner, reason=Reason.INVALID))
            return
        if c.post_only and self._crosses(c.side, c.price):
            out.append(C("REJECT", c.order_id, c.side, c.price, c.qty, c.owner, reason=Reason.POST_ONLY_WOULD_CROSS))
            return
        rem = self._match(c.side, c.price, c.qty, c.order_id, c.owner, out)
        if rem > 0:
            if c.ioc:
                out.append(C("EXPIRE", c.order_id, c.side, c.price, rem, c.owner, reason=Reason.IOC))
            else:
                self._rest(c.order_id, c.side, c.price, rem, c.owner, c.post_only)
                out.append(C("ADD", c.order_id, c.side, c.price, rem, c.owner, post_only=c.post_only))

    def _market(self, c: NewMarket, out):
        if c.order_id in self.seen:
            out.append(C("REJECT", c.order_id, c.side, 0, c.qty, c.owner, reason=Reason.DUPLICATE_ID))
            return
        self.seen.add(c.order_id)
        if c.qty < 1:
            out.append(C("REJECT", c.order_id, c.side, 0, c.qty, c.owner, reason=Reason.INVALID))
            return
        rem = self._match(c.side, None, c.qty, c.order_id, c.owner, out)
        if rem > 0:
            out.append(C("EXPIRE", c.order_id, c.side, 0, rem, c.owner, reason=Reason.MARKET_EXHAUSTED))

    def _modify(self, c: Modify, out):
        o = self.orders.get(c.order_id)
        if o is None:
            out.append(C("REJECT", c.order_id, 0, c.new_price, c.new_qty, reason=Reason.UNKNOWN_ORDER))
            return
        side, owner = o["side"], o["owner"]
        if c.new_qty < 1 or c.new_price < 1:
            out.append(C("REJECT", c.order_id, side, c.new_price, c.new_qty, owner, reason=Reason.INVALID))
            return
        if c.new_price == o["price"] and c.new_qty == o["qty"]:
            out.append(C("REJECT", c.order_id, side, c.new_price, c.new_qty, owner, reason=Reason.NO_CHANGE))
            return
        op, oq = o["price"], o["qty"]
        if c.new_price == op and c.new_qty < oq:
            o["qty"] = c.new_qty
            out.append(C("MODIFY", c.order_id, side, op, oq, owner, new_price=c.new_price,
                         new_qty=c.new_qty, keeps_priority=True, post_only=o["post_only"]))
            return
        if self._crosses(side, c.new_price):
            if o["post_only"]:
                out.append(C("REJECT", c.order_id, side, c.new_price, c.new_qty, owner,
                             reason=Reason.POST_ONLY_WOULD_CROSS))
                return
            del self.orders[c.order_id]
            out.append(C("CANCEL", c.order_id, side, op, oq, owner, reason=Reason.REPLACE))
            rem = self._match(side, c.new_price, c.new_qty, c.order_id, owner, out)
            if rem > 0:
                self._rest(c.order_id, side, c.new_price, rem, owner, False)
                out.append(C("ADD", c.order_id, side, c.new_price, rem, owner))
            return
        po = o["post_only"]
        del self.orders[c.order_id]
        self._rest(c.order_id, side, c.new_price, c.new_qty, owner, po)
        out.append(C("MODIFY", c.order_id, side, op, oq, owner, new_price=c.new_price,
                     new_qty=c.new_qty, keeps_priority=False, post_only=po))

    # -- state in the engine's canonical format
    def state(self) -> dict:
        def side_state(side, reverse):
            by_price: dict[int, list] = {}
            for o in sorted(self.orders.values(), key=lambda o: o["arr"]):
                if o["side"] == side:
                    by_price.setdefault(o["price"], []).append([o["id"], o["qty"], o["owner"], int(o["post_only"])])
            return [[p, by_price[p]] for p in sorted(by_price, reverse=reverse)]
        return {"bids": side_state(Side.BUY, True), "asks": side_state(Side.SELL, False)}
