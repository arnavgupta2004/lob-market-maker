"""Input commands accepted by an order book.

Commands are the *inputs* of the engine; events (``engine.events``) are its *outputs*. A
command stream is what the Python and C++ engines are fed in differential testing, and
what the simulator's participants ultimately produce.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Union

from engine.common import Side


@dataclass(frozen=True, slots=True)
class NewLimit:
    """Limit order. Marketable portions match immediately; any remainder rests (GTC) unless
    ``ioc``. ``post_only`` orders are rejected instead of crossing the book."""

    order_id: int
    side: Side
    price: int
    qty: int
    ts: int = 0
    owner: int = 0
    post_only: bool = False
    ioc: bool = False


@dataclass(frozen=True, slots=True)
class NewMarket:
    """Market order: sweeps the opposite side; any unfilled remainder is discarded."""

    order_id: int
    side: Side
    qty: int
    ts: int = 0
    owner: int = 0


@dataclass(frozen=True, slots=True)
class Cancel:
    """Cancel a resting order by id."""

    order_id: int
    ts: int = 0


@dataclass(frozen=True, slots=True)
class Modify:
    """Cancel-replace. A pure size decrease at the same price keeps queue priority; any other
    change (price change, size increase) sends the order to the back of the queue."""

    order_id: int
    new_price: int
    new_qty: int
    ts: int = 0


@dataclass(frozen=True, slots=True)
class SnapshotRequest:
    """Ask the book to emit a SNAPSHOT event (replay checkpoint)."""

    ts: int = 0


Command = Union[NewLimit, NewMarket, Cancel, Modify, SnapshotRequest]

_OPS = {
    "limit": NewLimit, "market": NewMarket, "cancel": Cancel,
    "modify": Modify, "snapshot": SnapshotRequest,
}
_NAMES = {v: k for k, v in _OPS.items()}


def command_to_dict(cmd: Command) -> dict:
    d = asdict(cmd)
    if "side" in d:
        d["side"] = cmd.side.code  # type: ignore[union-attr]
    d["op"] = _NAMES[type(cmd)]
    return d


def command_from_dict(d: dict) -> Command:
    d = dict(d)
    cls = _OPS[d.pop("op")]
    if "side" in d:
        d["side"] = Side.from_code(d["side"])
    return cls(**d)


def write_commands(path: str | Path, cmds: Iterable[Command]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for c in cmds:
            fh.write(json.dumps(command_to_dict(c), sort_keys=True, separators=(",", ":")))
            fh.write("\n")


def read_commands(path: str | Path) -> Iterator[Command]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield command_from_dict(json.loads(line))
