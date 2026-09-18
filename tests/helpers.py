"""Shared test doubles."""
from simulator.latency.models import LatencyConfig
from simulator.participant import Participant


class Scripted(Participant):
    """Sends fixed (time_ns, command) pairs and logs every feed delivery it receives."""

    def __init__(self, script, name="script", latency=LatencyConfig(), feed="all"):
        self.name, self.latency, self.feed = name, latency, feed
        self.script = sorted(script, key=lambda x: x[0])
        self.received: list[tuple[int, list]] = []

    def next_time(self):
        return self.script[0][0] if self.script else None

    def on_wakeup(self, sim, now):
        out = []
        while self.script and self.script[0][0] <= now:
            out.append(self.script.pop(0)[1])
        return out

    def on_events(self, sim, now, events):
        self.received.append((now, events))
        return []
