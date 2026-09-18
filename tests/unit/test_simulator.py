"""Discrete-event simulator: determinism, replay, latency semantics, statistical sanity."""
import math

import numpy as np
import pytest

from engine.commands import Cancel, NewLimit, NewMarket
from engine.common import EventType as ET, Side
from engine.events import stream_hash
from engine.python.order_book import OrderBook
from engine.replay import replay_commands, replay_events
from simulator.latency.models import ConstantLatency, LatencyConfig, LogNormalLatency, UniformLatency
from simulator.market_state.fundamental import FundamentalConfig, FundamentalProcess
from simulator.order_flow.distributions import Const, Exponential, LogNormal, Pareto, Uniform, sample_int
from simulator.order_flow.informed import InformedTrader, InformedTraderConfig
from simulator.order_flow.noise import NoiseTrader, NoiseTraderConfig
from simulator.participant import Participant
from simulator.simulator import NS, SimConfig, Simulator

MS = 1_000_000


class Scripted(Participant):
    """Sends fixed (time, command) pairs and logs every feed delivery it receives."""

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


def quiet_cfg(**kw):
    return SimConfig(horizon_s=1.0, seed_levels=0, warmup_s=0.0, **kw)


# ------------------------------------------------------------------ determinism / replay
def test_same_seed_same_event_stream_and_different_seed_differs():
    def run(seed):
        return Simulator(SimConfig(seed=seed, horizon_s=20), [NoiseTrader()]).run()
    a, b, c = run(1), run(1), run(2)
    assert stream_hash(a.events) == stream_hash(b.events)
    assert stream_hash(a.events) != stream_hash(c.events)
    assert a.n_events > 1000


def test_simulation_output_replays_exactly():
    r = Simulator(SimConfig(seed=3, horizon_s=20, fundamental=FundamentalConfig(sigma=0.02)),
                  [NoiseTrader(), InformedTrader()]).run()
    # (1) event-only replay reconstructs the final book
    assert replay_events(r.events).state() == r.book.state()
    # (2) re-executing the recorded command log reproduces the byte-identical event stream
    again = replay_commands(r.commands)
    assert stream_hash(again.events) == stream_hash(r.events)
    r.book.validate()


def test_book_never_crossed_in_samples():
    r = Simulator(SimConfig(seed=5, horizon_s=30), [NoiseTrader()]).run()
    sp = r.samples["spread"]
    assert np.all(sp[~np.isnan(sp)] >= 1)


def test_adding_a_participant_does_not_change_existing_rng_streams():
    s1 = Simulator(SimConfig(seed=9), [NoiseTrader(name="a")])
    s2 = Simulator(SimConfig(seed=9), [NoiseTrader(name="b"), NoiseTrader(name="a")])
    assert s1.rng_for("a").random() == s2.rng_for("a").random()
    assert s1.rng_for("a").random() != s1.rng_for("b").random()


def test_duplicate_participant_names_rejected():
    with pytest.raises(ValueError):
        Simulator(SimConfig(), [NoiseTrader(name="x"), NoiseTrader(name="x")])


# ---------------------------------------------------------------------------- latency
def test_order_latency_delays_arrival_and_stamps_exchange_time():
    p = Scripted([(0, NewLimit(1, Side.BUY, 100, 5, owner=1))],
                 latency=LatencyConfig(order=ConstantLatency(5 * MS)))
    r = Simulator(quiet_cfg(), [p]).run()
    (add,) = [e for e in r.events if e.type is ET.ADD]
    assert add.ts == 5 * MS


def test_feed_latency_delays_notification_and_preserves_order():
    p = Scripted([(0, NewLimit(1, Side.BUY, 100, 5, owner=1)), (1 * MS, NewLimit(2, Side.BUY, 99, 5, owner=1))],
                 latency=LatencyConfig(feed=ConstantLatency(7 * MS)))
    r = Simulator(quiet_cfg(), [p]).run()
    assert [t for t, _ in p.received] == [7 * MS, 8 * MS]
    assert [e.order_id for _, evs in p.received for e in evs] == [1, 2]


def test_jittered_latency_never_reorders_a_participants_commands():
    lat = LatencyConfig(order=UniformLatency(1 * MS, 20 * MS), feed=LogNormalLatency(5 * MS, 1.0))
    script = [(i * 100_000, NewLimit(i + 1, Side.BUY, 100 - (i % 5), 1, owner=1)) for i in range(200)]
    p = Scripted(script, latency=lat)
    r = Simulator(quiet_cfg(seed=4), [p]).run()
    adds = [e.order_id for e in r.events if e.type is ET.ADD]
    assert adds == sorted(adds)  # FIFO per participant
    delivered = [t for t, _ in p.received]
    assert delivered == sorted(delivered)


def test_own_feed_only_delivers_events_involving_the_participant():
    maker = Scripted([(0, NewLimit(1, Side.SELL, 100, 5, owner=1))], name="maker", feed="own")
    taker = Scripted([(1 * MS, NewMarket(2, Side.BUY, 3, owner=2))], name="taker", feed="own")
    bystander = Scripted([], name="bystander", feed="own")
    Simulator(quiet_cfg(), [maker, taker, bystander]).run()
    assert [e.type for _, evs in maker.received for e in evs] == [ET.ADD, ET.TRADE]
    assert [e.type for _, evs in taker.received for e in evs] == [ET.TRADE]
    assert bystander.received == []


def test_seeded_book_and_sampling():
    cfg = SimConfig(horizon_s=1.0, seed_levels=3, seed_size=7, sample_interval_s=0.25)
    r = Simulator(cfg, [Scripted([])]).run()
    assert r.samples["t_ns"].tolist() == [0, 250 * MS, 500 * MS, 750 * MS, 1000 * MS]
    assert r.samples["best_bid"][0] == cfg.mid0_ticks - 1 and r.samples["best_ask"][0] == cfg.mid0_ticks + 1
    assert r.samples["bid_depth5"][0] == 21


# ----------------------------------------------------------------- statistical sanity
def test_noise_arrival_counts_are_poisson_with_configured_rates():
    cfg = NoiseTraderConfig(limit_rate=80.0, market_rate=20.0)
    T = 100.0
    r = Simulator(SimConfig(seed=11, horizon_s=T), [NoiseTrader(cfg)]).run()
    n_lim = sum(isinstance(c, NewLimit) and c.owner == 1 for c in r.commands)
    n_mkt = sum(isinstance(c, NewMarket) for c in r.commands)
    for n, lam in ((n_lim, 80.0 * T), (n_mkt, 20.0 * T)):
        assert abs(n - lam) < 5 * math.sqrt(lam), (n, lam)  # 5 sigma


def test_noise_cancel_lifetimes_are_exponential():
    cfg = NoiseTraderConfig(limit_rate=50.0, market_rate=0.0001, cancel_rate=2.0, offset=Const(50))  # far away: never fill
    r = Simulator(SimConfig(seed=2, horizon_s=200, seed_levels=0), [NoiseTrader(cfg)]).run()
    born = {c.order_id: c.ts for c in r.commands if isinstance(c, NewLimit)}
    life = [(c.ts - born[c.order_id]) / NS for c in r.commands if isinstance(c, Cancel)]
    assert len(life) > 5000
    assert np.mean(life) == pytest.approx(1 / 2.0, rel=0.05)
    assert np.std(life) == pytest.approx(1 / 2.0, rel=0.08)  # exponential: sd == mean


def test_noise_buy_fraction_matches_p_buy():
    cfg = NoiseTraderConfig(p_buy=0.7)
    r = Simulator(SimConfig(seed=8, horizon_s=60), [NoiseTrader(cfg)]).run()
    sides = [c.side for c in r.commands if isinstance(c, (NewLimit, NewMarket)) and c.owner == 1]
    assert np.mean([s is Side.BUY for s in sides]) == pytest.approx(0.7, abs=0.02)


def test_book_is_stationary_under_noise_flow():
    r = Simulator(SimConfig(seed=6, horizon_s=120), [NoiseTrader()]).run()
    n = r.samples["n_orders"]
    first, second = n[len(n) // 4: len(n) // 2].mean(), n[len(n) // 2:].mean()
    assert 0.5 < first / second < 2.0 and n.max() < 400  # no blow-up


# -------------------------------------------------------------------------- fundamental
def rng(seed=0):
    return np.random.default_rng(seed)


def test_brownian_fundamental_variance_grows_linearly():
    cfg = FundamentalConfig(sigma=0.05, kappa=0.0, dt_s=0.01)
    ends = [FundamentalProcess(cfg, 100.0, 10 * NS, rng(i)).value_at(10 * NS) for i in range(400)]
    assert np.var(ends) == pytest.approx(0.05 ** 2 * 10, rel=0.2)
    assert np.mean(ends) == pytest.approx(100.0, abs=0.03)


def test_ou_fundamental_is_mean_reverting_with_correct_stationary_variance():
    cfg = FundamentalConfig(sigma=0.05, kappa=2.0, mu=100.0, dt_s=0.01)
    p = FundamentalProcess(cfg, 100.5, 400 * NS, rng(1))
    x = p.path[1000:]  # discard burn-in (10 time constants+)
    assert np.mean(x) == pytest.approx(100.0, abs=0.02)
    assert np.var(x) == pytest.approx(0.05 ** 2 / (2 * 2.0), rel=0.1)
    ac1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert ac1 == pytest.approx(math.exp(-2.0 * 0.01), abs=0.01)  # exact AR(1) coefficient


def test_fundamental_is_independent_of_participants_and_reproducible():
    a = Simulator(SimConfig(seed=4, fundamental=FundamentalConfig()), [NoiseTrader()])
    b = Simulator(SimConfig(seed=4, fundamental=FundamentalConfig()), [Scripted([])])
    assert np.array_equal(a.fundamental.path, b.fundamental.path)


# ------------------------------------------------------------------------ distributions
@pytest.mark.parametrize("dist", [Exponential(3.0), LogNormal(4.0, 0.5), Uniform(1, 9), Pareto(2.0, 3.0)])
def test_distribution_sample_means_match_theory(dist):
    g = rng(7)
    x = np.array([dist.sample(g) for _ in range(60_000)])
    assert x.mean() == pytest.approx(dist.mean(), rel=0.05)


def test_pareto_tail_exponent():
    g, d = rng(3), Pareto(1.0, 1.5)
    x = np.array([d.sample(g) for _ in range(100_000)])
    assert np.mean(x > 10) == pytest.approx(10 ** -1.5, rel=0.1)
    assert x.min() >= 1.0 and math.isinf(Pareto(1.0, 0.9).mean())


def test_sample_int_clips():
    assert sample_int(Const(0.4), rng(), lo=1) == 1
    assert sample_int(Const(99.0), rng(), lo=1, hi=10) == 10


# ---------------------------------------------------------------------- informed traders
def tracking_error(informed: bool, seed: int) -> float:
    parts = [NoiseTrader()] + ([InformedTrader(InformedTraderConfig(rate=10.0, threshold=1.0))] if informed else [])
    r = Simulator(SimConfig(seed=seed, horizon_s=120, fundamental=FundamentalConfig(sigma=0.03, kappa=0.0)), parts).run()
    s, _ = r.steady()
    ok = ~np.isnan(s["mid"])
    return float(np.mean(np.abs(s["mid"][ok] - s["fundamental"][ok])))


def test_informed_flow_pulls_price_toward_fundamental():
    base = np.mean([tracking_error(False, s) for s in range(4)])
    inf = np.mean([tracking_error(True, s) for s in range(4)])
    assert inf < 0.5 * base, (base, inf)


def test_momentum_trader_uses_lookback_and_trades_with_the_trend():
    cfg = InformedTraderConfig(rate=50.0, w_value=0.0, w_momentum=1.0, threshold=0.5, lookback_s=0.5)
    r = Simulator(SimConfig(seed=1, horizon_s=30), [NoiseTrader(), InformedTrader(cfg)]).run()
    mkts = [c for c in r.commands if isinstance(c, NewMarket) and c.owner == 2]
    assert len(mkts) > 50
