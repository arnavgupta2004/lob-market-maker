"""Per-environment calibration of the adaptive strategy's measured inputs, on seeds disjoint from evaluation.

Nothing here is tuned for performance: each quantity is *estimated by the same measurement code used in the research
experiments* and passed to the strategy as a fixed parameter.

* ``k``        fill-intensity decay - censored-MLE probe fit (Experiment as_assumptions);
* ``beta_obi`` OLS slope (ticks per unit touch-OBI) of the forward mid change at the typical quote lifetime (Experiment A);
* fill model   ``P(fill within 1 s | Q, d_touch)`` logistic fit on queue-probe data (Experiment B).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.baseline.as_assumptions import ALL_ENVS as ENVS, calibrate, env_fundamental
from experiments.common import pmap
from research.imbalance import session_stats
from research.queue_position import FillModel, QueueProbe, fit_fill_model
from simulator.market_state.fundamental import FundamentalConfig
from simulator.order_flow.informed import InformedTrader
from simulator.order_flow.noise import NoiseTrader
from simulator.simulator import SimConfig, Simulator

OBI_HORIZON_S = 0.25  # ~ mean lifetime of a market-maker quote (0.26 s in the Stage-4 baseline)


@dataclass(frozen=True)
class Job:
    env: str
    seed: int
    horizon_s: float


def _parts(env: str):
    return [NoiseTrader()] + ([InformedTrader(ENVS[env]["informed"])] if ENVS[env]["informed"] else [])


def _obi_slope(job: Job) -> float:
    cfg = SimConfig(seed=job.seed, horizon_s=job.horizon_s, sample_interval_s=0.01, fundamental=env_fundamental(job.env),
                    record_events=False, record_commands=False)
    res = Simulator(cfg, _parts(job.env)).run()
    return session_stats(res.samples, 0.01, [OBI_HORIZON_S], levels=1)[OBI_HORIZON_S]["slope"]


def _probe_dataset(job: Job) -> dict:
    cfg = SimConfig(seed=job.seed, horizon_s=job.horizon_s, fundamental=env_fundamental(job.env), record_events=False,
                    record_commands=False)
    probe = QueueProbe(dwell_s=3.0)
    Simulator(cfg, _parts(job.env) + [probe]).run()
    return probe.dataset()


def calibrate_environment(env: str, seeds: list[int], horizon_s: float = 120.0, workers: int = 1) -> dict:
    """Return ``{"k", "beta_obi", "fill_model", ...}`` for ``env`` from disjoint calibration seeds."""
    cal = calibrate(env, seeds, horizon_s=horizon_s, workers=workers)
    jobs = [Job(env, s, horizon_s) for s in seeds]
    slopes = np.array(pmap(_obi_slope, jobs, workers), dtype=float)
    fm = fit_fill_model(pmap(_probe_dataset, jobs, workers), delta_s=1.0, dwell_s=3.0)
    return {"k": cal["fit"]["k"], "k_ci": cal["fit"]["k_ci"], "beta_obi": float(np.nanmean(slopes)),
            "beta_obi_sd_across_sessions": float(np.nanstd(slopes, ddof=1)), "fill_model": fm, "seeds": seeds,
            "obi_horizon_s": OBI_HORIZON_S}
