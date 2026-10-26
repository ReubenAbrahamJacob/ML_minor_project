"""Cost-based maintenance decision from a RUL interval (all costs are assumptions, configs/costs.yaml).

Units: remaining life as a fraction of total life. R ~ Normal(median, sigma) truncated at 0, sigma from the
90% interval half-width. Actions: maintain_now / schedule / continue.
"""
import numpy as np
import yaml
from scipy import stats

from . import ROOT

COSTS = yaml.safe_load((ROOT / "configs" / "costs.yaml").read_text())
Z90 = stats.norm.ppf(0.95)


def _moments(med, lo, hi):
    sigma = np.maximum((hi - lo) / (2 * Z90), 1e-3)
    return med, sigma


def expected_costs(med, lo, hi, c=COSTS) -> dict[str, float]:
    mu, s = _moments(med, lo, hi)
    cp, cf, cw = c["planned_maintenance"], c["unplanned_failure"], c["waste_per_life_fraction"]
    W, d = c["lead_window"], c["inspection_interval"]
    ER = max(mu, 0.0)                                           # E[R], R = remaining life fraction
    ER_after = s * stats.norm.pdf((mu - W) / s) + (mu - W) * stats.norm.cdf((mu - W) / s)  # E[(R-W)+]
    return {
        "maintain_now": cp + cw * ER,
        "schedule": cp + cw * max(ER_after, 0.0) + cf * stats.norm.cdf((W - mu) / s),
        "continue": cf * stats.norm.cdf((d - mu) / s),
    }


def decide(med, lo, hi) -> str:
    costs = expected_costs(med, lo, hi)
    return min(costs, key=costs.get)


def simulate(life_frac, med, lo, hi, life_s, c=COSTS) -> dict:
    """Walk through one bearing's windows in time order; the first non-'continue' action ends the run."""
    cp, cf, cw, W = c["planned_maintenance"], c["unplanned_failure"], c["waste_per_life_fraction"], c["lead_window"]
    order = np.argsort(life_frac)
    for i in order:
        a = decide(med[i], lo[i], hi[i])
        f = life_frac[i]
        if a == "maintain_now":
            return {"cost": cp + cw * (1 - f), "failed": False, "action_at": f}
        if a == "schedule":
            done = f + W
            return {"cost": cf if done >= 1 else cp + cw * (1 - done), "failed": done >= 1, "action_at": f}
    return {"cost": cf, "failed": True, "action_at": 1.0}


def fixed_policy(life_s: float, mean_life_s: float, c=COSTS) -> dict:
    t = c["fixed_interval_fraction"] * mean_life_s
    if t >= life_s:
        return {"cost": c["unplanned_failure"], "failed": True}
    return {"cost": c["planned_maintenance"] + c["waste_per_life_fraction"] * (1 - t / life_s), "failed": False}
