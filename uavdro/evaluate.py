"""Evaluate a charger design against the nominal demand and the stressed support."""
from __future__ import annotations

import pandas as pd

from .queueing import simulate_p95, site_table, system_p95


def scenario_rates(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Wide scenario table -> {sid: {site: rate}}."""
    cols = [c for c in df.columns if c.startswith("lam_")]
    return {row["sid"]: {c[4:]: float(row[c]) for c in cols} for _, row in df.iterrows()}


def evaluate_design(c: dict, lam: dict, R: pd.DataFrame, mu: float, sim: dict | None = None) -> dict:
    p95, unstable = system_p95(lam, c, mu)
    tab = pd.DataFrame(site_table(lam, c, mu))
    stable = tab[~tab["unstable"]]
    out = {
        "total_c": int(sum(c.values())),
        "active_sites": int(sum(v > 0 for v in c.values())),
        "p95_min": p95,
        "unstable": unstable,
        "rho_max": float(stable["rho"].max()) if len(stable) else float("nan"),
        "wq_mean_min": float((stable["lam_hr"] * stable["Wq_mean_min"]).sum() / stable["lam_hr"].sum())
        if len(stable) else float("nan"),
    }
    worst = [system_p95(l, c, mu) for l in scenario_rates(R).values()]
    out["p95_worstR_min"] = max(p for p, _ in worst)
    out["unstable_worstR"] = max(u for _, u in worst)
    if sim:
        out.update(simulate_p95(lam, c, mu, sim["hours"], sim["seeds"]))
    return out
