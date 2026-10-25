"""Stage 2 — queue-constrained 1-Wasserstein DRO charger sizing (Section IV.E–F), as a MILP."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from pyomo.environ import (
    ConcreteModel, Constraint, NonNegativeIntegers, NonNegativeReals, Objective, Reals, Set,
    SolverFactory, Var, minimize, value,
)
from pyomo.opt import TerminationCondition


def lam_table(df: pd.DataFrame, sites) -> dict:
    """Wide scenario table (sid, lam_<site>...) -> {(sid, site): rate}; missing sites -> 0."""
    out = {}
    for _, row in df.iterrows():
        for j in sites:
            out[(row["sid"], j)] = float(row.get(f"lam_{j}", 0.0))
    return out


def l1_distances(S: pd.DataFrame, R: pd.DataFrame, Lbar: float) -> pd.DataFrame:
    """Ground metric (7): d(s, r) = ||lam_s - lam_r||_1 / Lambda_bar for every (s, r)."""
    cols = sorted({c for c in list(S.columns) + list(R.columns) if c.startswith("lam_")})
    s = S.reindex(columns=cols, fill_value=0.0).to_numpy(float)
    r = R.reindex(columns=cols, fill_value=0.0).to_numpy(float)
    d = np.abs(s[:, None, :] - r[None, :, :]).sum(axis=2) / float(Lbar)
    return pd.DataFrame([(si, ri, d[a, b]) for a, si in enumerate(S["sid"]) for b, ri in enumerate(R["sid"])],
                        columns=["s", "r", "d"])


def build_model(S: pd.DataFrame, R: pd.DataFrame, dist: pd.DataFrame, sites, eps: float, *,
                mu: float, beta: float, c_max: int, bkpts, coeffs, charger_cost: float = 1.0,
                total_chargers: int | None = None):
    K = range(len(bkpts))
    lamS, lamR = lam_table(S, sites), lam_table(R, sites)
    d = {(str(r.s), str(r.r)): float(r.d) for r in dist.itertuples()}

    m = ConcreteModel()
    m.J = Set(initialize=list(sites), ordered=True)
    m.S = Set(initialize=list(S["sid"]), ordered=True)
    m.R = Set(initialize=list(R["sid"]), ordered=True)
    m.K = Set(initialize=list(K), ordered=True)

    m.c = Var(m.J, domain=NonNegativeIntegers, bounds=(0, c_max))

    # (2) stability on every scenario in Omega = S u R:  lambda <= beta * mu * c
    m.stabS = Constraint(m.S, m.J, rule=lambda m, s, j: lamS[s, j] <= beta * mu * m.c[j])
    m.stabR = Constraint(m.R, m.J, rule=lambda m, r, j: lamR[r, j] <= beta * mu * m.c[j])
    if total_chargers is not None:
        m.budget = Constraint(expr=sum(m.c[j] for j in m.J) == int(total_chargers))

    # PWL queue surrogate with McCormick envelopes for w = c * u  (12)-(18)
    m.u = Var(m.R, m.J, m.K, bounds=(0.0, 1.0))
    m.w = Var(m.R, m.J, m.K, domain=NonNegativeReals)
    m.Qrj = Var(m.R, m.J, domain=NonNegativeReals)
    m.Qr = Var(m.R, domain=NonNegativeReals)
    m.usum = Constraint(m.R, m.J, rule=lambda m, r, j: sum(m.u[r, j, k] for k in m.K) == 1.0)
    m.mc2 = Constraint(m.R, m.J, m.K, rule=lambda m, r, j, k: m.w[r, j, k] <= m.c[j])
    m.mc3 = Constraint(m.R, m.J, m.K, rule=lambda m, r, j, k: m.w[r, j, k] <= c_max * m.u[r, j, k])
    m.mc4 = Constraint(m.R, m.J, m.K,
                       rule=lambda m, r, j, k: m.w[r, j, k] >= m.c[j] + c_max * m.u[r, j, k] - c_max)
    m.cap = Constraint(m.R, m.J, rule=lambda m, r, j:
                       lamR[r, j] <= mu * sum(bkpts[k] * m.w[r, j, k] for k in m.K))
    m.pen = Constraint(m.R, m.J, rule=lambda m, r, j:
                       m.Qrj[r, j] >= sum(coeffs[k] * m.u[r, j, k] for k in m.K))
    m.pen_r = Constraint(m.R, rule=lambda m, r: m.Qr[r] >= sum(m.Qrj[r, j] for j in m.J))

    # Wasserstein dual (11): alpha_s >= Q_r - gamma * d(s, r)
    m.alpha = Var(m.S, domain=Reals)
    m.gamma = Var(domain=NonNegativeReals)
    m.wcut = Constraint(m.S, m.R, rule=lambda m, s, r: m.alpha[s] >= m.Qr[r] - m.gamma * d.get((s, r), 0.0))
    if abs(eps) < 1e-12:
        m.gamma.fix(0.0)

    m.capex = charger_cost * sum(m.c[j] for j in m.J)
    m.obj = Objective(expr=m.capex + sum(m.alpha[s] for s in m.S) / len(S) + eps * m.gamma, sense=minimize)
    return m


def solve(m, solver: str = "highs", threads: int = 16, mip_gap: float = 0.02, tee: bool = False) -> dict:
    name = {"highs": "appsi_highs"}.get(solver, solver)
    opt = SolverFactory(name)
    if solver == "highs":
        opt.config.mip_gap = mip_gap
        opt.highs_options = {"threads": threads}
    elif solver == "gurobi":
        opt.options.update({"Threads": threads, "MIPGap": mip_gap})
    t0 = time.perf_counter()
    res = opt.solve(m) if solver == "highs" else opt.solve(m, tee=tee)
    secs = time.perf_counter() - t0
    # appsi results carry .termination_condition; classic pyomo results carry .solver.termination_condition
    tc = getattr(res, "termination_condition", None) or res.solver.termination_condition
    if "optimal" not in str(tc).lower() and tc != TerminationCondition.feasible:
        raise RuntimeError(f"Stage-2 solve failed: termination={tc}")
    return {
        "c": {j: int(round(value(m.c[j]))) for j in m.J},
        "objective": float(value(m.obj)),
        "capex": float(value(m.capex)),
        "gamma": float(value(m.gamma)),
        "solve_s": secs,
    }
