#!/usr/bin/env python3
"""Stage 1 — N-1 reliability-aware siting MILP (Appendix A).  Requires Gurobi.

  min  sum_j f_j y_j + Z
  s.t. Z >= sum_u w̄_u (pi_unmet x^unmet_{u,ξ} + sum_j tt_{u,j} x_{u,j,ξ})   for all contingencies ξ
       x^unmet_{u,ξ} + sum_j x_{u,j,ξ} = 1,  x_{u,j,ξ} <= y_j,  x_{u,j',ξ_j'} = 0,  sum_j y_j <= O_max

Solved in two passes (Appendix A.E): `quick` (top cells, K=4, 40 contingencies, 20% gap) then
`full` (K=6, 60 contingencies, 5% gap) warm-started from the quick footprint.
The full pass has ~6.4M rows and took ~5.4 h on 16 threads with Gurobi 12.0.3.

The paper's footprint is frozen in data/reference/footprint_stage1.csv; Stage 2 reads it from
`paths.footprint`. Point that path at results/stage1/footprint_full.csv to use a fresh solve.

Outputs: results/stage1/footprint_{quick,full}.csv, results/stage1/summary_{quick,full}.json
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from pyomo.environ import (
    Binary, ConcreteModel, Constraint, NonNegativeReals, Objective, Set, SolverFactory, Var, minimize, value,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uavdro.common import base_parser, ensure_dir, load_config, write_json  # noqa: E402
from uavdro.demand import load_arcs, mean_cell_demand  # noqa: E402


def build(cfg: dict, topk_use: int, max_contingencies: int, keep_top_u: int | None):
    s1 = cfg["stage1"]
    proc = cfg["paths"]["processed"]
    cand = pd.read_csv(proc / "candidate_sites.csv")
    J = cand["site_id"].astype(str).tolist()
    wbar = mean_cell_demand(proc)
    wbar = wbar[wbar > 0]
    if keep_top_u:
        wbar = wbar.sort_values(ascending=False).head(int(keep_top_u))
    arcs = load_arcs(proc, topk=topk_use)
    arcs = arcs[(arcs["tt_min"] < s1["pi_unmet"]) & arcs["grid_id"].isin(wbar.index)]
    U = wbar.index.tolist()

    Ju = defaultdict(list)
    for g, j, tt in arcs[["grid_id", "site_id", "tt_min"]].itertuples(index=False):
        Ju[g].append((j, float(tt)))
    P = [(u, j) for u in U for j, _ in Ju[u]]
    tt = {(u, j): t for u in U for j, t in Ju[u]}

    # contingencies: normal + outages of the sites with the largest rank-1 demand pull
    r1 = arcs[arcs["rank"] == 1]
    pull = r1.assign(w=r1["grid_id"].map(wbar)).groupby("site_id")["w"].sum().sort_values(ascending=False)
    ranked = pull.index.tolist() + [j for j in J if j not in set(pull.index)]
    C = ["normal"] + [f"down_{j}" for j in ranked][: max_contingencies - 1]
    down = {c: (c[5:] if c.startswith("down_") else None) for c in C}

    m = ConcreteModel("stage1")
    m.U, m.J, m.C = Set(initialize=U), Set(initialize=J), Set(initialize=C)
    m.P = Set(dimen=2, initialize=P)
    m.y = Var(m.J, domain=Binary)
    m.x = Var(m.P, m.C, domain=NonNegativeReals)
    m.xu = Var(m.U, m.C, domain=NonNegativeReals)
    m.Z = Var(domain=NonNegativeReals)
    m.cover = Constraint(m.U, m.C, rule=lambda m, u, c: m.xu[u, c] + sum(m.x[(u, j), c] for j, _ in Ju[u]) == 1.0)
    m.link = Constraint(m.P, m.C, rule=lambda m, u, j, c: m.x[(u, j), c] <= m.y[j])
    m.down = Constraint([(u, j, c) for c in C if down[c] for (u, j) in P if j == down[c]],
                        rule=lambda m, u, j, c: m.x[(u, j), c] == 0.0)
    m.epi = Constraint(m.C, rule=lambda m, c: m.Z >= sum(wbar[u] * s1["pi_unmet"] * m.xu[u, c] for u in U)
                       + sum(wbar[u] * tt[u, j] * m.x[(u, j), c] for (u, j) in P))
    m.cap = Constraint(expr=sum(m.y[j] for j in J) <= int(s1["open_cap"]))
    m.obj = Objective(expr=s1["install_cost"] * sum(m.y[j] for j in J) + m.Z, sense=minimize)

    # demand-pull warm start
    for j in pull.index[: int(s1["warm_start"])]:
        m.y[j].value = 1
    print(f"[stage1] |U|={len(U):,} |J|={len(J)} |C|={len(C)} |P|={len(P):,} -> x vars={len(P) * len(C):,}")
    return m


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--passes", nargs="+", default=["quick", "full"], choices=["quick", "full"])
    ap.add_argument("--solver", default="gurobi")
    ap.add_argument("--time_limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    s1 = cfg["stage1"]
    out = ensure_dir(cfg["paths"]["results"] / "stage1")

    warm = None
    for name in args.passes:
        p = s1[name]
        m = build(cfg, p["topk_use"], p["max_contingencies"], p.get("keep_top_u"))
        if warm is not None:
            for j in m.J:
                m.y[j].value = 1 if j in warm else 0
        opt = SolverFactory(args.solver)
        if args.solver == "gurobi":
            node_dir = ensure_dir(out / "nodefiles")
            opt.options.update({"MIPGap": p["mip_gap"], "Threads": s1["threads"], "NodefileStart": 0.5,
                                "NodefileDir": str(node_dir), "MIPFocus": 1, "Heuristics": 0.2, "Cuts": 1,
                                "Presolve": 2, "LogFile": str(out / f"gurobi_{name}.log")})
            if args.time_limit:
                opt.options["TimeLimit"] = args.time_limit
        t0 = time.perf_counter()
        opt.solve(m, tee=True, warmstart=(args.solver == "gurobi"))
        secs = time.perf_counter() - t0
        y = sorted(j for j in m.J if (m.y[j].value or 0) > 0.5)
        pd.DataFrame({"site_id": y}).to_csv(out / f"footprint_{name}.csv", index=False)
        write_json({"pass": name, "objective": float(value(m.obj)), "Z": float(value(m.Z)),
                    "open_count": len(y), "solve_s": secs, **p}, out / f"summary_{name}.json")
        print(f"[stage1:{name}] open={len(y)} obj={value(m.obj):.2f} t={secs:.0f}s")
        warm = set(y)


if __name__ == "__main__":
    main()
