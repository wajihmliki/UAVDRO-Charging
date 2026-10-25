#!/usr/bin/env python3
"""Stage 2 — size chargers on the fixed 78-site footprint (Sections IV.E–F, V.B, V.D, V.G).

Runs (each -> results/stage2/<run>/design.csv + summary.json; all -> results/stage2/runs.csv):
  dro_eps*        1-Wasserstein DRO, eps in {1e-6, 0.5, 1, 2} x eps_hat, budget sum c = 312
  dro_free        same model without the budget (cost-minimal design), eps = 0.5 eps_hat
  ro_inflated     classical RO: support = {1.2 x nominal}, budget 312
  ro_worstsite    classical RO: support = {per-site max over days}, budget 312 (may be infeasible)
  ro_*_free       the same RO models without the budget (cost-minimal stable design)
  sens_mu7/mu9    eps = 0.5 eps_hat with mu = 7, 9
  sens_q2         eps = 0.5 eps_hat with PWL coefficients x 2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uavdro.common import base_parser, ensure_dir, load_config, read_site_list, write_design, write_json  # noqa: E402
from uavdro.dro import build_model, l1_distances, solve  # noqa: E402


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--only", nargs="*", help="run only these run names")
    args = ap.parse_args()
    cfg = load_config(args.config)
    s2 = cfg["stage2"]
    scen = cfg["paths"]["results"] / "scenarios"
    out = ensure_dir(cfg["paths"]["results"] / "stage2")
    sites = read_site_list(cfg["paths"]["footprint"])

    S = pd.read_csv(scen / "scenarios_S.csv")
    R = pd.read_csv(scen / "scenarios_R.csv")
    D = pd.read_csv(scen / "dist_sr.csv")
    meta = json.loads((scen / "eps_hat.json").read_text())
    eps_hat, Lbar = meta["eps_hat"], meta["Lambda_bar"]
    budget = int(s2["total_chargers"])
    eps_half = 0.5 * eps_hat

    def eps_of(mult):
        return mult if mult < 1e-3 else mult * eps_hat

    runs = []
    for mult in s2["eps_multipliers"]:
        runs.append(dict(name=f"dro_eps{eps_of(mult):.6g}", kind="DRO", eps=eps_of(mult), R=R, D=D, budget=budget))
    runs.append(dict(name="dro_free", kind="DRO (no budget)", eps=eps_half, R=R, D=D, budget=None))
    # Classical RO = min capex + max_{r in R} Q_r: zero transport cost (d = 0) makes the dual cut
    # alpha_s >= Q_r for every r. (With the true d > eps the dual would be unbounded, since the
    # RO support does not contain the empirical point.)
    for nm, f in [("ro_inflated", "scenarios_R_inflated.csv"), ("ro_worstsite", "scenarios_R_worstsite.csv")]:
        Rro = pd.read_csv(scen / f)
        D0 = l1_distances(S, Rro, Lbar).assign(d=0.0)
        runs.append(dict(name=nm, kind="RO", eps=0.0, R=Rro, D=D0, budget=budget, may_fail=True))
        runs.append(dict(name=f"{nm}_free", kind="RO (no budget)", eps=0.0, R=Rro, D=D0, budget=None))
    for mu in s2.get("sensitivity_mu", cfg["evaluation"]["sensitivity_mu"]):
        if float(mu) != float(s2["mu"]):
            runs.append(dict(name=f"sens_mu{mu:g}", kind="sensitivity", eps=eps_half, R=R, D=D, budget=budget, mu=float(mu)))
    runs.append(dict(name="sens_q2", kind="sensitivity", eps=eps_half, R=R, D=D, budget=budget,
                     coeffs=[2 * q for q in s2["pwl_coeffs"]]))
    if args.only:
        runs = [r for r in runs if r["name"] in set(args.only)]

    rows = []
    for r in runs:
        mu = r.get("mu", s2["mu"])
        m = build_model(S, r["R"], r["D"], sites, r["eps"], mu=mu, beta=s2["beta"], c_max=s2["c_max"],
                        bkpts=s2["pwl_breakpoints"], coeffs=r.get("coeffs", s2["pwl_coeffs"]),
                        total_chargers=r["budget"])
        d = ensure_dir(out / r["name"])
        try:
            res = solve(m, s2["solver"], s2["threads"], s2["mip_gap"])
        except RuntimeError as e:
            if not r.get("may_fail"):
                raise
            summ = {"run": r["name"], "kind": r["kind"], "eps": r["eps"], "mu": mu, "budget": r["budget"],
                    "status": "infeasible", "note": f"no stable design within sum c = {r['budget']}"}
            write_json(summ, d / "summary.json")
            (d / "design.csv").unlink(missing_ok=True)
            rows.append(summ)
            print(f"[{r['name']:>16}] INFEASIBLE at budget {r['budget']} ({e.__class__.__name__})")
            continue
        write_design(res["c"], d / "design.csv")
        summ = {"run": r["name"], "kind": r["kind"], "eps": r["eps"], "mu": mu, "budget": r["budget"],
                "status": "optimal", "objective": res["objective"], "capex": res["capex"], "gamma": res["gamma"],
                "total_c": sum(res["c"].values()), "active_sites": sum(v > 0 for v in res["c"].values()),
                "solve_s": res["solve_s"], "solver": s2["solver"]}
        write_json(summ, d / "summary.json")
        rows.append(summ)
        print(f"[{r['name']:>16}] obj={res['objective']:.3f} total_c={summ['total_c']} "
              f"active={summ['active_sites']} t={res['solve_s']:.2f}s")
    if not args.only:
        pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)


if __name__ == "__main__":
    main()
