#!/usr/bin/env python3
"""Queue evaluation of every design (Sections V.A–B, V.D, V.G).

Designs: uniform c in {4, 6, 8} on the footprint, plus every results/stage2/<run>/design.csv.
For each: M/M/c system p95 (analytic mixture over stable sites), unstable sites (lam > 0 and
c = 0 or rho >= 1) under nominal S and the worst member of R, and the SimPy p95
(evaluation.sim_hours x evaluation.sim_seeds).

Outputs: results/eval/designs.csv, results/eval/site_detail_<design>.csv, results/eval/delta_c.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uavdro.common import base_parser, ensure_dir, load_config, read_design, read_site_list, write_design  # noqa: E402
from uavdro.evaluate import evaluate_design, scenario_rates  # noqa: E402
from uavdro.queueing import site_table  # noqa: E402


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--no_sim", action="store_true", help="skip the SimPy check (analytic only)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ev, s2 = cfg["evaluation"], cfg["stage2"]
    res = cfg["paths"]["results"]
    out = ensure_dir(res / "eval")
    sites = read_site_list(cfg["paths"]["footprint"])
    S = pd.read_csv(res / "scenarios" / "scenarios_S.csv")
    R = pd.read_csv(res / "scenarios" / "scenarios_R.csv")
    lam = scenario_rates(S)["nominal"]
    sim = None if args.no_sim else {"hours": ev["sim_hours"], "seeds": ev["sim_seeds"]}

    designs = {}
    for cu in (ev["baseline_c"], 6, 8):
        designs[f"uniform_c{cu}"] = {j: int(cu) for j in sites}
        write_design(designs[f"uniform_c{cu}"], ensure_dir(res / "designs") / f"uniform_c{cu}.csv")
    for f in sorted((res / "stage2").glob("*/design.csv")):
        designs[f.parent.name] = read_design(f)
    for upg in sorted((res / "designs").glob("upgrade_*.csv")):   # written by 06_upgrade_outage.py
        designs[upg.stem] = read_design(upg)

    rows = []
    for name, c in designs.items():
        mu = s2["mu"]
        if name.startswith("sens_mu"):
            mu = float(name[len("sens_mu"):])
        r = {"design": name, "mu": mu, **evaluate_design(c, lam, R, mu, sim)}
        rows.append(r)
        pd.DataFrame(site_table(lam, c, mu)).to_csv(out / f"site_detail_{name}.csv", index=False)
        print(f"[{name:>18}] c={r['total_c']:4d} p95={r['p95_min']:7.2f} unstable={r['unstable']:2d} "
              f"worstR: p95={r['p95_worstR_min']:7.2f} unst={r['unstable_worstR']:2d}"
              + (f"  sim p95={r['p95_sim_mean']:.2f}±{r['p95_sim_std']:.2f}" if sim else ""))
    pd.DataFrame(rows).to_csv(out / "designs.csv", index=False)

    # Per-site reallocation of the headline DRO design vs the uniform baseline (Fig. 2c)
    head = f"dro_eps{0.5 * float(pd.read_json(res / 'scenarios' / 'eps_hat.json', typ='series')['eps_hat']):.6g}"
    if head in designs:
        base = designs[f"uniform_c{ev['baseline_c']}"]
        dc = pd.DataFrame({"site_id": sites, "lam_hr": [lam.get(j, 0.0) for j in sites],
                           "c_base": [base[j] for j in sites], "c_dro": [designs[head][j] for j in sites]})
        dc["delta_c"] = dc["c_dro"] - dc["c_base"]
        dc.sort_values("delta_c", ascending=False).to_csv(out / "delta_c.csv", index=False)
        print(f"[delta_c] {head}: +{dc.delta_c.clip(lower=0).sum()} / {dc.delta_c.clip(upper=0).sum()} chargers moved")


if __name__ == "__main__":
    main()
