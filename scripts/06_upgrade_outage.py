#!/usr/bin/env python3
"""Targeted upgrades (Section V.C, Tables 3–4) and two-station outages (Table 2).

Each upgrade raises N hotspots to the smallest c with worst-case utilisation
max_r lam_rj / (mu c) <= upgrade_target_rho:
  upgrade_uniform  Table-3 rule: uniform baseline, hotspots = top-N sites by DRO reallocation Δc
  upgrade_dro      Table-4 row "DRO + targeted upgrade": headline DRO design (eps = 0.5 eps_hat),
                   hotspots = its N highest worst-case utilisations

Outages: for each pair of down sites (c = 0):
  no-reroute  their demand is not served (they count as unstable, all other sites unchanged)
  reroute     their cells move to the nearest remaining open site within max_travel_min
Reported for the baseline and the upgraded design: active sites, mean wait, p95, p99, unstable,
unserved demand share.

Outputs: results/designs/upgrade_{uniform,dro}.csv, results/eval/upgrade_*_plan.csv,
         results/eval/outages.csv
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uavdro.common import (  # noqa: E402
    base_parser, ensure_dir, load_config, read_design, read_site_list, write_design,
)
from uavdro.demand import assign_nearest, load_arcs, site_rate_panel, unserved_share  # noqa: E402
from uavdro.evaluate import scenario_rates  # noqa: E402
from uavdro.queueing import site_table, system_p95  # noqa: E402


def metrics(lam: dict, c: dict, mu: float) -> dict:
    p95, unstable = system_p95(lam, c, mu, 0.95)
    p99, _ = system_p95(lam, c, mu, 0.99)
    tab = pd.DataFrame(site_table(lam, c, mu))
    st = tab[~tab["unstable"] & (tab["lam_hr"] > 0)]
    mean = float((st["lam_hr"] * st["Wq_mean_min"]).sum() / st["lam_hr"].sum()) if len(st) else float("nan")
    return {"mean_wait_min": mean, "p95_min": p95, "p99_min": p99, "unstable": unstable}


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    ev, s2 = cfg["evaluation"], cfg["stage2"]
    res = cfg["paths"]["results"]
    mu, slot = s2["mu"], cfg["phase1"]["slot_min"]
    sites = read_site_list(cfg["paths"]["footprint"])
    lam = scenario_rates(pd.read_csv(res / "scenarios" / "scenarios_S.csv"))["nominal"]
    R = scenario_rates(pd.read_csv(res / "scenarios" / "scenarios_R.csv"))
    lam_worst = {j: max(r[j] for r in R.values()) for j in sites}

    # ---- upgrades ------------------------------------------------------------------------
    def upgrade(base: dict, hot: list[str]) -> tuple[dict, pd.DataFrame]:
        up, plan = dict(base), []
        for j in hot:
            need = math.ceil(lam_worst[j] / (mu * ev["upgrade_target_rho"]) - 1e-9)
            up[j] = min(max(base[j], need), s2["c_max"])
            plan.append({"site_id": j, "lam_nominal": lam[j], "lam_worstR": lam_worst[j],
                         "rho_worstR_before": lam_worst[j] / (mu * base[j]) if base[j] else math.inf,
                         "c_base": base[j], "c_final": up[j], "delta_c": up[j] - base[j]})
        return up, pd.DataFrame(plan)

    n = ev["upgrade_top_n"]
    base = {j: int(ev["baseline_c"]) for j in sites}
    head = f"dro_eps{0.5 * float(pd.read_json(res / 'scenarios' / 'eps_hat.json', typ='series')['eps_hat']):.6g}"
    dro = read_design(res / "stage2" / head / "design.csv")
    # (a) Table 3 rule: uniform baseline, reinforce the top-n sites by DRO reallocation Δc
    dc = pd.read_csv(res / "eval" / "delta_c.csv")
    hot_u = dc[dc["delta_c"] > 0].sort_values(["delta_c", "lam_hr"], ascending=False)["site_id"].head(n).tolist()
    # (b) "DRO + targeted upgrade" (Table 4): DRO design, reinforce its top-n worst-case utilisations
    hot_d = sorted(sites, key=lambda j: -(lam_worst[j] / (mu * dro[j]) if dro[j] else math.inf))[:n]
    designs = {"baseline": base, "dro": dro}
    for name, b, hot in [("upgrade_uniform", base, hot_u), ("upgrade_dro", dro, hot_d)]:
        up, plan = upgrade(b, hot)
        plan.to_csv(ensure_dir(res / "eval") / f"{name}_plan.csv", index=False)
        write_design(up, ensure_dir(res / "designs") / f"{name}.csv")
        designs[name] = up
        print(f"[{name}] +{plan.delta_c.sum()} chargers -> total {sum(up.values())}")
        print(plan.round(3).to_string(index=False))

    # ---- two-station outages -------------------------------------------------------------
    order = sorted(sites, key=lambda j: -lam[j])
    pairs = ev["outage_pairs"]
    if pairs == "auto":
        pairs = [[order[0], order[1]], [order[0], order[2]]]
    arcs = load_arcs(cfg["paths"]["processed"], max_tt=s2["max_travel_min"])
    wbar = pd.read_csv(res / "scenarios" / "cell_assignment.csv").set_index("grid_id")["wbar"]
    full = assign_nearest(arcs, sites)
    rows = []
    for design_name, c in designs.items():
        rows.append({"design": design_name, "scenario": "no outage", "down": "", "active_sites": len(sites),
                     "unserved_share": 0.0, **metrics(lam, c, mu)})
        for a, b in pairs:
            down = {a, b}
            # no-reroute: down sites keep their demand but have c = 0
            c0 = {j: (0 if j in down else v) for j, v in c.items()}
            rows.append({"design": design_name, "scenario": "no reroute", "down": f"{a}+{b}",
                         "active_sites": len(sites), "unserved_share": sum(lam[j] for j in down) / sum(lam.values()),
                         **metrics(lam, c0, mu)})
            # reroute: re-assign cells to the remaining open sites
            alive = [j for j in sites if j not in down]
            asg = assign_nearest(arcs, alive)
            panel, _ = site_rate_panel(cfg["paths"]["processed"], asg, alive, s2["rate_mode"], slot, s2["hours_per_day"])
            lam_r = panel.mean(axis=0).to_dict()
            rows.append({"design": design_name, "scenario": "reroute", "down": f"{a}+{b}",
                         "active_sites": len(alive),
                         "unserved_share": unserved_share(wbar, asg.reindex(full.index).dropna()),
                         **metrics(lam_r, {j: c[j] for j in alive}, mu)})
    out = pd.DataFrame(rows)
    out.to_csv(res / "eval" / "outages.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
