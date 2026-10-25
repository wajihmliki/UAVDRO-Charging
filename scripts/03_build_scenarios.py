#!/usr/bin/env python3
"""Stage-2 inputs (Section IV.A–D) from the Stage-1 footprint.

Every demand cell is served by its nearest open footprint site within `max_travel_min`.
Site arrival rates follow `stage2.rate_mode` (default: each site's peak hour, see
uavdro.demand.site_rate_panel). S is the mean over days; worst-site RO is the max over days.

Outputs (results/scenarios/):
  cell_assignment.csv     grid_id, site_id, wbar (mean daily deliveries)
  scenarios_S.csv         empirical set S: one row 'nominal' = mean over days     (|S| = 1)
  site_rates_daily.csv    per-day hourly rates (184 rows), used for worst-site RO
  site_peak_hour.csv      peak hour-of-day per site (rate_mode=peak_hour only)
  scenarios_R.csv         stressed support R: nominal + 5 hotspot pairs x {a3, a6} (|R| = 11)
  scenarios_R_inflated.csv  RO baseline: S x ro_inflation
  scenarios_R_worstsite.csv RO baseline: per-site max over days
  dist_sr.csv             d(s, r) = ||lam_s - lam_r||_1 / Lambda_bar
  eps_hat.json            eps_hat = median d(s, r) over pairs with d > 0
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uavdro.common import base_parser, ensure_dir, load_config, read_site_list, write_json  # noqa: E402
from uavdro.demand import (  # noqa: E402
    assign_nearest, daily_cell_demand, load_arcs, site_rate_panel, unserved_share,
)
from uavdro.dro import l1_distances  # noqa: E402


def wide(rows: dict[str, pd.Series], sites) -> pd.DataFrame:
    df = pd.DataFrame({sid: v.reindex(sites).fillna(0.0) for sid, v in rows.items()}).T
    df.columns = [f"lam_{j}" for j in df.columns]
    return df.rename_axis("sid").reset_index()


def shift_mass(base: pd.Series, hot: list[str], alpha: float) -> pd.Series:
    """Add alpha*Lambda split evenly over `hot`; remove the same mass uniformly from the
    other sites, never below zero (water-filling), so total demand is preserved (eqs. 5–6)."""
    v = base.copy()
    bump = alpha * base.sum()
    for h in hot:
        v[h] += bump / len(hot)
    pool = [j for j in v.index if j not in hot]
    need = bump
    while need > 1e-9:
        live = [j for j in pool if v[j] > 0]
        if not live:
            raise RuntimeError("not enough mass to remove")
        step = min(need / len(live), min(v[j] for j in live))
        for j in live:
            v[j] -= step
        need -= step * len(live)
    return v


def capped(v: pd.Series, cap: float, label: str) -> pd.Series:
    over = v[v > cap]
    if len(over):
        print(f"[warn] {label}: capping {len(over)} site(s) at {cap:.1f}/hr: {over.round(1).to_dict()}")
    return v.clip(upper=cap)


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    s2 = cfg["stage2"]
    out = ensure_dir(cfg["paths"]["results"] / "scenarios")
    sites = read_site_list(cfg["paths"]["footprint"])
    cap = s2["beta"] * s2["mu"] * s2["c_max"]          # 121.6 jobs/hr

    daily = daily_cell_demand(cfg["paths"]["processed"])
    wbar = daily.mean(axis=0)
    arcs = load_arcs(cfg["paths"]["processed"], max_tt=s2["max_travel_min"])
    asg = assign_nearest(arcs, sites)
    print(f"[assign] footprint={len(sites)} cells served={len(asg):,}/{len(wbar):,} "
          f"unserved demand share={unserved_share(wbar, asg):.4f}")
    pd.DataFrame({"grid_id": asg.index, "site_id": asg.values,
                  "wbar": wbar.reindex(asg.index).values}).to_csv(out / "cell_assignment.csv", index=False)

    per_day, peak = site_rate_panel(cfg["paths"]["processed"], asg, sites, s2["rate_mode"],
                                    cfg["phase1"]["slot_min"], s2["hours_per_day"])
    per_day.rename_axis("date").to_csv(out / "site_rates_daily.csv")
    if len(peak):
        peak.rename("peak_hour").rename_axis("site_id").to_csv(out / "site_peak_hour.csv")

    lam = capped(per_day.mean(axis=0), cap, "S nominal")
    S = wide({"nominal": lam}, sites)
    S.to_csv(out / "scenarios_S.csv", index=False)
    print(f"[S] rate_mode={s2['rate_mode']} days={len(per_day)} total={lam.sum():.2f}/hr  "
          f"max={lam.max():.2f} ({lam.idxmax()})  zero-demand sites={(lam == 0).sum()}")

    # Stressed support R
    hot = lam.sort_values(ascending=False).index[: s2["stress"]["topk"]].tolist()
    rows = {"nominal": lam}
    for h1, h2 in zip(hot[0::2], hot[1::2]):
        for a in s2["stress"]["alphas"]:
            rows[f"stress_{h1}_{h2}_a{round(100 * a)}"] = capped(shift_mass(lam, [h1, h2], a), cap, f"R {h1}+{h2} a={a}")
    R = wide(rows, sites)
    R.to_csv(out / "scenarios_R.csv", index=False)
    print(f"[R] |R|={len(R)} hotspots={hot}")

    # RO baselines
    wide({"inflated": capped(lam * s2["ro_inflation"], cap, "RO inflated")}, sites).to_csv(
        out / "scenarios_R_inflated.csv", index=False)
    wide({"worst_site": capped(per_day.max(axis=0), cap, "RO worst-site")}, sites).to_csv(
        out / "scenarios_R_worstsite.csv", index=False)

    # Ground metric and eps_hat
    Lbar = float(S.filter(like="lam_").mean().sum())
    dist = l1_distances(S, R, Lbar)
    dist.to_csv(out / "dist_sr.csv", index=False)
    eps_hat = float(dist.loc[dist.d > 0, "d"].median())
    write_json({"eps_hat": eps_hat, "Lambda_bar": float(Lbar), "definition": "median d(s,r) over S x R, d>0"},
               out / "eps_hat.json")
    print(f"[eps] Lambda_bar={Lbar:.2f}/hr  eps_hat={eps_hat:.6f}  -> {out}")


if __name__ == "__main__":
    main()
