#!/usr/bin/env python3
"""Solve-time scaling of the Stage-2 DRO MILP (Section V.E).

For k in evaluation.scaling_k, S and R are each replicated k times; every copy is multiplied
by independent U(0.98, 1.02) site-wise noise (seeded) so copies are distinct, and distances
are recomputed. Each instance is solved `scaling_repeats` times at eps = 0.5 eps_hat with the
312-charger budget and `scaling_mip_gap`; the median wall time is fitted with T(k) = a k^b.

Outputs: results/eval/scaling.csv, results/eval/scaling_fit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uavdro.common import base_parser, ensure_dir, load_config, read_site_list, write_json  # noqa: E402
from uavdro.dro import build_model, l1_distances, solve  # noqa: E402


def replicate(df: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.DataFrame:
    if k == 1:
        return df.copy()
    cols = [c for c in df.columns if c.startswith("lam_")]
    parts = []
    for i in range(k):
        p = df.copy()
        p[cols] = p[cols].to_numpy() * rng.uniform(0.98, 1.02, size=p[cols].shape)
        p["sid"] = p["sid"] + f"_k{i}"
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    ev, s2 = cfg["evaluation"], cfg["stage2"]
    res = cfg["paths"]["results"]
    sites = read_site_list(cfg["paths"]["footprint"])
    S0 = pd.read_csv(res / "scenarios" / "scenarios_S.csv")
    R0 = pd.read_csv(res / "scenarios" / "scenarios_R.csv")
    meta = json.loads((res / "scenarios" / "eps_hat.json").read_text())
    cap = s2["beta"] * s2["mu"] * s2["c_max"]

    rows = []
    for k in ev["scaling_k"]:
        rng = np.random.default_rng(1000 + k)
        S, R = replicate(S0, k, rng), replicate(R0, k, rng)
        for df in (S, R):
            cols = [c for c in df.columns if c.startswith("lam_")]
            df[cols] = df[cols].clip(upper=cap)
        D = l1_distances(S, R, meta["Lambda_bar"])
        times = []
        for _ in range(ev["scaling_repeats"]):
            m = build_model(S, R, D, sites, 0.5 * meta["eps_hat"], mu=s2["mu"], beta=s2["beta"],
                            c_max=s2["c_max"], bkpts=s2["pwl_breakpoints"], coeffs=s2["pwl_coeffs"],
                            total_chargers=s2["total_chargers"])
            times.append(solve(m, s2["solver"], s2["threads"], ev["scaling_mip_gap"])["solve_s"])
        rows.append({"k": k, "S": len(S), "R": len(R), "solve_s_median": float(np.median(times)),
                     "solve_s_all": times})
        print(f"[k={k:2d}] |S|={len(S):3d} |R|={len(R):3d} median={np.median(times):.2f}s  {np.round(times, 2)}")
    df = pd.DataFrame(rows)
    df.to_csv(ensure_dir(res / "eval") / "scaling.csv", index=False)

    x, y = np.log(df["k"].to_numpy(float)), np.log(df["solve_s_median"].to_numpy(float))
    b, loga = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (loga + b * x)) ** 2) / np.sum((y - y.mean()) ** 2)
    fit = {"a": float(np.exp(loga)), "b": float(b), "r2": float(r2), "solver": s2["solver"], "mip_gap": ev["scaling_mip_gap"],
           "threads": s2["threads"]}
    write_json(fit, res / "eval" / "scaling_fit.json")
    print(f"[fit] T(k) = {fit['a']:.2f} k^{fit['b']:.2f}  R^2 = {fit['r2']:.2f}")


if __name__ == "__main__":
    main()
