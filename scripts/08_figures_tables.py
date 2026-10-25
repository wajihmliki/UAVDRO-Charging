#!/usr/bin/env python3
"""Paper tables (CSV + LaTeX) and figures from results/.

Tables -> results/tables/: table1_dro_eps, table2_outages, table3_upgrade, table4_methods,
          table5_sensitivity, scaling
Figures -> results/figures/: fig2a_tail_vs_chargers, fig2b_eps_sweep, fig2c_delta_c,
          fig3_map, fig4_scaling   (PNG 300 dpi + PDF)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uavdro.common import base_parser, ensure_dir, load_config, read_design  # noqa: E402

# Categorical slots in fixed order (validated: CVD-safe on the light surface; slots 3–4 are
# below 3:1 contrast, so every mark also carries a direct label or a distinct marker).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blue_ramp", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "lines.linewidth": 2,
    "lines.markersize": 7, "legend.frameon": False,
})


def save(fig, out: Path, name: str):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def to_tex(df: pd.DataFrame, path: Path, fmt: dict | None = None):
    fmt = fmt or {}
    cols = list(df.columns)
    lines = [r"\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(c.replace("_", r"\_") for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("--" if np.isnan(v) else (fmt.get(c, "{:.2f}").format(v)))
            else:
                cells.append(str(v).replace("_", r"\_"))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    res = cfg["paths"]["results"]
    tabs, figs = ensure_dir(res / "tables"), ensure_dir(res / "figures")
    ev = pd.read_csv(res / "eval" / "designs.csv").set_index("design")
    runs = pd.read_csv(res / "stage2" / "runs.csv").set_index("run")
    eps_hat = json.loads((res / "scenarios" / "eps_hat.json").read_text())["eps_hat"]
    head = f"dro_eps{0.5 * eps_hat:.6g}"
    base = f"uniform_c{cfg['evaluation']['baseline_c']}"

    # ---- Table 1: DRO eps sweep (budget 312) ------------------------------------------------
    dro = [r for r in runs.index if r.startswith("dro_eps")]
    t1 = pd.DataFrame({
        "eps": [runs.loc[r, "eps"] for r in dro],
        "eps_over_eps_hat": [runs.loc[r, "eps"] / eps_hat for r in dro],
        "objective": [runs.loc[r, "objective"] for r in dro],
        "total_c": [int(ev.loc[r, "total_c"]) for r in dro],
        "p95_min": [ev.loc[r, "p95_min"] for r in dro],
        "p95_sim_min": [ev.loc[r, "p95_sim_mean"] for r in dro],
        "unstable": [int(ev.loc[r, "unstable"]) for r in dro],
        "p95_worstR_min": [ev.loc[r, "p95_worstR_min"] for r in dro],
        "unstable_worstR": [int(ev.loc[r, "unstable_worstR"]) for r in dro],
    }).sort_values("eps")
    t1.to_csv(tabs / "table1_dro_eps.csv", index=False)
    to_tex(t1, tabs / "table1_dro_eps.tex", {"eps": "{:.3g}", "eps_over_eps_hat": "{:.3g}"})

    # ---- Table 2: outages ---------------------------------------------------------------------
    t2 = pd.read_csv(res / "eval" / "outages.csv")
    t2.to_csv(tabs / "table2_outages.csv", index=False)
    to_tex(t2, tabs / "table2_outages.tex", {"unserved_share": "{:.3f}"})

    # ---- Table 3: upgrade plans ------------------------------------------------------------------
    for nm in ("upgrade_dro", "upgrade_uniform"):
        p = pd.read_csv(res / "eval" / f"{nm}_plan.csv")
        p = pd.concat([p, pd.DataFrame([{"site_id": "Total", "delta_c": p["delta_c"].sum(),
                                         "c_final": int(ev.loc[nm, "total_c"])}])], ignore_index=True)
        p.to_csv(tabs / f"table3_{nm}.csv", index=False)
        to_tex(p, tabs / f"table3_{nm}.tex")

    # ---- Table 4: method comparison ----------------------------------------------------------
    rows = [(base, "Deterministic (uniform c=4)"), ("ro_inflated", "RO - 1.2x inflation (312)"),
            ("ro_inflated_free", "RO - 1.2x inflation (no budget)"),
            ("ro_worstsite", "RO - worst-site (312)"), ("ro_worstsite_free", "RO - worst-site (no budget)"),
            (head, "DRO (eps = 0.5 eps_hat, 312)"), ("dro_free", "DRO (eps = 0.5 eps_hat, no budget)"),
            ("upgrade_dro", "DRO + targeted upgrade"), ("upgrade_uniform", "Uniform + targeted upgrade (Table 3 rule)")]
    t4 = []
    for key, label in rows:
        if key in ev.index:
            e = ev.loc[key]
            t4.append({"method": label, "total_c": int(e.total_c), "active_sites": int(e.active_sites),
                       "p95_min": e.p95_min, "p95_sim_min": e.p95_sim_mean, "unstable": int(e.unstable),
                       "unstable_worstR": int(e.unstable_worstR)})
        elif key in runs.index and runs.loc[key, "status"] == "infeasible":
            t4.append({"method": label + " - infeasible", "total_c": np.nan, "active_sites": np.nan,
                       "p95_min": np.nan, "p95_sim_min": np.nan, "unstable": np.nan, "unstable_worstR": np.nan})
    t4 = pd.DataFrame(t4)
    t4.to_csv(tabs / "table4_methods.csv", index=False)
    to_tex(t4, tabs / "table4_methods.tex", {"total_c": "{:.0f}", "active_sites": "{:.0f}",
                                             "unstable": "{:.0f}", "unstable_worstR": "{:.0f}"})

    # ---- Table 5: sensitivity ------------------------------------------------------------------
    sens = [head] + [r for r in ev.index if r.startswith("sens_")]
    t5 = ev.loc[sens, ["mu", "total_c", "p95_min", "p95_sim_mean", "unstable", "unstable_worstR"]].reset_index()
    t5.to_csv(tabs / "table5_sensitivity.csv", index=False)
    to_tex(t5, tabs / "table5_sensitivity.tex", {"mu": "{:.0f}"})

    # ---- Fig 2a: tail wait vs total chargers ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    uni = ev.loc[[f"uniform_c{c}" for c in (cfg["evaluation"]["baseline_c"], 6, 8)]]
    ax.plot(uni.total_c, uni.p95_min, "-o", color=BLUE, label="Uniform c = 4, 6, 8")
    pts = [(head, ORANGE, "s", "DRO (312)"), ("upgrade_dro", AQUA, "D", "DRO + upgrade"),
           ("ro_inflated", YELLOW, "^", "RO 1.2x (312)")]
    for key, col, mk, lab in pts:
        ax.plot(ev.loc[key, "total_c"], ev.loc[key, "p95_min"], mk, color=col, label=lab,
                markeredgecolor=SURFACE, markeredgewidth=1.5, markersize=9)
    offsets = {head: (10, 6), "upgrade_dro": (10, -12)}
    for key, e in ev.loc[list(uni.index) + [k for k, *_ in pts]].iterrows():
        dx, dy = offsets.get(key, (6, 4))
        ax.annotate(f"{int(e.unstable)} unstable", (e.total_c, e.p95_min), textcoords="offset points",
                    xytext=(dx, dy), fontsize=7, color=INK2, ha="right" if dx < 0 else "left")
    ax.set_xlabel("Total chargers")
    ax.set_ylabel("System p95 wait, stable sites (min)")
    ax.legend(fontsize=7, loc="upper right")
    save(fig, figs, "fig2a_tail_vs_chargers")

    # ---- Fig 2b: eps sweep, two panels (no dual axis) -----------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.8), sharex=True)
    a1.plot(t1.eps_over_eps_hat, t1.objective, "-o", color=BLUE)
    a1.set_ylabel("Objective")
    a2.plot(t1.eps_over_eps_hat, t1.p95_min, "-o", color=BLUE, label="nominal S")
    a2.plot(t1.eps_over_eps_hat, t1.p95_worstR_min, "--s", color=ORANGE, label="worst r in R")
    a2.set_ylabel("System p95 wait (min)")
    a2.legend(fontsize=7)
    for a in (a1, a2):
        a.set_xscale("symlog", linthresh=0.1)
        a.set_xlabel(r"$\varepsilon / \hat\varepsilon$")
    save(fig, figs, "fig2b_eps_sweep")

    # ---- Fig 2c: per-site reallocation Δc -----------------------------------------------------
    dc = pd.read_csv(res / "eval" / "delta_c.csv").sort_values("delta_c", ascending=False)
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    colors = [BLUE if v > 0 else ORANGE for v in dc.delta_c]
    ax.bar(range(len(dc)), dc.delta_c, color=colors, width=0.8, edgecolor=SURFACE, linewidth=0.5)
    ax.axhline(0, color=INK2, linewidth=0.8)
    for i, (sid, v) in enumerate(zip(dc.site_id, dc.delta_c)):
        if i < 7:
            ax.annotate(sid, (i, v), textcoords="offset points", xytext=(0, 2), ha="center",
                        fontsize=6, rotation=90, va="bottom", color=INK2)
    ax.set_xticks([])
    ax.set_xlabel("Footprint sites, sorted by change")
    ax.set_ylabel(r"$\Delta c_j$ (DRO $-$ uniform)")
    ax.set_ylim(dc.delta_c.min() - 1, dc.delta_c.max() + 4)
    save(fig, figs, "fig2c_delta_c")

    # ---- Fig 3: demand map + design ---------------------------------------------------------------
    cells = pd.read_csv(cfg["paths"]["processed"] / "grid_cells.csv")
    asg = pd.read_csv(res / "scenarios" / "cell_assignment.csv")
    cells = cells.merge(asg[["grid_id", "wbar"]], on="grid_id", how="inner")
    cand = pd.read_csv(cfg["paths"]["processed"] / "candidate_sites.csv").set_index("site_id")
    up = read_design(res / "designs" / "upgrade_dro.csv")
    dro_c = read_design(res / "stage2" / head / "design.csv")
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.grid(False)
    sc = ax.scatter(cells.centroid_lng, cells.centroid_lat, c=np.log10(cells.wbar.clip(lower=0.01)), s=2,
                    cmap=BLUES, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("Mean daily deliveries per cell (log10)")
    sites = cand.loc[list(up)]
    ax.scatter(sites.centroid_lng, sites.centroid_lat, s=[6 + 3 * up[j] for j in sites.index], c=INK,
               edgecolors=SURFACE, linewidths=0.8, label="Open site (size = chargers)")
    boosted = [j for j in up if up[j] > dro_c[j]]
    ax.scatter(cand.loc[boosted].centroid_lng, cand.loc[boosted].centroid_lat, s=90, facecolors="none",
               edgecolors=ORANGE, linewidths=1.6, label="Upgraded site")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1 / np.cos(np.radians(31.2)))
    ax.legend(fontsize=7, loc="lower left")
    save(fig, figs, "fig3_map")

    # ---- Fig 4 + table: scaling -----------------------------------------------------------------
    sp = res / "eval" / "scaling.csv"
    if sp.exists():
        sc_df = pd.read_csv(sp)
        fit = json.loads((res / "eval" / "scaling_fit.json").read_text())
        sc_df[["k", "S", "R", "solve_s_median"]].to_csv(tabs / "scaling.csv", index=False)
        fig, ax = plt.subplots(figsize=(4.0, 3.0))
        ax.loglog(sc_df.k, sc_df.solve_s_median, "o", color=BLUE, label="median of runs")
        kk = np.linspace(sc_df.k.min(), sc_df.k.max(), 50)
        ax.loglog(kk, fit["a"] * kk ** fit["b"], "-", color=INK2, linewidth=1,
                  label=f"T(k) = {fit['a']:.2f} k^{fit['b']:.2f}, R² = {fit['r2']:.2f}")
        ax.set_xlabel("Scenario multiplier k")
        ax.set_ylabel("Solve time (s)")
        ax.legend(fontsize=7)
        save(fig, figs, "fig4_scaling")

    print(f"[ok] tables -> {tabs}\n[ok] figures -> {figs}")


if __name__ == "__main__":
    main()
