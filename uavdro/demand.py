"""Cell demand, cell->site arcs and nearest-open-site assignment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import read_parquet_or_csv


def daily_cell_demand(processed: Path) -> pd.DataFrame:
    """Deliveries per (day, cell): DataFrame indexed by date, one column per grid_id (zeros filled)."""
    d = read_parquet_or_csv(Path(processed) / "demand_by_day_celltime.parquet")
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    return d.pivot_table(index="date", columns="grid_id", values="count", aggfunc="sum", fill_value=0.0)


def mean_cell_demand(processed: Path) -> pd.Series:
    """w̄_u: mean daily deliveries per cell over all observed days (Appendix A)."""
    return daily_cell_demand(processed).mean(axis=0).rename("wbar")


def load_arcs(processed: Path, max_tt: float | None = None, topk: int | None = None) -> pd.DataFrame:
    a = read_parquet_or_csv(Path(processed) / "cell_to_sites_topk.parquet")
    a["grid_id"] = a["grid_id"].astype(str)
    a["site_id"] = a["site_id"].astype(str)
    if topk is not None:
        a = a[a["rank"] <= int(topk)]
    if max_tt is not None:
        a = a[a["tt_min"] <= float(max_tt)]
    return a.reset_index(drop=True)


def assign_nearest(arcs: pd.DataFrame, open_sites) -> pd.Series:
    """For every cell, the nearest open site among its arcs (cells with none are absent)."""
    a = arcs[arcs["site_id"].isin(set(open_sites))]
    a = a.sort_values(["grid_id", "tt_min", "rank"]).drop_duplicates("grid_id", keep="first")
    return a.set_index("grid_id")["site_id"]


def site_rates(cell_weights: pd.Series, assignment: pd.Series, sites, hours_per_day: float) -> pd.Series:
    """Hourly arrival rate per site: sum of assigned cells' daily deliveries / hours_per_day."""
    w = cell_weights.reindex(assignment.index).fillna(0.0)
    lam = w.groupby(assignment.values).sum() / float(hours_per_day)
    return lam.reindex(list(sites)).fillna(0.0)


def daily_site_rates(daily: pd.DataFrame, assignment: pd.Series, sites, hours_per_day: float) -> pd.DataFrame:
    """Per-day hourly rates per site (rows = days)."""
    cols = [g for g in assignment.index if g in daily.columns]
    agg = daily[cols].T.groupby(assignment.loc[cols].values).sum().T / float(hours_per_day)
    return agg.reindex(columns=list(sites)).fillna(0.0)


def site_rate_panel(processed: Path, assignment: pd.Series, sites, mode: str,
                    slot_min: int, hours_per_day: float = 24.0) -> tuple[pd.DataFrame, pd.Series]:
    """Per-day hourly arrival rate at every site (rows = all observed days, zeros filled).

    mode='peak_hour': each site's peak hour h*_j is the hour-of-day with the highest mean
        deliveries over days; the day-d rate is the deliveries at j during h*_j on day d.
    mode='daily24'  : the day-d rate is the day's deliveries at j / hours_per_day.
    Returns (panel, peak_hour_per_site) — the second is empty for 'daily24'.
    """
    d = read_parquet_or_csv(Path(processed) / "demand_by_day_celltime.parquet")
    d["date"] = pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
    days = sorted(d["date"].unique())
    d = d[d["grid_id"].isin(assignment.index)].copy()
    d["site_id"] = assignment.loc[d["grid_id"]].to_numpy()
    sites = list(sites)
    if mode == "daily24":
        panel = d.pivot_table(index="date", columns="site_id", values="count", aggfunc="sum", fill_value=0.0)
        panel = panel.reindex(index=days, columns=sites, fill_value=0.0) / float(hours_per_day)
        return panel.astype(float), pd.Series(dtype=int)
    if mode != "peak_hour":
        raise ValueError(f"unknown rate mode {mode!r}")
    d["hour"] = d["t_index"] * int(slot_min) // 60
    sdh = d.groupby(["site_id", "date", "hour"])["count"].sum()
    mean_profile = sdh.groupby(["site_id", "hour"]).sum() / len(days)
    peak = mean_profile.groupby(level=0).idxmax().map(lambda t: t[1])
    at_peak = sdh.reset_index()
    at_peak = at_peak[at_peak["hour"] == at_peak["site_id"].map(peak)]
    panel = at_peak.pivot_table(index="date", columns="site_id", values="count", aggfunc="sum", fill_value=0.0)
    return panel.reindex(index=days, columns=sites, fill_value=0.0).astype(float), peak.reindex(sites)


def unserved_share(cell_weights: pd.Series, assignment: pd.Series) -> float:
    tot = float(cell_weights.sum())
    served = float(cell_weights.reindex(assignment.index).fillna(0.0).sum())
    return 0.0 if tot == 0 else 1.0 - served / tot
