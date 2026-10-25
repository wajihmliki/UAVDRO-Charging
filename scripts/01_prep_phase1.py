#!/usr/bin/env python3
"""Phase 1 — raw LaDe deliveries -> H3 demand cells, candidate sites, cell->site arcs.

Outputs (in paths.processed):
  grid_cells.csv                 grid_id, centroid_lat, centroid_lng       (demand cells U)
  candidate_sites.csv            site_id, grid_id, centroid_lat, centroid_lng, total, ...  (J)
  cell_to_sites_topk.parquet     grid_id, rank, site_id, tt_min            (arcs P, K nearest)
  demand_by_day_celltime.parquet date, grid_id, t_index, count
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uavdro.common import base_parser, ensure_dir, load_config  # noqa: E402

SH_BBOX = dict(lat_min=30.6, lat_max=31.6, lng_min=120.9, lng_max=122.2)
_CLOCK = re.compile(r"(\d{1,2}:\d{2}:\d{2})")


def serial_days_to_dates(serial: pd.Series, anchor: str) -> pd.Series:
    """Rank-order the unique day keys (e.g. 604, 907) onto consecutive dates from `anchor`."""
    sn = pd.to_numeric(serial, errors="coerce")
    rank = {v: i for i, v in enumerate(np.sort(sn.dropna().unique()))}
    offsets = sn.map(rank).astype("Int64")
    return (pd.to_datetime(anchor) + pd.to_timedelta(offsets, unit="D")).dt.date


def compose_timestamp(date_col: pd.Series, time_col: pd.Series, is_serial: bool, anchor: str) -> pd.Series:
    day = (serial_days_to_dates(date_col, anchor) if is_serial
           else pd.to_datetime(date_col, errors="coerce").dt.date).astype(str)
    clock = time_col.astype(str).str.extract(_CLOCK)[0].fillna("")
    return pd.to_datetime(day + " " + clock, errors="coerce")


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config)
    p1 = cfg["phase1"]
    out = ensure_dir(cfg["paths"]["processed"])
    raw = cfg["paths"]["raw_deliveries"]
    if not raw.exists():
        raise SystemExit(f"Raw file not found: {raw}. See data/README.md for how to download LaDe.")

    # The serial day mapping must see every day key at once, so read the needed columns whole.
    cols = ["order_id", p1["date_col"], p1["time_col"], p1["lat_col"], p1["lng_col"]]
    dv = pd.read_csv(raw, usecols=cols, low_memory=False)
    n_read = len(dv)
    dv["time"] = compose_timestamp(dv[p1["date_col"]], dv[p1["time_col"]],
                                   p1["date_is_serial"], p1["serial_anchor"])
    dv["lat"] = pd.to_numeric(dv[p1["lat_col"]], errors="coerce")
    dv["lng"] = pd.to_numeric(dv[p1["lng_col"]], errors="coerce")
    dv = dv.dropna(subset=["time", "lat", "lng"])
    dv = dv[dv["lat"].between(SH_BBOX["lat_min"], SH_BBOX["lat_max"])
            & dv["lng"].between(SH_BBOX["lng_min"], SH_BBOX["lng_max"])]
    dv = dv.sort_values("time").drop_duplicates(subset=["order_id"], keep="first")
    dv["date"] = dv["time"].dt.date
    dv["t_index"] = (dv["time"].dt.hour * 60 + dv["time"].dt.minute) // int(p1["slot_min"])
    print(f"[phase1] read={n_read:,} kept={len(dv):,} days={dv['date'].nunique()}")

    # H3 demand cells (order of first appearance, as in the original pipeline)
    res = int(p1["h3_res"])
    dv["grid_id"] = [h3.latlng_to_cell(a, b, res) for a, b in zip(dv["lat"].to_numpy(), dv["lng"].to_numpy())]
    cells = pd.DataFrame({"grid_id": pd.unique(dv["grid_id"])})
    cents = [h3.cell_to_latlng(g) for g in cells["grid_id"]]
    cells["centroid_lat"] = [c[0] for c in cents]
    cells["centroid_lng"] = [c[1] for c in cents]
    cells.to_csv(out / "grid_cells.csv", index=False)

    day_ct = dv.groupby(["date", "grid_id", "t_index"], as_index=False).agg(count=("order_id", "size"))
    day_ct.to_parquet(out / "demand_by_day_celltime.parquet", index=False)

    # Candidate sites = centroids of the top-k busiest cells
    tot = dv.groupby("grid_id", as_index=False).agg(total=("order_id", "size"))
    cand = (tot.sort_values("total", ascending=False).head(int(p1["topk_sites"]))
               .merge(cells, on="grid_id", how="left"))
    cand.insert(0, "site_id", [f"S{i:04d}" for i in range(len(cand))])
    cand = cand[["site_id", "grid_id", "centroid_lat", "centroid_lng", "total"]]
    cand.to_csv(out / "candidate_sites.csv", index=False)

    # Air travel time cell -> site: overhead + great-circle distance / speed; keep K nearest
    spd = float(p1["uav_speed_kmh"]) * 1000 / 3600
    uu = cells[["centroid_lat", "centroid_lng"]].to_numpy()
    jj = cand[["centroid_lat", "centroid_lng"]].to_numpy()
    T = float(p1["overhead_min"]) + haversine_m(uu[:, :1], uu[:, 1:], jj[None, :, 0], jj[None, :, 1]) / spd / 60.0
    T = T.astype(np.float32)
    k = int(p1["topk_map"])
    idx = np.argsort(T, axis=1)[:, :k]
    arcs = pd.DataFrame({
        "grid_id": np.repeat(cells["grid_id"].to_numpy(), k),
        "rank": np.tile(np.arange(1, k + 1), len(cells)),
        "site_id": cand["site_id"].to_numpy()[idx.ravel()],
        "tt_min": T[np.arange(len(cells))[:, None], idx].ravel().astype(float),
    })
    arcs.to_parquet(out / "cell_to_sites_topk.parquet", index=False)
    print(f"[phase1] cells={len(cells):,} sites={len(cand)} arcs={len(arcs):,} -> {out}")


if __name__ == "__main__":
    main()
