# Data

## Raw input (not committed)

The study uses the **Shanghai delivery table of LaDe**, released by Cainiao:

- Dataset: <https://huggingface.co/datasets/Cainiao-AI/LaDe>
- Paper: L. Wu *et al.*, "LaDe: The First Comprehensive Last-mile Express Dataset from
  Industry," *Proc. ACM SIGKDD*, 2024 (BibTeX in the top-level README).

The raw files are **not redistributed here**. Download the Shanghai **delivery** table and save it as

    data/raw/delivery_shanghai.csv

(about 218 MB, 1,483,864 rows). Columns used: `order_id`, `ds`, `delivery_time`,
`delivery_gps_lat`, `delivery_gps_lng`. Please check the dataset's licence before redistributing it.

It is only needed to rebuild `data/processed/` (`./run_all.sh --phase1`).

## Processed data (committed)

Output of `scripts/01_prep_phase1.py`, so everything from Stage 2 onward runs without the raw file.

| File | Content |
|---|---|
| `grid_cells.csv` | 15,067 H3 resolution-9 demand cells (`grid_id`, centroid) |
| `candidate_sites.csv` | 150 candidate sites = centroids of the busiest cells |
| `cell_to_sites_topk.parquet` | 6 nearest candidates per cell with UAV travel time (min) |
| `demand_by_day_celltime.parquet` | deliveries per (day, cell, 30-min slot); 184 days, 1,483,496 deliveries |

Re-running Phase 1 on the raw file reproduces these files exactly (same sites, arcs and counts).

## Reference (committed)

| File | Content |
|---|---|
| `footprint_stage1.csv` | the 78 sites opened by the Stage-1 MILP (Gurobi 12.0.3; f = 25, O_max = 80) |
| `stage1_solution_summary.json` | its objective (40,355.5 = 25 x 78 + Z) |
