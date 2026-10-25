# Queue-Constrained Wasserstein DRO for UAV Charging-Network Design


> W. Imliki, C. A. Deresa, H.-L. Choi, *Queue-Constrained Wasserstein Distributionally Robust
> Optimization for UAV Charging Network Design*, KAIST.

The framework has two stages:

1. **Siting.** An N−1 reliability-aware MILP picks the open sites (Appendix A).
2. **Sizing.** A queue-constrained 1-Wasserstein DRO MILP sizes the chargers at the open sites
   (Section IV). Every site must stay below the utilisation cap ρ ≤ 0.95 in every scenario.

Designs are then checked with M/M/c (Erlang-C) formulas and a SimPy simulation, under
nominal demand, stressed demand and two-station outages.

```
LaDe Shanghai deliveries ──01──▶ cells, 150 candidate sites, arcs ──02 (Gurobi)──▶ 78-site footprint
                                                                                     │ (frozen in data/reference)
                       ┌─────────────────────────────────────────────────────────────┘
                       ▼
          03 site arrival rates, S, R, d(s,r), ε̂ ──04──▶ DRO / RO / sensitivity designs
                                                         │
                     05 queue evaluation (analytic + SimPy) ◀┘ ──06──▶ upgrades, outages
                                                         │
                                  07 scaling ─── 08 tables + figures
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh                 # Stage 2 onward, ~5 min plus the scaling sweep
python -m pytest -q          # unit tests (pip install -r requirements-dev.txt)
```

Everything from Stage 2 onward runs from committed data using the free HiGHS solver.

- `./run_all.sh --phase1` rebuilds `data/processed/` from the raw LaDe file. See
  [data/README.md](data/README.md) for how to get it.
- `./run_all.sh --stage1` re-solves the Stage-1 MILP. This needs a Gurobi licence and takes
  about 6 h on 16 threads.

All parameters live in [`configs/shanghai.yaml`](configs/shanghai.yaml).

## Data source

The raw input is the Shanghai delivery table of **LaDe**, a public last-mile delivery dataset
released by Cainiao. It is **not included** in this repository. Download it from Hugging Face
([`Cainiao-AI/LaDe`](https://huggingface.co/datasets/Cainiao-AI/LaDe)) and place it at
`data/raw/delivery_shanghai.csv`; [data/README.md](data/README.md) has the details. The
repository ships only derived, aggregated data (`data/processed/`: delivery counts per H3 cell
and 30-min slot), so Stage 2 onward runs without the raw file.

If you use the data, please cite LaDe:

```bibtex
@inproceedings{wu2024lade,
  title     = {LaDe: The First Comprehensive Last-mile Express Dataset from Industry},
  author    = {Wu, Lixia and others},
  booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2024}
}
```

## Which script produces what

| Paper item | Script | Output |
|---|---|---|
| Data (Sec. IV.A) | `01_prep_phase1.py` | `data/processed/*` |
| Stage-1 siting (App. A) | `02_stage1_siting.py` | `results/stage1/footprint_full.csv` (frozen copy: `data/reference/footprint_stage1.csv`) |
| S, R, ground metric, ε̂ (Sec. IV.B–D) | `03_build_scenarios.py` | `results/scenarios/` |
| Table 1 (DRO ε sweep), Table 4 RO rows, Sec. V.G | `04_stage2_dro.py` | `results/stage2/<run>/` |
| Queue evaluation (Sec. V.A–B) | `05_evaluate.py` | `results/eval/designs.csv` |
| Tables 2–3 (outages, upgrade) | `06_upgrade_outage.py` | `results/eval/outages.csv`, `upgrade_*_plan.csv` |
| Sec. V.E scaling | `07_scaling.py` | `results/eval/scaling.csv`, `scaling_fit.json` |
| All tables (CSV + LaTeX) and Figs. 2–4 | `08_figures_tables.py` | `results/tables/`, `results/figures/` |

## Results (Shanghai, 78-site footprint, μ = 8/h, HiGHS 1.11)

| Design | Chargers | p95 wait, min (analytic / sim) | Unstable sites (nominal / worst r ∈ R) |
|---|---:|---:|---:|
| Uniform c = 4 | 312 | 5.77 / 5.83 | 6 / 8 |
| RO, 1.2× inflation | 312 | 2.94 / 2.83 | 0 / 2 |
| RO, worst-site | 312 | infeasible: needs ≥ 410 chargers | – |
| **DRO, ε = 0.5 ε̂** | **312** | **1.95 / 1.96** | **0 / 0** |
| DRO + targeted upgrade (+6) | 318 | 1.75 / 1.70 | 0 / 0 |

How to read these numbers:

- **Footprint and data.** The 78 sites come from the Stage-1 MILP. 1,483,496 deliveries over
  184 days were assigned to their nearest open site within 20 min, which serves 97.3% of demand.
  Each site's arrival rate is its **peak hour-of-day** load averaged over days.
- **Stability.** At a fixed budget of 312 chargers, the DRO design is stable at every site under
  every stressed scenario. Uniform allocation leaves 6 sites unstable, and 1.2× inflation RO
  leaves 2 unstable under spatial shifts.
- **Radius ε.** ε has little effect between 10⁻⁶ and 2 ε̂ (Table 1). The stability constraint does
  most of the work, and the radius mostly moves the objective.
- **Simulation check.** Analytic and SimPy p95 agree within the simulation spread (24 h, seeds
  123–127). For designs running near ρ = 0.95 the 24 h simulation sits below the analytic value,
  because it has not yet reached steady state.

## Changes from the first version of the paper

Rebuilding the pipeline from scratch turned up the problems below. This code fixes them, so its
numbers differ from the first draft.

1. **Arrival rates now count deliveries.** The draft's site rates summed *grid cells*, not
   deliveries: the evaluation script gave every cell weight 1, so Σλ·24 equalled the 15,067 cells.
   Rates now use real delivery counts. Dividing daily counts by 24 hides the 09:00–21:00 delivery
   peak and makes every design trivially stable, so the default rate is each site's peak hour
   (`stage2.rate_mode: peak_hour`). The draft's rule is still available as `daily24`.
2. **One consistent footprint.** The draft's footprint (`open_sites.txt`) came from a heuristic,
   not the Stage-1 MILP, and its demand table came from a *different* 75-site design. Only 52
   sites overlapped, which produced the draft's "52 active sites / 23 unstable" RO result. Both now
   come from the Stage-1 MILP footprint.
3. **The 312-charger budget is enforced.** The draft's DRO designs used 75–110 chargers despite
   the stated budget. All budgeted runs now satisfy Σc = 312 exactly.
4. **Stability holds on all scenarios.** The draft code applied constraint (2) to S only; it now
   covers Ω = S ∪ R, as the paper states.
5. **RO is plain robust optimisation.** The RO baselines are now min capex + max over r ∈ R of
   Q_r, i.e. zero transport cost. Under the Wasserstein dual with ε → 0 they would be unbounded,
   because their support does not contain the empirical point. Worst-site RO is infeasible within
   312 chargers at ρ ≤ 0.95; its cost-minimal design (410 chargers) is reported as well.
6. **Tight optimality gap.** With a fixed budget the objective is 312 plus a queue penalty of
   about 13, so a 2% gap allowed visibly sub-optimal allocations. Stage 2 uses a gap of 1e-4.
7. **ε̂ is fixed by construction.** With |S| = 1, every stress scenario sits at distance exactly
   2α ∈ {0.06, 0.12}, so ε̂ (the median) is 0.09 whatever the data.
8. **Candidate pool.** There are 150 candidate sites and the open-site cap is O_max = 80. The
   draft said |J| = 80.
9. **Scaling.** Scaling is measured for k ∈ {1, 2, 5, 10, 15}; the draft only had k ∈ {1, 5, 10}.
   With HiGHS, the 312-charger budget and a 2% gap, median solve times are 1.3, 5.1, 36, 202 and
   83 s, which fits T(k) ≈ 1.57 k^1.77 with R² = 0.92 (`results/eval/scaling_fit.json`, Fig. 4).
   Growth is super-linear, not the draft's k^0.53, and k = 15 beating k = 10 is ordinary
   branch-and-bound variance.
10. **Two upgrade variants.** "Targeted upgrade" is reported both ways:
    - on top of the DRO design (Table 4 row, `upgrade_dro`);
    - on top of uniform c = 4, following the draft's Table 3 rule (`upgrade_uniform`).


## Layout

```
configs/shanghai.yaml      all parameters
data/processed/            Phase-1 output (committed, ~5 MB)
data/reference/            frozen Stage-1 footprint
data/raw/                  put the LaDe file here (git-ignored)
uavdro/                    library: demand + assignment, queueing, DRO model, evaluation
scripts/01..08_*.py        pipeline steps
results/                   generated tables, figures and designs (committed for reference)
tests/                     unit tests
```

## License

Code: [MIT](LICENSE). The LaDe data remains under its own terms; see the dataset page.

## Solvers

- **Stage 2** uses HiGHS 1.11 through Pyomo by default. Set `stage2.solver: gurobi` to use Gurobi.
- **Stage 1** has about 6.4M constraints and is only practical with Gurobi (12.0.3 in the paper,
  with node files).
