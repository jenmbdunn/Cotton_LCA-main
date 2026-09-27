# Cotton LCA / Water Prediction — Code and Data

Code and data underlying "Environmental impacts of future cotton production in the United States."

## Setup

```
pip install -r requirements.txt
```

## Structure

- `data/` — raw inputs (USDA NASS/Census exports, CMIP6/CESM2 weather, ERA5 features, state shapefiles).
- `results/` — processed intermediate data shared across notebooks (`results/processed_data and results/`) and top-level LCA result workbooks.
- `code/` — analysis notebooks and scripts, numbered in pipeline order (see below).
- `outputs/` — figures and tables, one subfolder per pipeline stage (`outputs/water/`, `outputs/n2o/`, `outputs/energy_national/`, `outputs/energy_state/`, `outputs/ghg_state/`, `outputs/lca_4bar/`, `outputs/lca_17states/`, `outputs/fixed_notebook_3mlp/`, `outputs/strategy/`).
- `archive/` — superseded notebooks/files kept for provenance, not part of the current pipeline.

## Pipeline order

Files are numbered by stage, with a letter suffix where a stage has more than one file and a
fixed order between them (e.g. `3a` must run before `3d`); a bare number means that stage is a
single, self-contained file.

| Step | Script | Produces |
|---|---|---|
| 1a | `code/1a_data_process_CottonData.ipynb` | `PROCESS_GE2000_BASE.xlsx`, `PROCESS_GE2000_APPLICATIONS.xlsx` |
| 1b | `code/1b_data_process_CMIP6Data.ipynb` | `results_SSP*_8f.csv`, `results_his_8f.csv` (from raw CMIP6 NetCDF — slow; outputs already included, rerun only if the raw weather data changes) |
| 2 | `code/2_data_interpretation_CottonData.ipynb` | Water-consumption figures/tables (Fig. 1 panel, SI Fig. S3–S7) |
| 3a | `code/3a_data_interpretation_state_EnergyChemicals.ipynb` | `Energy_Analysis_Summary.xlsx` (state-level diesel/electricity, lint-allocated) — must run before 3d |
| 3b | `code/3b_N2O_estimation.ipynb` | Fig. 4 |
| 3c | `code/3c_data_interpretation_National_EnergyChemicals.ipynb` | Fig. 3 components |
| 3d | `code/3d_data_interpretation_state_Energy_lca.ipynb` | Fig. 5 (depends on 3a's `Energy_Analysis_Summary.xlsx` being fresh) |
| 4 | `code/4_Cotton_CMIP_Prediction.py` | Fig. 1/2 panels, SI Fig. S9–S15 (see "ML model" below) |
| 5a | `code/5a_Cotton_strategy.ipynb` | Fig. 2d |
| 5b | `code/5b_LCA_results_17states.ipynb` | SI Fig. S16 |
| 6 | `code/6_cotton_lca_4bar.py` | Fig. 6 |

Run notebooks with: `jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb`

## ML model (step 4)

`code/4_Cotton_CMIP_Prediction.py` (originally `run_fixed_notebook_cesm2.py`) is the model
behind the published figures: a stacked ensemble of 3 MLPs (plus individual RandomForest/KNN
baselines for comparison), trained on CMIP6/CESM2 weather features, with an honest
train/validation split (2000–2014 train, 2015–2023 validation) and a separate refit on the
full record for the 2024–2049 projection.

It imports `cotton_water_prediction.py`, `cotton_water_prediction_cmip_ensemble.py`,
`generate_era5_10figs.py`, and `run_original_notebook_cesm2.py` as local modules — all four
must stay alongside it in `code/`.

**Note on model selection:** an earlier, more elaborate ERA5-based model
(`cotton_water_prediction_era5_damped.py`, a per-state trend + damped extrapolation + KNN
residual model) was built and documented as a "final production model" in an internal summary
during development. It was subsequently superseded and reverted in favor of this simpler
CMIP6-only ensemble, which is the version that actually produced the published figures — do
not follow that earlier document if you find a copy of it elsewhere.

## Fig. 2d (step 5a, `5a_Cotton_strategy.ipynb`)

This notebook originally read a `predicted_water_consumption_changes.xlsx` file that was never
saved into this repository (an ML-derived intermediate from an earlier version of the analysis).
The manuscript's own Fig. 2 caption says panel d classifies states by their **historical**
(2000–2023) irrigation water trend — via Sen's slope, with Mann-Kendall significance — and
historical abandonment rate, not a future prediction. The notebook has been rewritten to compute
this directly from `water consumption.xlsx` using `pymannkendall`, with no ML dependency. The
resulting zone assignments (Arizona in Zone 1; Oklahoma/Texas/Kansas/New Mexico in Zone 2;
Arkansas/Georgia in Zone 3; South Carolina alone in Zone 4 with a non-significant trend) match
the manuscript text exactly.

## Known gaps

- `code/1b_data_process_CMIP6Data.ipynb` needs `netCDF4` and `geopandas` (heavier, platform-
  dependent installs) and reprocesses raw CMIP6 NetCDF files, which is slow. Its outputs
  (`results_*_8f.csv`) are already included under `data/Weather/CMIP/`, so it only needs to be
  rerun if the raw weather data changes.

## Notes on duplicate/superseded code found during cleanup

A few notebooks contained an older, active (not just commented-out) cell alongside a newer one
producing the same named figure with better styling — both ran and silently left the older,
plainer version overwritten by, or shadowing, the newer one. These have been resolved so each
figure is now produced by exactly one cell:
- `2_data_interpretation_CottonData.ipynb`: an older plain-`tab20`-palette version of the
  production/water-consumption percentage charts was replaced by the newer Nature-style-palette
  version; the older cell now only computes the `Production_Percent.xlsx`/`Water_Percent.xlsx`
  tables it also produces (still needed by `5a_Cotton_strategy.ipynb`).
- `3c_data_interpretation_National_EnergyChemicals.ipynb`: a duplicate cell recomputing and
  re-saving `merged_chemical_usage.tiff` (identical to the following cell, minus a diagnostic
  print) was removed.
