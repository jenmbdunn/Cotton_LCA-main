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


