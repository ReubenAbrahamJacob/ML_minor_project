# Multimodal, uncertainty-aware predictive maintenance for rolling bearings

Answers: bearing **state** (healthy/early/severe), **RUL with a conformal interval**, and a cost-based **maintain / schedule / continue** action; then tests generalisation (FEMTO <-> XJTU-SY, and C-MAPSS via a new adapter).

**Start here:** [reports/final_report.md](reports/final_report.md) (figures in `reports/figures/`, every number from `reports/tables/*.csv`).

## Run
```
make setup        # venv (Python 3.11) + dependencies
make data         # CWRU, FEMTO, C-MAPSS, XJTU-SY into data/raw/<key>/ (XJTU needs `unar`, see Makefile)
make features     # feature tables + frozen-ResNet18 embeddings -> data/processed/ (cached)
make eval         # all experiments, figures, report   (about 15 min on a laptop)
make test         # pytest + ruff
make app          # Streamlit replay dashboard
```

## Layout
- `src/pdm/` physics, ingest, features, images, models, splits, uncertainty, decision, eval, pipeline
- `scripts/` one script per experiment: `run_cwru`, `run_femto`, `run_uncertainty`, `run_explain`, `run_cross`, `run_cmapss`, `make_figures`, `make_report`
- `configs/` datasets (geometry, `verified` flags), models, costs (assumed)
- `docs/mechanics_notes.md` course formulas and the two discrepancies found

## Honest notes
Images are signal-derived (STFT/CWT), not photographs. Fusion helped only marginally on FEMTO and not on CWRU or C-MAPSS. Costs are assumptions. Unverified geometry is flagged in `configs/datasets.yaml`. Temperature was not used. Windows per bearing are thinned to 300.
`data/raw/xjtu/download/xjtu_full.rar` (4.4 GB) is a leftover from a failed extraction attempt and can be deleted by hand.
