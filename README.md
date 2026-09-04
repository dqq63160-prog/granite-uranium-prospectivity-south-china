# Granite-related uranium prospectivity in South China

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22297069.svg)](https://doi.org/10.5281/zenodo.22297069)

This repository contains the data, executable workflows, and canonical numerical
outputs supporting the analysis of uranium prospectivity, granite-type affinity,
and their cross-task geochemical interpretation in South China.

## Contents

- `data/source/Supplementary_Table_S1.xlsx`: the source compilation, geological-group registry, references, and data dictionary.
- `workflows/prospectivity`: source-connected uranium prospectivity modelling.
- `workflows/granite_classification`: source-connected I-, A-, and S-type granite classification.
- `workflows/coupling`: comparison of saved out-of-fold predictions and SHAP attributions.
- `results`: canonical machine-readable numerical outputs and source data for reported figures.

## Reproduction route

Create the environment from `environment.yml`, then run the notebooks in this order:

1. `workflows/prospectivity/notebooks/01_TaskA_Prospectivity_Optuna_SHAP.ipynb`
2. `workflows/granite_classification/notebooks/04_Granite_Classification_SHAP_v5.ipynb`
3. `workflows/coupling/notebooks/04_JGE_Coupled_Interpretation.ipynb`

The two modelling workflows use the `Dataset` and `Geological Groups` worksheets
of the current Supplementary Table S1 workbook. The coupling workflow reads the
curated canonical result bundle in `results/`; it does not refit either model.

## Interpretation boundary

SHAP values are model attributions, not direct evidence of petrogenetic or
metallogenic causality. The cross-task analysis provides a restricted comparison
of uranium prospectivity with S-type affinity and does not establish that granite
type alone predicts mineralization.

## Public result bundle

The repository does not contain duplicate rendered figures, runtime logs,
intermediate threshold grids, optimization databases, binary model objects, or
development records. It does contain the OOF predictions, SHAP summaries,
bootstrap/permutation evidence, and figure source tables required to inspect the
reported numerical results without rerunning long nested optimizations.

See `docs/REPRODUCIBILITY.md`, `docs/DATA_DICTIONARY.md`, and
`docs/RESULTS_AND_FIGURE_INDEX.md` for exact file-level provenance.
