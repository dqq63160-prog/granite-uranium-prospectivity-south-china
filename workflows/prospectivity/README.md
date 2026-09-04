# Uranium prospectivity workflow

This workflow performs source-connected validation, nested algorithm comparison,
out-of-fold prediction, and SHAP attribution for uranium prospectivity.

Input: `../../data/source/Supplementary_Table_S1.xlsx`.

Run the notebook in `notebooks/`. Saved numerical outputs are in
`../../results/prospectivity/`. The workflow retains the distinction between model
attribution and geological causality.
