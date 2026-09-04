# Reproducibility

## Environment

Create the conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate jge-granite-uranium-ml
```

## Inputs and outputs

Both modelling workflows read `data/source/Supplementary_Table_S1.xlsx`. All
data-dependent preprocessing is fitted within training partitions. The saved
results are a canonical numerical archive for inspection and figure regeneration.
Re-running nested optimization can require substantial computation and may show
small platform-dependent numerical variation.

## Workflow order

Run prospectivity modelling first, granite classification second, and cross-task
coupling last. The coupling workflow uses fixed OOF and attribution outputs, and
does not train either preceding model.

## Exclusions from the public archive

Runtime logs, cached studies, model binaries, duplicate SVG/PDF renderings,
checkpoints, and superseded development records are excluded because they do not
alter the reported numerical evidence and can be regenerated from the workflows.
