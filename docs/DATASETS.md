# Dataset sources

The local data files are intentionally excluded from Git. They are downloaded
from the official UCI packages below.

## Drug Induced Autoimmunity Prediction

- UCI dataset ID: 1104
- Source: https://archive.ics.uci.edu/dataset/1104/drug_induced_autoimmunity_prediction
- Required raw files: `DIA_trainingset_RDKit_descriptors.csv` and
  `DIA_testset_RDKit_descriptors.csv`.
- `RDKit_ChemDes.xlsx` is retained as descriptor documentation.
- The two CSV files may be merged for K-fold cross-validation. `SMILES` is an
  identifier and must not be supplied as a model feature; `Label` is the binary
  target.

## Abalone

- UCI dataset ID: 1
- Source: https://archive.ics.uci.edu/dataset/1/abalone
- Required raw files: `abalone.data` and `abalone.names`.
- The target is `Rings`, not the derived age `Rings + 1.5`.
