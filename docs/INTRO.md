
# DataMiningProj — Project Intro

## 1. Project Overview

This repository is for the 2026 Fall Data Mining course project.

Remote repository:

git@github.com:leo0302git/DataMiningProj.git

This repo only covers the course programming/project assignment.

Do NOT include:

- classroom presentation
- presentation PPT
- literature survey / final review report

The authoritative task requirements are contained in the two course PDFs provided locally.

Before making implementation decisions, read both PDFs carefully.

PDF paths:

1. Course project specification:
   <PATH_TO_数据挖掘课程大作业.pdf>
2. Course presentation / survey specification:
   <PATH_TO_数据挖掘前沿专题探究.pdf>

The second PDF is provided mainly for overall course context. For this repository, prioritize the requirements in the course project PDF.

If this INTRO conflicts with the PDFs, the PDFs are the source of truth.

---

## 2. Main Goal

The project is an interpretable data mining practice assignment.

We need to solve two tabular machine learning tasks:

### Task A: Drug Induced Autoimmunity Prediction

Type:
binary classification

Dataset:
UCI Drug Induced Autoimmunity Prediction

https://archive.ics.uci.edu/dataset/1104/drug_induced_autoimmunity_prediction

The data contains molecular descriptors calculated from drug molecular structures.

The original molecular structure is represented by SMILES, but according to the course requirement, SMILES does not need to be used as model input.

The main model input should therefore be the numerical molecular descriptors.

The dataset contains roughly:

- 500+ drug molecules in total
- about 195 RDKit molecular descriptors per molecule
- binary label: DIA positive / negative

The original train/test datasets may be merged and evaluated with K-fold cross validation, according to the course specification.

### Task B: Abalone

Type:
regression

Goal:
predict the target variable `Rings` from observable abalone features.

Estimated age can be calculated as:

age = Rings + 1.5

The model itself should predict `Rings`.

---

## 3. Required Work

The course project should include the following stages.

### 3.1 Data Analysis and Preprocessing

For both datasets:

- inspect dataset structure
- inspect feature types
- inspect missing values
- inspect abnormal values
- inspect feature scales
- inspect target distribution
- analyze possible problems that may affect model training
- perform appropriate preprocessing based on the analysis

Possible preprocessing includes, but is not limited to:

- data cleaning
- normalization / standardization
- missing value handling
- feature filtering
- handling class imbalance where appropriate

Do not blindly apply preprocessing.

Every important preprocessing decision should be justified by data analysis.

For the DIA dataset in particular, pay attention to:

- small sample size
- relatively high feature dimensionality
- class imbalance
- correlations / redundancy among molecular descriptors
- possible low-variance features

---

## 4. Model Requirements

The central interpretable model suggested by the course is RRL:

Rule Representation Learner

Official repository:

https://github.com/12wang3/rrl

Paper:

Learning Interpretable Rules for Scalable Data Representation and Classification

The course recommends using RRL and modifying its open-source implementation.

For the regression task, RRL needs to be adapted because the original implementation is classification-oriented.

In addition to the interpretable model:

- each task must include at least two other comparison models / algorithms
- each task must contain at least one manually implemented model

A single manually implemented model may be designed to support both classification and regression.

Alternatively, different manually implemented models may be used for the two tasks.

Using an existing sklearn estimator directly does NOT count as manual implementation.

Allowed building blocks include:

- numpy
- basic sklearn utilities
- torch
- custom `torch.nn.Module`

The manual implementation should be simple, clear, and easy to verify.

---

## 5. RRL Modification Requirement

The assignment is not satisfied by simply running the original RRL implementation.

At least one meaningful modification to the interpretable model is required.

The course gives examples such as:

- improving the binarization method
- modifying the logical activation function
- modifying the structure of conjunction / disjunction layers

If another interpretable model is used instead of RRL, it must also contain a meaningful structural modification.

Its explanations must be faithful to the model's actual decision process rather than being purely post-hoc explanations.

For now, use RRL as the default plan unless there is a strong technical reason not to.

---

## 6. RRL and macOS

Development will primarily happen locally on a MacBook Pro.

The datasets are small enough that compute is not expected to be a bottleneck.

However, the original RRL repository was written primarily for a Linux + NVIDIA CUDA environment.

Potential incompatibilities include:

- CUDA-specific code
- NCCL distributed training
- `.cuda()` calls
- GPU device assumptions
- old PyTorch / NumPy API usage

Therefore, do NOT assume the upstream RRL repository will run unchanged on macOS.

Preferred strategy:

1. first create a clean single-process CPU version
2. make device handling explicit
3. replace hard-coded CUDA usage with `.to(device)`
4. remove unnecessary distributed / NCCL logic for local experiments
5. only consider MPS acceleration later if useful

For these datasets, CPU execution is acceptable.

Correctness and reproducibility are more important than GPU acceleration.

Do not introduce unnecessary complexity just to use Apple GPU / MPS.

---

## 7. Experimental Design

For each task, define clearly:

- dataset split / cross-validation strategy
- preprocessing pipeline
- baseline models
- interpretable model
- manually implemented model
- evaluation metrics
- hyperparameter tuning strategy
- method for avoiding overfitting / underfitting

The same data split or CV folds should be reused across models whenever possible to make comparisons fair.

Random seeds should be fixed and recorded.

All experiments should be reproducible from code.

Avoid manually editing final result tables.

Results should preferably be generated from experiment outputs.

---

## 8. Evaluation

The final analysis should compare models from multiple perspectives.

At minimum:

### Performance

Use task-appropriate metrics.

For classification, do not rely only on accuracy, especially if the DIA dataset is imbalanced.

Possible metrics include:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- MCC

The final metric set should be chosen after inspecting the label distribution.

For regression, possible metrics include:

- MAE
- RMSE
- R²

### Interpretability

Analyze:

- whether explanations are understandable
- what rules / features the model uses
- whether explanations reflect the actual internal decision process

### Complexity

For RRL or tree-based models, model structural complexity should be measured where practical.

The original RRL code contains a `log(#edges)` style complexity calculation that may be useful.

Other reasonable complexity metrics may also be considered.

---

## 9. Recommended Project Structure

Keep the repository clean and modular.

A reasonable initial structure is:

DataMiningProj/
├── README.md
├── INTRO.md
├── pyproject.toml
├── uv.lock
├── data/
│   ├── dia/
│   └── abalone/
├── src/
│   └── dataminingproj/
│       ├── data/
│       │   ├── dia.py
│       │   ├── abalone.py
│       │   └── preprocessing.py
│       ├── models/
│       │   ├── baselines.py
│       │   ├── manual/
│       │   └── rrl/
│       ├── evaluation/
│       └── utils/
├── experiments/
│   ├── dia/
│   └── abalone/
├── notebooks/
├── results/
│   ├── figures/
│   ├── tables/
│   └── raw/
├── report/
└── tests/

This is only a suggested structure.

Do not create unnecessary abstraction before understanding the upstream RRL code and the datasets.

Prefer simple, readable code.

---

## 10. Environment

Use `uv` to manage the Python environment unless there is a strong reason not to.

Target local environment:

- macOS
- MacBook Pro
- Python
- CPU-first PyTorch
- numpy
- pandas
- scikit-learn
- matplotlib
- optional XGBoost / LightGBM if used as baselines

Avoid:

- CUDA-only dependencies
- unnecessary Docker usage
- unnecessary distributed training infrastructure

If upstream RRL has dependency conflicts with a modern Python environment, investigate the minimum compatibility changes rather than immediately downgrading the entire project.

---

## 11. Coding Principles

Please follow these principles while working on this repository.

1. Read the course PDF before implementing a requirement.
2. Do not silently invent assignment requirements.
3. Keep upstream RRL behavior intact before implementing our own modification.
4. Separate:

   - upstream-compatible RRL
   - macOS compatibility changes
   - assignment-specific RRL modifications
5. Make experimental changes traceable.
6. Keep datasets out of Git if they are large or easily downloadable.
7. Do not commit generated caches, temporary files, virtual environments, or large intermediate outputs.
8. Prefer scripts that can reproduce results from scratch.
9. Add tests for important custom implementations where practical.
10. Do not optimize code prematurely.

---

## 12. Suggested Milestones

### Milestone 1: Repository bootstrap

- inspect both PDFs
- inspect current repository
- initialize Python environment
- create sensible `.gitignore`
- create basic project structure
- document how to run the project

### Milestone 2: Dataset pipeline

DIA:

- download / load dataset
- inspect columns
- identify target
- exclude SMILES from model input
- perform EDA
- build preprocessing pipeline

Abalone:

- load dataset
- inspect target and features
- perform EDA
- build preprocessing pipeline

### Milestone 3: Baselines

Implement a consistent evaluation framework.

Run several simple baseline models for both tasks.

Do not spend much time on hyperparameter tuning yet.

The purpose is to establish a working end-to-end pipeline.

### Milestone 4: Manual model

Implement at least one model manually.

Prefer a simple algorithm whose implementation and behavior can be clearly explained.

Add unit tests where reasonable.

### Milestone 5: RRL reproduction

- inspect upstream RRL implementation
- reproduce one upstream example if possible
- make it run locally on macOS CPU
- preserve original behavior as much as possible

### Milestone 6: RRL on DIA

- convert DIA data into the format required by RRL
- run classification experiments
- extract interpretable rules
- compare against baselines

### Milestone 7: RRL regression adaptation

Adapt RRL to the Abalone regression task.

Likely areas requiring modification include:

- output layer
- loss
- prediction logic
- evaluation metrics

Do not assume changing only `CrossEntropyLoss` to `MSELoss` is sufficient. Inspect the whole classification-specific code path.

### Milestone 8: RRL structural improvement

Choose one meaningful modification required by the assignment.

Design:

- baseline RRL
- modified RRL

Run controlled comparison experiments.

### Milestone 9: Final experiments

Run reproducible final experiments with fixed seeds and consistent splits.

Generate:

- metrics tables
- figures
- interpretability examples
- complexity comparison

### Milestone 10: Report support

Prepare all experiment outputs needed for the final course report.

The report itself may be written later, but all tables, figures, experiment configurations and important observations should be reproducible from this repository.

---

## 13. What To Do First

When you first enter this repository:

1. Read this `INTRO.md`.
2. Read both provided PDFs.
3. Inspect the current repository state.
4. Inspect the upstream RRL repository:
   https://github.com/12wang3/rrl
5. Do NOT immediately start rewriting RRL.
6. First summarize:

   - exact assignment requirements
   - current repository state
   - expected technical risks
   - proposed implementation plan
   - proposed file structure
7. Then begin with repository bootstrap and dataset inspection.

If you discover ambiguity between this document and the original assignment PDF, stop and point it out rather than guessing.

---

## 14. Immediate Priority

The immediate goal is NOT to obtain the best model score.

The immediate goal is to establish a clean and reproducible end-to-end baseline:

dataset
→ preprocessing
→ model
→ evaluation
→ saved results

Once that pipeline is stable, move on to RRL adaptation and model improvement.
