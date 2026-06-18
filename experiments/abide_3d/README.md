# ABIDE 3D Tensor Classification

This directory contains 3D tensor-based classification experiments conducted on resting-state fMRI derivatives from the ABIDE Preprocessed project.

The main completed experiment described here uses **Regional Homogeneity (ReHo)** maps to distinguish participants with autism spectrum disorder (ASD) from control participants.

A separate fALFF experiment is also included in this module, but the detailed analysis below concerns the ReHo experiment.

## Directory structure

```text
abide_3d/
├── README.md
├── README_PL.md
│
├── docs/
│   └── legacy_abide_3d_results.md
│
├── reho/
│   ├── code/
│   │   ├── prepare_abide_reho_3d_subjectwise.py
│   │   └── cp_logistic_als_reho_3d_subjectwise.py
│   └── results/
│       ├── results_cp_logistic_als_reho_3d.csv
│       └── summary_cp_logistic_als_reho_3d.csv
│
└── falff/
    ├── code/
    │   ├── prepare_abide_falff_3d_subjectwise.py
    │   └── cp_logistic_als_falff_3d_subjectwise.py
    └── results/
        ├── results_cp_logistic_als_falff_3d.csv
        └── summary_cp_logistic_als_falff_3d.csv
```

## ReHo classification experiment

### Aim

The aim of this experiment was to assess whether a low-rank tensor logistic regression model can distinguish participants with ASD from control participants using spatially structured 3D ReHo maps.

Instead of flattening each brain volume into a single long feature vector, the method preserves the original 3D spatial arrangement of voxels. The regression coefficient tensor is represented using a low-rank CP decomposition.

### Data

* Dataset: **ABIDE Preprocessed**
* Processing pipeline: **CPAC**
* Functional derivative: **ReHo**
* Processing strategy: **filt_noglobal**
* Input tensor dimensions: **61 × 73 × 61**
* Number of participants: **884**

  * ASD: **408**
  * Controls: **476**

The source ABIDE data and prepared participant-level tensors are excluded from this repository because of their size.

### Data preparation

The ReHo preparation pipeline:

1. matches individual ReHo maps with phenotypic information;
2. assigns binary labels: ASD = 1 and control = 0;
3. creates one 3D tensor for each participant;
4. applies subject-wise z-score normalisation to non-zero voxels;
5. saves the prepared tensors and metadata for classification.

### Model

The model is a logistic regression classifier with a CP-structured coefficient tensor.

The coefficient tensor is approximated as:

[
\mathcal{B} \approx \sum_{r=1}^{R}
\mathbf{a}_r \circ \mathbf{b}_r \circ \mathbf{c}_r,
]

where (R) denotes the CP rank and (\circ) denotes the outer product.

Model parameters are estimated through alternating optimisation of the CP factor matrices and the logistic regression intercept.

## Evaluation procedure

The reported result was obtained using:

* CP rank: **9**
* Validation method: **stratified 10-fold cross-validation**
* Optimisation cycles per fold: **5**
* Logistic regularisation parameter: **C = 0.05**
* Random seed: **42**

The evaluation reports the mean performance across the 10 validation folds.

## Results

| Metric                   |  Mean | Standard deviation |
| ------------------------ | ----: | -----------------: |
| AUC                      | 0.612 |              0.048 |
| Accuracy                 | 0.584 |              0.037 |
| Sensitivity for ASD      | 0.505 |                  — |
| Specificity for controls | 0.651 |                  — |

The model achieved a mean AUC of **0.612**, indicating above-chance but modest discrimination between ASD and control participants.

The mean accuracy was **0.584**. Specificity was higher than sensitivity, which means that in this configuration the model classified control participants more effectively than participants with ASD.

## Interpretation

The result should be treated as exploratory rather than diagnostic.

ABIDE is a heterogeneous multi-site dataset. Classification performance may be influenced by site effects, differences in acquisition protocols, participant age and sex, head motion, sample composition, preprocessing choices, model rank, and regularisation strength.

The purpose of this experiment is to evaluate whether low-rank tensor modelling can use information contained in 3D neuroimaging derivatives while preserving their spatial structure.

## Reproducibility

To reproduce the ReHo experiment:

1. obtain ReHo maps from ABIDE Preprocessed generated with the CPAC pipeline and the `filt_noglobal` strategy;
2. prepare participant-level tensors with `reho/code/prepare_abide_reho_3d_subjectwise.py`;
3. update local file paths in the scripts where required;
4. run `reho/code/cp_logistic_als_reho_3d_subjectwise.py`;
5. inspect the output CSV files in `reho/results/`.

Raw ABIDE files and prepared subject-wise tensors are intentionally excluded from version control.
