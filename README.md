# Dynamic Pre-habilitation Targets for RN-IVCT: A Causal Machine Learning Study

## Overview
This repository contains the source code for the machine learning analysis and causal inference framework used in the manuscript:

> **Dynamic Pre-habilitation Targets for Renal Cell Carcinoma with Inferior Vena Cava Thrombus: A Multi-Center Validation Study Using Causal Machine Learning**

## Project Description
Radical nephrectomy with inferior vena cava thrombectomy (RN-IVCT) is a high-risk procedure. This study aims to shift risk stratification from static anatomical parameters (e.g., Mayo level) to dynamic, modifiable physiological targets.

We employed a two-stage **XGBoost** pipeline combined with **SHAP (SHapley Additive exPlanations)** and **Multivariate Causal Mediation Analysis** to:
1. Identify preoperative physiological buffers (Hemoglobin, Albumin, Uric Acid).
2. Disentangle the direct biological effects of patient physiology from the indirect effects mediated by surgical complexity (Operative Time, EBL).
3. Validate findings using a strict **Leave-One-Center-Out Cross-Validation (LOCO-CV)** strategy.

## Repository Contents
* `V2_CODE_FINAL.py`: The master analysis script including:
    * Data preprocessing and imputation (MICE/Median).
    * Two-Stage XGBoost model training.
    * SHAP value calculation and dependence plotting.
    * LOCO-CV spatial validation loop.
    * Causal Mediation Analysis (Statsmodels OLS) and Dumbbell Plot generation.

## Dependencies
The analysis was performed using Python 3.8+. Key libraries required:
* `pandas`
* `numpy`
* `matplotlib` & `seaborn`
* `xgboost`
* `shap`
* `statsmodels`
* `scikit-learn`

## Usage
To replicate the analysis (provided you have the de-identified dataset):
```bash
python V2_CODE_FINAL.py
