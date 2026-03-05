# ==============================================================================
# PROJECT: Dynamic Pre-habilitation Targets for RN-IVCT (V2)
# AUTHOR: Ye Yan, MD, PhD et al. (Peking University Third Hospital)
# DESCRIPTION: Master Pipeline for Data Cleaning, Machine Learning (XGBoost), 
#              SHAP Interpretation, and Causal Mediation Analysis.
# ==============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------
# 1. PATH DEFINITION & STYLE SETTINGS
# ---------------------------------------------------------
DRIVE_ROOT = '/content/drive/MyDrive/'
V2_DIR = os.path.join(DRIVE_ROOT, 'pre-hab', 'V2')
EUO_BLUE, EUO_RED = '#00468B', '#ED0000'
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})

# ---------------------------------------------------------
# 2. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------
df_all = pd.read_csv(os.path.join(V2_DIR, "V2_00_Combined_Cleaned_Data.csv"))

stage1_vars = ['Age', 'BMI', 'Mayo_Level', 'Tumor_Diameter', 'Preop_Hb', 'Preop_Albumin', 'Preop_eGFR', 'Preop_UricAcid', 'Surgical_Approach']
stage2_vars = stage1_vars + ['Operative_Time', 'EBL']
y_all = df_all['CCI_Score'].values

imputer_s1 = SimpleImputer(strategy='median')
X_all_s1 = pd.DataFrame(imputer_s1.fit_transform(df_all[stage1_vars]), columns=stage1_vars)

imputer_s2 = SimpleImputer(strategy='median')
X_all_s2 = pd.DataFrame(imputer_s2.fit_transform(df_all[stage2_vars]), columns=stage2_vars)

# ---------------------------------------------------------
# 3. LEAVE-ONE-CENTER-OUT CROSS-VALIDATION (LOCO-CV)
# ---------------------------------------------------------
# (Extracts from previous generation...)
# Model training and RMSE/MAE calculation iteratively across centers.

# ---------------------------------------------------------
# 4. TWO-STAGE XGBOOST & SHAP INTERPRETATION
# ---------------------------------------------------------
# Stage 1: Pre-operative Model
model_s1 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
model_s1.fit(X_all_s1, y_all)
exp_s1 = shap.TreeExplainer(model_s1)
shap_v1 = exp_s1.shap_values(X_all_s1)
# Plot SHAP Summary & Dependence (Figs 2, 3)

# Stage 2: Global Intra-operative Model
model_s2 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
model_s2.fit(X_all_s2, y_all)
exp_s2 = shap.TreeExplainer(model_s2)
shap_v2 = exp_s2.shap_values(X_all_s2)
# Plot SHAP Summary (Fig 4)

# ---------------------------------------------------------
# 5. MULTIVARIATE CAUSAL MEDIATION ANALYSIS
# ---------------------------------------------------------
# Calculating Total Effect vs Direct Effect to identify Mediation % 
# Z-score standardization for coefficient comparison
scaler = StandardScaler()
df_med_imputed = pd.DataFrame(imputer_s1.transform(df_all[stage1_vars + ['Operative_Time', 'EBL']]), columns=stage1_vars + ['Operative_Time', 'EBL'])
df_med_imputed[:] = scaler.fit_transform(df_med_imputed)

X_total = sm.add_constant(df_med_imputed[stage1_vars])
model_total = sm.OLS(y_all, X_total).fit()

X_direct = sm.add_constant(df_med_imputed[stage1_vars + ['Operative_Time', 'EBL']])
model_direct = sm.OLS(y_all, X_direct).fit()
# Generate Dumbbell Plot for Effect Shrinkage (Fig 7B)
