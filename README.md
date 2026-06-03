# Exploring environmental‑biodiversity relationships in abyssal polymetallic nodule fields using eDNA metabarcoding and interpretable machine learning

This repository contains the complete code and data for the manuscript:

> **"Exploring environmental‑biodiversity relationships in abyssal polymetallic nodule fields using eDNA metabarcoding and interpretable machine learning"**

We combine eDNA metabarcoding of benthic foraminifera with high‑dimensional environmental data (sediment grain‑size, major/trace elements, organic matter, bathymetry) and apply interpretable machine learning (linear models, ridge, random forest, XGBoost, LightGBM, GAM) to predict biodiversity metrics and quantify the influence of environmental drivers.

------

## 📁 Repository structure

text

```
.
├── data/                         # Raw and processed data (see below)
├── figures_R/                    # Output directory for R‑generated figures
├── output/                       # Results from non‑spatial cross‑validation (KFold)
├── output_group/                 # Results from spatial cross‑validation (GroupKFold by Area)
├── total/                        # Full analysis with 9 biodiversity targets (supplementary)
│   ├── output_total/             # Results for 9 targets (non‑spatial CV)
│   ├── output_group_total/       # Results for 9 targets (spatial CV)
│   └── figures_R_total/          # Figures for 9 targets
├── foram_ai.py                   # Main ML script – non‑spatial 5‑fold CV (3 targets)
├── foram_ai (group).py           # ML script with spatial GroupKFold by Area (3 targets)
├── foram_ai.R                    # R script for visualisation (all figures, tables)
├── README.md                     # This file
└── (other supporting files)
```



------

## 🔧 Requirements

### Python (≥3.8)

Install the required packages:

bash

```
pip install numpy pandas scikit-learn xgboost lightgbm pygam shap statsmodels openpyxl
```



Key libraries:

- `scikit-learn` – preprocessing, PCA, cross‑validation, models
- `xgboost`, `lightgbm` – gradient boosting
- `pygam` – generalised additive models
- `shap` – model interpretability
- `statsmodels` – VIF calculation

### R (≥4.0)

Run the following to install the required packages:

r

```
install.packages(c("ggplot2", "cowplot", "dplyr", "tidyr", "patchwork",
                   "viridis", "ggsci", "RColorBrewer", "scales", "grid",
                   "gtable", "corrplot", "stringr", "ggpubr"))
```



------

## 📊 Data description

All input data are stored in the `data/` folder.

| File                                  | Description                                                  |
| :------------------------------------ | :----------------------------------------------------------- |
| `foram_env_data_dy79.xlsx`            | Main table linking environmental variables (grain size, geochemistry, depth, nutrients) with foraminiferal biodiversity indices (reads, ZOTUs, Shannon, Simpson, taxonomic orders). Used in all ML scripts. |
| `zotu_table.txt`                      | ZOTU abundance table (raw metabarcoding counts).             |
| `zrep_seqs.fa`                        | Representative sequences of ZOTUs.                           |
| `rep_seqs_tax_assignments_blast.txt`  | Taxonomic assignment against PR2 database.                   |
| `rep_seqs_tax_assignments_blast2.txt` | Taxonomic assignment against BFR2 database.                  |

The environmental feature set includes:

- **Sedimentology**: Grain Size, Sorting, Skewness, Kurtosis, Clay/Silt/Sand Content
- **Major elements**: Na₂O, MgO, Al₂O₃, SiO₂, K₂O, CaO, Fe₂O₃
- **Trace elements**: Sc, Ti, V, Cr, Mn, Co, Ni, Cu, Zn, Ga, As, Rb
- **Organic / nutrient**: Chlorophyll a, N, C, S1, TOC, P, S2, Cl
- **Bathymetry**: Depth

Target variables (main analysis): `reads` (sequencing depth), `ZOTUs` (richness), `Monothalamida` (relative abundance of a key foraminiferal clade).

> For supplementary analysis, 9 targets are used (including Shannon, Simpson, Rotaliida, Textulariida, Miliolida, Other) – see the `total/` folder.

------

## 🤖 Machine learning scripts

Two Python scripts implement the modelling pipeline:

### `foram_ai.py` – Non‑spatial cross‑validation

- **CV strategy**: 5‑fold random KFold (shuffled)
- **Use case**: general predictive performance, no spatial grouping
- **Output directory**: `output/`

### `foram_ai (group).py` – Spatial cross‑validation

- **CV strategy**: 3‑fold GroupKFold based on `Area` column (samples from same region stay together)
- **Use case**: assess generalisability to unsampled areas, prevent spatial overfitting
- **Output directory**: `output_group/`

Both scripts share the same internal steps:

1. Load and clean data, select target variables.
2. For each target and each outer fold:
   - Split into train/validation.
   - Perform **grouped PCA** (four groups: Sediment, Major elements, Trace elements, Organic+Depth) – extract PC1 and PC2 from each group.
   - Train six models: Linear, Ridge (tuned), Random Forest (tuned), XGBoost (tuned), LightGBM (tuned), GAM (tuned).
   - Record performance (R², RMSE, MAE) on validation **and** training sets.
   - Compute SHAP values for tree‑based models (RF, XGB, LGB) on validation set.
   - Save PCA loadings and explained variances.
3. Aggregate results: performance summary, hyperparameters, SHAP importance, VIF analysis, stability metrics, OOF predictions for best models.

Run a script from the repository root:

bash

```
python "foram_ai.py"          # non‑spatial CV
python "foram_ai (group).py"  # spatial CV
```



------

## 📈 Visualisation (R script)

`foram_ai.R` generates all main and supplementary figures (`.tif` and `.pdf`) from the CSV outputs of the Python scripts.

### Main figures (for `output_group/` – spatial CV)

| Figure   | Content                                                      |      |      |
| :------- | :----------------------------------------------------------- | ---- | ---- |
| Figure 1 | Predicted vs observed scatter plots for all six models (three targets, faceted). |      |      |
| Figure 2 | Bar plot of cross‑validated R² ± SD across models and targets. |      |      |
| Figure 3 | SHAP heatmap (best‑performing model per target).             |      |      |
| Figure 4 | SHAP bar plot with grouped features (coloured by PCA group). |      |      |
| Figure 5 | Proportional contribution of each PCA group to total         | SHAP | .    |
| Figure 6 | Residual diagnostics (residuals vs fitted + Q‑Q plots) for best models. |      |      |

### Supplementary figures (S1–S10)

- S1 – PCA loadings (PC1 vs PC2) heatmap.
- S2 – Explained variance ratios (PC1 & PC2) across folds.
- S3 – VIF comparison between original features and PC features.
- S4 – SHAP importance variability (coefficient of variation).
- S5 – Cross‑fold stability of SHAP rankings (Spearman correlation).
- S6 – Train‑test R² gap (overfitting indicator).
- S7 – Hyperparameter distributions across folds.
- S8 – GAM hyperparameter stability (lambda, EDOF, CV R²).
- S9 – Feature importance CV (each feature).
- S10 – Frequency of features appearing in top‑2 importance.

### How to run the R script

1. Set working directory to repository root.
2. Ensure `output_group/` contains the required CSV files (generated by the Python script).
3. Execute:

r

```
source("foram_ai.R")
```



All figures will be saved in `figures_R/three_targets/`.

> You can modify `target_subset` inside the script to analyse other targets (e.g., nine targets for supplementary analysis).

------

## 📄 Output files description

### Main outputs (`output/` or `output_group/`)

| File                                    | Description                                                  |
| :-------------------------------------- | :----------------------------------------------------------- |
| `model_performance_summary.csv`         | R², RMSE, MAE for each fold, model, target (both train and test). |
| `scatter_data_all.csv`                  | Observed vs predicted values for all models and folds.       |
| `shap_all_models.csv`                   | Mean absolute SHAP values per feature, fold, model, target.  |
| `best_hyperparameters_all_folds.csv`    | Tuned hyperparameters for Ridge, RF, XGB, LGB per fold.      |
| `gam_lam_records.csv`                   | GAM optimal lambda, EDOF, and CV R² per fold.                |
| `pca_loadings_all.csv`                  | Loadings of original variables on PC1/PC2 per group, fold, target. |
| `pca_variances_all.csv`                 | Explained variance ratios for PC1/PC2.                       |
| `vif_original_features.csv`             | Variance Inflation Factor for raw environmental variables.   |
| `vif_pc_features.csv`                   | VIF for the 8 PC features (after PCA).                       |
| `feature_shap_importance_by_model.csv`  | Global SHAP importance (averaged across folds).              |
| `shap_group_contributions_by_model.csv` | Contribution of each PCA group to total SHAP.                |
| `shap_stability_mean_sd_cv.csv`         | Mean, SD, CV of SHAP values across folds.                    |
| `oof_predictions_best_models.csv`       | OOF predictions for the best‑performing model per target.    |
| `train_test_gap_analysis.csv`           | R² gap (train – test) as an overfitting metric.              |
| `Table1_Area_Distribution.csv`          | Sample count per geographic area.                            |
| `Table2_Model_Performance_Summary.csv`  | Average performance (R², RMSE, MAE) per target & model.      |
| `Table3_Best_Model_per_Target.csv`      | Best model (highest R²) for each target.                     |
| `Table4_SHAP_Top5_Features_XGBoost.csv` | Top‑5 SHAP features for XGBoost (example).                   |

(Other stability files – `shap_rank_spearman_stability.csv`, `model_gap_analysis.csv`, `residual_distribution_summary.csv`, `hyperparameter_stability_summary.csv`, `gam_stability_summary.csv`, `pca_loading_stability.csv`, `feature_importance_variation.csv`, `shap_top_frequency.csv` – are also generated.)

------

## 📦 Supplementary analysis (full 9 targets)

The `total/` folder contains **identical workflows** but using 9 biodiversity targets:
`reads`, `zotu`, `shannon`, `simpson`, `Rotaliida`, `Textulariida`, `Monothalamida`, `Miliolida`, `Other`.

- `foram_ai_total.py` – non‑spatial CV (5‑fold)
- `foram_ai (group)_total.py` – spatial CV (GroupKFold)
- `foram_ai_diversity.R` – R visualisation for diversity indices
- `foram_ai_composition.R` – R visualisation for taxonomic composition

All outputs are stored in `output_total/`, `output_group_total/`, and `figures_R_total/`.

This part provides additional evidence for the robustness of the environmental‑biodiversity relationships across a wider range of ecological metrics.

------

## 🚀 How to reproduce the full analysis

### Step 1: Prepare data

Place `foram_env_data_dy79.xlsx` and all other raw files inside the `data/` folder.
No further preprocessing is required – the scripts handle column cleaning and subscript conversion.

### Step 2: Run Python modelling

bash

```
python "foram_ai.py"          # non‑spatial, 3 targets → output/
python "foram_ai (group).py"  # spatial, 3 targets → output_group/
```



For the supplementary 9‑target analysis:

bash

```
cd total
python "foram_ai_total.py"
python "foram_ai (group)_total.py"
cd ..
```



### Step 3: Generate figures (R)

Open `foram_ai.R` in RStudio or run:

bash

```
Rscript foram_ai.R
```



Figures will appear in `figures_R/three_targets/`.
For the 9‑target figures, run the respective R scripts inside `total/`.

### Step 4: Explore outputs

All CSV files are human‑readable and can be opened in Excel or any data analysis software. They contain all numeric results needed to reconstruct tables and figures in the manuscript.

------

## 📜 License

This code is provided under the MIT License.

------

**If you use this code or data in your research, please cite our manuscript (citation details will be added upon publication).**
