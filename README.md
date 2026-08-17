**Continental Crop Yield Prediction and Cross-Country Transfer Limits Across Africa**
**Overview**

This project develops a machine learning framework for crop yield prediction across Africa, built on a 52-country panel (FAOSTAT + NASA POWER, 1990–2024, 1,770 observations). Eight ML architectures are benchmarked under both a standard temporal train/test split and leave-one-country-out (LOCO) cross-validation, to test whether models generalize across countries or merely memorize country-specific yield baselines. SHAP explainability is used to diagnose why models succeed or fail under each validation scheme.

**Objectives**
**Structured around four research questions:**

How much does historical yield (one-year lagged cereal yield) contribute to prediction accuracy, and does removing it reveal a generalization limit not visible under standard validation?
How does model architecture affect transferability to countries entirely absent from training, and what mechanism explains the difference between model families?
Do additional, globally available structural variables — specifically growing degree days — meaningfully improve cross-country generalization?
Which features drive predictions under SHAP, and does the answer change once historical yield is unavailable?
Dataset

Sources: FAOSTAT (yield, fertilizer, temperature), NASA POWER (precipitation, via representative country points derived from Natural Earth 1:10m boundaries)

**Coverage:**

52 of 54 African countries (Equatorial Guinea and Seychelles excluded — insufficient continuous cereal yield record)
1990–2024
1,770 country-year observations

The African panel was selected from a global audit of 248 FAO reporting entities: 151 had usable full-period cereal yield records, 32 had partial/late-start records, and 65 had no usable record. Africa was retained for the study because it offered near-complete national coverage and continuous records across the full period despite substantial climatic and agronomic heterogeneity.

**Feature Selection**

An initial pool of 24 candidate predictors was screened in two stages: retained if pooled Pearson correlation with cereal yield had |r| > 0.10 and p < 0.05, then checked pairwise for redundancy (excluded if |r| > 0.85 with an already-selected predictor) and for target leakage (e.g. soil nutrient balance variables derived from crop removal were dropped outright).

**Final feature set (6 variables):**

Total nitrogen fertilizer input
Total phosphorus fertilizer input
Total potassium fertilizer input
Total annual precipitation (NASA POWER, summed daily)
Annual temperature change
One-year lagged cereal yield ("Yield Lag1") — included in the operational feature set, excluded from the transferable feature set used for LOCO-without-yield experiments

Notable exclusions: nutrient removal by harvested crops, soil nutrient balance, and synthetic fertilizer application to soil were excluded as leakage or redundancy risks; production index, harvested area, and livestock abundance were excluded for negligible pooled association with yield (r ≈ 0).

Sensitivity test: growing degree days (GDD) were added to the transferable (no-yield-lag) feature set to test whether a low-cost thermal variable could close the cross-country generalization gap — see Results.

**Machine Learning Models**

The following eight architectures were benchmarked:

Linear Regression
Lasso Regression
Random Forest
Extra Trees
Gradient Boosting
XGBoost
LightGBM
CatBoost

**Model performance was evaluated using:**

R² Score
RMSE
MAE

**Two validation schemes were used:**

Temporal split — train ≤2017 (1,148 rows), test ≥2018 (290 rows), simulating real-world forecasting for countries already present in training.
Leave-one-country-out (LOCO) cross-validation — 7-fold, grouped by country, testing prediction for a country entirely absent from training.

Every model was run twice: once with one-year lagged cereal yield included ("operational" scenario) and once with it removed ("transferable" scenario), to separate in-sample forecasting skill from genuine cross-country generalization.

**Headline results:**

With historical yield: temporal-split R² up to 0.948 (Linear/Lasso); LOCO R² up to 0.853 (Linear/Lasso), down to 0.703 (CatBoost).
Without historical yield: temporal-split accuracy dropped only modestly (R² still above 0.77 for all eight models). LOCO accuracy collapsed for every tree-based model (R² as low as −1.21, LightGBM), while linear models stayed modestly positive (R² = 0.148).
Mean LOCO R² drop after removing historical yield: 1.60 for the six tree-based models vs. 0.70 for the two linear models (a 2.3× difference), statistically significant for all eight models (paired t-test, p < 0.05).
A per-country bias decomposition traced the tree-model collapse to a systematic offset (inability to relocate predictions toward an unfamiliar country's yield baseline) rather than random noise — offset magnitude correlated strongly with a held-out country's distance from the training distribution (r = 0.84–0.85, p < 0.0001).
**Model Explainability**

SHAP (SHapley Additive exPlanations) was applied to CatBoost and Linear, chosen as representative tree-based and linear models, separately for the scenarios with and without historical yield.

With historical yield: it dominated both models' attributions (70.8% of SHAP magnitude for CatBoost, 92.0% for Linear), with fertilizer and climate variables splitting the remainder.
Without historical yield: fertilizer inputs became the dominant SHAP category for both (74.2% CatBoost, 84.2% Linear). The two models disagreed on which fertilizer nutrient ranked highest (CatBoost: potassium; Linear: nitrogen) — attributed to moderate-to-high collinearity among the three fertilizer inputs (r = 0.60–0.82) rather than a genuine difference in driver importance.

**The repository includes:**

SHAP beeswarm plots (CatBoost and Linear, with/without historical yield)
Per-country bias decomposition plots (LOCO, with/without historical yield)
Predicted-vs-observed scatter plots for all eight models under LOCO validation

Statistical significance between the with/without-historical-yield scenarios was evaluated using paired t-tests across the seven LOCO folds (not Wilcoxon — corrected from the earlier draft), since fold composition is identical across scenarios.

Growing degree days (GDD) sensitivity test: GDD was added to the no-historical-yield feature set to test whether a low-cost thermal variable could close the cross-country generalization gap. Of eight models, only Gradient Boosting showed a statistically significant improvement (LOCO R² 0.818 → 0.823, p = 0.043); the other seven showed no significant change. This narrows, rather than closes, the search for the missing structural variable — the paper points toward irrigation share, drought indices, and vapor pressure deficit as stronger candidates for future work.

**Repository Structure**

Note: update folder/file names below to match your actual repo before publishing — these are placeholders carried over from the earlier draft.

Continental-Crop-Yield-Africa/
│
├── notebooks/
│   └── Continental_Crop_Yield_Africa.ipynb
│
├── figures/
│   ├── ml_comparison_temporal.png
│   ├── ml_comparison_loco.png
│   ├── shap_beeswarm.png
│   └── shap_bar.png
│
├── results/
│   ├── model_comparison_temporal.csv
│   ├── model_comparison_loco.csv
│   ├── shap_importance.csv
│   └── statistical_tests.csv
│
└── README.md
How to Run
Clone the repository.
Install the required Python libraries.
Open the notebook in the notebooks folder.
Update dataset paths if required.
Run all cells sequentially.
Key Highlights
Continental-scale (Africa) crop yield forecasting framework across 52 countries, 1990–2024
Benchmarking of eight ML architectures under both temporal-split and leave-one-country-out (LOCO) validation
Quantifies historical yield's contribution to accuracy (70.8–92.0% of SHAP magnitude) and its role in masking a cross-country generalization limit
Identifies a structural mechanism for tree-ensemble failure under LOCO: systematic per-country bias from inability to extrapolate beyond training-country yield baselines (2.3× larger R² drop than linear models)
Tests and largely rules out growing degree days as the missing generalization variable, pointing toward irrigation and drought-related features for future work
Statistical validation using paired t-tests across LOCO folds; feature redundancy and leakage screening prior to modeling
Author

Muhammad Shahzaib Mechatronics Engineer | Independent Researcher, Causal & Explainable Machine Learning for Climate and Agricultural Systems

Research Interests:

Causal Inference
Explainable Machine Learning
Climate-Adaptive Agriculture
Reinforcement Learning
Food Security
