# Global Crop Yield Forecasting using Ensemble Machine Learning and SHAP

## Overview

This project develops a machine learning framework for global crop yield prediction using multi-source FAOSTAT datasets covering 244 countries from 1990–2022. Climate, agricultural, and socio-economic indicators are integrated into a unified predictive model, with SHAP explainability used to interpret model predictions. 

---

## Objectives

- Predict crop yield using ensemble machine learning models.
- Integrate climate, agricultural, and socio-economic indicators.
- Develop novel engineered indices for improved prediction.
- Explain model decisions using SHAP.
- Compare multiple machine learning algorithms using statistical evaluation.

---

## Dataset

**Source:** FAOSTAT

**Coverage:**
- 244 countries
- 1990–2022
- 7,600+ observations

Data sources include:

- Climate indicators
- Crop production
- Livestock statistics
- Producer prices
- Agricultural emissions
- Employment indicators
- Nutrient indicators

---

## Feature Engineering

The project includes several engineered features, including:

- Climate Stress Index
- Soil Fertility Index
- Agricultural Intensity Index
- Crop Yield Lag Features
- Yield Growth Rate

---

## Machine Learning Models

The following models were evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Support Vector Regression (SVR)
- Random Forest
- Gradient Boosting
- XGBoost

Model performance was evaluated using:

- R² Score
- RMSE
- MAE
- 5-Fold Cross Validation

A temporal train-test split (1990–2015 training, 2016–2022 testing) was used to better simulate real-world forecasting.

---

## Model Explainability

The best-performing model was interpreted using SHAP (SHapley Additive exPlanations).

The repository includes:

- SHAP Beeswarm Plot
- SHAP Feature Importance Plot

Statistical significance between model performances was evaluated using the Wilcoxon Signed-Rank Test.

---

## Repository Structure

```
Global-Crop-Yield-Forecasting/
│
├── notebooks/
│   └── Global_Crop_Yield_Forecasting.ipynb
│
├── figures/
│   ├── ml_comparison_results.png
│   ├── shap_beeswarm.png
│   └── shap_bar.png
│
├── results/
│   ├── model_comparison_temporal.csv
│   ├── model_comparison_table.csv
│   ├── shap_importance.csv
│   └── statistical_tests.csv
│
└── README.md
```

---

## How to Run

1. Clone the repository.
2. Install the required Python libraries.
3. Open the notebook in the `notebooks` folder.
4. Update dataset paths if required.
5. Run all cells sequentially.

---

## Key Highlights

- Global-scale crop yield forecasting framework
- Ensemble machine learning comparison
- Novel feature engineering
- Explainable AI using SHAP
- Statistical validation using the Wilcoxon Signed-Rank Test

---

## Author

**Muhammad Shahzaib**

Mechatronics Engineer |  Researcher

Research Interests:
- Machine Learning
- Predictive Maintenance
- Industrial Robotics
- Marine Robotics
- Explainable AI
