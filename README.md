# Global-Crop-Yield-Forecasting
# Global Crop Yield Forecasting via Ensemble Machine Learning and SHAP Interpretability

## Overview
This project develops a multi-domain machine learning framework for global crop yield prediction using FAOSTAT data covering 244 countries over 33 years (1990–2022). The study integrates climate, agricultural, and socio-economic indicators into a unified predictive system with explainable AI methods.

## Key Features
- Multi-source FAOSTAT panel dataset (7,600+ observations)
- 22 engineered predictive features
- Three composite indices:
  - Climate Stress Index
  - Soil Fertility Index
  - Agricultural Intensity Index
- Temporal train-test split (1990–2015 train, 2016–2022 test)
- Ensemble ML models: Random Forest, XGBoost, Gradient Boosting
- Explainability using SHAP

## Methods
- Data preprocessing and feature engineering
- Ensemble machine learning regression
- Model evaluation using MAE, RMSE, R²
- Statistical validation using Wilcoxon signed-rank test
- Explainable AI using SHAP

## Results Summary
- R² up to 0.989 on test data
- Gradient Boosting achieved lowest MAE
- Feature importance dominated by temporal lag + soil nutrient variables

## Tools & Libraries
Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib

## Status
This project is part of an independent research pipeline and is being prepared for manuscript submission.

## Full Manuscript
The full research manuscript is available here:

[Paper](Main Manuscript.docx)

## How to Run This Project

### 1. Clone repository
git clone https://github.com/MuhammadShahzaib-PK/Global-Crop-Yield-Forecasting.git

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run model training
python src/main.py

## Author
Muhammad Shahzaib
Independent Researcher | Mechatronics Engineer
