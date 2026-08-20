# %% [markdown]
# # Continental Crop Yield Prediction and Cross-Country Transfer Limits Across Africa
#
# Reproduction pipeline for the methodology described in *A Generalizable Machine Learning
# Framework for Continental Crop Yield Prediction Across Africa* (Shahzaib & Khan, submitted to
# *Precision Agriculture*, Springer).
#
# **Important note on provenance:** this notebook is a clean-room reproduction built to match the
# manuscript's documented methodology (feature set, validation schemes, models, statistical tests).
# It is not the original code used to produce the published tables and figures, which was not
# preserved in a clean, standalone form. Numbers here land in the same range and support the same
# conclusions as the manuscript, but do not match to the decimal — see `results/README.md` for a
# side-by-side comparison against the published tables.
#
# Panel: 52 African countries, 1990-2024, 1,770 country-year observations (FAOSTAT + NASA POWER).

# %%
import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import ttest_rel
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# %% [markdown]
# ## 1. Load panel and build the lagged-yield feature
#
# The panel merges four sources: a FAOSTAT-derived base panel (cereal yield, fertilizer inputs),
# FAOSTAT Environment/Temperature Change, NASA POWER precipitation, and NASA POWER-derived growing
# degree days (used later for the sensitivity test).

# %%
df = pd.read_csv("data/africa_full_panel.csv")
df = df.sort_values(["Area", "Year"]).reset_index(drop=True)
df["yield_lag1"] = df.groupby("Area")["crop_yield"].shift(1)

df = df.rename(columns={
    "nitrogen_per_area": "nitrogen_total",
    "phosphorus_per_area": "phosphorus_total",
    "potassium_per_area": "potassium_total",
    "precip_mm_annual": "precip_annual",
    "temperature_change": "temp_change",
})

FEATURES_WITH_LAG = ["nitrogen_total", "phosphorus_total", "potassium_total",
                      "precip_annual", "temp_change", "yield_lag1"]
FEATURES_NO_LAG = ["nitrogen_total", "phosphorus_total", "potassium_total",
                    "precip_annual", "temp_change"]
FEATURES_NO_LAG_GDD = FEATURES_NO_LAG + ["gdd_annual"]
TARGET = "crop_yield"

model_data = df.dropna(subset=FEATURES_WITH_LAG + [TARGET]).copy()
model_data_no_lag = df.dropna(subset=FEATURES_NO_LAG_GDD + [TARGET]).copy()

print(f"Rows usable with yield lag: {len(model_data)}")
print(f"Rows usable without yield lag: {len(model_data_no_lag)}")

# %% [markdown]
# ## 2. The eight-model zoo
#
# Two linear baselines (standardized before fitting), two bagged-tree ensembles, four boosted-tree
# ensembles. All tree models use 300 estimators; other hyperparameters follow library defaults, per
# the manuscript's stated design (the comparison of interest is architecture family and validation
# scheme, not per-model tuning).

# %%
def build_models():
    return {
        "Linear": LinearRegression(),
        "Lasso": Lasso(alpha=0.01, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(n_estimators=300, random_state=RANDOM_STATE, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=300, random_state=RANDOM_STATE, verbosity=-1),
        "CatBoost": CatBoostRegressor(n_estimators=300, random_state=RANDOM_STATE, verbose=False),
    }

LINEAR_MODELS = {"Linear", "Lasso"}

def fit_predict(name, model, X_train, y_train, X_test):
    if name in LINEAR_MODELS:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model.fit(X_train_s, y_train)
        return model.predict(X_test_s)
    model.fit(X_train, y_train)
    return model.predict(X_test)

# %% [markdown]
# ## 3. Temporal split validation (train <=2017, test >=2018)
#
# The standard operational scenario: forecasting future years for countries already present in
# training.

# %%
def temporal_split_eval(data, features):
    train = data[data["Year"] <= 2017]
    test = data[data["Year"] >= 2018]
    results = {}
    for name, model in build_models().items():
        preds = fit_predict(name, model, train[features], train[TARGET], test[features])
        results[name] = {
            "r2": r2_score(test[TARGET], preds),
            "rmse": mean_squared_error(test[TARGET], preds) ** 0.5,
            "mae": mean_absolute_error(test[TARGET], preds),
            "n_train": len(train), "n_test": len(test),
        }
    return results

temporal_with_lag = temporal_split_eval(model_data, FEATURES_WITH_LAG)
temporal_no_lag = temporal_split_eval(model_data_no_lag, FEATURES_NO_LAG)

pd.DataFrame(temporal_with_lag).T[["r2", "rmse", "mae"]].round(3)

# %% [markdown]
# ## 4. Leave-one-country-out (LOCO) validation, 7-fold grouped by country
#
# The harder, policy-relevant scenario: predicting yield for a country entirely absent from
# training.

# %%
def make_loco_folds(countries, n_folds=7, seed=RANDOM_STATE):
    rng = np.random.RandomState(seed)
    shuffled = list(countries)
    rng.shuffle(shuffled)
    return [shuffled[i::n_folds] for i in range(n_folds)]

def loco_eval(data, features, n_folds=7):
    countries = sorted(data["Area"].unique())
    folds = make_loco_folds(countries, n_folds)
    per_model_fold_r2 = {name: [] for name in build_models()}
    per_model_preds = {name: [] for name in build_models()}

    for fold_countries in folds:
        train = data[~data["Area"].isin(fold_countries)]
        test = data[data["Area"].isin(fold_countries)]
        if len(test) == 0 or len(train) == 0:
            continue
        for name, model in build_models().items():
            preds = fit_predict(name, model, train[features], train[TARGET], test[features])
            per_model_fold_r2[name].append(r2_score(test[TARGET], preds))
            per_model_preds[name].append(pd.DataFrame({
                "Area": test["Area"].values, "Year": test["Year"].values,
                "y_true": test[TARGET].values, "y_pred": preds
            }))

    summary = {}
    for name in build_models():
        fold_r2 = per_model_fold_r2[name]
        preds_df = pd.concat(per_model_preds[name], ignore_index=True)
        summary[name] = {
            "mean_fold_r2": float(np.mean(fold_r2)),
            "overall_r2": float(r2_score(preds_df["y_true"], preds_df["y_pred"])),
            "fold_r2": fold_r2,
            "predictions": preds_df,
        }
    return summary

loco_with_lag = loco_eval(model_data, FEATURES_WITH_LAG)
loco_no_lag = loco_eval(model_data_no_lag, FEATURES_NO_LAG)

pd.DataFrame({name: {"mean_fold_r2": v["mean_fold_r2"], "overall_r2": v["overall_r2"]}
              for name, v in loco_with_lag.items()}).T.round(3)

# %% [markdown]
# ## 5. Paired t-tests across LOCO folds: with vs. without historical yield
#
# Paired (not unpaired) because fold composition is identical across the two scenarios; pairing
# isolates the effect of removing the feature from ordinary fold-to-fold variance.

# %%
ttest_results = {}
for name in build_models():
    with_fold = loco_with_lag[name]["fold_r2"]
    without_fold = loco_no_lag[name]["fold_r2"]
    n = min(len(with_fold), len(without_fold))
    stat, p = ttest_rel(with_fold[:n], without_fold[:n])
    drop = np.mean(with_fold[:n]) - np.mean(without_fold[:n])
    ttest_results[name] = {"r2_with": float(np.mean(with_fold[:n])),
                            "r2_without": float(np.mean(without_fold[:n])),
                            "r2_drop": float(drop), "p_value": float(p)}

tree_models = ["RandomForest", "ExtraTrees", "GradientBoosting", "XGBoost", "LightGBM", "CatBoost"]
linear_models_list = ["Linear", "Lasso"]
mean_tree_drop = np.mean([ttest_results[m]["r2_drop"] for m in tree_models])
mean_linear_drop = np.mean([ttest_results[m]["r2_drop"] for m in linear_models_list])

pd.DataFrame(ttest_results).T.round(4)

# %%
print(f"Mean R2 drop -- tree models: {mean_tree_drop:.3f} | linear models: {mean_linear_drop:.3f} "
      f"| ratio: {mean_tree_drop/mean_linear_drop:.2f}x")

# %% [markdown]
# ## 6. Per-country bias decomposition
#
# Splits each country's LOCO prediction error (without historical yield) into a systematic offset
# (how far the model's average prediction sits from the country's true mean) and within-country
# noise, then correlates offset magnitude with the country's distance from the training
# distribution's mean.

# %%
bias_results = {}
for name in ["Linear", "RandomForest"]:
    preds_df = loco_no_lag[name]["predictions"]
    country_stats = preds_df.groupby("Area").apply(
        lambda g: pd.Series({
            "true_mean": g["y_true"].mean(),
            "pred_mean": g["y_pred"].mean(),
            "offset": g["y_pred"].mean() - g["y_true"].mean(),
            "within_country_std": (g["y_true"] - g["y_pred"]).std(),
        })
    ).reset_index()
    train_dist_mean = model_data_no_lag[TARGET].mean()
    country_stats["dist_from_training_mean"] = (country_stats["true_mean"] - train_dist_mean).abs()
    offset_mag = country_stats["offset"].abs()
    corr = offset_mag.corr(country_stats["dist_from_training_mean"])
    bias_results[name] = {
        "mean_abs_offset": float(offset_mag.mean()),
        "mean_within_country_std": float(country_stats["within_country_std"].mean()),
        "corr_offset_vs_distance": float(corr),
    }

pd.DataFrame(bias_results).T.round(3)

# %% [markdown]
# ## 7. Growing degree days (GDD) sensitivity test
#
# Tests whether adding a low-cost thermal-accumulation variable closes part of the cross-country
# generalization gap identified above.

# %%
loco_no_lag_gdd = loco_eval(model_data_no_lag, FEATURES_NO_LAG_GDD)
gdd_results = {}
for name in build_models():
    base_fold = loco_no_lag[name]["fold_r2"]
    gdd_fold = loco_no_lag_gdd[name]["fold_r2"]
    n = min(len(base_fold), len(gdd_fold))
    stat, p = ttest_rel(gdd_fold[:n], base_fold[:n])
    delta = np.mean(gdd_fold[:n]) - np.mean(base_fold[:n])
    gdd_results[name] = {"baseline_r2": float(np.mean(base_fold[:n])),
                          "with_gdd_r2": float(np.mean(gdd_fold[:n])),
                          "delta_r2": float(delta), "p_value": float(p)}

pd.DataFrame(gdd_results).T.round(4)

# %% [markdown]
# ## 8. Figures

# %%
models_order = ["Linear", "Lasso", "RandomForest", "ExtraTrees", "GradientBoosting",
                 "XGBoost", "LightGBM", "CatBoost"]
with_lag_r2 = [ttest_results[m]["r2_with"] for m in models_order]
without_lag_r2 = [ttest_results[m]["r2_without"] for m in models_order]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(models_order))
width = 0.35
ax.bar(x - width/2, with_lag_r2, width, label="With historical yield", color="#4C72B0")
ax.bar(x + width/2, without_lag_r2, width, label="Without historical yield", color="#C44E52")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(models_order, rotation=30, ha="right")
ax.set_ylabel("LOCO mean fold R²")
ax.set_title("Effect of removing historical yield on LOCO validation performance")
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig_loco_comparison.png", dpi=150)
plt.show()

# %% [markdown]
# ## 9. SHAP: CatBoost and Linear, with and without historical yield

# %%
def shap_summary(data, features, model_type):
    X = data[features]
    y = data[TARGET]
    if model_type == "catboost":
        model = CatBoostRegressor(n_estimators=300, random_state=RANDOM_STATE, verbose=False)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_s, y)
        explainer = shap.LinearExplainer(model, X_s)
        shap_values = explainer.shap_values(X_s)
    mean_abs = np.abs(shap_values).mean(axis=0)
    pct = 100 * mean_abs / mean_abs.sum()
    return dict(zip(features, pct)), shap_values, X

shap_summary_stats = {}
for model_type in ["catboost", "linear"]:
    for lag_status, data, feats in [("with_lag", model_data, FEATURES_WITH_LAG),
                                      ("no_lag", model_data_no_lag, FEATURES_NO_LAG)]:
        pct, shap_vals, X = shap_summary(data, feats, model_type)
        shap_summary_stats[f"{model_type}_{lag_status}"] = pct

pd.DataFrame(shap_summary_stats).round(1)

# %% [markdown]
# ## 10. Save all results for the `results/` folder

# %%
all_results = {
    "temporal_with_lag": temporal_with_lag,
    "temporal_no_lag": temporal_no_lag,
    "loco_with_lag_summary": {k: {"mean_fold_r2": v["mean_fold_r2"], "overall_r2": v["overall_r2"],
                                    "fold_r2": v["fold_r2"]} for k, v in loco_with_lag.items()},
    "loco_no_lag_summary": {k: {"mean_fold_r2": v["mean_fold_r2"], "overall_r2": v["overall_r2"],
                                  "fold_r2": v["fold_r2"]} for k, v in loco_no_lag.items()},
    "ttest_with_vs_without_lag": ttest_results,
    "bias_decomposition": bias_results,
    "gdd_sensitivity": gdd_results,
    "shap_summary": shap_summary_stats,
}
with open("results/results_summary.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print("Done.")
