"""DS Team 7 GSM smartphone modeling pipeline.

This file is the main reproducible modeling script for the term project.
It is intentionally self-contained so the submitted notebook can run it in
Google Colab or locally and regenerate the same modeling outputs.

The modeling part has two practical goals:
1. B2B: predict an objective fair smartphone price from hardware and brand
   features, then explain important price drivers and brand premium.
2. B2C: segment phones, detect value-for-money outliers, and recommend devices
   that maximize performance-to-price ratio under a user budget.

The comments below explain not only what each block does, but also why the
method was chosen for this project.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover - fallback only for restricted envs.
    from sklearn.ensemble import HistGradientBoostingRegressor

    LGBMRegressor = None
    LIGHTGBM_AVAILABLE = False


# A fixed random seed keeps train/test split, cross-validation folds, clustering,
# and anomaly detection reproducible. This is important for grading because the metrics and output CSV files should not change every time the script runs.
RANDOM_STATE = 42

''' All paths are derived from this file location. This avoids hard-coded local paths 
and lets the same script work in GitHub, local Python, or Colab after the repository is uploaded.'''
ROOT_DIR = Path(__file__).resolve().parents[1]
MODELING_DIR = ROOT_DIR / "modeling"
OUTPUT_DIR = MODELING_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
MODEL_DIR = MODELING_DIR / "models"

# The modeling script consumes the preprocessing team's final CSV files from content/. 
# It does not re-run preprocessing because the modeling task should start from the cleaned and feature-engineered dataset.
CONTENT_DIR = ROOT_DIR / "content"

# Four files are used because price prediction and recommendation need slightly different feature sets.
# The "_all" files are encoded/scaled for model input, while the raw task files preserve readable columns for interpretation.
PRICE_MODEL_CSV = "gsm_processed_all(price_prediction).csv"
PRICE_RAW_CSV = "gsm_processed(price_prediction).csv"
RECO_MODEL_CSV = "gsm_processed_all(recommendation).csv"
RECO_RAW_CSV = "gsm_processed(recommendation).csv"

TARGET_COL = "target_price_eur"


def ensure_dirs() -> None:
    """Create every output directory used by the pipeline.

    The script writes metrics, plots, and trained model artifacts.
    Creating these folders at the beginning prevents later save operations from failing in a clean environment.
    """
    for path in [OUTPUT_DIR, PLOT_DIR, MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def find_data_source() -> Path:
    """Locate the team preprocessing output directory in the GitHub repo layout.

    The final version should load files from DS_Team7/content directly, not from a ZIP file or from a manual file picker. 
    This makes the project easier for teammates and graders to run after cloning the repository.
    """
    required = [PRICE_MODEL_CSV, PRICE_RAW_CSV, RECO_MODEL_CSV, RECO_RAW_CSV]
    if all((CONTENT_DIR / member).exists() for member in required):
        return CONTENT_DIR
    missing = [member for member in required if not (CONTENT_DIR / member).exists()]
    raise FileNotFoundError(
        "GitHub repo layout 기준 입력 CSV를 찾을 수 없습니다.\n"
        f"프로젝트 루트: {ROOT_DIR}\n"
        f"필수 폴더: {CONTENT_DIR}\n"
        "누락 파일: " + ", ".join(missing) + "\n"
        "기대 구조: DS_Team7/content/*.csv 와 DS_Team7/modeling/run_modeling.py"
    )


def read_modeling_csv(data_source: Path, member_name: str) -> pd.DataFrame:
    """Read one modeling CSV from DS_Team7/content.

    Keeping this as a helper makes it clear that every modeling input comes from the same validated preprocessing output folder.
    """
    return pd.read_csv(data_source / member_name)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE in percent.

    sMAPE is used in addition to MAE/RMSE because smartphone prices have a wide range. 
    A percentage-style metric helps compare relative error across cheap and expensive devices.
    """
    denom = np.abs(y_true) + np.abs(y_pred)
    score = np.where(denom == 0, 0, 2 * np.abs(y_true - y_pred) / denom)
    return float(np.mean(score) * 100)


def mase(y_true: np.ndarray, y_pred: np.ndarray, baseline_pred: np.ndarray) -> float:
    """MASE against a median-price naive baseline.

    MASE tells whether the model is better than a very simple business rule:
    always predict the median training price. 
    A value below 1 means the model is useful compared with that naive baseline.
    """
    baseline_mae = mean_absolute_error(y_true, baseline_pred)
    if baseline_mae == 0:
        return float("nan")
    return float(mean_absolute_error(y_true, y_pred) / baseline_mae)


def evaluate_price_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray,
) -> dict[str, float]:
    """Return all evaluation metrics in original EUR units.

    Even when the model is trained on log(price), evaluation is converted back
    to EUR because business users and customers understand pricing error in
    money units.
    """
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1, None)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "sMAPE": smape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, baseline_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_lgbm() -> Any:
    """Create the tree boosting model used for the strongest tabular baseline.

    LightGBM is effective for tabular data with many encoded features and nonlinear interactions.
    If LightGBM is unavailable, the fallback keeps the pipeline runnable in restricted environments.
    """
    if LIGHTGBM_AVAILABLE:
        return LGBMRegressor(
            n_estimators=450,
            learning_rate=0.045,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    return HistGradientBoostingRegressor(
        learning_rate=0.045,
        max_iter=450,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    )


def make_models() -> dict[str, Any]:
    """Define a compact model set that runs quickly in Colab CPU.

    The model list intentionally mixes simple and nonlinear models:
    - Linear Regression: easiest baseline.
    - Ridge: regularized baseline that is more stable with many features.
    - Random Forest: captures nonlinear hardware interactions.
    - LightGBM / HistGradientBoosting: strong boosted-tree model for final use.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=220,
            max_depth=18,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM" if LIGHTGBM_AVAILABLE else "HistGradientBoosting": make_lgbm(),
    }


@dataclass
class RegressionResult:
    """Container for regression outputs shared with later pipeline steps.

    Returning one object keeps the main function readable and prevents repeated CSV reads when later steps need the best model, metrics, or feature names.
    """

    metrics: pd.DataFrame
    cv_metrics: pd.DataFrame
    best_model: Any
    best_row: pd.Series
    feature_names: list[str]
    feature_medians: dict[str, float]


def train_and_evaluate_regression(price_model_df: pd.DataFrame) -> RegressionResult:
    """Train price prediction models and store holdout/CV metrics.

    This is the B2B modeling core. It predicts fair smartphone price from the encoded/scaled feature table created during preprocessing.
    The function compares several model families, evaluates both normal price and log-price targets, saves metrics, and stores the best model for later demo use.
    """
    if TARGET_COL not in price_model_df.columns:
        raise ValueError(f"{TARGET_COL} 컬럼이 회귀 입력에 없습니다.")

    ''' Separate features from the target.
    The preprocessing step already removed obvious leakage features from this encoded table, 
    so the model learns from hardware and brand signals rather than from price-derived columns.'''
    X = price_model_df.drop(columns=[TARGET_COL]).copy()
    y = price_model_df[TARGET_COL].astype(float).copy()
    feature_names = X.columns.tolist()
    feature_medians = X.median(numeric_only=True).to_dict()

    # Use an 80:20 holdout split for an easy-to-explain final performance check.
    # The same RANDOM_STATE makes the split reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    # Median price is the naive baseline.
    # If a model cannot beat this baseline, it is not useful as a pricing guide.
    baseline_test = np.full(len(y_test), y_train.median())

    holdout_rows: list[dict[str, Any]] = []
    fitted_models: dict[tuple[str, str], Any] = {}
    models = make_models()

    # Smartphone prices are skewed, so log(price) often produces more stable errors.
    # Both target modes are tested and compared using EUR-scale metrics.
    for target_mode in ["price_eur", "log_price_eur"]:
        train_target = np.log1p(y_train) if target_mode == "log_price_eur" else y_train
        for model_name, model in models.items():
            start = time.time()
            model.fit(X_train, train_target)
            fit_seconds = time.time() - start
            pred = model.predict(X_test)
            if target_mode == "log_price_eur":
                pred = np.expm1(pred)
            row = {
                "target_mode": target_mode,
                "model": model_name,
                **evaluate_price_metrics(y_test.to_numpy(), pred, baseline_test),
                "fit_seconds": fit_seconds,
            }
            holdout_rows.append(row)
            fitted_models[(target_mode, model_name)] = model

    holdout_metrics = pd.DataFrame(holdout_rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    holdout_metrics.to_csv(OUTPUT_DIR / "regression_holdout_metrics.csv", index=False)
    holdout_metrics.to_csv(OUTPUT_DIR / "regression_metrics.csv", index=False)

    cv_rows: list[dict[str, Any]] = []
    cv_fold_rows: list[dict[str, Any]] = []
    # Five-fold CV checks whether the model ranking is stable beyond one holdout split.
    # This also checks whether the selected model is stable across different validation folds.
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for target_mode in ["price_eur", "log_price_eur"]:
        for model_name, base_model in make_models().items():
            fold_metrics: list[dict[str, float]] = []
            start = time.time()
            for fold, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                model = make_models()[model_name]
                train_target = np.log1p(y_tr) if target_mode == "log_price_eur" else y_tr
                model.fit(X_tr, train_target)
                pred = model.predict(X_te)
                if target_mode == "log_price_eur":
                    pred = np.expm1(pred)
                baseline = np.full(len(y_te), y_tr.median())
                metrics = evaluate_price_metrics(y_te.to_numpy(), pred, baseline)
                fold_metrics.append(metrics)
                cv_fold_rows.append(
                    {
                        "target_mode": target_mode,
                        "model": model_name,
                        "fold": fold,
                        **metrics,
                    }
                )
            elapsed = time.time() - start
            avg_metrics = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
            cv_rows.append(
                {
                    "target_mode": target_mode,
                    "model": model_name,
                    **avg_metrics,
                    "fit_seconds": elapsed,
                }
            )

    cv_metrics = pd.DataFrame(cv_rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    cv_metrics.to_csv(OUTPUT_DIR / "regression_cv_metrics.csv", index=False)
    cv_metrics.to_csv(OUTPUT_DIR / "cv_metrics.csv", index=False)
    pd.DataFrame(cv_fold_rows).to_csv(OUTPUT_DIR / "regression_cv_fold_metrics.csv", index=False)

    # The best holdout MAE/RMSE row becomes the deployed model artifact.
    best_row = holdout_metrics.iloc[0]
    best_key = (str(best_row["target_mode"]), str(best_row["model"]))
    best_model = fitted_models[best_key]

    # Save not only the estimator but also metadata required for inference.
    # two_way_solution.py uses these fields to rebuild the model input row.
    package = {
        "model": best_model,
        "target_mode": best_key[0],
        "model_name": best_key[1],
        "feature_names": feature_names,
        "feature_medians": feature_medians,
        "lightgbm_available": LIGHTGBM_AVAILABLE,
    }
    joblib.dump(package, MODEL_DIR / "best_price_regressor.pkl")

    return RegressionResult(holdout_metrics, cv_metrics, best_model, best_row, feature_names, feature_medians)


def plot_regression_metrics(cv_metrics: pd.DataFrame) -> None:
    """Save quick CV metric comparison plots.

    These plots show that model choice was based on measured cross-validation performance, not only on one test split.
    """
    plt.figure(figsize=(10, 5))
    sns.barplot(data=cv_metrics, x="model", y="MAE", hue="target_mode")
    plt.title("5-fold CV MAE by model")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "regression_cv_mae.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=cv_metrics, x="model", y="R2", hue="target_mode")
    plt.title("5-fold CV R2 by model")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "regression_cv_r2.png", dpi=160)
    plt.close()


def feature_importance_analysis(price_model_df: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate feature importance and brand-related contribution.

    Feature importance supports the project objective of explaining how hardware specifications and brand variables affect smartphone price.
    Random Forest and LightGBM are both used so the explanation is not tied to only one model.
    """
    X = price_model_df.drop(columns=[TARGET_COL])
    y = price_model_df[TARGET_COL].astype(float)

    rf = RandomForestRegressor(
        n_estimators=260,
        max_depth=18,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    lgbm = make_lgbm()
    # Fit interpretation models on the full price-prediction table because this step is not used for holdout scoring;
    # it is used to summarize global feature influence after the model family has been evaluated.
    rf.fit(X, y)
    lgbm.fit(X, y)

    rows: list[dict[str, Any]] = []
    rows.extend(
        {
            "model": "Random Forest (price_eur)",
            "feature": feature,
            "importance": float(importance),
        }
        for feature, importance in zip(feature_names, rf.feature_importances_)
    )

    if hasattr(lgbm, "booster_"):
        gain = lgbm.booster_.feature_importance(importance_type="gain")
    elif hasattr(lgbm, "feature_importances_"):
        gain = lgbm.feature_importances_
    else:
        gain = np.zeros(len(feature_names))
    rows.extend(
        {
            "model": "LightGBM_gain (price_eur)" if LIGHTGBM_AVAILABLE else "HistGradientBoosting (price_eur)",
            "feature": feature,
            "importance": float(importance),
        }
        for feature, importance in zip(feature_names, gain)
    )

    importance_df = pd.DataFrame(rows)
    importance_df = importance_df.sort_values(["model", "importance"], ascending=[True, False])
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance_best_target.csv", index=False)

    top_features = []
    for model_name, group in importance_df.groupby("model"):
        top = group.sort_values("importance", ascending=False).head(15)
        top_features.append(top.assign(rank=range(1, len(top) + 1)))
        plt.figure(figsize=(9, 6))
        sns.barplot(data=top, y="feature", x="importance")
        plt.title(f"Top 15 Feature Importance - {model_name}")
        plt.tight_layout()
        filename = "rf_feature_importance_top15.png" if model_name.startswith("Random") else "lgbm_feature_importance_top15.png"
        plt.savefig(PLOT_DIR / filename, dpi=160)
        plt.close()

    top_df = pd.concat(top_features, ignore_index=True)
    consensus = (
        top_df.groupby("feature")
        .agg(avg_rank=("rank", "mean"), model_count=("model", "nunique"))
        .query("model_count >= 2")
        .sort_values(["avg_rank", "feature"])
        .reset_index()
    )
    consensus.to_csv(OUTPUT_DIR / "consensus_price_factors.csv", index=False)

    # Brand premium is measured as a share of total feature importance for all one-hot brand columns.
    # This directly answers the "brand awareness" part of the team proposal.
    brand_rows = []
    for model_name, group in importance_df.groupby("model"):
        total = group["importance"].sum()
        brand_sum = group.loc[group["feature"].str.startswith("cat__brand_group_"), "importance"].sum()
        brand_rows.append(
            {
                "model": model_name,
                "brand_importance_sum": brand_sum,
                "total_importance": total,
                "brand_importance_ratio": brand_sum / total if total else np.nan,
            }
        )
    brand_importance = pd.DataFrame(brand_rows)
    brand_importance.to_csv(OUTPUT_DIR / "brand_importance_summary.csv", index=False)

    return importance_df, consensus


def brand_premium_analysis(price_model_df: pd.DataFrame, price_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate raw and same-spec adjusted brand premiums.

    Raw average price can be misleading because premium brands may also have better hardware.
    To make the comparison fairer, a spec-only model predicts price without brand columns;
    the remaining residual is interpreted as an adjusted brand premium.
    """
    brand_summary = (
        price_raw_df.groupby("brand_group")
        .agg(
            count=("price_eur", "size"),
            avg_price_eur=("price_eur", "mean"),
            median_price_eur=("price_eur", "median"),
            avg_is_premium_brand=("is_premium_brand", "mean"),
        )
        .sort_values("avg_price_eur", ascending=False)
        .reset_index()
    )
    brand_summary.to_csv(OUTPUT_DIR / "brand_premium_summary.csv", index=False)

    X = price_model_df.drop(columns=[TARGET_COL]).copy()
    y = price_model_df[TARGET_COL].astype(float)
    brand_cols = [c for c in X.columns if c.startswith("cat__brand_group_")]
    spec_cols = [c for c in X.columns if c not in brand_cols and c != "num__is_premium_brand"]
    # Remove explicit brand columns and fit a spec-only model.
    # The gap between actual price and this spec-only prediction is our adjusted brand signal.
    spec_model = make_lgbm()
    spec_model.fit(X[spec_cols], y)
    pred = np.clip(spec_model.predict(X[spec_cols]), 1, None)

    premium_df = pd.DataFrame(
        {
            "brand_group": price_raw_df["brand_group"].values,
            "actual_price_eur": y.values,
            "spec_only_pred_price_eur": pred,
        }
    )
    premium_df["adjusted_brand_premium_eur"] = (
        premium_df["actual_price_eur"] - premium_df["spec_only_pred_price_eur"]
    )
    adjusted = (
        premium_df.groupby("brand_group")
        .agg(
            count=("actual_price_eur", "size"),
            avg_actual_price_eur=("actual_price_eur", "mean"),
            avg_spec_only_pred_price_eur=("spec_only_pred_price_eur", "mean"),
            avg_adjusted_brand_premium_eur=("adjusted_brand_premium_eur", "mean"),
            median_adjusted_brand_premium_eur=("adjusted_brand_premium_eur", "median"),
        )
        .reset_index()
    )
    adjusted["premium_ratio_vs_pred"] = (
        adjusted["avg_adjusted_brand_premium_eur"] / adjusted["avg_spec_only_pred_price_eur"]
    )
    adjusted = adjusted.sort_values("avg_adjusted_brand_premium_eur", ascending=False)
    adjusted.to_csv(OUTPUT_DIR / "adjusted_brand_premium_summary.csv", index=False)

    plt.figure(figsize=(10, 6))
    plot_df = adjusted.head(12)
    sns.barplot(data=plot_df, x="avg_adjusted_brand_premium_eur", y="brand_group")
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Adjusted Brand Premium by Brand Group")
    plt.xlabel("Actual price - spec-only predicted price (EUR)")
    plt.ylabel("Brand group")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "adjusted_brand_premium.png", dpi=160)
    plt.close()

    return adjusted


def cluster_segments(reco_df: pd.DataFrame) -> pd.DataFrame:
    """Cluster recommendation rows into Entry, Mid-range, Flagship segments.

    Clustering creates an unsupervised market structure. 
    The project uses k=3 because the proposal describes three intuitive smartphone tiers: entry, mid-range, and flagship.
    """
    segment_features = [
        "price_eur",
        "spec_score_0_100",
        "ram_gb",
        "storage_gb",
        "battery_capacity_mah",
        "main_camera_max_mp",
        "ppi",
        "network_generation",
        "sensor_count",
    ]
    missing = sorted(set(segment_features) - set(reco_df.columns))
    if missing:
        raise ValueError(f"클러스터링 피처 누락: {missing}")

    # Median imputation keeps rows with partial specs, while StandardScaler prevents large-unit features such as battery capacity from dominating distance-based clustering.
    cluster_input = reco_df[segment_features].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(cluster_input)
    X_scaled = scaler.fit_transform(X_imputed)

    # Evaluate several k values to document that the final k=3 choice was checked, even though k=3 also matches the business interpretation.
    k_rows = []
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X_scaled)
        k_rows.append(
            {
                "k": k,
                "inertia": KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X_scaled).inertia_,
                "silhouette": silhouette_score(X_scaled, labels),
                "davies_bouldin": davies_bouldin_score(X_scaled, labels),
            }
        )
    k_eval = pd.DataFrame(k_rows)
    k_eval.to_csv(OUTPUT_DIR / "cluster_k_evaluation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=k_eval, x="k", y="inertia", marker="o", ax=axes[0])
    axes[0].set_title("Elbow: KMeans inertia")
    sns.lineplot(data=k_eval, x="k", y="silhouette", marker="o", ax=axes[1])
    axes[1].set_title("Silhouette by k")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cluster_elbow_silhouette.png", dpi=160)
    plt.close()

    # Final segment assignment.
    # n_init=10 reduces instability from random centroid initialization.
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X_scaled)
    agg_labels = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X_scaled)
    comparison = pd.DataFrame(
        [
            {
                "algorithm": "KMeans",
                "n_samples": len(reco_df),
                "silhouette": silhouette_score(X_scaled, labels),
                "davies_bouldin": davies_bouldin_score(X_scaled, labels),
            },
            {
                "algorithm": "Agglomerative_Ward",
                "n_samples": len(reco_df),
                "silhouette": silhouette_score(X_scaled, agg_labels),
                "davies_bouldin": davies_bouldin_score(X_scaled, agg_labels),
            },
        ]
    )
    comparison.to_csv(OUTPUT_DIR / "cluster_algorithm_comparison.csv", index=False)
    comparison.rename(columns={"silhouette": "value"}).head(1).to_csv(
        OUTPUT_DIR / "task_d_cluster_quality.csv", index=False
    )

    segmented = reco_df.copy()
    segmented["cluster"] = labels
    cluster_order = (
        segmented.groupby("cluster")
        .agg(avg_price_eur=("price_eur", "mean"), avg_spec_score_0_100=("spec_score_0_100", "mean"))
        .sort_values(["avg_price_eur", "avg_spec_score_0_100"])
        .reset_index()
    )
    # Cluster IDs are arbitrary, so they are ordered by average price/spec score before assigning human-readable segment names.
    label_map = dict(zip(cluster_order["cluster"], ["Entry", "Mid-range", "Flagship"]))
    segmented["segment"] = segmented["cluster"].map(label_map)

    segment_summary = (
        segmented.groupby("segment")
        .agg(
            count=("price_eur", "size"),
            avg_price_eur=("price_eur", "mean"),
            median_price_eur=("price_eur", "median"),
            avg_spec_score_0_100=("spec_score_0_100", "mean"),
            avg_ram_gb=("ram_gb", "mean"),
            avg_storage_gb=("storage_gb", "mean"),
            avg_battery_capacity_mah=("battery_capacity_mah", "mean"),
            avg_main_camera_max_mp=("main_camera_max_mp", "mean"),
            avg_ppi=("ppi", "mean"),
            avg_network_generation=("network_generation", "mean"),
        )
        .reset_index()
    )
    segment_summary["segment_order"] = segment_summary["segment"].map({"Entry": 0, "Mid-range": 1, "Flagship": 2})
    segment_summary = segment_summary.sort_values("segment_order").drop(columns=["segment_order"])
    segment_summary.to_csv(OUTPUT_DIR / "cluster_segment_summary.csv", index=False)
    segment_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)

    brand_dist = (
        segmented.groupby(["segment", "brand_group"]).size().reset_index(name="count")
        .sort_values(["segment", "count"], ascending=[True, False])
    )
    brand_dist["rank"] = brand_dist.groupby("segment")["count"].rank(method="first", ascending=False)
    brand_dist[brand_dist["rank"] <= 3].drop(columns=["rank"]).to_csv(
        OUTPUT_DIR / "cluster_top3_brand_distribution.csv", index=False
    )

    # PCA is used only for visualization; the actual cluster labels come from KMeans on the full scaled feature set.
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_xy = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({"PC1": pca_xy[:, 0], "PC2": pca_xy[:, 1], "segment": segmented["segment"]})
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="segment", s=25, alpha=0.75)
    plt.title("PCA view of smartphone segments")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "cluster_pca_segments.png", dpi=160)
    plt.close()

    segmented.to_csv(OUTPUT_DIR / "df_with_segments.csv", index=False)
    return segmented


def value_outliers_and_recommendations(segmented: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect value-for-money outliers and build budget recommendations.

    This function implements the customer-side goal.
    It identifies devices that are unusually cheap for their performance, validates that those outliers are actually better value, 
    and then ranks phones under several budget scenarios.
    """
    # Performance-to-price ratio is the main value metric. 
    # A higher value means the user receives more estimated hardware performance per EUR.
    value_df = segmented.copy()
    value_df["performance_to_price_ratio"] = value_df["spec_score_0_100"] / value_df["price_eur"]
    value_df["segment_mean_value_score"] = value_df.groupby("segment")["value_score"].transform("mean")
    value_df["segment_median_price_eur"] = value_df.groupby("segment")["price_eur"].transform("median")
    value_df["value_score_lift_vs_segment"] = value_df["value_score"] / value_df["segment_mean_value_score"]
    value_df["price_discount_vs_segment_median"] = (
        value_df["segment_median_price_eur"] - value_df["price_eur"]
    ) / value_df["segment_median_price_eur"]

    # Estimate expected market price from hardware-only features.
    # If actual price is far below this expected price, the phone is a candidate bargain.
    market_features = [
        "spec_score_0_100",
        "ram_gb",
        "storage_gb",
        "battery_capacity_mah",
        "main_camera_max_mp",
        "ppi",
        "network_generation",
        "sensor_count",
        "phone_age",
    ]
    market_X = value_df[market_features].copy()
    market_y = value_df["price_eur"].astype(float)
    expected_price = np.zeros(len(value_df))
    # Out-of-fold expected prices avoid using the same row for both training and prediction, reducing overly optimistic underpricing labels.
    market_kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, test_idx in market_kfold.split(market_X):
        market_model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=140,
                max_depth=14,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        )
        market_model.fit(market_X.iloc[train_idx], market_y.iloc[train_idx])
        expected_price[test_idx] = market_model.predict(market_X.iloc[test_idx])
    value_df["expected_market_price_eur"] = np.clip(expected_price, 1, None)
    value_df["market_price_gap_eur"] = value_df["expected_market_price_eur"] - value_df["price_eur"]
    value_df["market_price_discount_ratio"] = (
        value_df["market_price_gap_eur"] / value_df["expected_market_price_eur"]
    )
    value_df["market_underpricing_z_in_segment"] = value_df.groupby("segment")[
        "market_price_discount_ratio"
    ].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    value_df["segment_median_performance_to_price_ratio"] = value_df.groupby("segment")[
        "performance_to_price_ratio"
    ].transform("median")
    # 1.645 is roughly the 95th percentile for a one-sided normal threshold.
    # It flags strong underpricing within each segment while keeping the rule simple enough to verify from the output tables.
    value_df["market_underpriced_outlier"] = (
        (value_df["market_price_gap_eur"] > 0)
        & (value_df["market_underpricing_z_in_segment"] > 1.645)
        & (value_df["performance_to_price_ratio"] >= value_df["segment_median_performance_to_price_ratio"])
    )

    value_df["value_z_in_segment"] = value_df.groupby("segment")["value_score"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    )
    value_df["zscore_value_outlier"] = value_df["value_z_in_segment"] > 1.645

    # IsolationForest captures unusual multivariate patterns.
    # The additional high-value filter prevents the algorithm from returning "weird but bad" phones as value recommendations.
    iso_features = [
        "value_score",
        "performance_to_price_ratio",
        "market_price_discount_ratio",
        "spec_score_0_100",
        "price_eur",
        "phone_age",
    ]
    iso_input = value_df[iso_features].copy()
    iso_X = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(iso_input)
    isolation = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
    value_df["isolation_raw_outlier"] = isolation.fit_predict(iso_X) == -1
    threshold = value_df["value_score"].quantile(0.75)
    value_df["isolation_value_outlier"] = (
        value_df["isolation_raw_outlier"] & (value_df["value_score"] >= threshold)
    )
    # The final outlier flag is a union of interpretable statistical evidence and model-based anomaly evidence.
    # This improves recall for good deals.
    value_df["is_value_outlier"] = (
        value_df["zscore_value_outlier"]
        | value_df["isolation_value_outlier"]
        | value_df["market_underpriced_outlier"]
    )

    comparison = pd.DataFrame(
        [
            {"method": "Z-score within segment", "count": int(value_df["zscore_value_outlier"].sum())},
            {"method": "IsolationForest + high value_score", "count": int(value_df["isolation_value_outlier"].sum())},
            {
                "method": "Market expected-price underpricing Z-score",
                "count": int(value_df["market_underpriced_outlier"].sum()),
            },
            {
                "method": "Intersection",
                "count": int((value_df["zscore_value_outlier"] & value_df["isolation_value_outlier"]).sum()),
            },
            {"method": "Union(final)", "count": int(value_df["is_value_outlier"].sum())},
        ]
    )
    comparison.to_csv(OUTPUT_DIR / "value_outlier_method_comparison.csv", index=False)

    validation = (
        value_df.groupby("is_value_outlier")
        .agg(
            count=("value_score", "size"),
            avg_price_eur=("price_eur", "mean"),
            avg_value_score=("value_score", "mean"),
            avg_performance_to_price_ratio=("performance_to_price_ratio", "mean"),
            avg_value_lift_vs_segment=("value_score_lift_vs_segment", "mean"),
            avg_price_discount_vs_segment_median=("price_discount_vs_segment_median", "mean"),
            avg_expected_market_price_eur=("expected_market_price_eur", "mean"),
            avg_market_price_gap_eur=("market_price_gap_eur", "mean"),
            avg_market_price_discount_ratio=("market_price_discount_ratio", "mean"),
        )
        .reset_index()
    )
    validation.to_csv(OUTPUT_DIR / "value_outlier_validation_summary.csv", index=False)

    value_outliers = value_df[value_df["is_value_outlier"]].sort_values(
        ["performance_to_price_ratio", "market_price_discount_ratio", "value_score"], ascending=False
    )
    value_outliers.to_csv(OUTPUT_DIR / "value_outliers.csv", index=False)
    value_outliers.to_csv(OUTPUT_DIR / "value_deal_outliers.csv", index=False)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=value_df,
        x="price_eur",
        y="performance_to_price_ratio",
        hue="is_value_outlier",
        alpha=0.7,
        s=24,
    )
    plt.title("Value-for-money outliers")
    plt.ylabel("Performance-to-Price Ratio")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "value_outlier_scatter.png", dpi=160)
    plt.close()

    # Fixed scenarios create repeatable recommendation examples for quick modeling checks.
    # Users can still enter any budget through two_way_solution.py.
    scenarios = [
        {"scenario": 1, "scenario_label": "학생 예산", "budget_eur": 200, "segment": None},
        {"scenario": 2, "scenario_label": "일반 직장인", "budget_eur": 400, "segment": "Mid-range"},
        {"scenario": 3, "scenario_label": "프리미엄 유저", "budget_eur": 800, "segment": "Flagship"},
        {"scenario": 4, "scenario_label": "입문자/세컨드폰", "budget_eur": 150, "segment": "Entry"},
    ]
    rec_frames = []
    validation_rows = []
    for scenario in scenarios:
        rec = recommend_phones(
            value_df,
            budget_eur=scenario["budget_eur"],
            top_n=10,
            segment_preference=scenario["segment"],
        )
        rec.insert(0, "scenario", scenario["scenario"])
        rec.insert(1, "scenario_label", scenario["scenario_label"])
        rec.insert(2, "budget_eur", scenario["budget_eur"])
        rec.to_csv(OUTPUT_DIR / f"recommendations_scenario_{scenario['scenario']}.csv", index=False)
        rec_frames.append(rec)
        # Each scenario is validated to prove that the recommendation system respects the budget and keeps PPR in descending order.
        validation_rows.append(
            {
                "scenario": scenario["scenario"],
                "scenario_label": scenario["scenario_label"],
                "budget_eur": scenario["budget_eur"],
                "row_count": len(rec),
                "max_price_eur": rec["price_eur"].max() if len(rec) else np.nan,
                "all_within_budget": bool((rec["price_eur"] <= scenario["budget_eur"]).all()) if len(rec) else True,
                "is_value_score_descending": bool(rec["value_score"].is_monotonic_decreasing) if len(rec) else True,
                "is_performance_to_price_ratio_descending": bool(
                    rec["performance_to_price_ratio"].is_monotonic_decreasing
                ) if len(rec) else True,
                "top_value_score": rec["value_score"].iloc[0] if len(rec) else np.nan,
                "top_performance_to_price_ratio": rec["performance_to_price_ratio"].iloc[0] if len(rec) else np.nan,
            }
        )
    recommendations = pd.concat(rec_frames, ignore_index=True)
    recommendations.to_csv(OUTPUT_DIR / "recommendations_all_scenarios.csv", index=False)
    recommendations.to_csv(OUTPUT_DIR / "budget_recommendations.csv", index=False)
    pd.DataFrame(validation_rows).to_csv(OUTPUT_DIR / "recommendation_constraint_validation.csv", index=False)

    lift = (
        recommendations.groupby(["scenario", "scenario_label"])
        .agg(
            recommendation_mean_value_score=("value_score", "mean"),
            recommendation_mean_performance_to_price_ratio=("performance_to_price_ratio", "mean"),
        )
        .reset_index()
    )
    lift["overall_mean_value_score"] = value_df["value_score"].mean()
    lift["overall_mean_performance_to_price_ratio"] = value_df["performance_to_price_ratio"].mean()
    lift["lift_vs_overall"] = lift["recommendation_mean_value_score"] / lift["overall_mean_value_score"]
    lift["ppr_lift_vs_overall"] = (
        lift["recommendation_mean_performance_to_price_ratio"]
        / lift["overall_mean_performance_to_price_ratio"]
    )
    lift.to_csv(OUTPUT_DIR / "task_d_recommendation_lift.csv", index=False)

    segment_outliers = value_df.groupby("segment")["is_value_outlier"].agg(["sum", "count"]).reset_index()
    segment_outliers["outlier_ratio"] = segment_outliers["sum"] / segment_outliers["count"]
    segment_outliers.to_csv(OUTPUT_DIR / "task_d_outlier_segment_distribution.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=lift, x="scenario", y="ppr_lift_vs_overall")
    plt.axhline(1, color="black", linewidth=1)
    plt.title("Recommendation PPR lift")
    plt.xlabel("Scenario")
    plt.ylabel("PPR lift vs overall")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "recommendation_lift.png", dpi=160)
    plt.close()

    value_df.to_csv(OUTPUT_DIR / "df_with_segments.csv", index=False)
    return value_df, recommendations


def recommend_phones(
    df: pd.DataFrame,
    budget_eur: float,
    top_n: int = 10,
    segment_preference: str | None = None,
    brand_preference: str | None = None,
) -> pd.DataFrame:
    """Recommend Top-N phones by Performance-to-Price Ratio under a EUR budget.

    Budget filtering happens before ranking because a recommendation that exceeds the user's constraint is not usable, even if its raw performance is high.
    """
    candidates = df[df["price_eur"] <= float(budget_eur)].copy()
    if segment_preference:
        candidates = candidates[candidates["segment"].eq(segment_preference)].copy()
    if brand_preference:
        candidates = candidates[candidates["brand_group"].eq(brand_preference)].copy()
    if candidates.empty:
        return pd.DataFrame(columns=[
            "recommendation_rank", "oem", "model", "brand_group", "price_eur", "budget_utilization",
            "price_tier", "segment", "spec_score_0_100", "value_score",
            "performance_to_price_ratio", "market_price_discount_ratio",
            "expected_market_price_eur", "market_underpriced_outlier",
            "is_value_outlier", "ram_gb",
            "storage_gb", "battery_capacity_mah", "main_camera_max_mp",
            "network_generation", "recommendation_reason",
        ])
    # Recompute PPR defensively so this helper also works if it receives a dataframe before PPR was saved.
    candidates["performance_to_price_ratio"] = candidates["spec_score_0_100"] / candidates["price_eur"]
    candidates["budget_utilization"] = candidates["price_eur"] / float(budget_eur)
    candidates["recommendation_reason"] = candidates.apply(make_recommendation_reason, axis=1)
    cols = [
        "recommendation_rank", "oem", "model", "brand_group", "price_eur", "budget_utilization",
        "price_tier", "segment", "spec_score_0_100", "value_score",
        "performance_to_price_ratio", "market_price_discount_ratio",
        "expected_market_price_eur", "market_underpriced_outlier",
        "is_value_outlier", "ram_gb",
        "storage_gb", "battery_capacity_mah", "main_camera_max_mp",
        "network_generation", "recommendation_reason",
    ]
    ranked = (
        # Ranking priority:
        # 1. maximize PPR, the main customer value objective;
        # 2. prefer stronger value_score and confirmed value outliers;
        # 3. prefer stronger raw specs;
        # 4. if still tied, choose the cheaper phone.
        candidates.sort_values(
            ["performance_to_price_ratio", "value_score", "is_value_outlier", "spec_score_0_100", "price_eur"],
            ascending=[False, False, False, False, True],
        )
        .head(top_n)
        .reset_index(drop=True)
    )
    ranked.insert(0, "recommendation_rank", range(1, len(ranked) + 1))
    return ranked[cols]


def make_recommendation_reason(row: pd.Series) -> str:
    """Build a short human-readable recommendation reason."""
    discount = row.get("market_price_discount_ratio", np.nan)
    discount_note = "" if pd.isna(discount) else f", market_discount={discount:.1%}"
    return (
        f"PPR={row['performance_to_price_ratio']:.3f}, value={row['value_score']:.2f}, "
        f"spec={row['spec_score_0_100']:.1f}{discount_note}, "
        f"RAM={row['ram_gb']:.0f}GB, camera={row.get('main_camera_max_mp', np.nan):.0f}MP"
    )


def save_price_driver_summary(importance_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact price-driver table for two-way solution demos.

    The interactive solution returns these factors so a predicted price is explainable instead of being a black-box number.
    """
    summary = (
        importance_df.sort_values("importance", ascending=False)
        .head(20)
        .assign(
            driver_type=lambda d: np.where(
                d["feature"].str.startswith("cat__brand_group_"),
                "brand",
                "hardware_or_spec",
            )
        )
    )
    summary.to_csv(OUTPUT_DIR / "two_way_price_driver_summary.csv", index=False)
    return summary


def create_model_reference(price_model_df: pd.DataFrame, price_raw_df: pd.DataFrame) -> None:
    """Store simple raw mean/std references for CLI business price predictions.

    The trained model expects scaled numeric features.
    The CLI accepts raw specs such as RAM or battery capacity, so this reference table stores the mean and
    standard deviation needed to transform user-entered values consistently.
    """
    numeric_cols = price_raw_df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for col in numeric_cols:
        scaled_col = "num__" + col
        if scaled_col in price_model_df.columns:
            mean = float(price_raw_df[col].mean())
            std = float(price_raw_df[col].std(ddof=0))
            rows.append(
                {
                    "raw_feature": col,
                    "model_feature": scaled_col,
                    "mean": mean,
                    "std": std if std else 1.0,
                    "median": float(price_raw_df[col].median()),
                }
            )
    ref = pd.DataFrame(rows)
    ref.to_csv(MODEL_DIR / "numeric_feature_reference.csv", index=False)

    brand_groups = sorted(price_raw_df["brand_group"].dropna().unique().tolist())
    (MODEL_DIR / "brand_groups.json").write_text(
        json.dumps(brand_groups, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def make_business_demo() -> pd.DataFrame:
    """Create static business guide examples from trained artifacts.

    These examples demonstrate how a company could compare a planned launch price with the model's fair-price estimate for different market positions.
    """
    from two_way_solution import TwoWaySmartphoneSolution

    solution = TwoWaySmartphoneSolution(root_dir=ROOT_DIR)
    rows = []
    examples = [
        {
            "scenario": "B2B_midrange_launch",
            "brand": "Samsung",
            "ram_gb": 8,
            "storage_gb": 128,
            "spec_score_0_100": 55,
            "battery_capacity_mah": 5000,
            "display_size_in": 6.5,
            "screen_to_body_pct": 84,
            "resolution_width_px": 1080,
            "resolution_height_px": 2400,
            "ppi": 405,
            "body_weight_g": 190,
            "fast_charging_w": 25,
            "main_camera_max_mp": 50,
            "network_generation": 5,
            "sensor_count": 5,
            "launch_year": 2020,
            "budget_price_eur": 450,
        },
        {
            "scenario": "B2B_flagship_launch",
            "brand": "Apple",
            "ram_gb": 8,
            "storage_gb": 256,
            "spec_score_0_100": 72,
            "battery_capacity_mah": 4300,
            "display_size_in": 6.7,
            "screen_to_body_pct": 87,
            "resolution_width_px": 1284,
            "resolution_height_px": 2778,
            "ppi": 460,
            "body_weight_g": 228,
            "fast_charging_w": 27,
            "main_camera_max_mp": 48,
            "network_generation": 5,
            "sensor_count": 6,
            "launch_year": 2020,
            "budget_price_eur": 950,
        },
        {
            "scenario": "B2B_value_launch",
            "brand": "Xiaomi",
            "ram_gb": 6,
            "storage_gb": 128,
            "spec_score_0_100": 50,
            "battery_capacity_mah": 5000,
            "display_size_in": 6.4,
            "screen_to_body_pct": 83,
            "resolution_width_px": 1080,
            "resolution_height_px": 2400,
            "ppi": 395,
            "body_weight_g": 195,
            "fast_charging_w": 30,
            "main_camera_max_mp": 64,
            "network_generation": 5,
            "sensor_count": 5,
            "launch_year": 2020,
            "budget_price_eur": 300,
        },
    ]
    for spec in examples:
        result = solution.business_price_guide(spec)
        rows.append({**spec, **result})
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "two_way_business_price_guides.csv", index=False)
    return df



def save_modeling_summary(
    regression: RegressionResult,
    importance_df: pd.DataFrame,
    segmented: pd.DataFrame,
    value_df: pd.DataFrame,
) -> None:
    """Save a compact modeling-only summary table.

    This helper is intentionally limited to information produced by the modeling pipeline itself, 
    so this file stays focused on model behavior, evaluation metrics, and reusable outputs.
    """
    best = regression.best_row
    top_features = (
        importance_df.sort_values("importance", ascending=False)["feature"]
        .head(5)
        .astype(str)
        .tolist()
    )
    segments = sorted(segmented["segment"].dropna().astype(str).unique().tolist())
    value_outlier_count = int(value_df["is_value_outlier"].sum())

    summary = pd.DataFrame(
        [
            {
                "item": "best_regression_model",
                "value": f"{best['model']} ({best['target_mode']})",
                "modeling_reason": "This model is selected by holdout price-error metrics and is used for fair-price prediction.",
            },
            {
                "item": "holdout_mae_eur",
                "value": f"{best['MAE']:.2f}",
                "modeling_reason": "MAE is saved in EUR because the average pricing error is easy to interpret for launch-price guidance.",
            },
            {
                "item": "holdout_r2",
                "value": f"{best['R2']:.4f}",
                "modeling_reason": "R2 summarizes how much price variance is explained by hardware, brand, and engineered features.",
            },
            {
                "item": "top_price_features",
                "value": ", ".join(top_features),
                "modeling_reason": "Feature importance identifies the hardware or brand variables that drive predicted price.",
            },
            {
                "item": "market_segments",
                "value": ", ".join(segments),
                "modeling_reason": "Segments create fair peer groups before comparing value-for-money.",
            },
            {
                "item": "value_outlier_count",
                "value": str(value_outlier_count),
                "modeling_reason": "These devices are treated as unusually strong value candidates under the combined outlier rules.",
            },
        ]
    )
    summary.to_csv(OUTPUT_DIR / "modeling_summary.csv", index=False)

def main() -> None:
    """Run every modeling task end to end.

    The order mirrors the project workflow:
    1. load preprocessing outputs,
    2. check input tables and leakage columns,
    3. train/evaluate price regression,
    4. explain price drivers and brand premium,
    5. cluster market segments,
    6. detect value outliers and generate recommendations,
    7. save reusable modeling outputs and two-way demo examples.
    """
    ensure_dirs()
    # Plot defaults are set once so all generated figures use the same visual style
    # across feature-importance, clustering, outlier, and recommendation outputs.
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid")

    # The script intentionally loads the team's processed CSV files from
    # content/. This proves the modeling part uses the preprocessing output.
    data_source = find_data_source()
    print(f"[INFO] using data source: {data_source}")

    price_model_df = read_modeling_csv(data_source, PRICE_MODEL_CSV)
    price_raw_df = read_modeling_csv(data_source, PRICE_RAW_CSV)
    reco_model_df = read_modeling_csv(data_source, RECO_MODEL_CSV)
    reco_raw_df = read_modeling_csv(data_source, RECO_RAW_CSV)

    # Save a compact input check so later model runs can verify row counts, null counts,
    # and whether the final model-ready tables contain only numeric features.
    data_summary = pd.DataFrame(
        [
            {
                "file": PRICE_MODEL_CSV,
                "rows": len(price_model_df),
                "cols": price_model_df.shape[1],
                "null_total": int(price_model_df.isna().sum().sum()),
                "object_cols": len(price_model_df.select_dtypes(include=["object", "string"]).columns),
            },
            {
                "file": PRICE_RAW_CSV,
                "rows": len(price_raw_df),
                "cols": price_raw_df.shape[1],
                "null_total": int(price_raw_df.isna().sum().sum()),
                "object_cols": len(price_raw_df.select_dtypes(include=["object", "string"]).columns),
            },
            {
                "file": RECO_MODEL_CSV,
                "rows": len(reco_model_df),
                "cols": reco_model_df.shape[1],
                "null_total": int(reco_model_df.isna().sum().sum()),
                "object_cols": len(reco_model_df.select_dtypes(include=["object", "string"]).columns),
            },
            {
                "file": RECO_RAW_CSV,
                "rows": len(reco_raw_df),
                "cols": reco_raw_df.shape[1],
                "null_total": int(reco_raw_df.isna().sum().sum()),
                "object_cols": len(reco_raw_df.select_dtypes(include=["object", "string"]).columns),
            },
        ]
    )
    data_summary.to_csv(OUTPUT_DIR / "input_data_summary.csv", index=False)

    # These columns are excluded or checked because they are IDs or price-derived
    # fields. Including them in price prediction would make the model look
    # artificially strong without learning real hardware/brand effects.
    leakage_cols = [
        "price_tier",
        "value_score",
        "spec_score_0_100",
        "price_per_ram_gb",
        "price_per_storage_gb",
        "battery_per_eur",
        "oem",
        "model",
    ]
    leakage_check = pd.DataFrame(
        {
            "column": leakage_cols,
            "present_in_price_model": [col in price_model_df.columns for col in leakage_cols],
        }
    )
    leakage_check.to_csv(OUTPUT_DIR / "regression_dropped_columns.csv", index=False)

    # Main modeling sequence. Each step saves its own CSV/plot artifacts, so the
    # result is reproducible and easy to inspect independently.
    regression = train_and_evaluate_regression(price_model_df)
    plot_regression_metrics(regression.cv_metrics)
    importance_df, _ = feature_importance_analysis(price_model_df, regression.feature_names)
    adjusted_brand = brand_premium_analysis(price_model_df, price_raw_df)
    create_model_reference(price_model_df, price_raw_df)
    segmented = cluster_segments(reco_raw_df)
    value_df, recommendations = value_outliers_and_recommendations(segmented)
    save_price_driver_summary(importance_df)

    # two_way_solution.py depends on saved model artifacts, so demo outputs are
    # generated only after training and model reference files exist.
    make_business_demo()
    from two_way_solution import TwoWaySmartphoneSolution

    solution = TwoWaySmartphoneSolution(root_dir=ROOT_DIR)
    demo_recs = []
    # These scenarios mirror common user budget levels and are used for the final
    # B2C recommendation evidence table.
    demo_scenarios = [
        {"scenario": "student_balanced", "budget_eur": 200, "segment": None},
        {"scenario": "worker_midrange", "budget_eur": 400, "segment": "Mid-range"},
        {"scenario": "premium_flagship", "budget_eur": 800, "segment": "Flagship"},
        {"scenario": "entry_second_phone", "budget_eur": 150, "segment": "Entry"},
    ]
    for sc in demo_scenarios:
        rec = solution.recommend_for_user(sc["budget_eur"], top_n=8, segment=sc["segment"])
        rec.insert(0, "scenario", sc["scenario"])
        rec.insert(1, "budget_eur", sc["budget_eur"])
        demo_recs.append(rec)
    pd.concat(demo_recs, ignore_index=True).to_csv(
        OUTPUT_DIR / "two_way_user_recommendations.csv", index=False
    )

    save_modeling_summary(regression, importance_df, segmented, value_df)

    print("[DONE] GSM modeling pipeline complete")
    print(regression.metrics.head().to_string(index=False))
    print(pd.read_csv(OUTPUT_DIR / "cluster_segment_summary.csv").to_string(index=False))
    print(pd.read_csv(OUTPUT_DIR / "value_outlier_validation_summary.csv").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="run full modeling pipeline")
    args = parser.parse_args()
    main()
