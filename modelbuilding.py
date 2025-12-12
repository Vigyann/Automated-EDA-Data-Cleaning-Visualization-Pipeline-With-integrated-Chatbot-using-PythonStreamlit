"""
modelbuilding.py

Automatically detect whether the target column requires regression or classification,
train multiple models accordingly, evaluate them with appropriate metrics, and
plot a side-by-side bar chart comparing model performance.

This version is robust to multiple scikit-learn versions and avoids
one-hot explosion by encoding high-cardinality categorical columns differently.
"""

import argparse
import warnings
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


def detect_task_type(y: pd.Series, threshold_unique_ratio: float = 0.05, max_unique_for_classification: int = 20) -> str:
    """Decide whether task is 'regression' or 'classification' using heuristics."""
    y_nonull = y.dropna()
    n = len(y_nonull)
    if n == 0:
        raise ValueError("Target column is empty after dropping NA")

    unique_count = y_nonull.nunique()

    # object or categorical -> classification
    if pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y):
        return "classification"

    # integer dtype with few unique values -> classification
    if pd.api.types.is_integer_dtype(y):
        if unique_count <= max_unique_for_classification:
            return "classification"

    # numeric but low unique ratio -> classification (e.g., rating 1-5)
    if pd.api.types.is_numeric_dtype(y):
        if unique_count / n <= threshold_unique_ratio:
            return "classification"
        else:
            return "regression"

    # fallback -> classification
    return "classification"


def build_preprocessor(X: pd.DataFrame, max_ohe_cardinality: int = 50) -> ColumnTransformer:
    """
    Build a preprocessing ColumnTransformer.

    - numeric columns: median impute + StandardScaler
    - categorical columns:
        - low-cardinality (unique < max_ohe_cardinality): SimpleImputer + OneHotEncoder
        - high-cardinality: SimpleImputer + OrdinalEncoder (to avoid explosion)
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Split categorical into low-card and high-card
    low_card_cols = [c for c in categorical_cols if X[c].nunique() < max_ohe_cardinality]
    high_card_cols = [c for c in categorical_cols if c not in low_card_cols]

    # Numeric pipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # OneHotEncoder with compatibility for sklearn versions
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    low_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    # OrdinalEncoder for high-cardinality categorical cols (avoids expanding to many columns)
    # Try to use handle_unknown param if available; else fallback to default behavior.
    try:
        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    except TypeError:
        # older versions might not support handle_unknown, fallback to simple OrdinalEncoder
        ord_enc = OrdinalEncoder()

    high_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", ord_enc),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if low_card_cols:
        transformers.append(("low_card_cat", low_card_pipeline, low_card_cols))
    if high_card_cols:
        transformers.append(("high_card_cat", high_card_pipeline, high_card_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def get_models(task: str, random_state: int = 42) -> Dict[str, Any]:
    """Return a dict of model name -> estimator appropriate for the task."""
    if task == "regression":
        return {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=random_state),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            "SVR": SVR(),
        }
    else:
        # For classification, use probability-capable models where possible to compute ROC AUC
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000, solver="lbfgs"),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=random_state),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            "SVC": SVC(probability=True),
        }


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    task: str,
    cv: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    max_ohe_cardinality: int = 50,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """
    Train and evaluate models. Returns a DataFrame with metrics and dict of fitted pipelines.

    Metrics:
      - Regression: RMSE, MAE, R2
      - Classification: Accuracy, F1 (weighted), Precision (weighted), Recall (weighted), ROC AUC (if possible)
    """
    # Build preprocessor using X (so encoders see the correct cardinalities)
    preprocessor = build_preprocessor(X, max_ohe_cardinality=max_ohe_cardinality)
    models = get_models(task, random_state=random_state)

    # Stratify when classification and more than 1 class
    stratify_arg = y if (task == "classification" and len(y.unique()) > 1) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )

    results = []
    fitted_pipelines = {}

    # Choose CV splitter (not used currently for metrics but kept for extension)
    if task == "classification":
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state) if len(y_train.unique()) > 1 else KFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    for name, estimator in models.items():
        # Build pipeline
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("estimator", estimator),
        ])

        # Fit and store pipeline
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        # Predict on test
        y_pred = pipe.predict(X_test)

        metrics = {"model": name}

        if task == "regression":
            # RMSE: handle sklearn versions with/without 'squared' kwarg
            try:
                rmse = mean_squared_error(y_test, y_pred, squared=False)
            except TypeError:
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metrics.update({"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)})

        else:
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            metrics.update({"Accuracy": float(acc), "F1": float(f1), "Precision": float(prec), "Recall": float(rec)})

            # Compute ROC AUC when possible
            roc_val = np.nan
            try:
                estimator_step = pipe.named_steps["estimator"]
                if hasattr(estimator_step, "predict_proba"):
                    proba = pipe.predict_proba(X_test)
                    # binary
                    if proba.shape[1] == 2:
                        roc_val = roc_auc_score(y_test, proba[:, 1])
                    else:
                        # multiclass: binarize labels
                        classes = np.unique(y_test)
                        y_test_b = label_binarize(y_test, classes=classes)
                        # If label_binarize returns shape (n_samples, n_classes)
                        roc_val = roc_auc_score(y_test_b, proba, average="weighted", multi_class="ovr")
                elif hasattr(estimator_step, "decision_function"):
                    # Some classifiers (SVC) may have decision_function
                    try:
                        dec = pipe.decision_function(X_test)
                        # binary
                        if dec.ndim == 1:
                            roc_val = roc_auc_score(y_test, dec)
                        else:
                            classes = np.unique(y_test)
                            y_test_b = label_binarize(y_test, classes=classes)
                            roc_val = roc_auc_score(y_test_b, dec, average="weighted", multi_class="ovr")
                    except Exception:
                        roc_val = np.nan
            except Exception:
                roc_val = np.nan

            metrics["ROC_AUC"] = float(roc_val) if not (roc_val is None or (isinstance(roc_val, float) and np.isnan(roc_val))) else np.nan

        results.append(metrics)

    # Create results DataFrame
    results_df = pd.DataFrame(results).set_index("model")
    return results_df, fitted_pipelines


def plot_results(results_df: pd.DataFrame, task: str, output_path: str = "model_comparison.png") -> None:
    """Create a grouped bar chart comparing models across metrics and save to output_path."""
    plot_df = results_df.copy()

    # Determine metrics order
    if task == "regression":
        metric_order = [c for c in ["RMSE", "MAE", "R2"] if c in plot_df.columns]
    else:
        metric_order = [c for c in ["Accuracy", "F1", "Precision", "Recall", "ROC_AUC"] if c in plot_df.columns]

    if not metric_order:
        raise ValueError("No metrics available to plot")

    n_models = plot_df.shape[0]
    n_metrics = len(metric_order)

    fig, ax = plt.subplots(figsize=(1.8 * n_metrics + 1, 4 + 0.5 * n_models))

    x = np.arange(n_metrics)
    width = 0.8 / max(1, n_models)

    for i, (model_name, row) in enumerate(plot_df.iterrows()):
        values = [row[m] if pd.notnull(row[m]) else 0 for m in metric_order]
        positions = x - 0.4 + i * width + width / 2
        ax.bar(positions, values, width=width, label=model_name, edgecolor="black")
        # annotate
        for xi, val in zip(positions, values):
            if pd.notnull(val):
                try:
                    txt = f"{val:.3f}"
                except Exception:
                    txt = str(val)
                ax.text(xi, (val if val is not None else 0) + 0.02 * (max(values) if values else 1), txt, ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_order)
    ax.set_title(f"Model comparison ({task})")
    ax.legend(loc="best", bbox_to_anchor=(1.01, 1))
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison chart to: {output_path}")


def load_data(data_path: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Auto model builder for regression or classification")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--target", type=str, required=True, help="Name of target column")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds (default 5)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction (default 0.2)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=str, default="model_comparison.png", help="Output chart path")
    parser.add_argument("--max-ohe-card", type=int, default=50, help="Max unique values for one-hot encoding")

    args = parser.parse_args()

    X, y = load_data(args.data, args.target)
    task = detect_task_type(y)
    print(f"Detected task type: {task}")

    results_df, pipelines = evaluate_models(
        X, y, task, cv=args.cv, test_size=args.test_size, random_state=args.random_state,
        max_ohe_cardinality=args.max_ohe_card
    )

    print("\nEvaluation results:\n")
    print(results_df)

    plot_results(results_df, task, output_path=args.output)


if __name__ == "__main__":
    main()
