"""
Download, prepare, train, evaluate, and export the credit-card default model.

Run from the project root:
    python train_model.py
"""

import importlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

try:
    fetch_ucirepo = importlib.import_module("ucimlrepo").fetch_ucirepo
except ModuleNotFoundError as exc:
    if exc.name != "ucimlrepo":
        raise
    fetch_ucirepo = None
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

TARGET = "default.payment.next.month"

# These are categorical codes in the original UCI dataset.
CATEGORICAL_FEATURES = [
    "SEX", "EDUCATION", "MARRIAGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "X2", "X3", "X4", "X6", "X7", "X8", "X9", "X10", "X11",
]

def main():
    if fetch_ucirepo is None:
        raise SystemExit(
            "Missing required dependency 'ucimlrepo'. Install project "
            "dependencies with `python -m pip install -r requirements.txt` "
            "using the same Python interpreter that runs this script."
        )

    print("1/8 Downloading the UCI dataset...")
    dataset = fetch_ucirepo(id=350)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    # Normalize the target name, regardless of the name returned by the API.
    y.columns = [TARGET]
    df = pd.concat([X, y], axis=1)

    print("2/8 Inspecting and preparing data...")
    df = df.drop_duplicates().copy()

    # Ensure target values are numeric and retain valid binary target rows.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)

    # Keep the feature names and order from the dataset.
    feature_columns = list(X.columns)
    categorical_features = [
        col for col in CATEGORICAL_FEATURES if col in feature_columns
    ]
    numerical_features = [
        col for col in feature_columns if col not in categorical_features
    ]

    # Save a prepared copy for inspection and coursework documentation.
    df.to_csv(DATA_DIR / "credit_card_default_clean.csv", index=False)

    X = df[feature_columns].copy()
    y = df[TARGET].copy()

    # Cast category codes to strings so OneHotEncoder treats them as categories.
    for column in categorical_features:
        X[column] = X[column].astype(str)

    print(f"   Dataset shape after basic cleaning: {df.shape}")
    print(f"   Missing values in target: {int(y.isna().sum())}")
    print(f"   Target distribution:\n{y.value_counts().sort_index()}")

    print("3/8 Splitting into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("4/8 Building preprocessing and model pipeline...")
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocess),
            ("classifier", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            )),
        ]
    )

    print("5/8 Training Logistic Regression...")
    pipeline.fit(X_train, y_train)

    print("6/8 Training Random Forest...")
    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocess)),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )
    random_forest_pipeline.fit(X_train, y_train)

    print("7/8 Training XGBoost...")
    xgboost_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocess)),
            ("classifier", XGBClassifier(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
            )),
        ]
    )
    xgboost_pipeline.fit(X_train, y_train)

    def evaluate(model):
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]
        return {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(y_test, predictions, zero_division=0)
            ),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
        }

    print("8/8 Evaluating and exporting models...")
    metrics = evaluate(pipeline)
    random_forest_metrics = evaluate(random_forest_pipeline)
    xgboost_metrics = evaluate(xgboost_pipeline)
    joblib.dump(pipeline, ARTIFACTS_DIR / "loan_default_pipeline.joblib")
    joblib.dump(
        xgboost_pipeline,
        ARTIFACTS_DIR / "xgboost_default_pipeline.joblib",
    )

    def comparison_row(name, values):
        return {
            "Model": name,
            "Accuracy": values["accuracy"],
            "Precision": values["precision"],
            "Recall": values["recall"],
            "F1": values["f1_score"],
            "ROC-AUC": values["roc_auc"],
        }

    comparison_rows = [
        comparison_row("Logistic Regression", metrics),
        comparison_row("Random Forest", random_forest_metrics),
        comparison_row("XGBoost", xgboost_metrics),
    ]
    pd.DataFrame(comparison_rows).to_csv(
        DATA_DIR / "model_comparison.csv",
        index=False,
    )

    info = {
        "model_name": "Logistic Regression with preprocessing pipeline",
        "dataset": "UCI Default of Credit Card Clients (ID 350)",
        "target": TARGET,
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "test_metrics": metrics,
        "random_forest_metrics": random_forest_metrics,
        "xgboost_metrics": xgboost_metrics,
        "test_size": 0.20,
        "random_state": 42,
        "classification_threshold": 0.50,
    }

    with open(ARTIFACTS_DIR / "model_info.json", "w", encoding="utf-8") as file:
        json.dump(info, file, indent=4)

    print("\nTraining complete!")
    print("Saved: artifacts/loan_default_pipeline.joblib")
    print("Saved: artifacts/xgboost_default_pipeline.joblib")
    print("Saved: artifacts/model_info.json")
    print("Saved: data/credit_card_default_clean.csv")
    print("Saved: data/model_comparison.csv")
    print("\nTest metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f}")
    print("\nXGBoost test metrics:")
    for name, value in xgboost_metrics.items():
        print(f"  {name}: {value:.3f}")
    print("\nRandom Forest test metrics:")
    for name, value in random_forest_metrics.items():
        print(f"  {name}: {value:.3f}")

if __name__ == "__main__":
    main()
