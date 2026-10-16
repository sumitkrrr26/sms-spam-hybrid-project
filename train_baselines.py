from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, precision_score,
    recall_score, ConfusionMatrixDisplay,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from hybrid_utils import load_sms_dataset
from model_adapters import XGBoostLabelAdapter

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results" / "classical"


def model_definitions():
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=42),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, weights="distance"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=100, min_samples_leaf=2,
            class_weight="balanced", random_state=42,
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    }

    try:
        from xgboost import XGBClassifier  # noqa: F401
        models["XGBoost"] = XGBoostLabelAdapter(
            n_estimators=200, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    except ImportError:
        print("WARNING: xgboost is not installed; seven baseline models will run.")
        print("Install it with: pip install xgboost")

    return models


def make_pipeline(estimator):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
            min_df=2, max_df=0.98, sublinear_tf=True,
        )),
        ("classifier", estimator),
    ])


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_sms_dataset().reset_index(drop=True)
    print(f"SMS samples after cleaning: {len(df):,}")
    print(df["label"].value_counts())

    split_path = ROOT / "results" / "shared_split.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Shared split not found: {split_path}. Run create_shared_split.py first."
        )
    split = json.loads(split_path.read_text(encoding="utf-8"))

    if len(df) != split["dataset_size"]:
        raise ValueError(
            "Dataset size does not match results/shared_split.json. "
            "Recreate the shared split before training."
        )

    fit_idx = split["fit_indices"]
    val_idx = split["validation_indices"]
    test_idx = split["test_indices"]

    x_fit, y_fit = df.iloc[fit_idx]["text"], df.iloc[fit_idx]["label"]
    x_val, y_val = df.iloc[val_idx]["text"], df.iloc[val_idx]["label"]

    # Refit the selected baseline on fit + validation data, excluding test data.
    train_idx = fit_idx + val_idx
    x_train, y_train = df.iloc[train_idx]["text"], df.iloc[train_idx]["label"]
    x_test, y_test = df.iloc[test_idx]["text"], df.iloc[test_idx]["label"]

    print(f"Shared split: fit={len(x_fit)}, validation={len(x_val)}, test={len(x_test)}")

    candidates, rows = {}, []
    for name, estimator in model_definitions().items():
        print(f"\nTraining baseline: {name}")
        model = make_pipeline(estimator)

        start = time.perf_counter()
        model.fit(x_fit, y_fit)
        elapsed = time.perf_counter() - start

        val_pred = model.predict(x_val)
        metrics = {
            "model": name,
            "validation_accuracy": accuracy_score(y_val, val_pred),
            "validation_spam_precision": precision_score(
                y_val, val_pred, pos_label="spam", zero_division=0
            ),
            "validation_spam_recall": recall_score(
                y_val, val_pred, pos_label="spam", zero_division=0
            ),
            "validation_spam_f1": f1_score(
                y_val, val_pred, pos_label="spam", zero_division=0
            ),
            "fit_seconds": elapsed,
        }
        rows.append(metrics)
        candidates[name] = model
        print(f"validation spam F1={metrics['validation_spam_f1']:.4f}; accuracy={metrics['validation_accuracy']:.4f}")

    results = pd.DataFrame(rows).sort_values(
        ["validation_spam_f1", "validation_spam_precision", "validation_accuracy"],
        ascending=False,
    ).reset_index(drop=True)
    results.to_csv(RESULT_DIR / "benchmark_results.csv", index=False)

    best_name = str(results.iloc[0]["model"])
    best_model = candidates[best_name]
    print(f"\nSelected by validation Spam F1: {best_name}")
    print("Refitting selected model on fit + validation data...")
    best_model.fit(x_train, y_train)

    test_pred = best_model.predict(x_test)
    test_metrics = {
        "selected_model": best_name,
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_spam_precision": float(precision_score(
            y_test, test_pred, pos_label="spam", zero_division=0
        )),
        "test_spam_recall": float(recall_score(
            y_test, test_pred, pos_label="spam", zero_division=0
        )),
        "test_spam_f1": float(f1_score(
            y_test, test_pred, pos_label="spam", zero_division=0
        )),
        "test_weighted_f1": float(f1_score(
            y_test, test_pred, average="weighted", zero_division=0
        )),
    }

    joblib.dump(best_model, MODEL_DIR / "classical_best.joblib")
    (MODEL_DIR / "classical_info.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    (RESULT_DIR / "classification_report.txt").write_text(
        f"Selected by validation Spam F1: {best_name}\n\n"
        + classification_report(
            y_test, test_pred, labels=["ham", "spam"], digits=4, zero_division=0
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, test_pred, labels=["ham", "spam"],
        display_labels=["Ham", "Spam"], cmap="Blues", ax=ax, colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print("\nBaseline benchmark:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nFinal held-out test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print("Saved model to models/classical_best.joblib")


if __name__ == "__main__":
    main()
