
"""Train and evaluate Titanic classification models.

Expected input: data/raw/train.csv from Kaggle's Titanic competition.
Run from the repository root: python -m src.train
"""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from src.features import add_features

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

TARGET = "Survived"
DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin", TARGET]


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download Kaggle Titanic train.csv and place it at data/raw/train.csv."
        )

    data = pd.read_csv(DATA_PATH)
    if TARGET not in data.columns:
        raise ValueError("train.csv must contain the target column 'Survived'.")

    print("Dataset shape:", data.shape)
    print("\nMissing values:\n", data.isna().sum().sort_values(ascending=False))
    print("\nTarget balance:\n", data[TARGET].value_counts(normalize=True).round(3))

    engineered = add_features(data)
    X = engineered.drop(columns=[c for c in DROP_COLUMNS if c in engineered.columns])
    y = engineered[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=5, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=3,
            random_state=42, class_weight="balanced"
        ),
    }

    fitted = {}
    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor(X_train)),
            ("classifier", estimator),
        ])
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        rows.append(evaluate_model(name, pipeline, X_test, y_test))
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        print(f"{name} 5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # A small, reproducible grid search for the Decision Tree.
    tree_pipeline = Pipeline([
        ("preprocessor", make_preprocessor(X_train)),
        ("classifier", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
    ])
    search = GridSearchCV(
        tree_pipeline,
        {
            "classifier__max_depth": [3, 4, 5, 6, None],
            "classifier__min_samples_leaf": [2, 5, 10],
        },
        cv=cv, scoring="roc_auc", n_jobs=-1
    )
    search.fit(X_train, y_train)
    fitted["Tuned Decision Tree"] = search.best_estimator_
    rows.append(evaluate_model("Tuned Decision Tree", search.best_estimator_, X_test, y_test))
    print("\nBest grid-search parameters:", search.best_params_)

    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics.to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    print("\nHold-out test metrics:\n", metrics.to_string(index=False))

    best_name = metrics.iloc[0]["model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, MODEL_DIR / "titanic_model.pkl")
    (MODEL_DIR / "best_model_name.txt").write_text(best_name, encoding="utf-8")

    # Confusion matrix and ROC curves for every fitted model.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, model in fitted.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[0], name=name)
    axes[0].set_title("ROC curves — hold-out test set")

    best_predictions = best_model.predict(X_test)
    sns.heatmap(
        confusion_matrix(y_test, best_predictions), annot=True, fmt="d",
        cmap="Blues", ax=axes[1], cbar=False
    )
    axes[1].set_title(f"Confusion matrix — {best_name}")
    axes[1].set_xlabel("Predicted label")
    axes[1].set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "evaluation_plots.png", dpi=160)
    plt.close(fig)

    # Save a readable report using the actual run's metrics.
    report_lines = [
        "# Model Evaluation Report", "",
        "Dataset: Kaggle Titanic training data. Hold-out split: 80% train / 20% test,",
        "stratified, random_state=42. Preprocessing is fitted inside each pipeline", 
        "to avoid fitting imputers, encoders, or scalers on the test set.", "",
        "## Hold-out test metrics", "",
        metrics.to_markdown(index=False), "",
        f"## Selected model", "",
        f"**{best_name}** was selected by the highest hold-out ROC-AUC in this run.", "",
        "## Interpretation and limitations", "",
        "- Accuracy alone can hide class-specific errors; precision, recall, F1 and ROC-AUC are also reported.",
        "- The dataset is historical and relatively small; results may not generalize to other populations or contexts.",
        "- Some columns have missing values, and imputation can introduce uncertainty.",
        "- Feature associations are not causal explanations.",
        "- The train/test result is one split; cross-validation on the training partition provides an additional stability check.",
        "",
        "## Reproducibility", "",
        "Run `python -m src.train` from the repository root after placing Kaggle `train.csv` in `data/raw/`.",
        "The selected fitted preprocessing + model pipeline is saved as `models/titanic_model.pkl`.",
    ]
    (REPORT_DIR / "EVALUATION_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\nClassification report for selected model:\n",
          classification_report(y_test, best_predictions, zero_division=0))
    print(f"\nSaved model: models/titanic_model.pkl")
    print(f"Selected model: {best_name}")


if __name__ == "__main__":
    main()
