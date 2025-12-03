# train_model.py

import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

import joblib

DATA_PATH = Path("data") / "framingham.csv"  
MODEL_PATH = Path("models") / "risk_model.joblib"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


# Columns in  CSV:
# male,age,education,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,
# prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose,TenYearCHD

TARGET_COL = "TenYearCHD"

FEATURE_COLS = [
    "male",
    "age",
    "education",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def train():
    print(f"Loading data from: {DATA_PATH}")
    df = load_data(DATA_PATH)

    # Quick sanity check
    print("Data shape:", df.shape)
    print("First few rows:")
    print(df.head())

    # Split X / y
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Define a simple numeric pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    solver="lbfgs",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training rows:", X_train.shape[0], "Test rows:", X_test.shape[0])

    # Fit model
    numeric_pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = numeric_pipeline.predict(X_test)
    y_proba = numeric_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("\n=== Evaluation ===")
    print("Accuracy:", round(acc, 4))
    print("ROC AUC:", round(auc, 4))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Save the entire pipeline (imputer + scaler + model)
    joblib.dump(
        {
            "pipeline": numeric_pipeline,
            "feature_cols": FEATURE_COLS,
            "target_col": TARGET_COL,
        },
        MODEL_PATH,
    )
    print(f"\nSaved model pipeline to: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    train()
