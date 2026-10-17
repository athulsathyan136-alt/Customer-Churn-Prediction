
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_clean.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "customer_churn_pipeline.joblib"


def main():
    data = pd.read_csv(DATA_PATH)

    # Remove the customer identifier
    data = data.drop(columns=["customerID"])

    # Separate target and original customer features
    y = data["Churn"].map({"No": 0, "Yes": 1})
    X = data.drop(columns=["Churn"])

    # Identify numeric and categorical columns
    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Prepare numeric features
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Prepare categorical features
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Combine preprocessing
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

    # Build one complete pipeline
    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    # Split before fitting preprocessing to prevent leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("===== CUSTOMER CHURN PIPELINE =====")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Stayed", "Churned"],
            zero_division=0,
        )
    )

    # Save the entire pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_names": X.columns.tolist(),
            "model_name": "Logistic Regression Pipeline",
        },
        MODEL_PATH,
    )

    print("Saved pipeline:", MODEL_PATH)
    print("File exists:", MODEL_PATH.is_file())


if __name__ == "__main__":
    main()
