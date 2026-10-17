
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH  = (PROJECT_ROOT/"data"/"processed"/"customer_churn_features.csv")

def evaluate_model(name,model,X_train,X_test,y_train,y_test):

    print(f"======{name}==========")

    model.fit(X_train,y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:,1]

    results = {
        "Model":name,
        "Accuracy":accuracy_score(y_test,predictions),
        "Precision":precision_score(y_test,predictions,zero_division=0),
        "Recall":recall_score(y_test,predictions,zero_division=0),
        "F1 Score":f1_score(y_test,predictions,zero_division=0),
        "ROC-AUC":roc_auc_score(y_test,probabilities),
    }

    for metric ,values in results.items():
        if metric != "Model":
            print(f"{metric}:{values:.4f}")

    return results

def main():

    data = pd.read_csv(DATA_PATH)

    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    models = [
    (
        "Logistic Regression (Scaled)",
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ),
    (
        "Random Forest",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
        ),
    ),
]
    all_results = []

    for name ,model in models:
        results = evaluate_model(name,model,X_train,X_test,y_train,y_test)
        all_results.append(results)

    comparison = pd.DataFrame(all_results)

    comparison = comparison.sort_values(by="ROC-AUC",ascending=False,)

    print("\n===== MODEL COMPARISON =====")
    print(comparison.round(4).to_string(index=False))

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    output_path = reports_dir / "model_comparison.csv"
    comparison.to_csv(output_path, index=False)

    print("\nComparison saved to:", output_path)


if __name__ == "__main__":
    main()


          