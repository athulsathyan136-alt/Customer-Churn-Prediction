
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (PROJECT_ROOT/"data"/"processed"/"customer_churn_features.csv")

MODEL_DIR = PROJECT_ROOT/"models"

MODEL_PATH = MODEL_DIR/"customer_churn_model.joblib"


def main():

    data = pd.read_csv(DATA_PATH)

    X = data.drop(columns=["Churn"])
    y =data["Churn"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("===== DATA SPLIT =====")
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=1
    )

    model.fit(X_train,y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:,1]

    print("\n===== MODEL EVALUATION =====")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

    print("\nClassification report:")
    print(classification_report(y_test,predictions,target_names=["Stayed (0)","Churned (1)"],
                                zero_division=0,))

    print("Confusion Matrix: ")
    print(confusion_matrix(y_test,predictions))

    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    joblib.dump({"model":model,"feature_names":X.columns.tolist(),},MODEL_PATH,)

    print("\nModel saved to:", MODEL_PATH)

if __name__ == "__main__":
    main()