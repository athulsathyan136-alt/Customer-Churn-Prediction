from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (PROJECT_ROOT/"data"/"processed"/"customer_churn_features.csv")

MODEL_DIR = PROJECT_ROOT /"models"

MODEL_PATH = MODEL_DIR/"best_churn_model.joblib"

def main():
    data = pd.read_csv(DATA_PATH)

    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        random_state=42,
        stratify=y,
        test_size=0.2


    )

    model = make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',random_state=42,),)

    model.fit(X_train,y_train)

    prediction = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:,1]
    print("===== FINAL MODEL EVALUATION =====")
    print("ROC-AUC: ",round(roc_auc_score(y_test,probabilities),4))
    print("Recall: ",round(recall_score(y_test,prediction),4))
    print("F1 Score: ",round(f1_score(y_test,prediction),4))

    MODEL_DIR.mkdir(parents=True,exist_ok=True)

    joblib.dump({
        "model":model,
        "feature_names":X.columns.tolist(),
        "model_name": "Scaled Logistic Regression",
    },MODEL_PATH,)

    print("\nModel saved to: ",MODEL_PATH)
    print("File exists: ",MODEL_PATH.is_file())

if __name__ == "__main__":
    main()

    