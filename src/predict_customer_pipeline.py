
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "customer_churn_pipeline.joblib"
)


def main():
    # Load the saved pipeline
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    package = joblib.load(MODEL_PATH)
    model = package["model"]
    feature_names = package["feature_names"]

    # Example customer using original dataset columns
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.00,
        "TotalCharges": 900.00,
    }

    # Create a complete input row with the expected columns
    customer_df = pd.DataFrame(
        [customer],
        columns=feature_names,
    )

    # Predict churn
    probability = model.predict_proba(customer_df)[0, 1]
    prediction = model.predict(customer_df)[0]

    print("===== CUSTOMER CHURN PREDICTION =====")
    print("Customer tenure:", customer["tenure"], "months")
    print("Contract:", customer["Contract"])
    print(f"Churn probability: {probability:.2%}")

    if prediction == 1:
        print("Prediction: Churn risk")
    else:
        print("Prediction: Likely to stay")


if __name__ == "__main__":
    main()
